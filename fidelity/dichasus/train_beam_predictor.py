"""
Train a beam predictor MLP on simulated channels, evaluate on real DICHASUS measurements.

For each captured config (.npz from run_beam_pred_experiment.py):
  - Compute best DFT beam index for each position from both sim and meas channels.
  - Train MLP: position -> sim_beam.
  - Evaluate on the held-out test split using meas_beam as ground truth (sim-to-real transfer).
  - Also train an "oracle" MLP on meas_beam to measure the ceiling.

Metrics:
  - top-1, top-3 accuracy against real labels
  - effective beam gain ratio |h_meas @ w_pred|^2 / |h_meas @ w_opt|^2

Usage:
  python train_beam_predictor.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "results" / "beam_pred"
OUT_DIR = BASE_DIR / "results" / "beam_pred"

# Fidelity metric per config (from all_results.json — computed on 30 different positions,
# but ranks the configs used here)
PWR_CORR_FROM_FIDELITY = {
    "calibrated":      0.290,
    "hw_tx_iso":       0.215,
    "rt_depth_3":      0.138,
    "mat_all_concrete":0.097,
    "hw_baseline":     0.087,
    "geo_full":        0.023,
    "mat_all_metal":   0.004,
}

N_ANT = 32           # use 32 antennas of array-1 (north)
N_BEAMS = 64         # 2x oversampled DFT codebook
HIDDEN = 256
EPOCHS = 300
PATIENCE = 30
BATCH = 64
LR = 1e-3
SEED = 42


def generate_dft_codebook(n_ant: int, n_beams: int) -> np.ndarray:
    k = np.arange(n_beams)[:, None]
    n = np.arange(n_ant)[None, :]
    return np.exp(1j * 2 * np.pi * k * n / n_beams) / np.sqrt(n_ant)


def compute_beam_labels(channels_narrowband: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """channels_narrowband: (N, n_ant) complex. Returns (N,) int."""
    power = np.abs(channels_narrowband @ codebook.conj().T) ** 2  # (N, n_beams)
    return np.argmax(power, axis=1)


def narrowband_from_cfr(channels: np.ndarray, n_ant: int, assign: np.ndarray) -> np.ndarray:
    """channels: (N, 64, 1024) complex. assign: indices into the 64 dim for the chosen array.
    Returns (N, n_ant) complex representing the average-subcarrier channel for beam selection.
    """
    # Pick the requested antennas (first n_ant of assignment)
    h = channels[:, assign[:n_ant], :]  # (N, n_ant, n_sub)
    # Narrowband representation: mean over subcarriers of complex amplitude
    # (equivalent to evaluating on the DC subcarrier after normalization)
    return h.mean(axis=-1)  # (N, n_ant) complex


class BeamPredictor(nn.Module):
    def __init__(self, n_in: int, n_beams: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_beams),
        )

    def forward(self, x):
        return self.net(x)


def make_loaders(X, y, tr_idx, va_idx, batch=BATCH):
    Xt = torch.from_numpy(X[tr_idx]).float()
    yt = torch.from_numpy(y[tr_idx]).long()
    Xv = torch.from_numpy(X[va_idx]).float()
    yv = torch.from_numpy(y[va_idx]).long()
    return (
        DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True),
        DataLoader(TensorDataset(Xv, yv), batch_size=batch),
    )


def train_model(model, tr_dl, va_dl, device, epochs=EPOCHS, patience=PATIENCE, lr=LR):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        sched.step(-acc)

        if acc > best_acc:
            best_acc, best_state, wait = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def eval_against_real(model, X_test, y_real, h_meas_nb, codebook, device):
    """Evaluate model predictions against REAL beam labels + real channel gain."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(device)).cpu()
    pred = logits.argmax(1).numpy()
    top3 = logits.topk(3, dim=1).indices.numpy()

    top1 = float(np.mean(pred == y_real) * 100)
    top3_acc = float(np.mean([y_real[i] in top3[i] for i in range(len(y_real))]) * 100)

    all_gains = np.abs(h_meas_nb @ codebook.conj().T) ** 2  # (N, n_beams)
    opt_gain = all_gains[np.arange(len(h_meas_nb)), y_real]
    pred_gain = all_gains[np.arange(len(h_meas_nb)), pred]
    valid = opt_gain > 1e-30
    gain_ratio = float(np.mean(pred_gain[valid] / opt_gain[valid]) * 100) if valid.any() else 0.0

    return {"top1_acc": top1, "top3_acc": top3_acc, "gain_ratio": gain_ratio, "n_test": int(len(y_real))}


def run_one_config(npz_path: Path, codebook: np.ndarray, device: str):
    d = np.load(npz_path, allow_pickle=True)
    name = str(d['config_name'])
    sim_ch = d['sim_channels']     # (N, 64, 1024) complex64
    meas_ch = d['meas_channels']   # (N, 64, 1024) complex64
    pos = d['positions']           # (N, 3) float32
    north = d['north_assign']      # (32,)

    n = len(sim_ch)
    print(f"\n=== {name} | N={n} ===")

    # Narrowband per-array channels
    h_sim_nb = narrowband_from_cfr(sim_ch, N_ANT, north)    # (N, 32) complex
    h_meas_nb = narrowband_from_cfr(meas_ch, N_ANT, north)  # (N, 32) complex

    # Beam labels
    y_sim = compute_beam_labels(h_sim_nb, codebook).astype(np.int64)
    y_meas = compute_beam_labels(h_meas_nb, codebook).astype(np.int64)
    agreement = float(np.mean(y_sim == y_meas) * 100)
    print(f"  Label agreement sim vs real: {agreement:.1f}%")

    # Features: normalized position
    mu = pos.mean(axis=0)
    sigma = pos.std(axis=0) + 1e-8
    X = ((pos - mu) / sigma).astype(np.float32)

    # Split
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    tr_idx = perm[:n_tr]
    va_idx = perm[n_tr:n_tr + n_va]
    te_idx = perm[n_tr + n_va:]

    # ---------- A) Train on SIM labels, evaluate against REAL ----------
    tr_dl, va_dl = make_loaders(X, y_sim, tr_idx, va_idx)
    model = BeamPredictor(X.shape[1], N_BEAMS)
    model, val_acc_sim = train_model(model, tr_dl, va_dl, device=device)
    sim_trained = eval_against_real(model, X[te_idx], y_meas[te_idx], h_meas_nb[te_idx], codebook, device)
    print(f"  [Train SIM] val(sim)={val_acc_sim*100:.1f}%  "
          f"-> real top1={sim_trained['top1_acc']:.1f}%, top3={sim_trained['top3_acc']:.1f}%, "
          f"gain={sim_trained['gain_ratio']:.1f}%")

    # ---------- B) Oracle: train on REAL labels (ceiling) ----------
    tr_dl, va_dl = make_loaders(X, y_meas, tr_idx, va_idx)
    model = BeamPredictor(X.shape[1], N_BEAMS)
    model, val_acc_real = train_model(model, tr_dl, va_dl, device=device)
    real_trained = eval_against_real(model, X[te_idx], y_meas[te_idx], h_meas_nb[te_idx], codebook, device)
    print(f"  [Oracle]    val(real)={val_acc_real*100:.1f}% "
          f"-> real top1={real_trained['top1_acc']:.1f}%, top3={real_trained['top3_acc']:.1f}%, "
          f"gain={real_trained['gain_ratio']:.1f}%")

    # ---------- C) Baseline: random beam selection ----------
    rng2 = np.random.RandomState(SEED)
    rand_pred = rng2.randint(0, N_BEAMS, size=len(te_idx))
    rand_top1 = float(np.mean(rand_pred == y_meas[te_idx]) * 100)
    rand_gains = np.abs(h_meas_nb[te_idx] @ codebook.conj().T) ** 2
    rand_opt = rand_gains[np.arange(len(te_idx)), y_meas[te_idx]]
    rand_prd = rand_gains[np.arange(len(te_idx)), rand_pred]
    valid_r = rand_opt > 1e-30
    rand_gain = float(np.mean(rand_prd[valid_r] / rand_opt[valid_r]) * 100) if valid_r.any() else 0.0

    return {
        "config": name,
        "n_positions": n,
        "n_test": int(len(te_idx)),
        "label_agreement_pct": agreement,
        "pwr_corr_fidelity": PWR_CORR_FROM_FIDELITY.get(name, None),
        "sim_trained_real_top1": sim_trained["top1_acc"],
        "sim_trained_real_top3": sim_trained["top3_acc"],
        "sim_trained_real_gain": sim_trained["gain_ratio"],
        "oracle_real_top1": real_trained["top1_acc"],
        "oracle_real_top3": real_trained["top3_acc"],
        "oracle_real_gain": real_trained["gain_ratio"],
        "random_top1": rand_top1,
        "random_gain": rand_gain,
        "val_acc_sim": val_acc_sim * 100,
        "val_acc_real": val_acc_real * 100,
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    codebook = generate_dft_codebook(N_ANT, N_BEAMS)
    print(f"Codebook: {N_BEAMS} beams x {N_ANT} antennas")

    results = []
    for npz in sorted(CAPTURE_DIR.glob("*.npz")):
        try:
            r = run_one_config(npz, codebook, device)
            results.append(r)
        except Exception as e:
            print(f"  FAIL {npz.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  Beam Prediction Summary")
    print(f"{'='*70}")
    print(f"{'Config':<20s} {'pwr_corr':>9s} {'Agree':>7s} {'Sim->Real':>11s} {'Oracle':>8s} {'Gain':>7s}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: -(x['pwr_corr_fidelity'] or 0)):
        pc = f"{r['pwr_corr_fidelity']:.3f}" if r['pwr_corr_fidelity'] is not None else "-"
        print(f"{r['config']:<20s} {pc:>9s} {r['label_agreement_pct']:>6.1f}% "
              f"{r['sim_trained_real_top1']:>10.1f}% {r['oracle_real_top1']:>7.1f}% "
              f"{r['sim_trained_real_gain']:>6.1f}%")

    out_path = OUT_DIR / "beam_pred_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
