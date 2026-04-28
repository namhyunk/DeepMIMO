#!/usr/bin/env python
"""
Run beam prediction analysis on gradient-calibrated channels.
Compares: grad_full_calibration vs grad_itu_baseline.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "results" / "beam_pred"
OUT_DIR = BASE_DIR / "results" / "beam_pred"

N_ANT = 32
N_BEAMS = 64
HIDDEN = 256
EPOCHS = 300
PATIENCE = 30
BATCH = 64
LR = 1e-3
SEED = 42
N_FOLDS = 5
COARSE_GROUPS = [8, 16]


def generate_dft_codebook(n_ant, n_beams):
    k = np.arange(n_beams)[:, None]
    n = np.arange(n_ant)[None, :]
    return np.exp(1j * 2 * np.pi * k * n / n_beams) / np.sqrt(n_ant)


def compute_beam_labels(h_nb, codebook):
    power = np.abs(h_nb @ codebook.conj().T) ** 2
    return np.argmax(power, axis=1)


def beam_power_spectrum(h_nb, codebook):
    return np.abs(h_nb @ codebook.conj().T) ** 2


def narrowband_from_cfr(channels, n_ant, assign):
    h = channels[:, assign[:n_ant], :]
    return h.mean(axis=-1)


def coarse_label(fine_label, n_fine, n_coarse):
    group_size = n_fine // n_coarse
    return fine_label // group_size


def direct_channel_metrics(h_sim_nb, h_meas_nb, codebook):
    n = len(h_sim_nb)
    y_sim = compute_beam_labels(h_sim_nb, codebook)
    y_meas = compute_beam_labels(h_meas_nb, codebook)

    agreement = float(np.mean(y_sim == y_meas) * 100)

    coarse_agreement = {}
    for ng in COARSE_GROUPS:
        y_sim_c = coarse_label(y_sim, N_BEAMS, ng)
        y_meas_c = coarse_label(y_meas, N_BEAMS, ng)
        coarse_agreement[ng] = float(np.mean(y_sim_c == y_meas_c) * 100)

    pwr_sim = beam_power_spectrum(h_sim_nb, codebook)
    pwr_meas = beam_power_spectrum(h_meas_nb, codebook)

    topk_overlap = {}
    for k in [3, 5, 10]:
        top_sim = np.argsort(-pwr_sim, axis=1)[:, :k]
        top_meas = np.argsort(-pwr_meas, axis=1)[:, :k]
        overlap = np.array([len(set(top_sim[i]) & set(top_meas[i])) / k for i in range(n)])
        topk_overlap[k] = float(np.mean(overlap) * 100)

    norm_sim = pwr_sim / (np.linalg.norm(pwr_sim, axis=1, keepdims=True) + 1e-30)
    norm_meas = pwr_meas / (np.linalg.norm(pwr_meas, axis=1, keepdims=True) + 1e-30)
    cosine_sim = float(np.mean(np.sum(norm_sim * norm_meas, axis=1)) * 100)

    all_gains = np.abs(h_meas_nb @ codebook.conj().T) ** 2
    opt_gain = all_gains[np.arange(n), y_meas]
    sim_gain = all_gains[np.arange(n), y_sim]
    valid = opt_gain > 1e-30
    direct_gain_ratio = float(np.mean(sim_gain[valid] / opt_gain[valid]) * 100)

    rng = np.random.RandomState(SEED)
    rand_beams = rng.randint(0, N_BEAMS, size=n)
    rand_gain = all_gains[np.arange(n), rand_beams]
    random_gain_ratio = float(np.mean(rand_gain[valid] / opt_gain[valid]) * 100)

    return {
        "beam_agreement_pct": agreement,
        "coarse_agreement": coarse_agreement,
        "topk_overlap_pct": topk_overlap,
        "beam_cosine_sim_pct": cosine_sim,
        "direct_gain_ratio_pct": direct_gain_ratio,
        "random_gain_ratio_pct": random_gain_ratio,
    }


class BeamPredictor(nn.Module):
    def __init__(self, n_in, n_classes, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


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
            crit(model(x), y).backward()
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
    return model


def kfold_experiment(X, y_sim, y_meas, h_meas_nb, codebook, device, n_classes, label_name):
    n = len(X)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    fold_size = n // N_FOLDS

    all_top1, all_gain = [], []
    oracle_top1_all = []

    for fold in range(N_FOLDS):
        te_start = fold * fold_size
        te_end = te_start + fold_size if fold < N_FOLDS - 1 else n
        te_idx = perm[te_start:te_end]
        tr_va_idx = np.concatenate([perm[:te_start], perm[te_end:]])

        n_val = max(1, len(tr_va_idx) // 5)
        va_idx = tr_va_idx[:n_val]
        tr_idx = tr_va_idx[n_val:]

        Xt = torch.from_numpy(X[tr_idx]).float()
        yt = torch.from_numpy(y_sim[tr_idx]).long()
        Xv = torch.from_numpy(X[va_idx]).float()
        yv = torch.from_numpy(y_sim[va_idx]).long()
        tr_dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(TensorDataset(Xv, yv), batch_size=BATCH)

        model = BeamPredictor(X.shape[1], n_classes)
        model = train_model(model, tr_dl, va_dl, device)

        # Eval on real labels
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X[te_idx]).float().to(device)).cpu()
        pred = logits.argmax(1).numpy()
        top1 = float(np.mean(pred == y_meas[te_idx]) * 100)
        all_top1.append(top1)

        if n_classes == N_BEAMS:
            all_gains = np.abs(h_meas_nb @ codebook.conj().T) ** 2
            opt_g = all_gains[te_idx][np.arange(len(te_idx)), y_meas[te_idx]]
            pred_g = all_gains[te_idx][np.arange(len(te_idx)), pred]
            valid = opt_g > 1e-30
            if valid.any():
                all_gain.append(float(np.mean(pred_g[valid] / opt_g[valid]) * 100))

        # Oracle
        yt_real = torch.from_numpy(y_meas[tr_idx]).long()
        yv_real = torch.from_numpy(y_meas[va_idx]).long()
        tr_dl_r = DataLoader(TensorDataset(Xt, yt_real), batch_size=BATCH, shuffle=True)
        va_dl_r = DataLoader(TensorDataset(Xv, yv_real), batch_size=BATCH)
        model_o = BeamPredictor(X.shape[1], n_classes)
        model_o = train_model(model_o, tr_dl_r, va_dl_r, device)
        model_o.eval()
        with torch.no_grad():
            logits_o = model_o(torch.from_numpy(X[te_idx]).float().to(device)).cpu()
        pred_o = logits_o.argmax(1).numpy()
        oracle_top1_all.append(float(np.mean(pred_o == y_meas[te_idx]) * 100))

    return {
        f"{label_name}_sim2real_top1_mean": float(np.mean(all_top1)),
        f"{label_name}_sim2real_top1_std": float(np.std(all_top1)),
        f"{label_name}_oracle_top1_mean": float(np.mean(oracle_top1_all)),
        f"{label_name}_oracle_top1_std": float(np.std(oracle_top1_all)),
        f"{label_name}_sim2real_gain_mean": float(np.mean(all_gain)) if all_gain else None,
    }


def run_one_config(npz_path, codebook, device):
    d = np.load(npz_path, allow_pickle=True)
    name = str(d["config_name"])
    sim_ch = d["sim_channels"]
    meas_ch = d["meas_channels"]
    pos = d["positions"]
    north = d["north_assign"]
    n = len(sim_ch)

    print(f"\n{'='*60}")
    print(f"  {name} | N={n}")
    print(f"{'='*60}")

    h_sim_nb = narrowband_from_cfr(sim_ch, N_ANT, north)
    h_meas_nb = narrowband_from_cfr(meas_ch, N_ANT, north)

    # Direct metrics
    direct = direct_channel_metrics(h_sim_nb, h_meas_nb, codebook)
    print(f"  Beam agreement:     {direct['beam_agreement_pct']:.1f}%")
    for ng, v in direct["coarse_agreement"].items():
        print(f"  Coarse-{ng} agreement: {v:.1f}%")
    print(f"  Direct gain ratio:  {direct['direct_gain_ratio_pct']:.1f}%")
    print(f"  Cosine similarity:  {direct['beam_cosine_sim_pct']:.1f}%")

    # K-fold
    mu, sigma = pos.mean(axis=0), pos.std(axis=0) + 1e-8
    X = ((pos - mu) / sigma).astype(np.float32)
    y_sim_fine = compute_beam_labels(h_sim_nb, codebook).astype(np.int64)
    y_meas_fine = compute_beam_labels(h_meas_nb, codebook).astype(np.int64)

    print("  K-fold CV: 64 beams...")
    kf_fine = kfold_experiment(X, y_sim_fine, y_meas_fine, h_meas_nb, codebook, device, N_BEAMS, "fine64")
    print(f"  Sim→Real top-1: {kf_fine['fine64_sim2real_top1_mean']:.1f}%")
    print(f"  Oracle top-1:   {kf_fine['fine64_oracle_top1_mean']:.1f}%")

    kf_coarse = {}
    for ng in COARSE_GROUPS:
        print(f"  K-fold CV: {ng} coarse groups...")
        y_sim_c = coarse_label(y_sim_fine, N_BEAMS, ng).astype(np.int64)
        y_meas_c = coarse_label(y_meas_fine, N_BEAMS, ng).astype(np.int64)
        kf = kfold_experiment(X, y_sim_c, y_meas_c, h_meas_nb, codebook, device, ng, f"coarse{ng}")
        kf_coarse[ng] = kf
        print(f"  Sim→Real top-1: {kf[f'coarse{ng}_sim2real_top1_mean']:.1f}%")

    return {
        "config": name,
        "n_positions": n,
        "direct": direct,
        "kfold_fine64": kf_fine,
        "kfold_coarse8": kf_coarse.get(8, {}),
        "kfold_coarse16": kf_coarse.get(16, {}),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    codebook = generate_dft_codebook(N_ANT, N_BEAMS)

    files = ["grad_full_calibration.npz", "grad_itu_baseline.npz"]
    results = []
    for f in files:
        npz_path = CAPTURE_DIR / f
        if npz_path.exists():
            r = run_one_config(npz_path, codebook, device)
            results.append(r)

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25s} {'DirGain':>8s} {'Cos':>6s} {'Top5':>6s} {'C8Agr':>6s} {'KF-Gain':>8s}")
    print("-" * 65)
    for r in results:
        dg = f"{r['direct']['direct_gain_ratio_pct']:.1f}"
        cs = f"{r['direct']['beam_cosine_sim_pct']:.1f}"
        t5 = f"{r['direct']['topk_overlap_pct'][5]:.1f}"
        c8 = f"{r['direct']['coarse_agreement'][8]:.1f}"
        kg = f"{r['kfold_fine64']['fine64_sim2real_gain_mean']:.1f}" if r['kfold_fine64'].get('fine64_sim2real_gain_mean') else "-"
        print(f"{r['config']:<25s} {dg:>8s} {cs:>6s} {t5:>6s} {c8:>6s} {kg:>8s}")

    out_path = OUT_DIR / "beam_pred_gradient_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
