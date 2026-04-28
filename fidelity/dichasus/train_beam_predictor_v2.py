"""
Improved beam prediction analysis: coarse beams, K-fold CV, direct channel metrics.

Reuses existing .npz captures (sim_channels, meas_channels, positions) from
run_beam_pred_experiment.py.  No new RT simulations needed.

Key improvements over v1:
  1. Coarse beam groups (64 → 8/16 groups) — less noisy classification.
  2. 5-fold cross-validation — uses all 300 positions instead of 45 test.
  3. Direct channel-based metrics (no MLP):
       - beam label agreement
       - top-K beam overlap
       - beam power spectrum cosine similarity
       - gain ratio statistics
  4. Proper correlation statistics (Pearson/Spearman with p-values).

Usage:
  python train_beam_predictor_v2.py
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

# Fidelity metric per config (from all_results.json)
PWR_CORR = {
    "calibrated":       0.290,
    "hw_tx_iso":        0.215,
    "rt_depth_3":       0.138,
    "mat_all_concrete": 0.097,
    "hw_baseline":      0.087,
    "geo_full":         0.023,
    "mat_all_metal":    0.004,
}

N_ANT = 32
N_BEAMS = 64
HIDDEN = 256
EPOCHS = 300
PATIENCE = 30
BATCH = 64
LR = 1e-3
SEED = 42
N_FOLDS = 5

COARSE_GROUPS = [8, 16]  # group 64 beams into 8 or 16 sectors


# ---------------------------------------------------------------------------
# Codebook & beam utilities
# ---------------------------------------------------------------------------

def generate_dft_codebook(n_ant: int, n_beams: int) -> np.ndarray:
    k = np.arange(n_beams)[:, None]
    n = np.arange(n_ant)[None, :]
    return np.exp(1j * 2 * np.pi * k * n / n_beams) / np.sqrt(n_ant)


def compute_beam_labels(h_nb: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """h_nb: (N, n_ant) complex → (N,) int beam index."""
    power = np.abs(h_nb @ codebook.conj().T) ** 2
    return np.argmax(power, axis=1)


def beam_power_spectrum(h_nb: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """(N, n_ant) → (N, n_beams) power per beam."""
    return np.abs(h_nb @ codebook.conj().T) ** 2


def narrowband_from_cfr(channels: np.ndarray, n_ant: int, assign: np.ndarray) -> np.ndarray:
    h = channels[:, assign[:n_ant], :]
    return h.mean(axis=-1)


def coarse_label(fine_label: np.ndarray, n_fine: int, n_coarse: int) -> np.ndarray:
    """Map fine beam index to coarse group index."""
    group_size = n_fine // n_coarse
    return fine_label // group_size


# ---------------------------------------------------------------------------
# Direct channel-based metrics (no MLP)
# ---------------------------------------------------------------------------

def direct_channel_metrics(h_sim_nb, h_meas_nb, codebook):
    """Compute metrics directly from sim/meas channels, no learning."""
    n = len(h_sim_nb)

    # Beam labels
    y_sim = compute_beam_labels(h_sim_nb, codebook)
    y_meas = compute_beam_labels(h_meas_nb, codebook)

    # 1. Beam label agreement (exact match)
    agreement = float(np.mean(y_sim == y_meas) * 100)

    # 2. Coarse beam agreement
    coarse_agreement = {}
    for ng in COARSE_GROUPS:
        y_sim_c = coarse_label(y_sim, N_BEAMS, ng)
        y_meas_c = coarse_label(y_meas, N_BEAMS, ng)
        coarse_agreement[ng] = float(np.mean(y_sim_c == y_meas_c) * 100)

    # 3. Top-K beam overlap
    pwr_sim = beam_power_spectrum(h_sim_nb, codebook)
    pwr_meas = beam_power_spectrum(h_meas_nb, codebook)
    topk_overlap = {}
    for k in [3, 5, 10]:
        top_sim = np.argsort(-pwr_sim, axis=1)[:, :k]
        top_meas = np.argsort(-pwr_meas, axis=1)[:, :k]
        overlap = np.array([len(set(top_sim[i]) & set(top_meas[i])) / k
                            for i in range(n)])
        topk_overlap[k] = float(np.mean(overlap) * 100)

    # 4. Beam power spectrum cosine similarity
    norm_sim = pwr_sim / (np.linalg.norm(pwr_sim, axis=1, keepdims=True) + 1e-30)
    norm_meas = pwr_meas / (np.linalg.norm(pwr_meas, axis=1, keepdims=True) + 1e-30)
    cosine_sim = float(np.mean(np.sum(norm_sim * norm_meas, axis=1)) * 100)

    # 5. Direct gain ratio (use sim's best beam on real channel)
    all_gains = np.abs(h_meas_nb @ codebook.conj().T) ** 2
    opt_gain = all_gains[np.arange(n), y_meas]
    sim_gain = all_gains[np.arange(n), y_sim]
    valid = opt_gain > 1e-30
    direct_gain_ratio = float(np.mean(sim_gain[valid] / opt_gain[valid]) * 100)

    # 6. Random baseline gain ratio
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


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------

class BeamPredictor(nn.Module):
    def __init__(self, n_in: int, n_classes: int, hidden: int = HIDDEN):
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


def eval_model(model, X_test, y_real, h_meas_nb, codebook, device, n_classes):
    """Evaluate predictions against real labels."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float().to(device)).cpu()
    pred = logits.argmax(1).numpy()

    top1 = float(np.mean(pred == y_real) * 100)

    # Gain ratio (only for fine beams)
    if n_classes == N_BEAMS:
        all_gains = np.abs(h_meas_nb @ codebook.conj().T) ** 2
        opt_gain = all_gains[np.arange(len(h_meas_nb)), y_real]
        pred_gain = all_gains[np.arange(len(h_meas_nb)), pred]
        valid = opt_gain > 1e-30
        gain_ratio = float(np.mean(pred_gain[valid] / opt_gain[valid]) * 100) if valid.any() else 0.0
    else:
        gain_ratio = None

    return {"top1_acc": top1, "gain_ratio": gain_ratio}


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def kfold_experiment(X, y_sim, y_meas, h_meas_nb, codebook, device, n_classes, label_name):
    """Run K-fold CV: train on sim labels, evaluate on real labels."""
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

        # Split train/val from remaining
        n_val = max(1, len(tr_va_idx) // 5)
        va_idx = tr_va_idx[:n_val]
        tr_idx = tr_va_idx[n_val:]

        # Train on sim labels
        Xt = torch.from_numpy(X[tr_idx]).float()
        yt = torch.from_numpy(y_sim[tr_idx]).long()
        Xv = torch.from_numpy(X[va_idx]).float()
        yv = torch.from_numpy(y_sim[va_idx]).long()
        tr_dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
        va_dl = DataLoader(TensorDataset(Xv, yv), batch_size=BATCH)

        model = BeamPredictor(X.shape[1], n_classes)
        model = train_model(model, tr_dl, va_dl, device)
        res = eval_model(model, X[te_idx], y_meas[te_idx], h_meas_nb[te_idx], codebook, device, n_classes)
        all_top1.append(res["top1_acc"])
        if res["gain_ratio"] is not None:
            all_gain.append(res["gain_ratio"])

        # Oracle: train on real labels
        yt_real = torch.from_numpy(y_meas[tr_idx]).long()
        yv_real = torch.from_numpy(y_meas[va_idx]).long()
        tr_dl_real = DataLoader(TensorDataset(Xt, yt_real), batch_size=BATCH, shuffle=True)
        va_dl_real = DataLoader(TensorDataset(Xv, yv_real), batch_size=BATCH)
        model_oracle = BeamPredictor(X.shape[1], n_classes)
        model_oracle = train_model(model_oracle, tr_dl_real, va_dl_real, device)
        res_oracle = eval_model(model_oracle, X[te_idx], y_meas[te_idx], h_meas_nb[te_idx], codebook, device, n_classes)
        oracle_top1_all.append(res_oracle["top1_acc"])

    return {
        f"{label_name}_sim2real_top1_mean": float(np.mean(all_top1)),
        f"{label_name}_sim2real_top1_std": float(np.std(all_top1)),
        f"{label_name}_oracle_top1_mean": float(np.mean(oracle_top1_all)),
        f"{label_name}_oracle_top1_std": float(np.std(oracle_top1_all)),
        f"{label_name}_sim2real_gain_mean": float(np.mean(all_gain)) if all_gain else None,
        f"{label_name}_sim2real_gain_std": float(np.std(all_gain)) if all_gain else None,
    }


# ---------------------------------------------------------------------------
# Per-config analysis
# ---------------------------------------------------------------------------

def run_one_config(npz_path: Path, codebook: np.ndarray, device: str):
    d = np.load(npz_path, allow_pickle=True)
    name = str(d["config_name"])
    sim_ch = d["sim_channels"]
    meas_ch = d["meas_channels"]
    pos = d["positions"]
    north = d["north_assign"]

    n = len(sim_ch)
    print(f"\n{'='*60}")
    print(f"  {name} | N={n} | pwr_corr={PWR_CORR.get(name, '?')}")
    print(f"{'='*60}")

    h_sim_nb = narrowband_from_cfr(sim_ch, N_ANT, north)
    h_meas_nb = narrowband_from_cfr(meas_ch, N_ANT, north)

    # --- Direct channel metrics (no MLP) ---
    print("  Computing direct channel metrics...")
    direct = direct_channel_metrics(h_sim_nb, h_meas_nb, codebook)
    print(f"    Beam agreement:     {direct['beam_agreement_pct']:.1f}%")
    for ng, v in direct["coarse_agreement"].items():
        print(f"    Coarse-{ng} agreement: {v:.1f}%")
    for k, v in direct["topk_overlap_pct"].items():
        print(f"    Top-{k} overlap:      {v:.1f}%")
    print(f"    Cosine similarity:  {direct['beam_cosine_sim_pct']:.1f}%")
    print(f"    Direct gain ratio:  {direct['direct_gain_ratio_pct']:.1f}%")
    print(f"    Random gain ratio:  {direct['random_gain_ratio_pct']:.1f}%")

    # --- Features ---
    mu, sigma = pos.mean(axis=0), pos.std(axis=0) + 1e-8
    X = ((pos - mu) / sigma).astype(np.float32)

    # --- Fine beam labels ---
    y_sim_fine = compute_beam_labels(h_sim_nb, codebook).astype(np.int64)
    y_meas_fine = compute_beam_labels(h_meas_nb, codebook).astype(np.int64)

    # --- K-fold: fine beams (64) ---
    print("  K-fold CV: 64 beams...")
    kf_fine = kfold_experiment(X, y_sim_fine, y_meas_fine, h_meas_nb, codebook, device,
                               N_BEAMS, "fine64")
    print(f"    Sim→Real top-1: {kf_fine['fine64_sim2real_top1_mean']:.1f}% "
          f"(±{kf_fine['fine64_sim2real_top1_std']:.1f})")
    print(f"    Oracle top-1:   {kf_fine['fine64_oracle_top1_mean']:.1f}% "
          f"(±{kf_fine['fine64_oracle_top1_std']:.1f})")
    if kf_fine["fine64_sim2real_gain_mean"] is not None:
        print(f"    Sim→Real gain:  {kf_fine['fine64_sim2real_gain_mean']:.1f}% "
              f"(±{kf_fine['fine64_sim2real_gain_std']:.1f})")

    # --- K-fold: coarse beams ---
    kf_coarse = {}
    for ng in COARSE_GROUPS:
        print(f"  K-fold CV: {ng} coarse groups...")
        y_sim_c = coarse_label(y_sim_fine, N_BEAMS, ng).astype(np.int64)
        y_meas_c = coarse_label(y_meas_fine, N_BEAMS, ng).astype(np.int64)
        kf = kfold_experiment(X, y_sim_c, y_meas_c, h_meas_nb, codebook, device,
                              ng, f"coarse{ng}")
        kf_coarse[ng] = kf
        print(f"    Sim→Real top-1: {kf[f'coarse{ng}_sim2real_top1_mean']:.1f}% "
              f"(±{kf[f'coarse{ng}_sim2real_top1_std']:.1f})")
        print(f"    Oracle top-1:   {kf[f'coarse{ng}_oracle_top1_mean']:.1f}% "
              f"(±{kf[f'coarse{ng}_oracle_top1_std']:.1f})")

    result = {
        "config": name,
        "n_positions": n,
        "pwr_corr": PWR_CORR.get(name),
        "direct": direct,
        "kfold_fine64": kf_fine,
    }
    for ng, kf in kf_coarse.items():
        result[f"kfold_coarse{ng}"] = kf

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # --- Correlation analysis ---
    print(f"\n{'='*70}")
    print("  Correlation Analysis: pwr_corr vs metrics")
    print(f"{'='*70}")

    valid = [r for r in results if r["pwr_corr"] is not None]
    valid.sort(key=lambda r: r["pwr_corr"])
    pwr = np.array([r["pwr_corr"] for r in valid])

    from scipy.stats import pearsonr, spearmanr

    metrics_to_correlate = [
        ("Direct gain ratio", [r["direct"]["direct_gain_ratio_pct"] for r in valid]),
        ("Beam cosine sim", [r["direct"]["beam_cosine_sim_pct"] for r in valid]),
        ("Top-3 overlap", [r["direct"]["topk_overlap_pct"][3] for r in valid]),
        ("Top-5 overlap", [r["direct"]["topk_overlap_pct"][5] for r in valid]),
        ("Top-10 overlap", [r["direct"]["topk_overlap_pct"][10] for r in valid]),
        ("Beam agreement (64)", [r["direct"]["beam_agreement_pct"] for r in valid]),
        ("Coarse-8 agreement", [r["direct"]["coarse_agreement"][8] for r in valid]),
        ("Coarse-16 agreement", [r["direct"]["coarse_agreement"][16] for r in valid]),
        ("KF fine64 gain", [r["kfold_fine64"]["fine64_sim2real_gain_mean"] for r in valid]),
        ("KF fine64 top-1", [r["kfold_fine64"]["fine64_sim2real_top1_mean"] for r in valid]),
        ("KF coarse8 top-1", [r["kfold_coarse8"]["coarse8_sim2real_top1_mean"] for r in valid]),
        ("KF coarse16 top-1", [r["kfold_coarse16"]["coarse16_sim2real_top1_mean"] for r in valid]),
    ]

    print(f"\n{'Metric':<25s} {'Pearson r':>10s} {'p-val':>8s} {'Spearman r':>11s} {'p-val':>8s}")
    print("-" * 65)
    for name, vals in metrics_to_correlate:
        v = np.array([x if x is not None else np.nan for x in vals])
        mask = ~np.isnan(v)
        if mask.sum() < 3:
            print(f"{name:<25s}  (insufficient data)")
            continue
        pr, pp = pearsonr(pwr[mask], v[mask])
        sr, sp = spearmanr(pwr[mask], v[mask])
        print(f"{name:<25s} {pr:>10.3f} {pp:>8.3f} {sr:>11.3f} {sp:>8.3f}")

    # --- Summary table ---
    print(f"\n{'='*90}")
    print("  Summary Table")
    print(f"{'='*90}")
    print(f"{'Config':<18s} {'pwr_corr':>8s} {'DirGain':>8s} {'Cos':>6s} "
          f"{'Top5':>6s} {'C8Agr':>6s} {'KF-Gain':>8s} {'KF-C8':>7s}")
    print("-" * 90)
    for r in valid:
        pc = f"{r['pwr_corr']:.3f}"
        dg = f"{r['direct']['direct_gain_ratio_pct']:.1f}"
        cs = f"{r['direct']['beam_cosine_sim_pct']:.1f}"
        t5 = f"{r['direct']['topk_overlap_pct'][5]:.1f}"
        c8 = f"{r['direct']['coarse_agreement'][8]:.1f}"
        kg = f"{r['kfold_fine64']['fine64_sim2real_gain_mean']:.1f}" if r['kfold_fine64']['fine64_sim2real_gain_mean'] else "-"
        kc = f"{r['kfold_coarse8']['coarse8_sim2real_top1_mean']:.1f}"
        print(f"{r['config']:<18s} {pc:>8s} {dg:>8s} {cs:>6s} {t5:>6s} {c8:>6s} {kg:>8s} {kc:>7s}")

    # Save
    out_path = OUT_DIR / "beam_pred_results_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
