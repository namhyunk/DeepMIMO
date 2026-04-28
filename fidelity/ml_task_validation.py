"""
Step 6: ML Task Validation
Evaluate whether channel NMSE thresholds translate to ML task performance degradation.

Two tasks:
  1. Beam Prediction:  position (x,y,z) -> best beam index (DFT codebook)
  2. Localization:     channel fingerprint -> position (x,y)

Experiment: Train on degraded (digital-twin) data, test on baseline (reality) data.
This answers: "If my DT has fidelity X, how well does an ML model trained on it work in reality?"

Usage:
    python -m fidelity.ml_task_validation
    python -m fidelity.ml_task_validation --configs geo_noise_0_5m geo_noise_1m
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm


# ============================================================================
# Config: which degraded configs to evaluate and their baselines
# ============================================================================

SELECTED_CONFIGS = [
    # (config_name, baseline_name, tag, param_value, param_label)
    # Geometry (baseline_ds used: original baseline overwritten by Munich run)
    ("geo_noise_0_1m",   "baseline_ds", "geometry", 0.1, "0.1m"),
    ("geo_noise_0_5m",   "baseline_ds", "geometry", 0.5, "0.5m"),
    ("geo_noise_1m",     "baseline_ds", "geometry", 1.0, "1m"),
    ("geo_noise_2m",     "baseline_ds", "geometry", 2.0, "2m"),
    ("geo_noise_5m",     "baseline_ds", "geometry", 5.0, "5m"),
    # Material
    ("mat_all_concrete", "baseline_ds", "material", 1, "concrete"),
    ("mat_all_metal",    "baseline_ds", "material", 2, "metal"),
    # RT depth
    ("rt_depth_2",       "rt_depth_10", "rt_depth", 2, "depth=2"),
    ("rt_depth_3",       "rt_depth_10", "rt_depth", 3, "depth=3"),
    ("rt_depth_5",       "rt_depth_10", "rt_depth", 5, "depth=5"),
    ("rt_depth_7",       "rt_depth_10", "rt_depth", 7, "depth=7"),
    # RT rays (baseline_ds used: original baseline overwritten by Munich run)
    ("rt_5k_rays",       "baseline_ds", "rt_rays", 5, "5K"),
    ("rt_50k_rays",      "baseline_ds", "rt_rays", 50, "50K"),
    ("rt_200k_rays",     "baseline_ds", "rt_rays", 200, "200K"),
    # Hardware
    ("hw_4x4_spacing04", "hw_baseline_4x4", "hardware", 1, "spacing 0.4λ"),
    ("hw_4x4_polh",      "hw_baseline_4x4", "hardware", 2, "pol H"),
    ("hw_4x4_dipole",    "hw_baseline_4x4", "hardware", 3, "dipole"),
    ("hw_4x4_iso",       "hw_baseline_4x4", "hardware", 4, "isotropic"),
]

N_BEAMS = 32  # DFT codebook size (oversampled for 8 TX antennas)


# ============================================================================
# DFT Codebook and beam labels
# ============================================================================

def generate_dft_codebook(n_antennas: int, n_beams: int) -> np.ndarray:
    """Oversampled DFT codebook. Returns (n_beams, n_antennas) complex."""
    k = np.arange(n_beams)[:, None]
    n = np.arange(n_antennas)[None, :]
    return np.exp(1j * 2 * np.pi * k * n / n_beams) / np.sqrt(n_antennas)


def compute_beam_labels(channels: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Best beam index per user. channels: (n_ue, n_rx, n_tx, n_sc)."""
    h = channels[:, 0, :, 0]  # (n_ue, n_tx)
    beam_power = np.abs(h @ codebook.conj().T) ** 2  # (n_ue, n_beams)
    return np.argmax(beam_power, axis=1)


# ============================================================================
# Feature extraction
# ============================================================================

def get_beam_features(ds) -> np.ndarray:
    """Beam prediction input: normalized user position."""
    pos = ds.rx_pos[:, :3].astype(np.float32)
    return (pos - pos.mean(axis=0)) / (pos.std(axis=0) + 1e-8)


MAX_PATHS = 25  # Fixed feature dimension across all configs


def _pad_paths(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Pad or truncate path dimension to MAX_PATHS."""
    n_ue, n_paths = arr.shape
    if n_paths >= MAX_PATHS:
        return arr[:, :MAX_PATHS]
    pad = np.full((n_ue, MAX_PATHS - n_paths), fill, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def get_loc_features(ds) -> np.ndarray:
    """Localization input: path-level features (power, delay, angles)."""
    pwr = _pad_paths(np.nan_to_num(ds.pwr, nan=-200.0), -200.0).astype(np.float32)
    toa = _pad_paths(np.nan_to_num(ds.toa * 1e9, nan=0.0), 0.0).astype(np.float32)
    aoa_az = _pad_paths(np.nan_to_num(ds.aoa_az, nan=0.0), 0.0).astype(np.float32)
    aoa_el = _pad_paths(np.nan_to_num(ds.aoa_el, nan=0.0), 0.0).astype(np.float32)
    feats = np.concatenate([pwr, toa, aoa_az, aoa_el], axis=-1)  # (n_ue, MAX_PATHS*4)
    # Normalize per feature
    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    return ((feats - mu) / std).astype(np.float32)


def get_loc_labels(ds) -> np.ndarray:
    """Localization output: 2D position (x, y)."""
    return ds.rx_pos[:, :2].astype(np.float32)


# ============================================================================
# Models
# ============================================================================

class BeamPredictor(nn.Module):
    def __init__(self, n_in: int, n_beams: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_beams),
        )

    def forward(self, x):
        return self.net(x)


class Localizer(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training / evaluation
# ============================================================================

def make_loaders(X: np.ndarray, y: np.ndarray,
                 train_idx, val_idx, batch_size: int = 128):
    """Create train and val DataLoaders."""
    Xt = torch.from_numpy(X[train_idx])
    yt = torch.from_numpy(y[train_idx])
    Xv = torch.from_numpy(X[val_idx])
    yv = torch.from_numpy(y[val_idx])
    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size)
    return train_dl, val_dl


def train_model(model, train_dl, val_dl, task: str,
                epochs: int = 200, lr: float = 1e-3, device: str = "cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    criterion = nn.CrossEntropyLoss() if task == "beam" else nn.MSELoss()

    best_metric = -float("inf")
    best_state = None
    patience, wait = 20, 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            if task == "beam":
                correct = total = 0
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
                metric = correct / total
            else:
                errs = []
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    errs.append(((model(x) - y) ** 2).sum(1).sqrt().mean().item())
                metric = -np.mean(errs)

        scheduler.step(-metric)

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def eval_beam(model, X: np.ndarray, y: np.ndarray,
              channels: np.ndarray, codebook: np.ndarray,
              device: str = "cpu"):
    """Evaluate beam prediction: top-1/3 accuracy + effective beam gain ratio."""
    model.eval()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y)
    with torch.no_grad():
        logits = model(Xt).cpu()

    pred_beams = logits.argmax(1).numpy()
    top1 = (logits.argmax(1) == yt).float().mean().item() * 100
    top3_idx = logits.topk(3, dim=1).indices
    top3 = (top3_idx == yt.unsqueeze(1)).any(1).float().mean().item() * 100

    # Effective beam gain ratio: |h @ w_pred|^2 / |h @ w_opt|^2
    h = channels[:, 0, :, 0]  # (n, n_tx)
    all_gains = np.abs(h @ codebook.conj().T) ** 2  # (n, n_beams)
    y_np = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
    opt_gain = all_gains[np.arange(len(h)), y_np]
    pred_gain = all_gains[np.arange(len(h)), pred_beams]
    # Avoid division by zero for users with no signal
    valid = opt_gain > 1e-30
    gain_ratio = np.mean(pred_gain[valid] / opt_gain[valid]) if valid.any() else 0.0

    return {"top1_acc": top1, "top3_acc": top3, "gain_ratio": gain_ratio * 100}


def eval_loc(model, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
    """Evaluate localization: RMSE and median error in meters."""
    model.eval()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y)
    with torch.no_grad():
        pred = model(Xt).cpu()
    errors = ((pred - yt) ** 2).sum(1).sqrt().numpy()
    return {
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
    }


# ============================================================================
# Channel NMSE (for comparison in output table)
# ============================================================================

def compute_nmse(bl_channels, dg_channels) -> float:
    """Channel NMSE in dB between baseline and degraded."""
    n = min(bl_channels.shape[0], dg_channels.shape[0])
    bl = bl_channels[:n].flatten()
    dg = dg_channels[:n].flatten()
    mse = np.mean(np.abs(bl - dg) ** 2)
    norm = np.mean(np.abs(bl) ** 2)
    if norm < 1e-30:
        return float("nan")
    return float(10 * np.log10(mse / norm))


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results: list[dict], output_dir: Path):
    """Plot ML accuracy vs fidelity parameter for each tag."""
    tags = sorted(set(r["tag"] for r in results))
    tag_colors = {
        "geometry": "tab:red", "material": "tab:blue",
        "rt_depth": "tab:green", "rt_rays": "tab:olive",
        "hardware": "tab:orange",
    }

    # --- Beam Top-1 vs NMSE ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for tag in tags:
        rows = sorted([r for r in results if r["tag"] == tag],
                       key=lambda r: r["nmse_dB"])
        x = [r["nmse_dB"] for r in rows]
        y = [r["beam_top1"] for r in rows]
        ax.plot(x, y, "o-", color=tag_colors.get(tag, "gray"),
                label=tag, markersize=6, alpha=0.8)
    # Also plot baselines
    bl_rows = [r for r in results if r.get("is_baseline")]
    for r in bl_rows:
        ax.axhline(r["beam_top1"], color="black", linestyle=":", alpha=0.3)

    ax.set_xlabel("Channel NMSE (dB)")
    ax.set_ylabel("Beam Top-1 Accuracy (%)")
    ax.set_title("Beam Prediction vs Channel NMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Localization RMSE vs NMSE ---
    ax = axes[1]
    for tag in tags:
        rows = sorted([r for r in results if r["tag"] == tag],
                       key=lambda r: r["nmse_dB"])
        x = [r["nmse_dB"] for r in rows]
        y = [r["loc_rmse"] for r in rows]
        ax.plot(x, y, "s-", color=tag_colors.get(tag, "gray"),
                label=tag, markersize=6, alpha=0.8)
    for r in bl_rows:
        ax.axhline(r["loc_rmse"], color="black", linestyle=":", alpha=0.3)

    ax.set_xlabel("Channel NMSE (dB)")
    ax.set_ylabel("Localization RMSE (m)")
    ax.set_title("Localization vs Channel NMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ml_task_vs_nmse.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {output_dir / 'ml_task_vs_nmse.png'}")
    plt.close()

    # --- Per-tag detailed plots ---
    for tag in tags:
        rows = sorted([r for r in results if r["tag"] == tag],
                       key=lambda r: r["param_value"])
        if len(rows) < 2:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        labels = [r["param_label"] for r in rows]
        x = np.arange(len(labels))

        # NMSE
        ax = axes[0]
        nmse = [r["nmse_dB"] for r in rows]
        bars = ax.bar(x, nmse, color=tag_colors.get(tag, "gray"), alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Channel NMSE (dB)")
        ax.set_title(f"{tag}: Channel NMSE")
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Beam accuracy
        ax = axes[1]
        top1 = [r["beam_top1"] for r in rows]
        top3 = [r["beam_top3"] for r in rows]
        ax.bar(x - 0.15, top1, 0.3, label="Top-1", color=tag_colors.get(tag, "gray"), alpha=0.8)
        ax.bar(x + 0.15, top3, 0.3, label="Top-3", color=tag_colors.get(tag, "gray"), alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{tag}: Beam Prediction")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Localization
        ax = axes[2]
        rmse = [r["loc_rmse"] for r in rows]
        med = [r["loc_median"] for r in rows]
        ax.bar(x - 0.15, rmse, 0.3, label="RMSE", color=tag_colors.get(tag, "gray"), alpha=0.8)
        ax.bar(x + 0.15, med, 0.3, label="Median", color=tag_colors.get(tag, "gray"), alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{tag}: Localization")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"ML Task Performance: {tag.replace('_', ' ').title()}", fontsize=13)
        plt.tight_layout()
        plt.savefig(output_dir / f"ml_{tag}_detail.png", dpi=200, bbox_inches="tight")
        print(f"Saved: {output_dir / f'ml_{tag}_detail.png'}")
        plt.close()


# ============================================================================
# Main experiment loop
# ============================================================================

def run_experiment(configs: list[tuple], output_dir: Path,
                   n_beams: int = N_BEAMS, seed: int = 42, device: str = "cpu"):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Fixed data split (same indices for all configs) ---
    n_ue = 2601
    idx = np.random.permutation(n_ue)
    n_train = int(0.70 * n_ue)
    n_val = int(0.15 * n_ue)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # --- Codebook ---
    n_tx = 8
    codebook = generate_dft_codebook(n_tx, n_beams)

    # --- Load and cache baselines ---
    baseline_names = sorted(set(bl for _, bl, *_ in configs))
    baselines = {}
    for bl_name in baseline_names:
        print(f"Loading baseline: {bl_name}")
        ds = dm.load(bl_name)
        baselines[bl_name] = {
            "channels": ds.channels,
            "beam_features": get_beam_features(ds),
            "beam_labels": compute_beam_labels(ds.channels, codebook),
            "loc_features": get_loc_features(ds),
            "loc_labels": get_loc_labels(ds),
        }

    # --- Run baseline self-evaluation first ---
    results = []
    for bl_name, bl_data in baselines.items():
        print(f"\n{'='*60}")
        print(f"  Baseline self-eval: {bl_name}")
        print(f"{'='*60}")

        # Beam prediction: train & test on same baseline
        model_b = BeamPredictor(bl_data["beam_features"].shape[1], n_beams)
        tr_dl, va_dl = make_loaders(
            bl_data["beam_features"], bl_data["beam_labels"].astype(np.int64),
            train_idx, val_idx)
        model_b = train_model(model_b, tr_dl, va_dl, "beam", device=device)
        beam_res = eval_beam(model_b, bl_data["beam_features"][test_idx],
                             bl_data["beam_labels"][test_idx].astype(np.int64),
                             bl_data["channels"][test_idx], codebook, device)

        # Localization: train & test on same baseline
        model_l = Localizer(bl_data["loc_features"].shape[1])
        tr_dl, va_dl = make_loaders(
            bl_data["loc_features"], bl_data["loc_labels"],
            train_idx, val_idx)
        model_l = train_model(model_l, tr_dl, va_dl, "loc", device=device)
        loc_res = eval_loc(model_l, bl_data["loc_features"][test_idx],
                           bl_data["loc_labels"][test_idx], device)

        print(f"  Beam: top1={beam_res['top1_acc']:.1f}%, top3={beam_res['top3_acc']:.1f}%, gain={beam_res['gain_ratio']:.1f}%")
        print(f"  Loc:  RMSE={loc_res['rmse_m']:.3f}m, median={loc_res['median_m']:.3f}m")

        results.append({
            "config": bl_name, "baseline": bl_name, "tag": "baseline",
            "param_value": 0, "param_label": bl_name,
            "nmse_dB": float("-inf"),
            "beam_top1": beam_res["top1_acc"], "beam_top3": beam_res["top3_acc"],
            "beam_gain": beam_res["gain_ratio"],
            "loc_rmse": loc_res["rmse_m"], "loc_median": loc_res["median_m"],
            "loc_p90": loc_res["p90_m"],
            "is_baseline": True,
        })

    # --- Run each degraded config ---
    for cfg_name, bl_name, tag, param_val, param_lbl in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {cfg_name}  (baseline: {bl_name})")
        print(f"{'='*60}")

        bl_data = baselines[bl_name]
        ds = dm.load(cfg_name)

        # Channel NMSE
        nmse_dB = compute_nmse(bl_data["channels"], ds.channels)
        print(f"  NMSE: {nmse_dB:+.2f} dB")

        # --- Beam prediction ---
        # Train on degraded labels, test on baseline labels
        dg_beam_features = get_beam_features(ds)
        dg_beam_labels = compute_beam_labels(ds.channels, codebook).astype(np.int64)

        model_b = BeamPredictor(dg_beam_features.shape[1], n_beams)
        tr_dl, va_dl = make_loaders(dg_beam_features, dg_beam_labels, train_idx, val_idx)
        model_b = train_model(model_b, tr_dl, va_dl, "beam", device=device)

        # Test on baseline (reality)
        beam_res = eval_beam(model_b, bl_data["beam_features"][test_idx],
                             bl_data["beam_labels"][test_idx].astype(np.int64),
                             bl_data["channels"][test_idx], codebook, device)
        print(f"  Beam: top1={beam_res['top1_acc']:.1f}%, top3={beam_res['top3_acc']:.1f}%, gain={beam_res['gain_ratio']:.1f}%")

        # --- Localization ---
        # Train on degraded channels, test on baseline channels
        dg_loc_features = get_loc_features(ds)
        dg_loc_labels = get_loc_labels(ds)

        model_l = Localizer(dg_loc_features.shape[1])
        tr_dl, va_dl = make_loaders(dg_loc_features, dg_loc_labels, train_idx, val_idx)
        model_l = train_model(model_l, tr_dl, va_dl, "loc", device=device)

        # Test on baseline (reality)
        loc_res = eval_loc(model_l, bl_data["loc_features"][test_idx],
                           bl_data["loc_labels"][test_idx], device)
        print(f"  Loc:  RMSE={loc_res['rmse_m']:.3f}m, median={loc_res['median_m']:.3f}m")

        results.append({
            "config": cfg_name, "baseline": bl_name, "tag": tag,
            "param_value": param_val, "param_label": param_lbl,
            "nmse_dB": nmse_dB,
            "beam_top1": beam_res["top1_acc"], "beam_top3": beam_res["top3_acc"],
            "beam_gain": beam_res["gain_ratio"],
            "loc_rmse": loc_res["rmse_m"], "loc_median": loc_res["median_m"],
            "loc_p90": loc_res["p90_m"],
            "is_baseline": False,
        })

    return results


def print_results_table(results: list[dict]):
    print(f"\n{'='*100}")
    print(f"  Step 6: ML Task Validation Results")
    print(f"{'='*100}")
    print(f"{'Config':<22s} {'Tag':<11s} {'NMSE':>7s} {'Beam T1':>8s} {'Beam T3':>8s} "
          f"{'Gain%':>6s} {'Loc RMSE':>9s} {'Loc Med':>8s}")
    print("-" * 100)

    for r in results:
        nmse = f"{r['nmse_dB']:+.1f}" if r["nmse_dB"] > -100 else "ref"
        gain = f"{r['beam_gain']:.1f}" if "beam_gain" in r else "---"
        print(f"{r['config']:<22s} {r['tag']:<11s} {nmse:>7s} "
              f"{r['beam_top1']:>7.1f}% {r['beam_top3']:>7.1f}% "
              f"{gain:>6s} {r['loc_rmse']:>8.3f}m {r['loc_median']:>7.3f}m")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 6: ML Task Validation")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config names to run (default: all selected)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "results"
    if args.output:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter configs if specified
    configs = SELECTED_CONFIGS
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs]

    print("=" * 60)
    print("  Step 6: ML Task Validation")
    print(f"  Configs: {len(configs)}")
    print(f"  Codebook: {N_BEAMS} DFT beams, 8 TX antennas")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # Run
    results = run_experiment(configs, output_dir, seed=args.seed, device=args.device)

    # Print table
    print_results_table(results)

    # Save results
    out_path = output_dir / "ml_task_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Plot
    plot_results(results, output_dir)

    print("\n" + "=" * 60)
    print("  Step 6 Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
