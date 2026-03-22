"""Channel Analysis for Fidelity Experiments.

Compare degraded vs baseline CSI datasets and compute channel accuracy metrics.

Usage:
    python -m fidelity.analyze_channels --baseline results/baseline --degraded results/geo_noise_5m
    python -m fidelity.analyze_channels --results-dir results/  # compare all vs baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deepmimo as dm  # noqa: E402


# ============================================================================
# Channel Metrics
# ============================================================================


def compute_los_accuracy(baseline_los: np.ndarray, degraded_los: np.ndarray) -> dict:
    """Compare LoS probability between baseline and degraded.

    Returns:
        Dict with LoS accuracy, false positive rate, false negative rate.
    """
    bl = baseline_los.astype(bool).flatten()
    dg = degraded_los.astype(bool).flatten()

    n = len(bl)
    accuracy = np.mean(bl == dg)
    false_pos = np.mean(dg & ~bl)  # degraded says LoS but baseline says NLoS
    false_neg = np.mean(~dg & bl)  # degraded says NLoS but baseline says LoS

    return {
        "los_accuracy": float(accuracy),
        "los_false_positive_rate": float(false_pos),
        "los_false_negative_rate": float(false_neg),
        "baseline_los_fraction": float(bl.mean()),
        "degraded_los_fraction": float(dg.mean()),
        "n_samples": int(n),
    }


def compute_path_loss_error(
    baseline_pwr: np.ndarray, degraded_pwr: np.ndarray
) -> dict:
    """Compute path loss error between baseline and degraded.

    Args:
        baseline_pwr: Baseline power [N_UE, N_paths] (linear scale).
        degraded_pwr: Degraded power [N_UE, N_paths] (linear scale).

    Returns:
        Dict with RMSE, MAE, and correlation of total received power.
    """
    # Total received power (sum over paths)
    bl_total = baseline_pwr.sum(axis=-1) if baseline_pwr.ndim > 1 else baseline_pwr
    dg_total = degraded_pwr.sum(axis=-1) if degraded_pwr.ndim > 1 else degraded_pwr

    # Convert to dB (avoid log of zero)
    eps = 1e-30
    bl_db = 10 * np.log10(np.maximum(bl_total, eps))
    dg_db = 10 * np.log10(np.maximum(dg_total, eps))

    # Metrics
    diff = bl_db - dg_db
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    # Correlation
    valid = np.isfinite(bl_db) & np.isfinite(dg_db)
    if valid.sum() > 2:
        corr = float(np.corrcoef(bl_db[valid], dg_db[valid])[0, 1])
    else:
        corr = float("nan")

    return {
        "path_loss_rmse_dB": rmse,
        "path_loss_mae_dB": mae,
        "path_loss_bias_dB": bias,
        "path_loss_correlation": corr,
    }


def compute_delay_spread_error(
    baseline_ds, degraded_ds,
) -> dict:
    """Compute delay spread deviation between baseline and degraded datasets.

    Uses the DeepMIMO dataset's delay_spread attribute if available,
    otherwise computes from ToA and power.
    """
    bl = np.array(baseline_ds).flatten()
    dg = np.array(degraded_ds).flatten()

    # Filter valid entries
    valid = np.isfinite(bl) & np.isfinite(dg) & (bl > 0) & (dg > 0)
    if valid.sum() < 2:
        return {"delay_spread_rmse_ns": float("nan"), "delay_spread_correlation": float("nan")}

    bl_v, dg_v = bl[valid], dg[valid]

    diff = bl_v - dg_v
    rmse = float(np.sqrt(np.mean(diff**2))) * 1e9  # Convert to ns

    corr = float(np.corrcoef(bl_v, dg_v)[0, 1])

    return {
        "delay_spread_rmse_ns": rmse,
        "delay_spread_correlation": corr,
    }


def compute_channel_nmse(
    baseline_channels: np.ndarray, degraded_channels: np.ndarray
) -> dict:
    """Compute Normalized Mean Squared Error between channel matrices.

    NMSE = ||H_base - H_deg||^2 / ||H_base||^2

    Args:
        baseline_channels: Baseline channel coefficients [N_UE, N_RX_ant, N_TX_ant, N_paths].
        degraded_channels: Degraded channel coefficients [N_UE, N_RX_ant, N_TX_ant, N_paths].
    """
    n_ue = min(baseline_channels.shape[0], degraded_channels.shape[0])
    bl = baseline_channels[:n_ue]
    dg = degraded_channels[:n_ue]

    # Per-user NMSE
    nmse_per_user = []
    for i in range(n_ue):
        bl_norm = np.linalg.norm(bl[i]) ** 2
        if bl_norm > 1e-30:
            nmse = np.linalg.norm(bl[i] - dg[i]) ** 2 / bl_norm
            nmse_per_user.append(nmse)

    if not nmse_per_user:
        return {"channel_nmse_dB": float("nan"), "channel_nmse_linear": float("nan")}

    nmse_arr = np.array(nmse_per_user)
    mean_nmse = float(nmse_arr.mean())
    median_nmse = float(np.median(nmse_arr))

    return {
        "channel_nmse_linear": mean_nmse,
        "channel_nmse_dB": float(10 * np.log10(max(mean_nmse, 1e-30))),
        "channel_nmse_median_dB": float(10 * np.log10(max(median_nmse, 1e-30))),
    }


# ============================================================================
# Main Comparison
# ============================================================================


def compare_datasets(baseline_name: str, degraded_name: str) -> dict:
    """Load two DeepMIMO datasets and compute all comparison metrics.

    Args:
        baseline_name: DeepMIMO scenario name for the baseline.
        degraded_name: DeepMIMO scenario name for the degraded variant.

    Returns:
        Dict of all computed metrics.
    """
    print(f"Loading baseline: {baseline_name}")
    bl_ds = dm.load(baseline_name)

    print(f"Loading degraded: {degraded_name}")
    dg_ds = dm.load(degraded_name)

    results = {
        "baseline": baseline_name,
        "degraded": degraded_name,
        "n_ue_baseline": int(bl_ds.n_ue),
        "n_ue_degraded": int(dg_ds.n_ue),
    }

    # LoS comparison
    if hasattr(bl_ds, "los") and hasattr(dg_ds, "los"):
        results["los"] = compute_los_accuracy(bl_ds.los, dg_ds.los)

    # Path loss comparison
    if hasattr(bl_ds, "pwr") and hasattr(dg_ds, "pwr"):
        results["path_loss"] = compute_path_loss_error(bl_ds.pwr, dg_ds.pwr)

    # Channel NMSE
    if hasattr(bl_ds, "channels") and hasattr(dg_ds, "channels"):
        results["channel_nmse"] = compute_channel_nmse(bl_ds.channels, dg_ds.channels)

    # Delay spread (if available)
    if hasattr(bl_ds, "toa") and hasattr(dg_ds, "toa"):
        # Compute RMS delay spread from ToA and power
        try:
            bl_toa = bl_ds.toa
            dg_toa = dg_ds.toa
            bl_pwr = bl_ds.pwr
            dg_pwr = dg_ds.pwr

            # RMS delay spread per user
            def rms_delay_spread(toa, pwr):
                """Compute RMS delay spread."""
                total_pwr = pwr.sum(axis=-1, keepdims=True)
                total_pwr = np.maximum(total_pwr, 1e-30)
                mean_delay = (toa * pwr).sum(axis=-1, keepdims=True) / total_pwr
                ds = np.sqrt(
                    (pwr * (toa - mean_delay) ** 2).sum(axis=-1) / total_pwr.squeeze()
                )
                return ds

            bl_ds_val = rms_delay_spread(bl_toa, bl_pwr)
            dg_ds_val = rms_delay_spread(dg_toa, dg_pwr)
            results["delay_spread"] = compute_delay_spread_error(bl_ds_val, dg_ds_val)
        except Exception as e:
            print(f"  Warning: Could not compute delay spread: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and degraded fidelity datasets.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline DeepMIMO scenario name",
    )
    parser.add_argument(
        "--degraded",
        type=str,
        nargs="+",
        required=True,
        help="Degraded DeepMIMO scenario name(s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    all_results = {}
    for degraded in args.degraded:
        print(f"\n{'=' * 50}")
        print(f"  Comparing: {args.baseline} vs {degraded}")
        print(f"{'=' * 50}")

        results = compare_datasets(args.baseline, degraded)
        all_results[degraded] = results

        # Print summary
        print(f"\n  Summary for {degraded}:")
        if "los" in results:
            print(f"    LoS accuracy: {results['los']['los_accuracy']:.3f}")
        if "path_loss" in results:
            print(f"    Path loss RMSE: {results['path_loss']['path_loss_rmse_dB']:.2f} dB")
            print(f"    Path loss correlation: {results['path_loss']['path_loss_correlation']:.4f}")
        if "channel_nmse" in results:
            print(f"    Channel NMSE: {results['channel_nmse']['channel_nmse_dB']:.2f} dB")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = str(Path(__file__).resolve().parent / "results" / "channel_analysis.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
