"""
Step 5: Correlation Analysis
Compute correlations between physical fidelity metrics and channel NMSE.
Identifies which physical metrics best predict channel accuracy degradation.

Usage:
    python -m fidelity.correlation_analysis
    python -m fidelity.correlation_analysis --scenario simple_street_canyon
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm
from fidelity.analyze_channels import compute_channel_nmse
from fidelity.quantify_fidelity import (
    compute_geometry_metrics,
    compute_hardware_metrics,
    compute_material_metrics,
    compute_ray_tracing_metrics,
)

# ============================================================================
# Baseline selection logic (same as quantify_fidelity.py)
# ============================================================================

def get_baseline_for_config(cfg_name: str) -> str:
    """Determine the appropriate baseline for a given config."""
    if "mat_" in cfg_name:
        return "baseline_ds"
    if "hw_" in cfg_name:
        return "hw_baseline_4x4"
    if "rt_depth_" in cfg_name:
        return "rt_depth_10"
    return "baseline"


# ============================================================================
# Data collection
# ============================================================================

def collect_metrics(scenario: str) -> list[dict]:
    """Load all config pairs and compute both physical metrics and channel NMSE.

    Returns a list of dicts, one per (baseline, degraded) pair, with:
      - config_name, tag
      - Physical: blockage_error, toa_rmse_ns, nlos_pl_rmse_dB,
                  path_count_ratio, energy_dev_dB, capacity_ratio
      - Component scores: geo_score, mat_score, rt_score, hw_score, unified_score
      - Channel: channel_nmse_dB
    """
    configs = [
        # Geometry
        ("geo_noise_0_1m", "geometry"), ("geo_noise_0_5m", "geometry"),
        ("geo_noise_1m", "geometry"), ("geo_noise_2m", "geometry"),
        ("geo_noise_3m", "geometry"), ("geo_noise_4m", "geometry"),
        ("geo_noise_5m", "geometry"), ("geo_noise_7m", "geometry"),
        ("geo_noise_10m", "geometry"),
        ("geo_height_noise_3m", "geometry"),
        ("geo_remove_small", "geometry"), ("geo_remove_30pct", "geometry"),
        # Material
        ("mat_all_concrete", "material"), ("mat_all_glass", "material"),
        ("mat_all_metal", "material"), ("mat_all_wood", "material"),
        ("mat_all_marble", "material"), ("mat_all_brick", "material"),
        ("mat_no_scattering", "material"),
        # Ray Tracing — depth
        ("rt_depth_0", "ray_tracing"), ("rt_depth_1", "ray_tracing"),
        ("rt_depth_2", "ray_tracing"), ("rt_depth_3", "ray_tracing"),
        ("rt_depth_4", "ray_tracing"), ("rt_depth_5", "ray_tracing"),
        ("rt_depth_6", "ray_tracing"), ("rt_depth_7", "ray_tracing"),
        ("rt_depth_8", "ray_tracing"), ("rt_depth_9", "ray_tracing"),
        # Ray Tracing — rays
        ("rt_500k_rays", "ray_tracing"), ("rt_200k_rays", "ray_tracing"),
        ("rt_low_rays", "ray_tracing"), ("rt_50k_rays", "ray_tracing"),
        ("rt_20k_rays", "ray_tracing"), ("rt_very_low_rays", "ray_tracing"),
        ("rt_5k_rays", "ray_tracing"), ("rt_1k_rays", "ray_tracing"),
        ("rt_with_diffraction", "ray_tracing"),
        # Hardware
        ("hw_4x4_dipole", "hardware"), ("hw_4x4_iso", "hardware"),
        ("hw_4x4_spacing04", "hardware"), ("hw_4x4_polh", "hardware"),
    ]

    rows = []
    for cfg_name, tag in configs:
        bl_name = get_baseline_for_config(cfg_name)
        try:
            bl_ds = dm.load(cfg_name.replace(cfg_name, bl_name))
            dg_ds = dm.load(cfg_name)
        except Exception as e:
            print(f"  Skip {cfg_name}: {e}")
            continue

        # Physical metrics
        geo = compute_geometry_metrics(bl_ds, dg_ds)
        mat = compute_material_metrics(bl_ds, dg_ds)
        rt = compute_ray_tracing_metrics(bl_ds, dg_ds)
        hw = compute_hardware_metrics(bl_ds, dg_ds)

        # Channel NMSE
        nmse_res = compute_channel_nmse(bl_ds.channels, dg_ds.channels)
        nmse_dB = nmse_res.get("channel_nmse_dB", float("nan"))

        unified = (geo["score"] + mat["score"] + rt["score"] + hw["score"]) / 4.0

        rows.append({
            "config_name": cfg_name,
            "baseline": bl_name,
            "tag": tag,
            # Physical metrics (raw)
            "blockage_error": geo["blockage_error_rate"],
            "toa_rmse_ns": geo["toa_rmse_ns"],
            "nlos_pl_rmse_dB": mat["nlos_power_rmse_dB"],
            "path_count_ratio": rt["path_count_ratio"],
            "energy_dev_dB": abs(rt["global_energy_deviation_dB"]),
            "capacity_ratio": hw["capacity_ratio"],
            # Component scores
            "geo_score": geo["score"],
            "mat_score": mat["score"],
            "rt_score": rt["score"],
            "hw_score": hw["score"],
            "unified_score": unified,
            # Channel accuracy
            "channel_nmse_dB": nmse_dB,
        })
        print(f"  {cfg_name:<22s}  NMSE={nmse_dB:+7.2f} dB  unified={unified:.3f}")

    return rows


# ============================================================================
# Correlation computation
# ============================================================================

def compute_correlations(rows: list[dict]) -> dict:
    """Compute Pearson and Spearman correlations between metrics and channel NMSE."""
    physical_metrics = [
        "blockage_error", "toa_rmse_ns", "nlos_pl_rmse_dB",
        "energy_dev_dB", "unified_score",
        "geo_score", "mat_score", "rt_score", "hw_score",
    ]

    nmse = np.array([r["channel_nmse_dB"] for r in rows])
    valid = np.isfinite(nmse)

    correlations = {}
    for metric in physical_metrics:
        vals = np.array([r[metric] for r in rows])
        mask = valid & np.isfinite(vals)
        if mask.sum() < 4:
            correlations[metric] = {"pearson_r": float("nan"), "spearman_rho": float("nan")}
            continue

        x, y = vals[mask], nmse[mask]
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        correlations[metric] = {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
            "n_samples": int(mask.sum()),
        }

    return correlations


def compute_per_tag_correlations(rows: list[dict]) -> dict:
    """Compute correlations within each component tag."""
    tags = sorted(set(r["tag"] for r in rows))
    # Map tag -> most relevant physical metric
    tag_metric = {
        "geometry": ["blockage_error", "toa_rmse_ns"],
        "material": ["nlos_pl_rmse_dB"],
        "ray_tracing": ["energy_dev_dB", "path_count_ratio"],
        "hardware": ["capacity_ratio"],
    }

    per_tag = {}
    for tag in tags:
        tag_rows = [r for r in rows if r["tag"] == tag]
        nmse = np.array([r["channel_nmse_dB"] for r in tag_rows])
        valid = np.isfinite(nmse)
        metrics = tag_metric.get(tag, [])

        per_tag[tag] = {"n_configs": len(tag_rows)}
        for metric in metrics:
            vals = np.array([r[metric] for r in tag_rows])
            mask = valid & np.isfinite(vals)
            if mask.sum() < 4:
                per_tag[tag][metric] = {"pearson_r": float("nan")}
                continue
            x, y = vals[mask], nmse[mask]
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            per_tag[tag][metric] = {
                "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_rho": float(sr), "spearman_p": float(sp),
            }

    return per_tag


# ============================================================================
# Plotting
# ============================================================================

def plot_correlation_matrix(rows: list[dict], output_dir: Path):
    """Plot scatter matrix: physical metrics vs Channel NMSE, colored by tag."""
    tag_colors = {
        "geometry": "tab:red", "material": "tab:blue",
        "ray_tracing": "tab:green", "hardware": "tab:orange",
    }
    tag_markers = {
        "geometry": "o", "material": "s",
        "ray_tracing": "^", "hardware": "D",
    }

    metrics = [
        ("unified_score", "Unified Fidelity Score"),
        ("blockage_error", "Blockage Error Rate"),
        ("toa_rmse_ns", "ToA RMSE (ns)"),
        ("nlos_pl_rmse_dB", "NLoS PL RMSE (dB)"),
        ("energy_dev_dB", "|Energy Deviation| (dB)"),
    ]

    nmse = np.array([r["channel_nmse_dB"] for r in rows])

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for tag in tag_colors:
            tag_rows = [r for r in rows if r["tag"] == tag]
            if not tag_rows:
                continue
            x = [r[metric_key] for r in tag_rows]
            y = [r["channel_nmse_dB"] for r in tag_rows]
            ax.scatter(x, y, c=tag_colors[tag], marker=tag_markers[tag],
                       label=tag.replace("_", " ").title(), alpha=0.7, s=40)

        # Trend line (all data)
        vals = np.array([r[metric_key] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(nmse)
        if mask.sum() >= 3:
            z = np.polyfit(vals[mask], nmse[mask], 1)
            p = np.poly1d(z)
            x_range = np.linspace(vals[mask].min(), vals[mask].max(), 50)
            ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.5)
            pr, _ = stats.pearsonr(vals[mask], nmse[mask])
            ax.set_title(f"{metric_label}\nr={pr:.3f}", fontsize=10)
        else:
            ax.set_title(metric_label, fontsize=10)

        ax.set_xlabel(metric_label, fontsize=9)
        ax.set_ylabel("Channel NMSE (dB)", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_scatter.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {output_dir / 'correlation_scatter.png'}")
    plt.close()


def plot_component_scores_vs_nmse(rows: list[dict], output_dir: Path):
    """Plot each component score vs Channel NMSE."""
    tag_colors = {
        "geometry": "tab:red", "material": "tab:blue",
        "ray_tracing": "tab:green", "hardware": "tab:orange",
    }

    score_metrics = [
        ("geo_score", "Geometry Score"),
        ("mat_score", "Material Score"),
        ("rt_score", "Ray Tracing Score"),
        ("hw_score", "Hardware Score"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (score_key, score_label) in zip(axes, score_metrics):
        for tag in tag_colors:
            tag_rows = [r for r in rows if r["tag"] == tag]
            if not tag_rows:
                continue
            x = [r[score_key] for r in tag_rows]
            y = [r["channel_nmse_dB"] for r in tag_rows]
            ax.scatter(x, y, c=tag_colors[tag], marker="o", alpha=0.7, s=40,
                       label=tag.replace("_", " ").title())

        # Compute correlation for this score across all data
        vals = np.array([r[score_key] for r in rows])
        nmse = np.array([r["channel_nmse_dB"] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(nmse)
        if mask.sum() >= 3:
            pr, pp = stats.pearsonr(vals[mask], nmse[mask])
            ax.set_title(f"{score_label}\nr={pr:.3f} (p={pp:.1e})", fontsize=10)
        else:
            ax.set_title(score_label, fontsize=10)

        ax.set_xlabel(score_label)
        ax.set_ylabel("Channel NMSE (dB)")
        ax.grid(True, alpha=0.3)

    axes[-1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "component_scores_vs_nmse.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {output_dir / 'component_scores_vs_nmse.png'}")
    plt.close()


def plot_unified_score_vs_nmse(rows: list[dict], output_dir: Path):
    """Single focused plot: Unified Fidelity Score vs Channel NMSE."""
    tag_colors = {
        "geometry": "tab:red", "material": "tab:blue",
        "ray_tracing": "tab:green", "hardware": "tab:orange",
    }
    tag_markers = {
        "geometry": "o", "material": "s",
        "ray_tracing": "^", "hardware": "D",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for tag in tag_colors:
        tag_rows = [r for r in rows if r["tag"] == tag]
        if not tag_rows:
            continue
        x = [r["unified_score"] for r in tag_rows]
        y = [r["channel_nmse_dB"] for r in tag_rows]
        ax.scatter(x, y, c=tag_colors[tag], marker=tag_markers[tag],
                   label=tag.replace("_", " ").title(), alpha=0.8, s=60)

    # Trend line
    us = np.array([r["unified_score"] for r in rows])
    nmse = np.array([r["channel_nmse_dB"] for r in rows])
    mask = np.isfinite(us) & np.isfinite(nmse)
    if mask.sum() >= 3:
        pr, pp = stats.pearsonr(us[mask], nmse[mask])
        sr, sp = stats.spearmanr(us[mask], nmse[mask])
        z = np.polyfit(us[mask], nmse[mask], 1)
        p = np.poly1d(z)
        x_range = np.linspace(us[mask].min(), us[mask].max(), 50)
        ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.6, linewidth=2)
        ax.set_title(f"Unified Score vs Channel NMSE\nPearson r={pr:.3f} (p={pp:.1e}), "
                     f"Spearman ρ={sr:.3f}", fontsize=11)

    ax.set_xlabel("Unified Fidelity Score", fontsize=11)
    ax.set_ylabel("Channel NMSE (dB)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "unified_score_vs_nmse.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {output_dir / 'unified_score_vs_nmse.png'}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Step 5: Correlation Analysis")
    parser.add_argument("--scenario", type=str, default="simple_street_canyon")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / "results"
    if args.output:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 5: Correlation Analysis")
    print(f"  Scenario: {args.scenario}")
    print("=" * 60)

    # 1. Collect all metrics
    print("\n[1/4] Collecting physical metrics + Channel NMSE...")
    rows = collect_metrics(args.scenario)
    print(f"\n  Collected {len(rows)} config pairs")

    if len(rows) < 5:
        print("Error: Not enough data points for correlation analysis.")
        return

    # 2. Compute correlations
    print("\n[2/4] Computing correlations...")
    correlations = compute_correlations(rows)
    per_tag = compute_per_tag_correlations(rows)

    # Print summary
    print("\n  Global Correlations (metric vs Channel NMSE):")
    print(f"  {'Metric':<22s}  {'Pearson r':>10s}  {'Spearman ρ':>11s}  {'n':>4s}")
    print("  " + "-" * 55)
    for metric, vals in sorted(correlations.items(),
                                key=lambda x: abs(x[1].get("pearson_r", 0)), reverse=True):
        pr = vals.get("pearson_r", float("nan"))
        sr = vals.get("spearman_rho", float("nan"))
        n = vals.get("n_samples", 0)
        print(f"  {metric:<22s}  {pr:>+10.4f}  {sr:>+11.4f}  {n:>4d}")

    print("\n  Per-Tag Correlations:")
    for tag, tvals in per_tag.items():
        print(f"\n  [{tag}] ({tvals['n_configs']} configs)")
        for k, v in tvals.items():
            if k == "n_configs":
                continue
            pr = v.get("pearson_r", float("nan"))
            sr = v.get("spearman_rho", float("nan"))
            print(f"    {k:<22s}  r={pr:+.4f}  ρ={sr:+.4f}")

    # 3. Generate plots
    print("\n[3/4] Generating plots...")
    plot_correlation_matrix(rows, output_dir)
    plot_component_scores_vs_nmse(rows, output_dir)
    plot_unified_score_vs_nmse(rows, output_dir)

    # 4. Save results
    print("\n[4/4] Saving results...")
    results = {
        "scenario": args.scenario,
        "n_configs": len(rows),
        "global_correlations": correlations,
        "per_tag_correlations": per_tag,
        "raw_data": rows,
    }
    out_path = output_dir / "correlation_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("  Correlation Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
