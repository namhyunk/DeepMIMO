"""
Step 3: Fidelity Quantification
Compute physical/geometric metrics and rank fidelity components.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm


def to_linear(pwr_dB):
    """DeepMIMO pwr is stored as 20*log10(amp) which is 10*log10(power)."""
    return 10 ** (pwr_dB / 10.0)


def compute_geometry_metrics(bl_ds, dg_ds):
    # 1. Blockage Error (LoS discrepancy rate)
    bl_los = bl_ds.los.flatten()
    dg_los = dg_ds.los.flatten()
    blockage_error = np.mean(bl_los != dg_los)

    # 2. ToA / Path Length RMSE (Distance Distribution Error)
    eps = 1e-30
    bl_pwr_lin = to_linear(bl_ds.pwr)
    dg_pwr_lin = to_linear(dg_ds.pwr)
    
    bl_pwr = np.nansum(bl_pwr_lin, axis=-1, keepdims=True) + eps
    dg_pwr = np.nansum(dg_pwr_lin, axis=-1, keepdims=True) + eps

    bl_mean_toa = np.nansum(bl_ds.toa * bl_pwr_lin, axis=-1, keepdims=True) / bl_pwr
    dg_mean_toa = np.nansum(dg_ds.toa * dg_pwr_lin, axis=-1, keepdims=True) / dg_pwr

    # valid UEs where both have some power
    valid = (bl_pwr.squeeze() > 1e-25) & (dg_pwr.squeeze() > 1e-25)
    if np.sum(valid) > 0:
        toa_rmse = np.sqrt(np.mean((bl_mean_toa[valid] - dg_mean_toa[valid])**2)) * 1e9  # in ns
    else:
        toa_rmse = float('nan')

    score = max(0.0, 1.0 - blockage_error)
    # penalty for large timing errors: 100ns difference = 0 score
    if not np.isnan(toa_rmse):
        score *= max(0.0, 1.0 - (toa_rmse / 100.0))

    return {
        "blockage_error_rate": float(blockage_error),
        "toa_rmse_ns": float(toa_rmse),
        "score": float(score)
    }


def compute_material_metrics(bl_ds, dg_ds):
    # NLoS Power Deviation — use intersection of NLoS masks
    # so only users that are NLoS in BOTH datasets are compared
    bl_los = bl_ds.los.flatten().astype(bool)
    dg_los = dg_ds.los.flatten().astype(bool)
    nlos_mask = (~bl_los) & (~dg_los)  # NLoS in both baseline AND degraded

    if np.sum(nlos_mask) == 0:
        return {"nlos_power_rmse_dB": 0.0, "n_nlos_compared": 0, "score": 1.0}

    bl_pwr_lin = to_linear(bl_ds.pwr)
    dg_pwr_lin = to_linear(dg_ds.pwr)
    
    bl_nlos_pwr = np.nansum(bl_pwr_lin[nlos_mask], axis=-1)
    dg_nlos_pwr = np.nansum(dg_pwr_lin[nlos_mask], axis=-1)

    eps = 1e-30
    bl_dB = 10 * np.log10(np.maximum(bl_nlos_pwr, eps))
    dg_dB = 10 * np.log10(np.maximum(dg_nlos_pwr, eps))

    valid = np.isfinite(bl_dB) & np.isfinite(dg_dB)
    if np.sum(valid) > 0:
        rmse_dB = np.sqrt(np.mean((bl_dB[valid] - dg_dB[valid])**2))
    else:
        rmse_dB = float('nan')

    # penalty scales with 20dB being "very bad" (0 score)
    return {
        "nlos_power_rmse_dB": float(rmse_dB),
        "n_nlos_compared": int(np.sum(nlos_mask)),
        "score": float(max(0, 1.0 - (rmse_dB / 20.0))) if not np.isnan(rmse_dB) else 0.0 
    }


def compute_ray_tracing_metrics(bl_ds, dg_ds):
    # Path Count Difference. Ignore padded nans
    bl_paths = np.sum(~np.isnan(bl_ds.pwr), axis=-1)
    dg_paths = np.sum(~np.isnan(dg_ds.pwr), axis=-1)

    mean_bl_paths = np.mean(bl_paths)
    mean_dg_paths = np.mean(dg_paths)

    path_count_ratio = mean_dg_paths / max(mean_bl_paths, 1e-5)

    # Energy capture difference
    bl_total_energy = np.nansum(to_linear(bl_ds.pwr))
    dg_total_energy = np.nansum(to_linear(dg_ds.pwr))

    eps = 1e-30
    energy_dev_dB = 10 * np.log10(max(dg_total_energy, eps) / max(bl_total_energy, eps))

    # Score: if we lose paths, ratio drops below 1. If energy deviates, penalty.
    score_paths = min(1.0, path_count_ratio)
    score_energy = max(0.0, 1.0 - abs(energy_dev_dB)/20.0) # 20dB loss = 0 score
    return {
        "path_count_ratio": float(path_count_ratio),
        "mean_paths_baseline": float(mean_bl_paths),
        "mean_paths_degraded": float(mean_dg_paths),
        "global_energy_deviation_dB": float(energy_dev_dB),
        "score": float((score_paths + score_energy) / 2.0)
    }


def compute_hardware_metrics(bl_ds, dg_ds):
    # Pattern Mismatch / SNR -> Channel Capacity Ratio
    n_ue = min(bl_ds.channels.shape[0], dg_ds.channels.shape[0])
    bl = bl_ds.channels[:n_ue]
    dg = dg_ds.channels[:n_ue]

    snr_linear = 10**(10 / 10.0) # 10 dB SNR 
    bl_capacities = []
    dg_capacities = []
    
    for i in range(n_ue):
        # average over subcarriers if present
        bl_h = np.mean(bl[i], axis=-1) if bl[i].ndim >= 3 else bl[i]
        dg_h = np.mean(dg[i], axis=-1) if dg[i].ndim >= 3 else dg[i]
        
        # ensure 2D: [N_RX, N_TX]
        bl_h = np.atleast_2d(bl_h.squeeze())
        dg_h = np.atleast_2d(dg_h.squeeze())
        
        # capacity: log2(det(I + SNR * H * H^H))
        bl_I = np.eye(bl_h.shape[0])
        dg_I = np.eye(dg_h.shape[0])
        bl_cap = np.real(np.log2(np.linalg.det(bl_I + snr_linear * bl_h @ bl_h.conj().T)))
        dg_cap = np.real(np.log2(np.linalg.det(dg_I + snr_linear * dg_h @ dg_h.conj().T)))
        
        bl_capacities.append(max(0.0, float(bl_cap)))
        dg_capacities.append(max(0.0, float(dg_cap)))

    bl_c_mean = np.mean(bl_capacities)
    dg_c_mean = np.mean(dg_capacities)
    ratio = dg_c_mean / max(bl_c_mean, 1e-10)

    # Score based on capacity deviation (1.0 is exactly identical capacity)
    score = max(0.0, 1.0 - abs(1.0 - ratio))

    return {
        "capacity_mean": dg_c_mean,
        "capacity_ratio": ratio,
        "score": float(score)
    }


def main():
    parser = argparse.ArgumentParser(description="Quantify fidelity components.")
    parser.add_argument("--baseline", type=str, default="baseline")
    parser.add_argument("--scenario", type=str, default="simple_street_canyon")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results" / args.scenario

    def quantify_scenario(bl_name, dg_name):
        try:
            bl_ds = dm.load(str(results_dir / bl_name))
            dg_ds = dm.load(str(results_dir / dg_name))

            geo = compute_geometry_metrics(bl_ds, dg_ds)
            mat = compute_material_metrics(bl_ds, dg_ds)
            rt = compute_ray_tracing_metrics(bl_ds, dg_ds)
            hw = compute_hardware_metrics(bl_ds, dg_ds)

            unified_score = (geo["score"] + mat["score"] + rt["score"] + hw["score"]) / 4.0

            return {
                "config_name": dg_name,
                "unified_fidelity_score": float(unified_score),
                "geometry": geo,
                "material": mat,
                "ray_tracing": rt,
                "hardware": hw
            }
        except Exception as e:
            print(f"Error processing {dg_name}: {e}")
            return None

    configs = [
        # Geometry
        "geo_noise_0_1m", "geo_noise_0_5m", "geo_noise_1m", "geo_noise_2m",
        "geo_noise_3m", "geo_noise_4m", "geo_noise_5m", "geo_noise_7m", "geo_noise_10m",
        "geo_height_noise_3m", "geo_remove_small", "geo_remove_30pct",
        # Material
        "baseline_ds", "mat_all_concrete", "mat_all_glass", "mat_all_metal",
        "mat_all_wood", "mat_all_marble", "mat_all_brick", "mat_no_scattering",
        # Ray Tracing
        "rt_depth_0", "rt_depth_1", "rt_depth_2", "rt_depth_3", "rt_depth_4",
        "rt_depth_5", "rt_depth_6", "rt_depth_7", "rt_depth_8", "rt_depth_9", "rt_depth_10",
        "rt_500k_rays", "rt_200k_rays", "rt_low_rays", "rt_50k_rays",
        "rt_20k_rays", "rt_very_low_rays", "rt_5k_rays", "rt_1k_rays",
        "rt_with_diffraction",
        # Hardware
        "hw_baseline_4x4", "hw_4x4_dipole", "hw_4x4_iso",
        "hw_4x4_spacing04", "hw_4x4_polh"
    ]
    
    all_results = {}
    print("=" * 105)
    print(f"{'Config Name':<20} | {'Fid Score':<12} | {'Domain Gap Cause'}")
    print("=" * 105)

    for cfg in configs:
        # Determine baseline for this config
        current_baseline = args.baseline
        if "mat_" in cfg:
            current_baseline = "baseline_ds"
        elif "hw_" in cfg:
            current_baseline = "hw_baseline_4x4"
        elif "rt_depth_" in cfg:
            current_baseline = "rt_depth_10"

        res = quantify_scenario(current_baseline, cfg)
        if res is None:
            continue
        
        # Override baseline against itself so it doesn't get score 0
        if cfg == current_baseline:
            res = {
                "config_name": current_baseline,
                "unified_fidelity_score": 1.0,
                "geometry": {"score": 1.0},
                "material": {"score": 1.0},
                "ray_tracing": {"score": 1.0},
                "hardware": {"score": 1.0}
            }
            
        all_results[cfg] = res
        
        scores = {
            "Geometry": res["geometry"]["score"],
            "Material": res["material"]["score"],
            "RayTracing": res["ray_tracing"]["score"],
            "Hardware": res["hardware"]["score"],
        }
        # Dominant source of error is the one with the lowest score
        dominant = min(scores, key=scores.get)

        print(f"{cfg:<20} | {res['unified_fidelity_score']:<12.3f} | {dominant:<10} (Limiting Factor Score: {scores[dominant]:.3f})")

    out_path = args.output
    if not out_path:
        out_path = str(results_dir / "fidelity_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 105)
    print(f"Results successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
