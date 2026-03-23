"""
Step 4: Channel Impact Analysis
Generate Sensitivity Curve plots for Geometry, Ray Tracing, Material, and Hardware.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import deepmimo as dm
from fidelity.quantify_fidelity import (
    compute_geometry_metrics,
    compute_material_metrics,
    compute_ray_tracing_metrics,
    compute_hardware_metrics,
)
from fidelity.analyze_channels import compute_channel_nmse


def load_all_metrics(baseline_name, degraded_name):
    """Load datasets once and compute all metric categories."""
    bl_ds = dm.load(baseline_name)
    dg_ds = dm.load(degraded_name)

    geo = compute_geometry_metrics(bl_ds, dg_ds)
    mat = compute_material_metrics(bl_ds, dg_ds)
    rt = compute_ray_tracing_metrics(bl_ds, dg_ds)
    hw = compute_hardware_metrics(bl_ds, dg_ds)

    # Compute actual channel NMSE (not capacity ratio)
    if hasattr(bl_ds, "channels") and hasattr(dg_ds, "channels"):
        nmse_res = compute_channel_nmse(bl_ds.channels, dg_ds.channels)
        ch_nmse_dB = nmse_res.get("channel_nmse_dB", float('nan'))
    else:
        ch_nmse_dB = float('nan')

    return {
        "Channel_NMSE_dB": ch_nmse_dB,
        "Capacity_Ratio": hw["capacity_ratio"],
        "ToA_RMSE_ns": geo["toa_rmse_ns"],
        "Blockage_Error": geo["blockage_error_rate"] * 100,  # percentage
        "NLoS_PL_RMSE_dB": mat["nlos_power_rmse_dB"],
        "Energy_Dev_dB": rt["global_energy_deviation_dB"],
        "Path_Count_Ratio": rt["path_count_ratio"],
    }


def main():
    results_dir = Path(__file__).resolve().parent / "results"

    # --- 1. Geometry Analysis ---
    geo_names = [
        "baseline", "geo_noise_0_1m", "geo_noise_0_5m", "geo_noise_1m",
        "geo_noise_2m", "geo_noise_3m", "geo_noise_4m", "geo_noise_5m",
        "geo_noise_7m", "geo_noise_10m"
    ]
    geo_x = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    geo_nmse = [-80.0]  # baseline vs baseline floor
    geo_toa = [0.0]
    geo_blockage = [0.0]
    geo_nlos = [0.0]

    print("Analyzing Geometry Sensitivity...")
    for g in geo_names[1:]:
        res = load_all_metrics("baseline", g)
        geo_nmse.append(res["Channel_NMSE_dB"])
        geo_toa.append(res["ToA_RMSE_ns"])
        geo_blockage.append(res["Blockage_Error"])
        geo_nlos.append(res["NLoS_PL_RMSE_dB"])

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.plot(geo_x, geo_nmse, 'o-', color='red')
    # plt.axhline(y=-80, color='black', linestyle='--', alpha=0.3)
    plt.title('Channel NMSE vs. Position Noise')
    plt.xlabel('Gaussian Position Noise (m)')
    plt.ylabel('NMSE (dB)')
    # plt.ylim(bottom=-90) # Removed arbitrary floor
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(geo_x, geo_blockage, 's-', color='orange')
    plt.title('Blockage Error vs. Noise')
    plt.xlabel('Gaussian Position Noise (m)')
    plt.ylabel('LoS Discrepancy Rate (%)')
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(geo_x, geo_nlos, 'd-', color='purple')
    plt.title('NLoS PL RMSE vs. Noise')
    plt.xlabel('Gaussian Position Noise (m)')
    plt.ylabel('PL RMSE (dB)')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(geo_x, geo_toa, 'x-', color='blue')
    plt.title('Mean ToA RMSE vs. Noise')
    plt.xlabel('Gaussian Position Noise (m)')
    plt.ylabel('ToA RMSE (ns)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(results_dir / "geometry_sensitivity.png", dpi=200, bbox_inches='tight')
    print("Saved: geometry_sensitivity.png")

    # --- 2A. RT Analysis: Reflection Depth ---
    rt_depth_names = [
        "rt_depth_10", "rt_depth_9", "rt_depth_8", "rt_depth_7", "rt_depth_6",
        "rt_depth_5", "rt_depth_4", "rt_depth_3", "rt_depth_2", "rt_depth_1", "rt_depth_0"
    ]
    rt_depth_x = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    rt_depth_nmse = []
    rt_depth_energy = []

    print("Analyzing Ray Tracing Sensitivity (Depth)...")
    for r in rt_depth_names[1:]: # Skip baseline (depth 10) for plotting
        res = load_all_metrics("rt_depth_10", r)
        rt_depth_nmse.append(res["Channel_NMSE_dB"])
        rt_depth_energy.append(res["Energy_Dev_dB"])

    # Update x-axis to match the sliced names
    rt_depth_x = rt_depth_x[1:]

    print(f"Depth NMSE Values: {rt_depth_nmse}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rt_depth_x, rt_depth_nmse, 'o-', color='red', linewidth=2)
    plt.title('NMSE vs. Depth (Baseline: 10)')
    plt.xlabel('Max Reflections')
    plt.ylabel('NMSE (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rt_depth_x, rt_depth_energy, 's-', color='green', linewidth=2)
    plt.title('Energy Dev. vs. Depth (Baseline: 10)')
    plt.xlabel('Max Reflections')
    plt.ylabel('Deviation (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(results_dir / "rt_depth_sensitivity.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: rt_depth_sensitivity.png")

    # --- 2B. RT Analysis: Ray Count ---
    rt_ray_names = [
        "baseline", "rt_500k_rays", "rt_200k_rays", "rt_low_rays",
        "rt_50k_rays", "rt_20k_rays", "rt_very_low_rays", "rt_5k_rays", "rt_1k_rays"
    ]
    rt_ray_x = [1000000, 500000, 200000, 100000, 50000, 20000, 10000, 5000, 1000]

    rt_ray_nmse = []
    rt_ray_energy = []

    print("Analyzing Ray Tracing Sensitivity (Rays)...")
    for r in rt_ray_names[1:]: # Skip baseline (1M rays) for plotting
        res = load_all_metrics("baseline", r)
        rt_ray_nmse.append(res["Channel_NMSE_dB"])
        rt_ray_energy.append(res["Energy_Dev_dB"])

    # Update x-axis to match sliced names
    rt_ray_x = rt_ray_x[1:]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rt_ray_x, rt_ray_nmse, 'o-', color='purple', linewidth=2)
    plt.xscale('log')
    plt.title('NMSE vs. Rays (Baseline: 1M)')
    plt.xlabel('Number of Rays per TX')
    plt.ylabel('NMSE (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(rt_ray_x, rt_ray_energy, 's-', color='orange', linewidth=2)
    plt.xscale('log')
    plt.title('Energy Dev. vs. Rays (Baseline: 1M)')
    plt.xlabel('Number of Rays per TX')
    plt.ylabel('Deviation (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_dir / "rt_ray_sensitivity.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: rt_ray_sensitivity.png")

    # --- 2C. RT Analysis: Combined 4-panel Plot ---
    plt.figure(figsize=(10, 8))
    
    # 1. Depth NMSE
    plt.subplot(2, 2, 1)
    plt.plot(rt_depth_x, rt_depth_nmse, 'o-', color='red', linewidth=2)
    plt.title('NMSE vs. Depth (Baseline: 10)')
    plt.xlabel('Max Reflections')
    plt.ylabel('NMSE (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True)

    # 2. Depth Energy
    plt.subplot(2, 2, 2)
    plt.plot(rt_depth_x, rt_depth_energy, 's-', color='green', linewidth=2)
    plt.title('Energy Dev. vs. Depth (Baseline: 10)')
    plt.xlabel('Max Reflections')
    plt.ylabel('Deviation (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True)

    # 3. Ray NMSE
    plt.subplot(2, 2, 3)
    plt.plot(rt_ray_x, rt_ray_nmse, 'o-', color='purple', linewidth=2)
    plt.xscale('log')
    plt.title('NMSE vs. Rays (Baseline: 1M)')
    plt.xlabel('Number of Rays')
    plt.ylabel('NMSE (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 4. Ray Energy
    plt.subplot(2, 2, 4)
    plt.plot(rt_ray_x, rt_ray_energy, 's-', color='orange', linewidth=2)
    plt.xscale('log')
    plt.title('Energy Dev. vs. Rays (Baseline: 1M)')
    plt.xlabel('Number of Rays')
    plt.ylabel('Deviation (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle('Ray Tracing Sensitivity Analysis', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(results_dir / "ray_tracing_sensitivity.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: ray_tracing_sensitivity.png")

    # --- 3. Material Analysis ---
    mat_names = [
        "mat_all_concrete", "mat_all_glass", "mat_all_metal",
        "mat_all_wood", "mat_all_marble", "mat_no_scattering"
    ]
    mat_labels = ["Concrete", "Glass", "Metal", "Wood", "Marble", "No Scat"]
    mat_nmse = []
    mat_nlos = []

    print("Analyzing Material Sensitivity...")
    for m in mat_names:
        res = load_all_metrics("baseline_ds", m)
        nmse_val = res["Channel_NMSE_dB"]
        nlos_val = res["NLoS_PL_RMSE_dB"]
        mat_nmse.append(nmse_val if not np.isnan(nmse_val) else -80.0)
        mat_nlos.append(nlos_val if not np.isnan(nlos_val) else 0.0)

    x_pos = np.arange(len(mat_labels))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.bar(x_pos, mat_nmse, color=['orange', 'deepskyblue', 'silver', 'peru', 'plum', 'gray'])
    plt.xticks(x_pos, mat_labels, rotation=15)
    plt.title('Channel NMSE vs. Material (relative to baseline)')
    plt.ylabel('NMSE (dB)')
    plt.grid(axis='y')

    plt.subplot(1, 2, 2)
    plt.bar(x_pos, mat_nlos, color=['orange', 'deepskyblue', 'silver', 'peru', 'plum', 'gray'])
    plt.xticks(x_pos, mat_labels, rotation=15)
    plt.title('NLoS PL RMSE vs. Material (relative to baseline)')
    plt.ylabel('PL RMSE (dB)')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(results_dir / "material_sensitivity.png", dpi=200, bbox_inches='tight')
    print("Saved: material_sensitivity.png")

    # --- 4. Hardware Analysis ---
    hw_names = ["hw_baseline_4x4", "hw_4x4_dipole", "hw_4x4_iso", "hw_4x4_spacing04", "hw_4x4_polh"]
    hw_labels = ["4x4 TR38901 (Base)", "Dipole Pat.", "Iso Pat.", "Spacing 0.4l", "Pol H"]
    hw_cap_ratio = []
    hw_nmse = []

    print("Analyzing Hardware Sensitivity...")
    for h in hw_names:
        res = load_all_metrics("hw_baseline_4x4", h)
        hw_cap_ratio.append(res["Capacity_Ratio"])
        nmse_val = res["Channel_NMSE_dB"]
        hw_nmse.append(nmse_val if not np.isnan(nmse_val) else -150.0)

    x_pos = np.arange(len(hw_labels))

    # Plot 1: Capacity Ratio
    plt.figure(figsize=(8, 4))
    plt.bar(x_pos, hw_cap_ratio, color='tab:blue', alpha=0.8)
    plt.xlabel('Hardware Config')
    plt.ylabel('C_degraded / C_baseline')
    plt.xticks(x_pos, hw_labels, rotation=15)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect match')
    plt.title('Hardware Sensitivity: Capacity Preservation Ratio')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "hardware_capacity.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: hardware_capacity.png")

    # Plot 2: Channel NMSE
    plt.figure(figsize=(8, 4))
    plt.bar(x_pos, hw_nmse, color='tab:red', alpha=0.8)
    plt.xlabel('Hardware Config')
    plt.ylabel('NMSE (dB)')
    plt.xticks(x_pos, hw_labels, rotation=15)
    # plt.axhline(y=-80, color='black', linestyle='--', alpha=0.5, label='Baseline Floor')
    # Set y limit dynamically
    plt.ylim(top=max(25, max(hw_nmse) + 5))
    plt.title('Hardware Sensitivity: Channel NMSE')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "hardware_nmse.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: hardware_nmse.png")
    print("Saved: hardware_sensitivity.png")


if __name__ == "__main__":
    main()
