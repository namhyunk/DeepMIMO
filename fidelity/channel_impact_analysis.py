"""
Step 4: Channel Impact Analysis
Generate Sensitivity Curve plots for Geometry, Ray Tracing, Material, and Hardware.
Supports multiple scenarios (e.g., simple vs. complex) in a dual-row format.
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


def load_all_metrics(scenario, baseline_name, degraded_name):
    """Load datasets once and compute all metric categories for a given scenario."""
    results_dir = Path(__file__).resolve().parent / "results" / scenario
    
    bl_path = str(results_dir / baseline_name)
    dg_path = str(results_dir / degraded_name)
    
    if not os.path.exists(bl_path) or not os.path.exists(dg_path):
        return None

    bl_ds = dm.load(bl_path)
    dg_ds = dm.load(dg_path)

    geo = compute_geometry_metrics(bl_ds, dg_ds)
    mat = compute_material_metrics(bl_ds, dg_ds)
    rt = compute_ray_tracing_metrics(bl_ds, dg_ds)
    hw = compute_hardware_metrics(bl_ds, dg_ds)

    # Compute actual channel NMSE
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
    results_root = Path(__file__).resolve().parent / "results"
    scenarios = ["simple_street_canyon", "munich"]
    
    # Check which scenarios are available
    available_scenarios = [s for s in scenarios if (results_root / s).exists()]
    if not available_scenarios:
        print("Error: No scenario results found in results/")
        return
    
    print(f"Analyzing scenarios: {available_scenarios}")

    # --- 1. Geometry Analysis (2 rows) ---
    geo_names = [
        "baseline", "geo_noise_0_1m", "geo_noise_0_5m", "geo_noise_1m",
        "geo_noise_2m", "geo_noise_3m", "geo_noise_4m", "geo_noise_5m",
        "geo_noise_7m", "geo_noise_10m"
    ]
    geo_x = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]

    fig, axes = plt.subplots(len(available_scenarios), 4, figsize=(18, 4 * len(available_scenarios)))
    if len(available_scenarios) == 1: axes = [axes] # Ensure 2D-like indexing

    for s_idx, scenario in enumerate(available_scenarios):
        geo_nmse = [-150.0]
        geo_toa = [0.0]
        geo_blockage = [0.0]
        geo_nlos = [0.0]

        print(f"[{scenario}] Analyzing Geometry...")
        for g in geo_names[1:]:
            res = load_all_metrics(scenario, "baseline", g)
            if res:
                geo_nmse.append(res["Channel_NMSE_dB"])
                geo_toa.append(res["ToA_RMSE_ns"])
                geo_blockage.append(res["Blockage_Error"])
                geo_nlos.append(res["NLoS_PL_RMSE_dB"])
            else:
                for arr in [geo_nmse, geo_toa, geo_blockage, geo_nlos]: arr.append(np.nan)

        # Plot Row
        ax_row = axes[s_idx]
        ax_row[0].plot(geo_x, geo_nmse, 'o-', color='red', label=scenario)
        ax_row[0].set_title(f'{scenario.capitalize()}: NMSE vs. Noise')
        ax_row[1].plot(geo_x, geo_blockage, 's-', color='orange')
        ax_row[1].set_title('Blockage Error (%)')
        ax_row[2].plot(geo_x, geo_nlos, 'd-', color='purple')
        ax_row[2].set_title('NLoS PL RMSE (dB)')
        ax_row[3].plot(geo_x, geo_toa, 'x-', color='blue')
        ax_row[3].set_title('ToA RMSE (ns)')
        
        for ax in ax_row: ax.grid(True)

    plt.tight_layout()
    plt.savefig(results_root / "geometry_comparative.png", dpi=200, bbox_inches='tight')
    print("Saved: geometry_comparative.png")

    # --- 2. Ray Tracing Analysis (2 rows, 4 panels each) ---
    rt_depth_names = ["rt_depth_10", "rt_depth_9", "rt_depth_8", "rt_depth_7", "rt_depth_6", "rt_depth_5", "rt_depth_4", "rt_depth_3", "rt_depth_2", "rt_depth_1", "rt_depth_0"]
    rt_depth_x = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    rt_ray_names = ["baseline", "rt_500k_rays", "rt_200k_rays", "rt_low_rays", "rt_50k_rays", "rt_20k_rays", "rt_very_low_rays", "rt_5k_rays", "rt_1k_rays"]
    rt_ray_x = [1000000, 500000, 200000, 100000, 50000, 20000, 10000, 5000, 1000]

    fig, axes = plt.subplots(len(available_scenarios), 4, figsize=(18, 4 * len(available_scenarios)))
    if len(available_scenarios) == 1: axes = [axes]

    for s_idx, scenario in enumerate(available_scenarios):
        d_nmse, d_energy = [], []
        r_nmse, r_energy = [], []

        print(f"[{scenario}] Analyzing Ray Tracing...")
        # Depth
        for dn in rt_depth_names[1:]:
            res = load_all_metrics(scenario, "rt_depth_10", dn)
            d_nmse.append(res["Channel_NMSE_dB"] if res else np.nan)
            d_energy.append(res["Energy_Dev_dB"] if res else np.nan)
        # Rays
        for rn in rt_ray_names[1:]:
            res = load_all_metrics(scenario, "baseline", rn)
            r_nmse.append(res["Channel_NMSE_dB"] if res else np.nan)
            r_energy.append(res["Energy_Dev_dB"] if res else np.nan)

        ax_row = axes[s_idx]
        ax_row[0].plot(rt_depth_x[1:], d_nmse, 'o-', color='red')
        ax_row[0].set_title(f'{scenario.capitalize()}: NMSE vs. Depth')
        ax_row[0].invert_xaxis()
        
        ax_row[1].plot(rt_depth_x[1:], d_energy, 's-', color='green')
        ax_row[1].set_title('Energy Dev (dB) vs. Depth')
        ax_row[1].invert_xaxis()

        ax_row[2].plot(rt_ray_x[1:], r_nmse, 'o-', color='purple')
        ax_row[2].set_xscale('log')
        ax_row[2].set_title('NMSE vs. Ray Count')
        ax_row[2].invert_xaxis()

        ax_row[3].plot(rt_ray_x[1:], r_energy, 's-', color='orange')
        ax_row[3].set_xscale('log')
        ax_row[3].set_title('Energy Dev vs. Ray Count')
        ax_row[3].invert_xaxis()
        
        for ax in ax_row: ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_root / "ray_tracing_comparative.png", dpi=200, bbox_inches='tight')
    print("Saved: ray_tracing_comparative.png")

    # --- 3. Material Analysis (2 rows) ---
    mat_names = ["mat_all_concrete", "mat_all_glass", "mat_all_metal", "mat_all_wood", "mat_all_marble", "mat_no_scattering"]
    mat_labels = ["Concrete", "Glass", "Metal", "Wood", "Marble", "No Scat"]
    
    fig, axes = plt.subplots(len(available_scenarios), 2, figsize=(12, 4 * len(available_scenarios)))
    if len(available_scenarios) == 1: axes = [axes]

    for s_idx, scenario in enumerate(available_scenarios):
        m_nmse, m_nlos = [], []
        for mn in mat_names:
            res = load_all_metrics(scenario, "baseline_ds", mn)
            m_nmse.append(res["Channel_NMSE_dB"] if res else -150.0)
            m_nlos.append(res["NLoS_PL_RMSE_dB"] if res else 0.0)
        
        ax_row = axes[s_idx]
        x_pos = np.arange(len(mat_labels))
        colors = ['orange', 'deepskyblue', 'silver', 'peru', 'plum', 'gray']
        ax_row[0].bar(x_pos, m_nmse, color=colors)
        ax_row[0].set_xticks(x_pos, mat_labels, rotation=15)
        ax_row[0].set_title(f'{scenario.capitalize()}: NMSE vs. Material')
        
        ax_row[1].bar(x_pos, m_nlos, color=colors)
        ax_row[1].set_xticks(x_pos, mat_labels, rotation=15)
        ax_row[1].set_title('NLoS PL RMSE (dB)')
        
        for ax in ax_row: ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_root / "material_comparative.png", dpi=200, bbox_inches='tight')
    print("Saved: material_comparative.png")

    # --- 4. Hardware Analysis (2 rows) ---
    hw_names = ["hw_baseline_4x4", "hw_4x4_dipole", "hw_4x4_iso", "hw_4x4_spacing04", "hw_4x4_polh"]
    hw_labels = ["4x4 TR38901", "Dipole", "Iso", "0.4l Spacing", "Pol H"]

    fig, axes = plt.subplots(len(available_scenarios), 2, figsize=(12, 4 * len(available_scenarios)))
    if len(available_scenarios) == 1: axes = [axes]

    for s_idx, scenario in enumerate(available_scenarios):
        h_cap, h_nmse = [], []
        for hn in hw_names:
            res = load_all_metrics(scenario, "hw_baseline_4x4", hn)
            h_cap.append(res["Capacity_Ratio"] if res else 1.0)
            h_nmse.append(res["Channel_NMSE_dB"] if res else -150.0)
        
        ax_row = axes[s_idx]
        x_pos = np.arange(len(hw_labels))
        ax_row[0].bar(x_pos, h_cap, color='tab:blue', alpha=0.7)
        ax_row[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax_row[0].set_xticks(x_pos, hw_labels, rotation=15)
        ax_row[0].set_title(f'{scenario.capitalize()}: Capacity Ratio')
        
        ax_row[1].bar(x_pos, h_nmse, color='tab:red', alpha=0.7)
        ax_row[1].set_xticks(x_pos, hw_labels, rotation=15)
        ax_row[1].set_title('Channel NMSE (dB)')
        
        for ax in ax_row: ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_root / "hardware_comparative.png", dpi=200, bbox_inches='tight')
    print("Saved: hardware_comparative.png")


if __name__ == "__main__":
    main()
