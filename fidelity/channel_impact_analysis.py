"""
Step 4: Channel Impact Analysis
Generate Sensitivity Curve plots for Geometry and Ray Tracing.
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

def load_metrics(baseline_name, degraded_name):
    bl_ds = dm.load(baseline_name)
    dg_ds = dm.load(degraded_name)

    geo = compute_geometry_metrics(bl_ds, dg_ds)
    mat = compute_material_metrics(bl_ds, dg_ds)
    rt = compute_ray_tracing_metrics(bl_ds, dg_ds)
    hw = compute_hardware_metrics(bl_ds, dg_ds)

    return {
        "NMSE_dB": hw["channel_nmse_dB"],
        "ToA_RMSE_ns": geo["toa_rmse_ns"],
        "Blockage_Error": geo["blockage_error_rate"] * 100, # percentage
        "NLoS_PL_RMSE_dB": mat["nlos_power_rmse_dB"]
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
    
    geo_nmse = [0.0]
    geo_toa = [0.0]
    geo_blockage = [0.0]
    geo_nlos = [0.0]

    print("Analyzing Geometry Sensitivity...")
    for idx, g in enumerate(geo_names[1:]):
        res = load_metrics("baseline", g)
        geo_nmse.append(res["NMSE_dB"])
        geo_toa.append(res["ToA_RMSE_ns"])
        geo_blockage.append(res["Blockage_Error"])
        geo_nlos.append(res["NLoS_PL_RMSE_dB"])

    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(geo_x, geo_nmse, 'o-', color='red')
    plt.title('Channel NMSE vs. Position Noise')
    plt.xlabel('Gaussian Position Noise (m)')
    plt.ylabel('NMSE (dB)')
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
    geo_plt_path = results_dir / "geometry_sensitivity.png"
    plt.savefig(geo_plt_path, dpi=200)
    print(f"Saved: {geo_plt_path}")


    # --- 2. RT Analysis ---
    rt_names = ["baseline", "rt_depth_3", "rt_depth_1"]
    rt_x = [5, 3, 1] # reflection depth
    
    rt_nmse = [0.0]
    rt_energy = [0.0]

    print("Analyzing Ray Tracing Sensitivity...")
    for idx, r in enumerate(rt_names[1:]):
        bl_ds = dm.load("baseline")
        dg_ds = dm.load(r)
        
        # Use proper NMSE for ray tracing
        if hasattr(bl_ds, "channels") and hasattr(dg_ds, "channels"):
            nmse_res = compute_channel_nmse(bl_ds.channels, dg_ds.channels)
            nmse_val = nmse_res.get("channel_nmse_dB", float('nan'))
        else:
            nmse_val = float('nan')
            
        rt = compute_ray_tracing_metrics(bl_ds, dg_ds)
        
        rt_nmse.append(nmse_val)
        rt_energy.append(rt["global_energy_deviation_dB"])

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rt_x, rt_nmse, 'o-', color='red')
    plt.title('Channel NMSE vs. Reflection Depth')
    plt.xlabel('Max Reflections')
    plt.ylabel('NMSE (dB)')
    plt.gca().invert_xaxis() # Lower depth = more right
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rt_x, rt_energy, 's-', color='green')
    plt.title('Global Energy Deviation vs. Depth')
    plt.xlabel('Max Reflections')
    plt.ylabel('Deviation (dB)')
    plt.gca().invert_xaxis()
    plt.grid(True)

    plt.tight_layout()
    rt_plt_path = results_dir / "ray_tracing_sensitivity.png"
    plt.savefig(rt_plt_path, dpi=200)
    print(f"Saved: {rt_plt_path}")

    # --- 3. Material Analysis ---
    mat_names = ["baseline", "mat_all_concrete", "mat_no_scattering"]
    mat_labels = ["Baseline", "Concrete", "No Scat"]
    
    mat_nmse = []
    mat_nlos = []

    print("Analyzing Material Sensitivity...")
    for idx, m in enumerate(mat_names):
        res = load_metrics("baseline", m)
        mat_nmse.append(res["NMSE_dB"])
        mat_nlos.append(res["NLoS_PL_RMSE_dB"] if not np.isnan(res["NLoS_PL_RMSE_dB"]) else 0.0)

    x_pos = np.arange(len(mat_labels))
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(x_pos, mat_nmse, color=['gray', 'orange', 'cyan'])
    plt.xticks(x_pos, mat_labels)
    plt.title('Channel NMSE vs. Material')
    plt.ylabel('NMSE (dB)')
    plt.grid(axis='y')

    plt.subplot(1, 2, 2)
    plt.bar(x_pos, mat_nlos, color=['gray', 'purple', 'cyan'])
    plt.xticks(x_pos, mat_labels)
    plt.title('NLoS PL RMSE vs. Material')
    plt.ylabel('PL RMSE (dB)')
    plt.grid(axis='y')

    plt.tight_layout()
    mat_plt_path = results_dir / "material_sensitivity.png"
    plt.savefig(mat_plt_path, dpi=200)
    print(f"Saved: {mat_plt_path}")

    # --- 4. Hardware Analysis ---
    hw_names = ["baseline", "hw_dipole", "hw_tr38901", "hw_4x4_array"]
    hw_labels = ["Iso (Base)", "Dipole", "TR38901", "4x4 UPA"]
    
    hw_nmse = []

    print("Analyzing Hardware Sensitivity...")
    for idx, h in enumerate(hw_names):
        res = load_metrics("baseline", h)
        hw_nmse.append(res["NMSE_dB"])

    plt.figure(figsize=(6, 4))
    x_pos2 = np.arange(len(hw_labels))
    # NMSE key was repurposed as Capacity Ratio
    plt.bar(x_pos2, hw_nmse, color=['gray', 'blue', 'red', 'green'])
    plt.xticks(x_pos2, hw_labels)
    plt.title('Capacity Preservation Ratio vs. Hardware')
    plt.ylabel('C_degraded / C_baseline')
    plt.grid(axis='y')

    plt.tight_layout()
    hw_plt_path = results_dir / "hardware_sensitivity.png"
    plt.savefig(hw_plt_path, dpi=200)
    print(f"Saved: {hw_plt_path}")

if __name__ == "__main__":
    main()
