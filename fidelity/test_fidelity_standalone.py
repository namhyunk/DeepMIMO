#!/usr/bin/env python3
"""Standalone fidelity experiment runner.

Runs directly with Sionna RT, bypassing DeepMIMO pipeline imports.
Designed to test each fidelity config one at a time in a single process.

Usage:
    conda activate lwm
    python fidelity/test_fidelity_standalone.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

# Fix OptiX loading in WSL2 
os.environ["DRJIT_LIBOPTIX_PATH"] = "/usr/lib/wsl/lib/libnvoptix.so.1"
# Force CPU execution to bypass broken OptiX in WSL2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================================
# Step 0: Sionna RT imports (slow, do once)
# ============================================================================
print("=" * 60)
print("  Fidelity Experiment — Standalone Runner")
print("=" * 60)

t0 = time.time()
print("\n[0/7] Importing Sionna RT (this is slow, ~1-2 min)...")

import numpy as np
import sionna
print(f"  Sionna version: {sionna.__version__}")

import sionna.rt as srt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

print(f"  Sionna RT imported in {time.time() - t0:.1f}s")

# Check available built-in scenes
print("\n  Available built-in scenes:")
for attr in dir(srt.scene):
    if not attr.startswith("_"):
        val = getattr(srt.scene, attr, None)
        if isinstance(val, str) and val.endswith(".xml"):
            print(f"    - {attr}")

# ============================================================================
# Step 1: Define fidelity configs inline (avoid import chain issues)
# ============================================================================
print("\n[1/7] Defining fidelity configurations...")

CONFIGS = {
    "baseline": {
        "max_depth": 5,
        "description": "Full fidelity: 5 reflections, 1M rays, iso antenna",
        "samples_per_src": 1_000_000,
    },
    "rt_depth_1": {
        "max_depth": 1,
        "description": "RT: max 1 reflection",
        "samples_per_src": 1_000_000,
    },
    "rt_depth_3": {
        "max_depth": 3,
        "description": "RT: max 3 reflections",
        "samples_per_src": 1_000_000,
    },
    "rt_low_rays": {
        "max_depth": 5,
        "description": "RT: 100K rays (10x fewer)",
        "samples_per_src": 100_000,
    },
    "rt_very_low_rays": {
        "max_depth": 5,
        "description": "RT: 10K rays (100x fewer)",
        "samples_per_src": 10_000,
    },
}

print(f"  Defined {len(CONFIGS)} configs: {list(CONFIGS.keys())}")

# ============================================================================
# Step 2: Load scene
# ============================================================================
print("\n[2/7] Loading scene...")

# Use a built-in Sionna scene
try:
    scene = load_scene(srt.scene.simple_street_canyon)
    scene_name = "simple_street_canyon"
except Exception:
    try:
        scene = load_scene(srt.scene.munich)
        scene_name = "munich"
    except Exception:
        scene = load_scene()  # empty scene
        scene_name = "empty"

print(f"  Scene: {scene_name}")
print(f"  Objects: {len(scene.objects)}")

# Configure antenna arrays
scene.tx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V",
)
scene.rx_array = PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V",
)

# Set frequency
scene.frequency = 3.5e9
freq_arr = scene.frequency.numpy() if hasattr(scene.frequency, 'numpy') else np.array(scene.frequency)
freq_val = float(np.ravel(freq_arr)[0])
print(f"  Frequency: {freq_val / 1e9:.1f} GHz")

# ============================================================================
# Step 3: Set up TX/RX positions
# ============================================================================
print("\n[3/7] Setting up TX/RX positions...")

# Add transmitter
tx = Transmitter(name="tx", position=[0, 0, 10])
scene.add(tx)
print(f"  TX position: [0, 0, 10]")

# Generate RX grid
rx_positions = []
for x in np.arange(-20, 21, 5):
    for y in np.arange(-20, 21, 5):
        rx_positions.append([float(x), float(y), 1.5])

print(f"  RX positions: {len(rx_positions)} users")

# ============================================================================
# Step 4: Run each config
# ============================================================================
print("\n[4/7] Running fidelity experiments...")

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

p_solver = PathSolver()
all_results = {}

for config_name, config in CONFIGS.items():
    print(f"\n  {'=' * 50}")
    print(f"  Config: {config_name}")
    print(f"  Description: {config['description']}")
    print(f"  {'=' * 50}")

    # Remove existing receivers
    rx_names = [name for name in scene.objects if name.startswith("rx_")]
    for name in rx_names:
        scene.remove(name)

    # Add receivers with unique names per config
    for i, pos in enumerate(rx_positions):
        scene.add(Receiver(name=f"rx_{config_name}_{i}", position=pos))

    # Build path solver params
    rt_params = {
        "scene": scene,
        "max_depth": config["max_depth"],
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": False,
        "synthetic_array": True,
        "samples_per_src": config["samples_per_src"],
        "max_num_paths_per_src": config["samples_per_src"],
    }

    # Run ray tracing
    t_start = time.time()
    try:
        paths = p_solver(**rt_params)
        paths.normalize_delays = False
        t_elapsed = time.time() - t_start

        # Extract channel info
        # paths.a: complex path coefficients
        # paths.tau: propagation delays
        a = paths.a.numpy() if hasattr(paths.a, 'numpy') else np.array(paths.a)
        tau = paths.tau.numpy() if hasattr(paths.tau, 'numpy') else np.array(paths.tau)

        print(f"  ✓ Completed in {t_elapsed:.2f}s")
        print(f"  ✓ paths.a shape: {a.shape}")
        print(f"  ✓ paths.tau shape: {tau.shape}")

        # Compute basic channel statistics
        path_powers = np.abs(a) ** 2

        # Total received power per user (sum over paths)
        total_power_per_user = path_powers.sum(axis=-1).sum(axis=-2)  # sum over paths and antennas
        mean_power = total_power_per_user.mean()
        
        # Mean number of significant paths per user
        significant_paths = (path_powers.max(axis=(-3, -4)) > 1e-15).sum(axis=-1)
        mean_paths = significant_paths.mean()

        # Delay spread
        if tau.size > 0:
            mean_delay = float(tau.mean())
            max_delay = float(tau.max())
        else:
            mean_delay = max_delay = 0

        result = {
            "config_name": config_name,
            "description": config["description"],
            "max_depth": config["max_depth"],
            "samples_per_src": config["samples_per_src"],
            "runtime_seconds": round(t_elapsed, 2),
            "a_shape": list(a.shape),
            "tau_shape": list(tau.shape),
            "mean_total_power": float(mean_power),
            "mean_total_power_dB": float(10 * np.log10(max(mean_power, 1e-30))),
            "mean_significant_paths": float(mean_paths),
            "mean_delay_s": mean_delay,
            "max_delay_s": max_delay,
            "status": "success",
        }

        print(f"  ✓ Mean total power: {result['mean_total_power_dB']:.2f} dB")
        print(f"  ✓ Mean significant paths: {result['mean_significant_paths']:.1f}")
        print(f"  ✓ Max delay: {result['max_delay_s'] * 1e9:.2f} ns")

    except Exception as e:
        t_elapsed = time.time() - t_start
        print(f"  ✗ FAILED after {t_elapsed:.2f}s: {e}")
        traceback.print_exc()
        result = {
            "config_name": config_name,
            "description": config["description"],
            "runtime_seconds": round(t_elapsed, 2),
            "status": "error",
            "error": str(e),
        }

    all_results[config_name] = result

    # Clean up receivers for next iteration
    rx_names = [name for name in scene.objects if name.startswith("rx_")]
    for name in rx_names:
        scene.remove(name)

# ============================================================================
# Step 5: Compare results against baseline
# ============================================================================
print("\n\n[5/7] Comparing against baseline...")

if "baseline" in all_results and all_results["baseline"]["status"] == "success":
    bl = all_results["baseline"]
    print(f"\n  {'Config':<20s} {'Power(dB)':>10s} {'ΔPower(dB)':>12s} {'Paths':>8s} {'ΔPaths':>8s} {'Time(s)':>8s}")
    print(f"  {'-'*66}")
    for name, res in all_results.items():
        if res["status"] != "success":
            print(f"  {name:<20s} {'ERROR':>10s}")
            continue
        delta_pwr = res["mean_total_power_dB"] - bl["mean_total_power_dB"]
        delta_paths = res["mean_significant_paths"] - bl["mean_significant_paths"]
        print(
            f"  {name:<20s} {res['mean_total_power_dB']:>10.2f} {delta_pwr:>+12.2f} "
            f"{res['mean_significant_paths']:>8.1f} {delta_paths:>+8.1f} {res['runtime_seconds']:>8.2f}"
        )

# ============================================================================
# Step 6: Save results
# ============================================================================
print("\n[6/7] Saving results...")

results_path = output_dir / "fidelity_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"  Saved to: {results_path}")

# ============================================================================
# Step 7: Summary
# ============================================================================
print(f"\n[7/7] Done!")
total_time = time.time() - t0
print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Successful: {sum(1 for r in all_results.values() if r['status'] == 'success')}/{len(all_results)}")
