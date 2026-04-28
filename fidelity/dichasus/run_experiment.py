"""
Orchestrator: run each fidelity config as a separate subprocess to avoid DrJIT crashes.
Resumes from checkpoint - skips configs that already have results.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "dichasus_fidelity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = RESULTS_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

SCENE_SIMPLE = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")
SCENE_DETAILED_NB = str(BASE_DIR / "dcxx/inue_detailed/inue_no_blockers.xml")
SCENE_DETAILED_FULL = str(BASE_DIR / "dcxx/inue_detailed/inue_detailed.xml")


def make_config(name, tag, scene_path, coord_system="raw",
                max_depth=5, samples=500000, diffuse=False,
                tx_pattern="tr38901", rx_pattern="dipole",
                tx_spacing=0.5, polarization="V",
                material_override=None, n_positions=30):
    return {
        "name": name, "tag": tag, "scene_path": scene_path,
        "coord_system": coord_system,
        "max_depth": max_depth, "samples_per_src": samples,
        "diffuse_reflection": diffuse,
        "tx_pattern": tx_pattern, "rx_pattern": rx_pattern,
        "tx_spacing": tx_spacing, "polarization": polarization,
        "material_override": material_override,
        "n_positions": n_positions,
    }


CONFIGS = [
    # Geometry (3)
    make_config("geo_simple", "geometry", SCENE_SIMPLE, coord_system="simple"),
    make_config("geo_detailed", "geometry", SCENE_DETAILED_NB),
    make_config("geo_full", "geometry", SCENE_DETAILED_FULL),
    # Material (6)
    make_config("mat_baseline", "material", SCENE_DETAILED_NB),
    make_config("mat_all_concrete", "material", SCENE_DETAILED_NB, material_override="itu_concrete"),
    make_config("mat_all_metal", "material", SCENE_DETAILED_NB, material_override="itu_metal"),
    make_config("mat_all_glass", "material", SCENE_DETAILED_NB, material_override="itu_glass"),
    make_config("mat_all_plasterboard", "material", SCENE_DETAILED_NB, material_override="itu_plasterboard"),
    make_config("mat_all_wood", "material", SCENE_DETAILED_NB, material_override="itu_wood"),
    # RT (9)
    make_config("rt_depth_1", "rt", SCENE_DETAILED_NB, max_depth=1),
    make_config("rt_depth_2", "rt", SCENE_DETAILED_NB, max_depth=2),
    make_config("rt_depth_3", "rt", SCENE_DETAILED_NB, max_depth=3),
    make_config("rt_depth_5", "rt", SCENE_DETAILED_NB, max_depth=5),
    make_config("rt_depth_7", "rt", SCENE_DETAILED_NB, max_depth=7),
    make_config("rt_samples_5k", "rt", SCENE_DETAILED_NB, samples=5000),
    make_config("rt_samples_50k", "rt", SCENE_DETAILED_NB, samples=50000),
    make_config("rt_samples_500k", "rt", SCENE_DETAILED_NB, samples=500000),
    make_config("rt_diffuse_on", "rt", SCENE_DETAILED_NB, diffuse=True),
    # Hardware (6)
    make_config("hw_baseline", "hardware", SCENE_DETAILED_NB),
    make_config("hw_tx_dipole", "hardware", SCENE_DETAILED_NB, tx_pattern="dipole"),
    make_config("hw_tx_iso", "hardware", SCENE_DETAILED_NB, tx_pattern="iso"),
    make_config("hw_spacing_04", "hardware", SCENE_DETAILED_NB, tx_spacing=0.4),
    make_config("hw_pol_H", "hardware", SCENE_DETAILED_NB, polarization="H"),
    make_config("hw_rx_iso", "hardware", SCENE_DETAILED_NB, rx_pattern="iso"),
]

# Load existing results
all_results_path = RESULTS_DIR / "all_results.json"
if all_results_path.exists():
    with open(all_results_path) as f:
        all_results = json.load(f)
    print(f"Loaded existing results: {len(all_results)} configs")
else:
    all_results = {}

t_total = time.time()
for ci, config in enumerate(CONFIGS):
    cname = config["name"]

    # Skip if already done
    if cname in all_results and all_results[cname].get("n_valid", 0) >= config["n_positions"]:
        print(f"[{ci+1}/{len(CONFIGS)}] {cname}: already done, skipping")
        continue

    print(f"\n[{ci+1}/{len(CONFIGS)}] {cname} ({config['tag']})")

    # Write config to temp file
    config_path = str(CONFIG_DIR / f"{cname}.json")
    output_path = str(RESULTS_DIR / f"{cname}_result.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run as subprocess
    t_start = time.time()
    cmd = [sys.executable, str(BASE_DIR / "run_single_config.py"), config_path, output_path]
    proc = subprocess.run(cmd, timeout=1200)  # 20 min timeout per config

    elapsed = time.time() - t_start

    if proc.returncode != 0:
        print(f"  FAILED (return code {proc.returncode}, {elapsed:.0f}s)")
        continue

    # Load result and merge
    try:
        with open(output_path) as f:
            result = json.load(f)
        if result.get("n_valid", 0) > 0:
            all_results[cname] = result
            print(f"  OK: {result['n_valid']} valid, {elapsed:.0f}s, "
                  f"pwr_corr={result['aggregate'].get('pwr_corr_mean', 0):.3f}")
        else:
            print(f"  No valid positions ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  Error loading result: {e}")

    # Save checkpoint
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

total_time = time.time() - t_total
print(f"\n{'='*60}")
print(f"EXPERIMENT COMPLETE: {len(all_results)}/{len(CONFIGS)} configs in {total_time:.0f}s")
print(f"Results: {all_results_path}")
print(f"{'='*60}")
