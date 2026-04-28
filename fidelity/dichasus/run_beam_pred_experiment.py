"""
Orchestrator for beam prediction channel capture.
Each config is split into chunks of ~50 positions, and each chunk runs in its own
subprocess (DrJIT state reset between chunks avoids GPU memory accumulation).
Chunks are merged into a single .npz per config.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "beam_pred"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG_DIR = OUT_DIR / "configs"
CFG_DIR.mkdir(exist_ok=True)
CHUNK_DIR = OUT_DIR / "chunks"
CHUNK_DIR.mkdir(exist_ok=True)

SCENE_SIMPLE = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")
SCENE_DETAILED_NB = str(BASE_DIR / "dcxx/inue_detailed/inue_no_blockers.xml")
SCENE_DETAILED_FULL = str(BASE_DIR / "dcxx/inue_detailed/inue_detailed.xml")

N_POSITIONS = 300
CHUNK_SIZE = 50  # positions per subprocess


def make_config(name, tag, scene_path, coord_system="raw",
                max_depth=5, samples=500000, diffuse=False,
                tx_pattern="tr38901", rx_pattern="dipole",
                tx_spacing=0.5, polarization="V",
                material_override=None):
    return {
        "name": name, "tag": tag, "scene_path": scene_path,
        "coord_system": coord_system,
        "max_depth": max_depth, "samples_per_src": samples,
        "diffuse_reflection": diffuse,
        "tx_pattern": tx_pattern, "rx_pattern": rx_pattern,
        "tx_spacing": tx_spacing, "polarization": polarization,
        "material_override": material_override,
        "n_positions": N_POSITIONS,
    }


# Representative configs spanning the fidelity spectrum
CONFIGS = [
    make_config("hw_tx_iso", "hardware", SCENE_DETAILED_NB, tx_pattern="iso"),        # 0.215
    make_config("rt_depth_3", "rt", SCENE_DETAILED_NB, max_depth=3),                  # 0.138
    make_config("mat_all_concrete", "material", SCENE_DETAILED_NB, material_override="itu_concrete"),  # 0.097
    make_config("hw_baseline", "hardware", SCENE_DETAILED_NB),                        # 0.087
    make_config("geo_full", "geometry", SCENE_DETAILED_FULL),                         # 0.023
    make_config("mat_all_metal", "material", SCENE_DETAILED_NB, material_override="itu_metal"),  # 0.004
]


def merge_chunks(chunk_paths, final_path):
    """Concatenate chunk .npz files into one final .npz."""
    chunks = [np.load(p, allow_pickle=True) for p in chunk_paths]
    merged = {
        "sim_channels": np.concatenate([c["sim_channels"] for c in chunks]),
        "meas_channels": np.concatenate([c["meas_channels"] for c in chunks]),
        "positions": np.concatenate([c["positions"] for c in chunks]),
        "pos_indices": np.concatenate([c["pos_indices"] for c in chunks]),
        "north_assign": chunks[0]["north_assign"],
        "south_assign": chunks[0]["south_assign"],
        "config_name": chunks[0]["config_name"],
        "config_tag": chunks[0]["config_tag"],
    }
    np.savez_compressed(final_path, **merged)
    return merged["sim_channels"].shape[0]


def main():
    t_total = time.time()
    done, skipped, failed = [], [], []

    for ci, config in enumerate(CONFIGS):
        cname = config["name"]
        final_path = OUT_DIR / f"{cname}.npz"

        if final_path.exists():
            print(f"[{ci+1}/{len(CONFIGS)}] {cname}: already exists, skipping")
            skipped.append(cname)
            continue

        print(f"\n[{ci+1}/{len(CONFIGS)}] {cname} ({config['tag']}) [N={N_POSITIONS}]")
        t_cfg_start = time.time()
        chunk_paths = []
        chunk_failed = False

        for chunk_idx, chunk_start in enumerate(range(0, N_POSITIONS, CHUNK_SIZE)):
            chunk_end = min(chunk_start + CHUNK_SIZE, N_POSITIONS)
            chunk_cfg = dict(config)
            chunk_cfg["chunk_start"] = chunk_start
            chunk_cfg["chunk_end"] = chunk_end

            cfg_path = str(CFG_DIR / f"{cname}_chunk{chunk_idx}.json")
            chunk_out = CHUNK_DIR / f"{cname}_chunk{chunk_idx}.npz"
            with open(cfg_path, 'w') as f:
                json.dump(chunk_cfg, f, indent=2)

            print(f"  chunk {chunk_idx+1}: positions [{chunk_start}:{chunk_end}]", flush=True)
            t_c = time.time()
            cmd = [sys.executable, "-u", str(BASE_DIR / "run_beam_pred_config.py"),
                   cfg_path, str(chunk_out)]
            proc = subprocess.run(cmd, timeout=600)
            elapsed_c = time.time() - t_c

            if proc.returncode != 0 or not chunk_out.exists():
                print(f"  chunk {chunk_idx+1} FAILED (rc={proc.returncode}, {elapsed_c:.0f}s)")
                chunk_failed = True
                break

            print(f"  chunk {chunk_idx+1} OK ({elapsed_c:.0f}s)")
            chunk_paths.append(chunk_out)

        elapsed = time.time() - t_cfg_start

        if chunk_failed or not chunk_paths:
            print(f"  {cname}: FAILED ({elapsed:.0f}s)")
            failed.append(cname)
            continue

        n_total = merge_chunks(chunk_paths, final_path)
        print(f"  {cname}: OK — {n_total} valid positions, {elapsed:.0f}s")
        done.append(cname)

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"BEAM PRED CAPTURE: {len(done)} done, {len(skipped)} skipped, {len(failed)} failed")
    print(f"Total: {total:.0f}s ({total/60:.1f} min)")
    print(f"Outputs: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
