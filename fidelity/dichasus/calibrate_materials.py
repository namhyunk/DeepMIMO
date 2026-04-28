"""
Calibrate material parameters via coordinate descent.
Each evaluation runs as a subprocess (eval_material_params.py).

Uses first 2000 positions for training (subsample to 20 for each eval).
Saves calibrated params to results/calibration/calibrated_materials.json.
"""
import json
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = OUT_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

SCENE_PATH = str(BASE_DIR / "dcxx/inue_detailed/inue_no_blockers.xml")

# Training positions: subsample 20 from first 2000
rng = np.random.RandomState(42)
TRAIN_POOL = 2000
N_EVAL = 20
EVAL_INDICES = sorted(rng.choice(TRAIN_POOL, size=N_EVAL, replace=False).tolist())

# Get initial material list from scene
def get_initial_materials():
    """Load scene once to discover objects and their materials."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import drjit as dr
    from sionna.rt import load_scene
    scene = load_scene(SCENE_PATH)
    scene.frequency = 3.438e9
    mat_map = {}  # obj_name -> mat_name
    init_params = {}  # mat_name -> {rel_perm, cond}
    for oname in scene.objects:
        rm = scene.objects[oname].radio_material
        if rm is not None:
            mat_map[oname] = rm.name
            if rm.name not in init_params:
                init_params[rm.name] = {
                    "relative_permittivity": float(rm.relative_permittivity[0]),
                    "conductivity": float(rm.conductivity[0]),
                }
    del scene
    dr.flush_malloc_cache()
    return init_params, mat_map


def eval_params(materials, tag=""):
    """Evaluate material params via subprocess, return mean pwr_corr."""
    cfg = {
        "scene_path": SCENE_PATH,
        "materials": materials,
        "position_indices": EVAL_INDICES,
    }
    cfg_path = str(TMP_DIR / f"eval_{tag}.json")
    out_path = str(TMP_DIR / f"result_{tag}.json")
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    cmd = [sys.executable, "-u", str(BASE_DIR / "eval_material_params.py"), cfg_path, out_path]
    proc = subprocess.run(cmd, timeout=600, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  eval FAILED (rc={proc.returncode}): {proc.stderr[-200:]}")
        return -999.0

    with open(out_path) as f:
        result = json.load(f)
    return result.get("pwr_corr_mean", -999.0)


def main():
    print("Discovering scene materials...", flush=True)
    init_params, mat_map = get_initial_materials()

    print(f"\nMaterials ({len(init_params)}):")
    for mn, p in init_params.items():
        print(f"  {mn}: eps_r={p['relative_permittivity']:.2f}, sigma={p['conductivity']:.4f}")
    print(f"\nEval positions: {N_EVAL} from first {TRAIN_POOL}")
    print(f"Indices: {EVAL_INDICES[:5]}...{EVAL_INDICES[-5:]}\n")

    # Evaluate ITU baseline
    print("Evaluating ITU baseline...", flush=True)
    t0 = time.time()
    baseline_corr = eval_params(init_params, tag="itu_baseline")
    print(f"  ITU baseline pwr_corr: {baseline_corr:.4f} ({time.time()-t0:.0f}s)\n")

    best_params = {m: dict(p) for m, p in init_params.items()}
    best_corr = baseline_corr

    # Coordinate descent
    MAX_EPOCHS = 10
    for epoch in range(MAX_EPOCHS):
        t_epoch = time.time()
        improved = False

        for mname in sorted(init_params.keys()):
            for param_type in ["relative_permittivity", "conductivity"]:
                current_val = best_params[mname][param_type]

                if param_type == "relative_permittivity":
                    # Try a range of values
                    candidates = [1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0]
                    candidates = [c for c in candidates if abs(c - current_val) > 0.3]
                else:
                    # Conductivity on log scale
                    candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0]
                    candidates = [c for c in candidates if abs(np.log(c) - np.log(max(current_val, 1e-6))) > 0.3]

                best_cand_corr = best_corr
                best_cand_val = current_val

                for cval in candidates:
                    trial = {m: dict(p) for m, p in best_params.items()}
                    trial[mname][param_type] = cval
                    tag = f"e{epoch}_{mname}_{param_type}_{cval}"
                    trial_corr = eval_params(trial, tag=tag)
                    if trial_corr > best_cand_corr:
                        best_cand_corr = trial_corr
                        best_cand_val = cval

                if best_cand_val != current_val:
                    best_params[mname][param_type] = best_cand_val
                    best_corr = best_cand_corr
                    improved = True
                    print(f"  [{mname}] {param_type}: {current_val:.4f} -> {best_cand_val:.4f} "
                          f"(corr: {best_corr:.4f})", flush=True)

        elapsed_epoch = time.time() - t_epoch
        print(f"\nEpoch {epoch+1}: pwr_corr={best_corr:.4f} ({elapsed_epoch:.0f}s)", flush=True)

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "initial_params": init_params,
            "calibrated_params": best_params,
            "initial_pwr_corr": baseline_corr,
            "calibrated_pwr_corr": best_corr,
        }
        with open(OUT_DIR / "calibration_checkpoint.json", 'w') as f:
            json.dump(ckpt, f, indent=2)

        if not improved:
            print("No improvement, stopping early.")
            break

    # Save final
    final = {
        "initial_params": init_params,
        "calibrated_params": best_params,
        "initial_pwr_corr": float(baseline_corr),
        "calibrated_pwr_corr": float(best_corr),
        "n_eval_positions": N_EVAL,
        "eval_indices": EVAL_INDICES,
    }
    out_path = OUT_DIR / "calibrated_materials.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Calibration complete")
    print(f"  ITU pwr_corr:        {baseline_corr:.4f}")
    print(f"  Calibrated pwr_corr: {best_corr:.4f}")
    print(f"  Improvement:         {best_corr - baseline_corr:+.4f}")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
