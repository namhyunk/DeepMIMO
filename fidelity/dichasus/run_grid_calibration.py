#!/usr/bin/env python
"""
Grid-based calibration: sweep material params, scattering, antenna patterns.
Each config runs as a subprocess to avoid memory leaks.
"""
import json, subprocess, sys, time, os
import numpy as np
from pathlib import Path
from itertools import product

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "full_calibration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = RESULTS_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable
EVAL_SCRIPT = str(BASE_DIR / "eval_single_config.py")

# Fixed positions for evaluation
rng = np.random.RandomState(42)
ALL_POS = rng.permutation(3282)
TRAIN_POS = sorted(ALL_POS[:30].tolist())  # 30 train positions
TEST_POS = sorted(ALL_POS[300:350].tolist())  # 50 test positions

def eval_config(cfg, label):
    """Run eval_single_config.py as subprocess."""
    cfg_path = str(TMP_DIR / f"cfg_{label}.json")
    out_path = str(TMP_DIR / f"out_{label}.json")
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    cmd = [PYTHON, "-u", EVAL_SCRIPT, cfg_path, out_path]
    try:
        proc = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[-200:]}", flush=True)
            return None
        with open(out_path) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {label}", flush=True)
        return None

def make_cfg(materials, tx_pat="dipole", rx_pat="dipole", diffuse=True,
             scatt_pat="lambertian", samples=50000, positions=None):
    return {
        'materials': materials,
        'tx_pattern': tx_pat,
        'rx_pattern': rx_pat,
        'diffuse': diffuse,
        'scatt_pattern': scatt_pat,
        'max_depth': 5,
        'samples': samples,
        'position_indices': positions or TRAIN_POS,
    }

def main():
    all_results = {}
    t_start = time.time()

    print("=" * 70)
    print("GRID-BASED CALIBRATION SWEEP")
    print("=" * 70)

    # ========================================
    # Phase 1: Material grid search (no scattering)
    # ========================================
    print("\n[Phase 1] Material parameter grid search (no scattering)...\n")

    # Key insight from prior work: plasterboard (walls) dominates.
    # Concrete and ceiling board have secondary effects.
    plasterboard_eps = [1.0, 1.2, 1.5, 2.0, 2.73]  # ITU default is 2.73
    concrete_eps = [1.5, 3.0, 4.5, 5.24, 7.0]  # ITU default is 5.24
    ceiling_eps = [1.48, 5.0, 15.0, 25.0]  # ITU default is 1.48

    best_pwr = -1.0
    best_mat = None
    phase1_results = []

    # First: sweep plasterboard (most important) with concrete fixed at 4.5
    print("  Sweeping plasterboard eps_r (concrete=4.5, ceiling=25.0)...")
    for pe in plasterboard_eps:
        materials = {
            'itu_concrete': {'eps_r': 4.5, 'sigma': 0.05},
            'itu_ceiling_board': {'eps_r': 25.0, 'sigma': 0.001},
            'itu_plasterboard': {'eps_r': pe, 'sigma': 0.027},
        }
        label = f"mat_pe{pe}"
        t0 = time.time()
        cfg = make_cfg(materials, diffuse=False, samples=50000)
        r = eval_config(cfg, label)
        if r and r.get('pwr_corr_mean', -1) > best_pwr:
            best_pwr = r['pwr_corr_mean']
            best_mat = materials.copy()
        pc = r['pwr_corr_mean'] if r else -1
        print(f"    plast_eps={pe:5.2f}: pwr={pc:.4f} ({time.time()-t0:.0f}s)"
              f"{'  *BEST*' if r and pc == best_pwr else ''}", flush=True)
        if r: all_results[label] = r

    best_plast_eps = best_mat['itu_plasterboard']['eps_r']
    print(f"  Best plasterboard eps_r: {best_plast_eps}")

    # Sweep concrete with best plasterboard
    print(f"\n  Sweeping concrete eps_r (plaster={best_plast_eps})...")
    for ce in concrete_eps:
        materials = {
            'itu_concrete': {'eps_r': ce, 'sigma': 0.05},
            'itu_ceiling_board': {'eps_r': 25.0, 'sigma': 0.001},
            'itu_plasterboard': {'eps_r': best_plast_eps, 'sigma': 0.027},
        }
        label = f"mat_ce{ce}"
        t0 = time.time()
        cfg = make_cfg(materials, diffuse=False, samples=50000)
        r = eval_config(cfg, label)
        if r and r.get('pwr_corr_mean', -1) > best_pwr:
            best_pwr = r['pwr_corr_mean']
            best_mat = materials.copy()
        pc = r['pwr_corr_mean'] if r else -1
        print(f"    conc_eps={ce:5.2f}: pwr={pc:.4f} ({time.time()-t0:.0f}s)"
              f"{'  *BEST*' if r and pc == best_pwr else ''}", flush=True)
        if r: all_results[label] = r

    best_conc_eps = best_mat['itu_concrete']['eps_r']
    print(f"  Best concrete eps_r: {best_conc_eps}")

    # Sweep ceiling
    print(f"\n  Sweeping ceiling eps_r (plaster={best_plast_eps}, concrete={best_conc_eps})...")
    for cle in ceiling_eps:
        materials = {
            'itu_concrete': {'eps_r': best_conc_eps, 'sigma': 0.05},
            'itu_ceiling_board': {'eps_r': cle, 'sigma': 0.001},
            'itu_plasterboard': {'eps_r': best_plast_eps, 'sigma': 0.027},
        }
        label = f"mat_cle{cle}"
        t0 = time.time()
        cfg = make_cfg(materials, diffuse=False, samples=50000)
        r = eval_config(cfg, label)
        if r and r.get('pwr_corr_mean', -1) > best_pwr:
            best_pwr = r['pwr_corr_mean']
            best_mat = materials.copy()
        pc = r['pwr_corr_mean'] if r else -1
        print(f"    ceil_eps={cle:5.2f}: pwr={pc:.4f} ({time.time()-t0:.0f}s)"
              f"{'  *BEST*' if r and pc == best_pwr else ''}", flush=True)
        if r: all_results[label] = r

    print(f"\n  Phase 1 best: pwr={best_pwr:.4f}")
    print(f"  Materials: {json.dumps(best_mat, indent=2)}")
    all_results['phase1_best'] = {'pwr_corr_mean': best_pwr, 'materials': best_mat}
    save(all_results, "grid_phase1.json")

    # ========================================
    # Phase 2: Scattering coefficient + pattern sweep
    # ========================================
    print("\n[Phase 2] Scattering sweep on calibrated materials...\n")

    scatt_coeffs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    scatt_patterns = ['lambertian', 'backscatter_5_5', 'backscatter_3_3',
                      'backscatter_10_3', 'directive_5']

    best_combined = -1.0
    best_scat_coeff = 0.0
    best_scat_pat = 'lambertian'

    for sc in scatt_coeffs:
        if sc == 0.0:
            # Just test no scattering
            materials = {m: {**v, 'scatt_coeff': 0.0} for m, v in best_mat.items()}
            label = f"scat_0.0"
            t0 = time.time()
            cfg = make_cfg(materials, diffuse=False, samples=50000)
            r = eval_config(cfg, label)
            if r:
                score = r['pwr_corr_mean']
                if score > best_combined:
                    best_combined = score; best_scat_coeff = 0.0; best_scat_pat = 'lambertian'
                print(f"  scat=0.00 (no diffuse): pwr={score:.4f} ({time.time()-t0:.0f}s)"
                      f"{'  *BEST*' if score == best_combined else ''}", flush=True)
                all_results[label] = r
            continue

        for sp in scatt_patterns:
            materials = {m: {**v, 'scatt_coeff': sc} for m, v in best_mat.items()}
            label = f"scat_{sc}_{sp}"
            t0 = time.time()
            cfg = make_cfg(materials, diffuse=True, scatt_pat=sp, samples=50000)
            r = eval_config(cfg, label)
            if r:
                score = r['pwr_corr_mean']
                if score > best_combined:
                    best_combined = score; best_scat_coeff = sc; best_scat_pat = sp
                print(f"  scat={sc:.2f} {sp:20s}: pwr={score:.4f} pdp={r['pdp_corr_mean']:.4f} "
                      f"({time.time()-t0:.0f}s)"
                      f"{'  *BEST*' if score == best_combined else ''}", flush=True)
                all_results[label] = r

    print(f"\n  Phase 2 best: pwr={best_combined:.4f}, scat_coeff={best_scat_coeff}, pattern={best_scat_pat}")
    all_results['phase2_best'] = {'pwr_corr_mean': best_combined, 'scatt_coeff': best_scat_coeff,
                                   'scatt_pattern': best_scat_pat}
    save(all_results, "grid_phase2.json")

    # ========================================
    # Phase 3: Antenna pattern sweep
    # ========================================
    print("\n[Phase 3] Antenna pattern sweep...\n")

    final_materials = {m: {**v, 'scatt_coeff': best_scat_coeff} for m, v in best_mat.items()}
    best_ant_pwr = -1.0
    best_tx = "dipole"
    best_rx = "dipole"

    for tx, rx in [("dipole", "dipole"), ("iso", "dipole"), ("iso", "iso"),
                   ("tr38901", "dipole"), ("dipole", "iso")]:
        label = f"ant_{tx}_{rx}"
        t0 = time.time()
        cfg = make_cfg(final_materials, tx_pat=tx, rx_pat=rx,
                        diffuse=(best_scat_coeff > 0), scatt_pat=best_scat_pat, samples=50000)
        r = eval_config(cfg, label)
        if r:
            score = r['pwr_corr_mean']
            if score > best_ant_pwr:
                best_ant_pwr = score; best_tx = tx; best_rx = rx
            print(f"  TX={tx:10s} RX={rx:10s}: pwr={score:.4f} pdp={r['pdp_corr_mean']:.4f} "
                  f"({time.time()-t0:.0f}s)"
                  f"{'  *BEST*' if score == best_ant_pwr else ''}", flush=True)
            all_results[label] = r

    print(f"\n  Phase 3 best: TX={best_tx}, RX={best_rx}, pwr={best_ant_pwr:.4f}")
    save(all_results, "grid_phase3.json")

    # ========================================
    # Phase 4: Final evaluation on test set
    # ========================================
    print("\n[Phase 4] Final evaluation on test set (50 positions, high quality)...\n")

    final_cfg = {
        'materials': final_materials,
        'tx_pattern': best_tx,
        'rx_pattern': best_rx,
        'diffuse': best_scat_coeff > 0,
        'scatt_pattern': best_scat_pat,
        'max_depth': 5,
        'samples': 200000,
        'position_indices': TEST_POS,
    }

    print("  Evaluating final config on test set...")
    t0 = time.time()
    test_result = eval_config(final_cfg, "final_test")
    if test_result:
        print(f"  Test result ({time.time()-t0:.0f}s):")
        print(f"    pwr_corr: {test_result['pwr_corr_mean']:.4f} (±{test_result['pwr_corr_std']:.4f})")
        print(f"    pdp_corr: {test_result['pdp_corr_mean']:.4f} (±{test_result['pdp_corr_std']:.4f})")
        print(f"    rms_err:  {test_result['rms_delay_err_ns_mean']:.1f}ns")
        print(f"    n_valid:  {test_result['n_valid']}")
        all_results['final_test'] = test_result

    # Baseline on test set
    print("\n  Baseline comparison on test set...")
    itu_cfg = make_cfg(
        {'itu_concrete': {'eps_r': 5.24, 'sigma': 0.121},
         'itu_ceiling_board': {'eps_r': 1.48, 'sigma': 0.004},
         'itu_plasterboard': {'eps_r': 2.73, 'sigma': 0.027}},
        diffuse=False, samples=200000, positions=TEST_POS)
    itu_result = eval_config(itu_cfg, "test_itu")
    if itu_result:
        print(f"    ITU baseline: pwr={itu_result['pwr_corr_mean']:.4f}")
        all_results['test_itu'] = itu_result

    # ========================================
    # Summary
    # ========================================
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"CALIBRATION COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"  Materials: {json.dumps(final_materials, indent=4)}")
    print(f"  TX: {best_tx}, RX: {best_rx}")
    print(f"  Scattering: coeff={best_scat_coeff}, pattern={best_scat_pat}")
    if test_result:
        print(f"  Test pwr_corr: {test_result['pwr_corr_mean']:.4f}")
        if itu_result:
            ratio = test_result['pwr_corr_mean'] / max(itu_result['pwr_corr_mean'], 0.001)
            print(f"  Improvement vs ITU: {ratio:.1f}x")

    all_results['final_config'] = {
        'materials': final_materials,
        'tx_pattern': best_tx,
        'rx_pattern': best_rx,
        'scatt_coeff': best_scat_coeff,
        'scatt_pattern': best_scat_pat,
    }
    save(all_results, "grid_calibration_results.json")

def save(results, filename):
    def ser(obj):
        if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(RESULTS_DIR / filename, 'w') as f:
        json.dump(ser(results), f, indent=2)

if __name__ == "__main__":
    main()
