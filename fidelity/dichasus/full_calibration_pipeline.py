#!/usr/bin/env python
"""
Full Calibration Pipeline for DICHASUS dc41
============================================
Jointly optimizes material properties (eps_r, sigma, scattering_coefficient),
scattering patterns, and antenna patterns using CMA-ES.

Key improvements over prior approaches:
  1. Scattering enabled and optimized (scattering_coefficient per material)
  2. CMA-ES global optimizer (not Nelder-Mead or grid search)
  3. Multi-objective loss: pwr_corr + pdp_corr + rms_delay_err
  4. More training positions (30 vs 15-20)
  5. Antenna pattern + scattering pattern jointly optimized
  6. Staged optimization: materials -> scattering -> antenna -> joint refinement
  7. Full evaluation with beam prediction

Usage:
  conda activate nfsim
  python full_calibration_pipeline.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import json
import gc
import time
import csv
import sys
import cma
import drjit as dr
from pathlib import Path
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       PathSolver, RadioMaterial, BackscatteringPattern,
                       DirectivePattern, LambertianPattern)

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "full_calibration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENE_PATH = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")
BANDWIDTH = 50e6
N_SUB = 1024
FREQ = 3.438e9

# ============================================================
# Coordinate transform (matches dichasus_fidelity_experiment.py)
# ============================================================
data_dir = str(BASE_DIR / "diff-rt-calibration/data")
data_dict = {}
with open(os.path.join(data_dir, 'coordinates.csv'), mode='r') as f:
    for row in csv.DictReader(f):
        data_dict[row['Name']] = {k: v for k, v in row.items() if k != 'Name'}

poi_raw = {}
for name, pos in data_dict.items():
    if pos['South'] != 'noLoS':
        poi_raw[name] = np.array([float(pos['West']), float(pos['South']),
                                   float(pos['Height'])], dtype=np.float64)
poi_raw['array_1'] = np.array([7.480775, -20.9824, 1.39335])
poi_raw['array_2'] = np.array([-6.390425, 24.440075, 1.4197])

center = poi_raw['AU'].copy()
poi_centered = {k: v - center for k, v in poi_raw.items()}
nwu = poi_centered['NWU']
rot_angle = PI / 2 - np.arctan2(nwu[1], nwu[0])
rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                     [np.sin(rot_angle), np.cos(rot_angle), 0],
                     [0, 0, 1]], dtype=np.float64)
poi = {k: rot_mat @ v for k, v in poi_centered.items()}


def transform_pos(pos_raw):
    return rot_mat @ (pos_raw - center)


# Antenna assignments
with open(os.path.join(data_dir, 'spec.json'), 'r') as f:
    spec = json.load(f)
north_assign = np.array(spec['antennas'][0]['assignments']).flatten()
south_assign = np.array(spec['antennas'][1]['assignments']).flatten()

# ============================================================
# Load measurements
# ============================================================
print("=" * 70)
print("FULL CALIBRATION PIPELINE FOR DICHASUS DC41")
print("=" * 70)
print("\n[1/7] Loading DICHASUS dc41 measurements...")

tfrecord_path = str(BASE_DIR / "dcxx/dichasus-dc41.tfrecords")
with open(str(BASE_DIR / "dcxx/reftx-offsets-dichasus-dc41.json"), 'r') as f:
    offsets = json.load(f)

feature_desc = {
    'csi': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'pos-tachy': tf.io.FixedLenFeature([], tf.string, default_value=''),
}


def parse_fn(proto):
    record = tf.io.parse_single_example(proto, feature_desc)
    csi = tf.ensure_shape(tf.io.parse_tensor(record['csi'], out_type=tf.float32), (64, 1024, 2))
    csi = tf.signal.fftshift(csi, axes=1)
    csi = tf.complex(csi[..., 0], csi[..., 1])
    n_sub = tf.shape(csi)[1]
    sto = tf.tensordot(
        tf.constant(offsets['sto'], tf.float32),
        2 * np.pi * tf.range(n_sub, dtype=tf.float32) / tf.cast(n_sub, tf.float32), axes=0)
    cpo = tf.tensordot(
        tf.constant(offsets['cpo'], tf.float32),
        tf.ones(n_sub, dtype=tf.float32), axes=0)
    csi = csi * tf.exp(tf.complex(0.0, sto + cpo))
    pos = tf.ensure_shape(tf.io.parse_tensor(record['pos-tachy'], out_type=tf.float64), (3,))
    return csi, pos


ds = tf.data.TFRecordDataset([tfrecord_path])
print("  Loading records...", end='', flush=True)
all_samples = []
for i, item in enumerate(ds.map(parse_fn)):
    all_samples.append(item)
    if (i + 1) % 500 == 0:
        print(f"{i+1}...", end='', flush=True)
    if i >= 3281:
        break
total_samples = len(all_samples)
print(f" done! {total_samples} samples", flush=True)

freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())
time_ns = np.arange(N_SUB) / BANDWIDTH * 1e9

# Split: train / test
rng = np.random.RandomState(42)
TRAIN_SIZE = 300
all_indices = rng.permutation(total_samples)
train_indices = sorted(all_indices[:TRAIN_SIZE].tolist())
test_indices = sorted(all_indices[TRAIN_SIZE:TRAIN_SIZE + 100].tolist())
train_samples = [all_samples[i] for i in train_indices]
test_samples = [all_samples[i] for i in test_indices]
print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

# ============================================================
# Materials config
# ============================================================
MATERIAL_NAMES = ['itu_concrete', 'itu_ceiling_board', 'itu_plasterboard']
OBJ_MAT_MAP = {
    'floor': 'itu_concrete',
    'ceiling': 'itu_ceiling_board',
    'walls': 'itu_plasterboard',
}
ITU_DEFAULTS = {
    'itu_concrete': {'eps_r': 5.24, 'sigma': 0.121},
    'itu_ceiling_board': {'eps_r': 1.48, 'sigma': 0.004},
    'itu_plasterboard': {'eps_r': 2.73, 'sigma': 0.027},
}


# ============================================================
# Metrics
# ============================================================
def compute_metrics(h_sim, h_meas):
    """Compute all fidelity metrics between simulated and measured CSI."""
    # Spatial power correlation
    sim_pwr = np.array([np.mean(np.abs(h_sim[i]) ** 2) for i in range(64)])
    meas_pwr = np.array([np.mean(np.abs(h_meas[i]) ** 2) for i in range(64)])
    sp = sim_pwr.sum()
    mp = meas_pwr.sum()
    if sp > 0 and mp > 0:
        pwr_corr = np.corrcoef(sim_pwr / sp, meas_pwr / mp)[0, 1]
    else:
        pwr_corr = 0.0

    # PDP correlation
    pdp_sim = np.mean(np.abs(np.fft.ifft(h_sim, axis=1)) ** 2, axis=0)
    pdp_meas = np.mean(np.abs(np.fft.ifft(h_meas, axis=1)) ** 2, axis=0)
    pdp_sim_n = pdp_sim / (pdp_sim.max() + 1e-30)
    pdp_meas_n = pdp_meas / pdp_meas.max()
    pdp_corr = np.corrcoef(pdp_sim_n[:200], pdp_meas_n[:200])[0, 1]

    # RMS delay spread error
    def rms_delay(pdp, t):
        p = pdp / (pdp.sum() + 1e-30)
        mt = np.sum(t * p)
        return np.sqrt(np.sum((t - mt) ** 2 * p))
    rms_sim = rms_delay(pdp_sim[:200], time_ns[:200])
    rms_meas = rms_delay(pdp_meas[:200], time_ns[:200])
    rms_err = abs(rms_sim - rms_meas)

    # Per-antenna CFR correlation
    ant_corrs = []
    for i in range(64):
        s, m = h_sim[i], h_meas[i]
        ns, nm = np.linalg.norm(s), np.linalg.norm(m)
        if ns > 0 and nm > 0:
            ant_corrs.append(np.abs(np.vdot(s, m)) / (ns * nm))
    mean_ant_corr = np.mean(ant_corrs) if ant_corrs else 0.0

    return {
        'pwr_corr': float(pwr_corr) if not np.isnan(pwr_corr) else 0.0,
        'pdp_corr': float(pdp_corr) if not np.isnan(pdp_corr) else 0.0,
        'rms_delay_err_ns': float(rms_err),
        'mean_ant_corr': float(mean_ant_corr),
    }


# ============================================================
# RT simulation
# ============================================================
def run_rt(pos_raw_np, material_params, tx_pattern="dipole", rx_pattern="dipole",
           max_depth=5, samples=50000, diffuse=True, scatt_pattern=None):
    """Run RT for one position with given material parameters."""
    pos_np = transform_pos(pos_raw_np)

    scene = load_scene(SCENE_PATH)
    scene.frequency = FREQ

    scene.tx_array = PlanarArray(
        num_rows=4, num_cols=8,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern=tx_pattern, polarization="V")
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern=rx_pattern, polarization="V")

    # Set custom material parameters (including scattering)
    for obj_name, mat_name in OBJ_MAT_MAP.items():
        if mat_name in material_params:
            mp = material_params[mat_name]
            rm = RadioMaterial(
                f"cal_{mat_name}",
                relative_permittivity=float(mp['eps_r']),
                conductivity=float(mp['sigma']))
            # Set scattering parameters
            sc = mp.get('scatt_coeff', 0.0)
            if sc > 0.01:
                rm.scattering_coefficient = float(sc)
                xpd = mp.get('xpd_coeff', 0.0)
                rm.xpd_coefficient = float(xpd)
                if scatt_pattern is not None:
                    rm.scattering_pattern = scatt_pattern
            scene.objects[obj_name].radio_material = rm

    # Place TX/RX
    for tx_ind in [1, 2]:
        p = poi[f'array_{tx_ind}'].tolist()
        o = [float(-PI / 2 if tx_ind == 1 else PI / 2), 0.0, 0.0]
        scene.add(Transmitter(name=f'tx-{tx_ind}', position=p, orientation=o))
    scene.add(Receiver(name='rx-0', position=pos_np.tolist()))

    solver = PathSolver()
    try:
        paths = solver(scene=scene, max_depth=max_depth, los=True,
                       specular_reflection=True, diffuse_reflection=diffuse,
                       refraction=False, synthetic_array=True,
                       samples_per_src=samples, max_num_paths_per_src=samples)
    except Exception:
        del scene, solver
        dr.flush_malloc_cache()
        gc.collect()
        return None

    n_paths = paths.tau.shape[-1]
    if n_paths == 0:
        del scene, solver, paths
        dr.flush_malloc_cache()
        gc.collect()
        return None

    cfr = paths.cfr(freq_dr, normalize_delays=False, out_type='numpy')
    cfr_np = np.squeeze(cfr)

    h_sim = np.zeros((64, N_SUB), dtype=complex)
    if cfr_np.ndim == 3:
        for ant in range(32):
            h_sim[north_assign[ant]] = cfr_np[0, ant]
            h_sim[south_assign[ant]] = cfr_np[1, ant]
    elif cfr_np.ndim == 2:
        for ant in range(min(32, cfr_np.shape[0])):
            h_sim[north_assign[ant]] = cfr_np[ant]

    del scene, solver, paths, cfr, cfr_np
    dr.flush_malloc_cache()
    gc.collect()
    return h_sim


def evaluate_config(material_params, positions, n_max=30,
                    tx_pattern="dipole", rx_pattern="dipole",
                    max_depth=5, samples=50000, diffuse=True,
                    scatt_pattern=None, verbose=False):
    """Evaluate a parameter configuration on a set of positions."""
    metrics_list = []
    for i, (csi_meas, pos_raw) in enumerate(positions):
        if i >= n_max:
            break
        h_meas = csi_meas.numpy()
        h_sim = run_rt(pos_raw.numpy(), material_params,
                        tx_pattern=tx_pattern, rx_pattern=rx_pattern,
                        max_depth=max_depth, samples=samples,
                        diffuse=diffuse, scatt_pattern=scatt_pattern)
        if h_sim is None:
            continue
        m = compute_metrics(h_sim, h_meas)
        metrics_list.append(m)
        if verbose and (i + 1) % 10 == 0:
            print(f"    [{i+1}/{min(n_max, len(positions))}] "
                  f"pwr={m['pwr_corr']:.3f} pdp={m['pdp_corr']:.3f}", flush=True)

    if not metrics_list:
        return {'pwr_corr_mean': -1, 'pdp_corr_mean': -1,
                'rms_delay_err_ns_mean': 999, 'mean_ant_corr_mean': 0, 'n_valid': 0}

    result = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list]
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std'] = float(np.std(vals))
    result['n_valid'] = len(metrics_list)
    return result


def combined_loss(result):
    """Multi-objective loss: maximize pwr_corr + pdp_corr, minimize rms_delay."""
    pc = result.get('pwr_corr_mean', 0)
    dc = result.get('pdp_corr_mean', 0)
    rms = result.get('rms_delay_err_ns_mean', 50)
    # Normalize rms to [0, 1] range (50ns = worst)
    rms_norm = min(rms, 50) / 50
    loss = -(0.5 * pc + 0.3 * dc - 0.2 * rms_norm)
    return loss


# ============================================================
# CMA-ES parameter space
# ============================================================
# Params: for each of 3 materials: log(eps_r), log(sigma), scatt_coeff
# Total: 9 parameters

def params_from_vec(x):
    """Convert CMA-ES vector [9] to material params dict."""
    params = {}
    for i, mat_name in enumerate(MATERIAL_NAMES):
        eps_r = np.exp(np.clip(x[3 * i], np.log(1.01), np.log(50)))
        sigma = np.exp(np.clip(x[3 * i + 1], np.log(1e-4), np.log(10)))
        scatt = np.clip(x[3 * i + 2], 0.0, 0.8)
        params[mat_name] = {
            'eps_r': float(eps_r),
            'sigma': float(sigma),
            'scatt_coeff': float(scatt),
        }
    return params


def vec_from_itu():
    """Starting point from ITU defaults (scattering = 0)."""
    x = []
    for mat_name in MATERIAL_NAMES:
        d = ITU_DEFAULTS[mat_name]
        x.extend([np.log(d['eps_r']), np.log(d['sigma']), 0.1])
    return np.array(x)


def vec_from_prior_best():
    """Starting point from prior best (diffrt_calibration refined result)."""
    prior = {
        'itu_concrete': {'eps_r': 4.5, 'sigma': 0.05},
        'itu_ceiling_board': {'eps_r': 25.0, 'sigma': 0.001},
        'itu_plasterboard': {'eps_r': 1.2, 'sigma': 0.027},
    }
    x = []
    for mat_name in MATERIAL_NAMES:
        d = prior[mat_name]
        x.extend([np.log(d['eps_r']), np.log(d['sigma']), 0.1])
    return np.array(x)


# ============================================================
# Main pipeline
# ============================================================
def main():
    all_results = {}
    t_start = time.time()

    # Subsample train positions for optimization (faster)
    n_opt_train = 15  # positions per CMA-ES evaluation (keep small for speed)
    n_opt_eval = 30   # positions for evaluation/sweeps
    opt_train = [train_samples[i] for i in
                 sorted(rng.choice(len(train_samples), size=n_opt_eval, replace=False))]
    opt_train_fast = opt_train[:n_opt_train]  # subset for fast CMA-ES

    # --------------------------------------------------------
    # Stage 1: ITU Baseline evaluation
    # --------------------------------------------------------
    print("\n[2/7] Evaluating baselines...")
    itu_params = {m: {'eps_r': d['eps_r'], 'sigma': d['sigma'], 'scatt_coeff': 0.0}
                  for m, d in ITU_DEFAULTS.items()}

    # Baseline: ITU + dipole + no scattering
    t0 = time.time()
    itu_no_scat = evaluate_config(itu_params, opt_train, n_max=30,
                                   tx_pattern="dipole", diffuse=False,
                                   samples=50000, verbose=True)
    all_results['itu_baseline_no_scat'] = itu_no_scat
    print(f"  ITU (no scat):  pwr={itu_no_scat['pwr_corr_mean']:.4f} "
          f"pdp={itu_no_scat['pdp_corr_mean']:.4f} "
          f"rms={itu_no_scat['rms_delay_err_ns_mean']:.1f}ns "
          f"({time.time()-t0:.0f}s)")

    # Baseline: ITU + dipole + scattering enabled
    t0 = time.time()
    itu_params_scat = {m: {'eps_r': d['eps_r'], 'sigma': d['sigma'], 'scatt_coeff': 0.3}
                       for m, d in ITU_DEFAULTS.items()}
    itu_with_scat = evaluate_config(itu_params_scat, opt_train, n_max=30,
                                     tx_pattern="dipole", diffuse=True,
                                     samples=50000, verbose=True)
    all_results['itu_baseline_with_scat'] = itu_with_scat
    print(f"  ITU (scat=0.3): pwr={itu_with_scat['pwr_corr_mean']:.4f} "
          f"pdp={itu_with_scat['pdp_corr_mean']:.4f} "
          f"rms={itu_with_scat['rms_delay_err_ns_mean']:.1f}ns "
          f"({time.time()-t0:.0f}s)")

    # Prior best (from diffrt_calibration)
    t0 = time.time()
    prior_params = {
        'itu_concrete': {'eps_r': 4.5, 'sigma': 0.05, 'scatt_coeff': 0.0},
        'itu_ceiling_board': {'eps_r': 25.0, 'sigma': 0.001, 'scatt_coeff': 0.0},
        'itu_plasterboard': {'eps_r': 1.2, 'sigma': 0.027, 'scatt_coeff': 0.0},
    }
    prior_result = evaluate_config(prior_params, opt_train, n_max=30,
                                    tx_pattern="dipole", diffuse=False,
                                    samples=50000, verbose=True)
    all_results['prior_best'] = prior_result
    print(f"  Prior best:     pwr={prior_result['pwr_corr_mean']:.4f} "
          f"pdp={prior_result['pdp_corr_mean']:.4f} "
          f"rms={prior_result['rms_delay_err_ns_mean']:.1f}ns "
          f"({time.time()-t0:.0f}s)")

    # Save checkpoint
    save_results(all_results, "checkpoint_baselines.json")

    # --------------------------------------------------------
    # Stage 2a: CMA-ES Material-Only Optimization (NO scattering — fast)
    # --------------------------------------------------------
    print("\n[3/7] CMA-ES material optimization (no scattering, fast)...")
    print(f"  Params: 6 (3 materials x [log_eps_r, log_sigma])")
    print(f"  Train positions: {n_opt_train}")

    eval_count = [0]
    best_loss = [999.0]
    best_params_found = [None]

    def cma_objective_mat(x):
        """CMA-ES objective: material-only (no scattering)."""
        params = {}
        for i, mat_name in enumerate(MATERIAL_NAMES):
            eps_r = np.exp(np.clip(x[2 * i], np.log(1.01), np.log(50)))
            sigma = np.exp(np.clip(x[2 * i + 1], np.log(1e-4), np.log(10)))
            params[mat_name] = {'eps_r': float(eps_r), 'sigma': float(sigma), 'scatt_coeff': 0.0}
        try:
            result = evaluate_config(params, opt_train_fast, n_max=n_opt_train,
                                      tx_pattern="dipole", diffuse=False,
                                      samples=50000)
        except Exception as e:
            print(f"  eval ERROR: {e}", flush=True)
            return 1.0
        loss = -result.get('pwr_corr_mean', 0)  # maximize pwr_corr only (fast stage)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            best_params_found[0] = params.copy()
        if eval_count[0] % 5 == 0:
            pc = result.get('pwr_corr_mean', 0)
            param_str = ', '.join(
                f'{k.replace("itu_","")}:e={v["eps_r"]:.1f}/s={v["sigma"]:.3f}'
                for k, v in params.items())
            print(f"  eval {eval_count[0]:3d}: pwr={pc:.3f} "
                  f"best={-best_loss[0]:.3f} [{param_str}]", flush=True)
        return loss

    # Start from prior best
    x0_mat = []
    prior = {'itu_concrete': (4.5, 0.05), 'itu_ceiling_board': (25.0, 0.001), 'itu_plasterboard': (1.2, 0.027)}
    for mat_name in MATERIAL_NAMES:
        e, s = prior[mat_name]
        x0_mat.extend([np.log(e), np.log(s)])
    x0_mat = np.array(x0_mat)

    opts = cma.CMAOptions()
    opts.set('maxiter', 60)
    opts.set('popsize', 12)
    opts.set('tolx', 1e-3)
    opts.set('verb_disp', 0)
    opts.set('bounds', [
        [np.log(1.01), np.log(1e-4)] * 3,
        [np.log(50), np.log(10)] * 3,
    ])

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0_mat, 0.5, opts)
    es.optimize(cma_objective_mat)
    print(f"\n  Stage 2a done ({eval_count[0]} evals, {time.time()-t0:.0f}s)")

    # Extract best material params
    best_x_mat = es.result.xbest
    best_mat_only = {}
    for i, mat_name in enumerate(MATERIAL_NAMES):
        best_mat_only[mat_name] = {
            'eps_r': float(np.exp(np.clip(best_x_mat[2*i], np.log(1.01), np.log(50)))),
            'sigma': float(np.exp(np.clip(best_x_mat[2*i+1], np.log(1e-4), np.log(10)))),
            'scatt_coeff': 0.0,
        }
    print(f"  Best material-only params:")
    for k, v in best_mat_only.items():
        print(f"    {k}: eps_r={v['eps_r']:.3f}, sigma={v['sigma']:.5f}")

    mat_only_result = evaluate_config(best_mat_only, opt_train, n_max=30,
                                       tx_pattern="dipole", diffuse=False,
                                       samples=100000, verbose=True)
    all_results['cma_mat_only'] = mat_only_result
    all_results['cma_mat_only']['params'] = best_mat_only
    print(f"  Material-only:  pwr={mat_only_result['pwr_corr_mean']:.4f} "
          f"pdp={mat_only_result['pdp_corr_mean']:.4f}")

    save_results(all_results, "checkpoint_cma_mat.json")

    # --------------------------------------------------------
    # Stage 2b: Scattering coefficient sweep on top of calibrated materials
    # --------------------------------------------------------
    print("\n  Stage 2b: Scattering coefficient sweep...")
    best_scat_pwr = mat_only_result['pwr_corr_mean']
    best_scat_coeffs = {m: 0.0 for m in MATERIAL_NAMES}

    for sc_val in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        test_p = {m: {**v, 'scatt_coeff': sc_val} for m, v in best_mat_only.items()}
        t0 = time.time()
        r = evaluate_config(test_p, opt_train, n_max=20,
                             tx_pattern="dipole", diffuse=True, samples=50000)
        improved = ""
        if r['pwr_corr_mean'] > best_scat_pwr:
            best_scat_pwr = r['pwr_corr_mean']
            best_scat_coeffs = {m: sc_val for m in MATERIAL_NAMES}
            improved = " *BEST*"
        print(f"    scatt={sc_val:.2f}: pwr={r['pwr_corr_mean']:.4f} "
              f"pdp={r['pdp_corr_mean']:.4f} ({time.time()-t0:.0f}s){improved}", flush=True)
        all_results[f'scat_sweep_{sc_val}'] = r

    # Apply best scattering coefficients
    best_material_params = {m: {**v, 'scatt_coeff': best_scat_coeffs[m]}
                            for m, v in best_mat_only.items()}
    print(f"  Best scattering coefficients: {best_scat_coeffs}")

    # Evaluate combined result
    cma_result = evaluate_config(best_material_params, opt_train, n_max=30,
                                  tx_pattern="dipole", diffuse=True,
                                  samples=100000, verbose=True)
    all_results['cma_optimized'] = cma_result
    all_results['cma_optimized']['params'] = best_material_params
    print(f"  CMA+scatt result: pwr={cma_result['pwr_corr_mean']:.4f} "
          f"pdp={cma_result['pdp_corr_mean']:.4f} "
          f"rms={cma_result['rms_delay_err_ns_mean']:.1f}ns")
    best_x = np.array([np.log(best_material_params[m]['eps_r']) for m in MATERIAL_NAMES
                        for _ in range(1)] +
                       [np.log(best_material_params[m]['sigma']) for m in MATERIAL_NAMES
                        for _ in range(1)])  # placeholder for joint refinement

    save_results(all_results, "checkpoint_cma.json")

    # --------------------------------------------------------
    # Stage 3: Scattering Pattern Sweep
    # --------------------------------------------------------
    print("\n[4/7] Scattering pattern sweep...")
    best_scat_result = cma_result
    best_scat_pattern = None
    best_scat_label = "lambertian_default"

    scat_configs = [
        ("lambertian", LambertianPattern()),
        ("directive_5", DirectivePattern(alpha_r=5)),
        ("directive_10", DirectivePattern(alpha_r=10)),
        ("backscatter_3_3", BackscatteringPattern(alpha_r=3, alpha_i=3)),
        ("backscatter_5_5", BackscatteringPattern(alpha_r=5, alpha_i=5)),
        ("backscatter_10_3", BackscatteringPattern(alpha_r=10, alpha_i=3)),
        ("backscatter_3_10", BackscatteringPattern(alpha_r=3, alpha_i=10)),
    ]

    for label, pattern in scat_configs:
        t0 = time.time()
        r = evaluate_config(best_material_params, opt_train, n_max=20,
                             tx_pattern="dipole", diffuse=True,
                             samples=50000, scatt_pattern=pattern)
        loss = combined_loss(r)
        improved = " *BEST*" if loss < combined_loss(best_scat_result) else ""
        if loss < combined_loss(best_scat_result):
            best_scat_result = r
            best_scat_pattern = pattern
            best_scat_label = label
        print(f"  {label:25s}: pwr={r['pwr_corr_mean']:.4f} "
              f"pdp={r['pdp_corr_mean']:.4f} "
              f"rms={r['rms_delay_err_ns_mean']:.1f}ns "
              f"({time.time()-t0:.0f}s){improved}")
        all_results[f'scat_{label}'] = r

    print(f"  Best scattering: {best_scat_label}")
    save_results(all_results, "checkpoint_scattering.json")

    # --------------------------------------------------------
    # Stage 4: Antenna Pattern Sweep
    # --------------------------------------------------------
    print("\n[5/7] Antenna pattern sweep...")
    best_ant_result = best_scat_result
    best_tx_pattern = "dipole"
    best_rx_pattern = "dipole"

    ant_configs = [
        ("dipole_dipole", "dipole", "dipole"),
        ("iso_dipole", "iso", "dipole"),
        ("iso_iso", "iso", "iso"),
        ("tr38901_dipole", "tr38901", "dipole"),
        ("dipole_iso", "dipole", "iso"),
    ]

    for label, tx_pat, rx_pat in ant_configs:
        t0 = time.time()
        r = evaluate_config(best_material_params, opt_train, n_max=20,
                             tx_pattern=tx_pat, rx_pattern=rx_pat,
                             diffuse=True, samples=50000,
                             scatt_pattern=best_scat_pattern)
        loss = combined_loss(r)
        improved = " *BEST*" if loss < combined_loss(best_ant_result) else ""
        if loss < combined_loss(best_ant_result):
            best_ant_result = r
            best_tx_pattern = tx_pat
            best_rx_pattern = rx_pat
        print(f"  {label:25s}: pwr={r['pwr_corr_mean']:.4f} "
              f"pdp={r['pdp_corr_mean']:.4f} "
              f"rms={r['rms_delay_err_ns_mean']:.1f}ns "
              f"({time.time()-t0:.0f}s){improved}")
        all_results[f'ant_{label}'] = r

    print(f"  Best antenna: TX={best_tx_pattern}, RX={best_rx_pattern}")
    save_results(all_results, "checkpoint_antenna.json")

    # --------------------------------------------------------
    # Stage 5: Joint refinement with best scattering + antenna
    # --------------------------------------------------------
    print("\n[6/7] Joint refinement with best antenna + scattering...")

    eval_count[0] = 0
    best_loss[0] = 999.0

    def refine_objective(x):
        params = params_from_vec(x)
        try:
            result = evaluate_config(params, opt_train_fast, n_max=n_opt_train,
                                      tx_pattern=best_tx_pattern,
                                      rx_pattern=best_rx_pattern,
                                      diffuse=True, samples=30000,
                                      scatt_pattern=best_scat_pattern)
        except Exception as e:
            print(f"  refine ERROR: {e}", flush=True)
            return 1.0
        loss = combined_loss(result)
        eval_count[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
            best_params_found[0] = params.copy()
        if eval_count[0] % 5 == 0:
            pc = result.get('pwr_corr_mean', 0)
            dc = result.get('pdp_corr_mean', 0)
            print(f"  refine eval {eval_count[0]:3d}: loss={loss:.4f} "
                  f"pwr={pc:.3f} pdp={dc:.3f} best={best_loss[0]:.4f}", flush=True)
        return loss

    # Build starting vector from best_material_params
    x0_refine = []
    for mat_name in MATERIAL_NAMES:
        mp = best_material_params[mat_name]
        x0_refine.extend([np.log(mp['eps_r']), np.log(mp['sigma']), mp['scatt_coeff']])
    x0_refine = np.array(x0_refine)

    opts_refine = cma.CMAOptions()
    opts_refine.set('maxiter', 40)
    opts_refine.set('popsize', 12)
    opts_refine.set('tolx', 1e-3)
    opts_refine.set('verb_disp', 0)
    opts_refine.set('bounds', [
        [np.log(1.01), np.log(1e-4), 0.0] * 3,
        [np.log(50), np.log(10), 0.8] * 3,
    ])

    t0 = time.time()
    es2 = cma.CMAEvolutionStrategy(x0_refine, 0.3, opts_refine)
    es2.optimize(refine_objective)
    print(f"  Refinement done ({eval_count[0]} evals, {time.time()-t0:.0f}s)")

    final_x = es2.result.xbest
    final_material_params = params_from_vec(final_x)

    print(f"  Final material params:")
    for k, v in final_material_params.items():
        print(f"    {k}: eps_r={v['eps_r']:.3f}, sigma={v['sigma']:.5f}, "
              f"scatt={v['scatt_coeff']:.3f}")

    # --------------------------------------------------------
    # Stage 6: Full evaluation on test set
    # --------------------------------------------------------
    print("\n[7/7] Full evaluation on test set (high-quality RT)...")

    final_config = {
        'material_params': final_material_params,
        'tx_pattern': best_tx_pattern,
        'rx_pattern': best_rx_pattern,
        'scatt_pattern': best_scat_label,
        'diffuse': True,
        'max_depth': 5,
    }

    # Evaluate on test set with high-quality RT
    t0 = time.time()
    test_result = evaluate_config(final_material_params, test_samples, n_max=50,
                                   tx_pattern=best_tx_pattern,
                                   rx_pattern=best_rx_pattern,
                                   diffuse=True, samples=200000,
                                   scatt_pattern=best_scat_pattern,
                                   verbose=True)
    test_result['config'] = final_config
    all_results['final_test'] = test_result
    print(f"\n  Test set result ({time.time()-t0:.0f}s):")
    print(f"    pwr_corr:      {test_result['pwr_corr_mean']:.4f} "
          f"(±{test_result['pwr_corr_std']:.4f})")
    print(f"    pdp_corr:      {test_result['pdp_corr_mean']:.4f} "
          f"(±{test_result['pdp_corr_std']:.4f})")
    print(f"    rms_delay_err: {test_result['rms_delay_err_ns_mean']:.1f}ns "
          f"(±{test_result['rms_delay_err_ns_std']:.1f})")
    print(f"    ant_corr:      {test_result['mean_ant_corr_mean']:.4f}")
    print(f"    n_valid:       {test_result['n_valid']}")

    # Also evaluate baselines on same test set for fair comparison
    print("\n  Baseline comparisons on test set:")
    for label, params, diff in [
        ("ITU_no_scat", itu_params, False),
        ("ITU_with_scat", itu_params_scat, True),
        ("prior_best", prior_params, False),
    ]:
        r = evaluate_config(params, test_samples, n_max=50,
                             tx_pattern="dipole", diffuse=diff, samples=200000)
        all_results[f'test_{label}'] = r
        print(f"    {label:20s}: pwr={r['pwr_corr_mean']:.4f} "
              f"pdp={r['pdp_corr_mean']:.4f} rms={r['rms_delay_err_ns_mean']:.1f}ns")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"CALIBRATION COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\n  Best config:")
    print(f"    TX pattern:  {best_tx_pattern}")
    print(f"    RX pattern:  {best_rx_pattern}")
    print(f"    Scattering:  {best_scat_label}")
    print(f"    Materials:")
    for k, v in final_material_params.items():
        print(f"      {k}: eps_r={v['eps_r']:.3f}, sigma={v['sigma']:.5f}, "
              f"scatt={v['scatt_coeff']:.3f}")
    print(f"\n  Test results:")
    print(f"    pwr_corr:  {test_result['pwr_corr_mean']:.4f}")
    print(f"    pdp_corr:  {test_result['pdp_corr_mean']:.4f}")
    print(f"    rms_err:   {test_result['rms_delay_err_ns_mean']:.1f}ns")

    # Improvement over baselines
    itu_pc = all_results.get('test_ITU_no_scat', {}).get('pwr_corr_mean', 0)
    prior_pc = all_results.get('test_prior_best', {}).get('pwr_corr_mean', 0)
    final_pc = test_result['pwr_corr_mean']
    if itu_pc > 0:
        print(f"\n  Improvement vs ITU:   {final_pc / itu_pc:.1f}x pwr_corr")
    if prior_pc > 0:
        print(f"  Improvement vs prior: {final_pc / prior_pc:.1f}x pwr_corr")

    save_results(all_results, "full_calibration_results.json")
    print(f"\n  Results saved to {RESULTS_DIR}/full_calibration_results.json")


def save_results(results, filename):
    """Save results with numpy type conversion."""
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)


if __name__ == "__main__":
    main()
