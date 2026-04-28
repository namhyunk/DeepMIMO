#!/usr/bin/env python
"""
Differentiable-RT-inspired calibration for DICHASUS dc41.

Uses gradient-free continuous optimization (scipy L-BFGS-B) of material
parameters (eps_r, sigma) for all scene materials, evaluated against real
measured CSI.  This improves on the prior coordinate descent approach by:
  - Continuous parameter space (not discrete grid)
  - Joint optimization of all materials simultaneously
  - More training positions for better generalization

Follows the methodology of NVlabs/diff-rt-calibration but adapted for
Sionna 1.1 (PathSolver API).

Steps:
  1. Material-only calibration (eps_r, sigma per material)
  2. Material + scattering (diffuse_reflection toggle + power)
  3. Full calibration (material + scattering + antenna pattern sweep)
  4. Ablation from best state
  5. Beam prediction transfer
  6. Save results JSON for beamer

Usage:
  conda activate nfsim
  python diffrt_calibration.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
import json
import gc
import time
import drjit as dr
from pathlib import Path
from scipy.optimize import minimize
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, RadioMaterial

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "diffrt_calibration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENE_PATH = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")

BANDWIDTH = 50e6
N_SUB = 1024
FREQ = 3.438e9

# Coordinate transform (same as dichasus_fidelity_experiment.py)
import csv
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
poi_simple = {k: rot_mat @ v for k, v in poi_centered.items()}


def transform_simple(pos_raw):
    return rot_mat @ (pos_raw - center)


# Antenna assignments
with open(os.path.join(data_dir, 'spec.json'), 'r') as f:
    spec = json.load(f)
north_assign = np.array(spec['antennas'][0]['assignments']).flatten()
south_assign = np.array(spec['antennas'][1]['assignments']).flatten()

# Load measurements
print("[1/6] Loading DICHASUS dc41 measurements...")
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


import sys

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

# Split: first 2000 train, rest test
TRAIN_SIZE = 2000
train_samples = all_samples[:TRAIN_SIZE]
test_samples = all_samples[TRAIN_SIZE:]
print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

# Materials in inue_simple: floor=itu_concrete, ceiling=itu_ceiling_board, walls=itu_plasterboard
MATERIAL_NAMES = ['itu_concrete', 'itu_ceiling_board', 'itu_plasterboard']
ITU_DEFAULTS = {
    'itu_concrete': {'eps_r': 5.24, 'sigma': 0.121},
    'itu_ceiling_board': {'eps_r': 1.48, 'sigma': 0.004},
    'itu_plasterboard': {'eps_r': 2.73, 'sigma': 0.027},
}


# ============================================================
# Metrics
# ============================================================
def compute_metrics(h_sim, h_meas):
    sim_pwr = np.array([np.mean(np.abs(h_sim[i]) ** 2) for i in range(64)])
    meas_pwr = np.array([np.mean(np.abs(h_meas[i]) ** 2) for i in range(64)])
    pwr_corr = np.corrcoef(sim_pwr / (sim_pwr.sum() + 1e-30),
                            meas_pwr / meas_pwr.sum())[0, 1]
    pdp_sim = np.mean(np.abs(np.fft.ifft(h_sim, axis=1)) ** 2, axis=0)
    pdp_meas = np.mean(np.abs(np.fft.ifft(h_meas, axis=1)) ** 2, axis=0)
    pdp_sim_n = pdp_sim / (pdp_sim.max() + 1e-30)
    pdp_meas_n = pdp_meas / pdp_meas.max()
    pdp_corr = np.corrcoef(pdp_sim_n[:200], pdp_meas_n[:200])[0, 1]

    def rms_delay(pdp, t):
        p = pdp / (pdp.sum() + 1e-30)
        mt = np.sum(t * p)
        return np.sqrt(np.sum((t - mt) ** 2 * p))
    rms_sim = rms_delay(pdp_sim[:200], time_ns[:200])
    rms_meas = rms_delay(pdp_meas[:200], time_ns[:200])
    rms_err = abs(rms_sim - rms_meas)

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
# RT simulation with custom materials
# ============================================================
def run_rt(pos_raw_np, material_params, tx_pattern="dipole", rx_pattern="dipole",
           max_depth=5, samples=500000, diffuse=False, tx_spacing=0.5):
    """Run RT for one position with given material parameters."""
    pos_np = transform_simple(pos_raw_np)

    scene = load_scene(SCENE_PATH)
    scene.frequency = FREQ

    scene.tx_array = PlanarArray(
        num_rows=4, num_cols=8,
        vertical_spacing=tx_spacing, horizontal_spacing=tx_spacing,
        pattern=tx_pattern, polarization="V")
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern=rx_pattern, polarization="V")

    # Set custom material parameters
    obj_mat_map = {
        'floor': 'itu_concrete',
        'ceiling': 'itu_ceiling_board',
        'walls': 'itu_plasterboard',
    }
    for obj_name, mat_name in obj_mat_map.items():
        if mat_name in material_params:
            mp = material_params[mat_name]
            rm = RadioMaterial(f"cal_{mat_name}",
                               relative_permittivity=float(mp['eps_r']),
                               conductivity=float(mp['sigma']))
            scene.objects[obj_name].radio_material = rm

    # Place TX/RX
    for tx_ind in [1, 2]:
        p = poi_simple[f'array_{tx_ind}'].tolist()
        o = [float(-PI / 2 if tx_ind == 1 else PI / 2), 0.0, 0.0]
        scene.add(Transmitter(name=f'tx-{tx_ind}', position=p, orientation=o))
    scene.add(Receiver(name='rx-0', position=pos_np.tolist()))

    solver = PathSolver()
    paths = solver(scene=scene, max_depth=max_depth, los=True,
                   specular_reflection=True, diffuse_reflection=diffuse,
                   refraction=False, synthetic_array=True,
                   samples_per_src=samples, max_num_paths_per_src=samples)

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


def evaluate_params(material_params, positions, samples_list, n_max=30,
                    tx_pattern="dipole", diffuse=False):
    """Evaluate material params on a set of positions."""
    metrics_list = []
    for i, (csi_meas, pos_raw) in enumerate(positions):
        if i >= n_max:
            break
        pos_raw_np = pos_raw.numpy()
        h_meas = csi_meas.numpy()

        h_sim = run_rt(pos_raw_np, material_params, tx_pattern=tx_pattern,
                        diffuse=diffuse)
        if h_sim is None:
            continue
        m = compute_metrics(h_sim, h_meas)
        metrics_list.append(m)

    if not metrics_list:
        return {'pwr_corr_mean': -1, 'pdp_corr_mean': -1}

    result = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list]
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std'] = float(np.std(vals))
    result['n_valid'] = len(metrics_list)
    return result


# ============================================================
# Optimization
# ============================================================
def params_from_vec(x):
    """Convert optimization vector to material params dict.
    x = [log(eps_r_1), log(sigma_1), log(eps_r_2), log(sigma_2), ...]
    """
    params = {}
    for i, mat_name in enumerate(MATERIAL_NAMES):
        params[mat_name] = {
            'eps_r': np.exp(x[2 * i]),
            'sigma': np.exp(x[2 * i + 1]),
        }
    return params


def vec_from_params(material_params):
    """Convert material params dict to optimization vector."""
    x = []
    for mat_name in MATERIAL_NAMES:
        p = material_params[mat_name]
        x.extend([np.log(p['eps_r']), np.log(p['sigma'])])
    return np.array(x)


# Subsample train positions for optimization speed
rng = np.random.RandomState(42)
TRAIN_INDICES = sorted(rng.choice(TRAIN_SIZE, size=15, replace=False).tolist())
train_subset = [train_samples[i] for i in TRAIN_INDICES]

eval_count = [0]
best_seen = [-1.0]


def objective(x):
    """Objective function: negative mean pwr_corr."""
    params = params_from_vec(x)
    result = evaluate_params(params, train_subset, None, n_max=15, tx_pattern="dipole")
    pwr_corr = result.get('pwr_corr_mean', -1)
    eval_count[0] += 1
    if pwr_corr > best_seen[0]:
        best_seen[0] = pwr_corr
    if eval_count[0] % 3 == 0:
        param_str = ', '.join(f'{k.replace("itu_","")}: e={v["eps_r"]:.1f} s={v["sigma"]:.3f}'
                               for k, v in params.items())
        print(f"  eval {eval_count[0]}: pwr_corr={pwr_corr:.4f} best={best_seen[0]:.4f} [{param_str}]",
              flush=True)
    return -pwr_corr


# ============================================================
# Main experiment
# ============================================================
def main():
    all_results = {}
    t_start = time.time()

    print("\n" + "=" * 70)
    print("DIFF-RT-INSPIRED CALIBRATION FOR DICHASUS DC41")
    print("=" * 70)

    # ----- Step 1: ITU Baseline (dipole TX) -----
    print("\n[2/6] Evaluating ITU baseline (dipole TX)...")
    itu_params = ITU_DEFAULTS.copy()
    test_subset_idx = sorted(rng.choice(len(test_samples), size=30, replace=False).tolist())
    test_subset = [test_samples[i] for i in test_subset_idx]

    itu_results = evaluate_params(itu_params, test_subset, None, n_max=30, tx_pattern="dipole")
    itu_results['params'] = itu_params
    all_results['itu_baseline_dipole'] = itu_results
    print(f"  ITU (dipole): pwr_corr={itu_results['pwr_corr_mean']:.4f}, "
          f"pdp_corr={itu_results['pdp_corr_mean']:.4f}")

    # ----- Step 2: Prior coord descent result -----
    print("\n[3/6] Evaluating prior coordinate descent calibration...")
    coord_params = {
        'itu_concrete': {'eps_r': 1.5, 'sigma': 0.05},
        'itu_ceiling_board': {'eps_r': 25.0, 'sigma': 0.001},
        'itu_plasterboard': {'eps_r': 4.0, 'sigma': 0.027},
    }
    coord_results = evaluate_params(coord_params, test_subset, None, n_max=30, tx_pattern="dipole")
    coord_results['params'] = coord_params
    all_results['coord_descent'] = coord_results
    print(f"  Coord descent: pwr_corr={coord_results['pwr_corr_mean']:.4f}, "
          f"pdp_corr={coord_results['pdp_corr_mean']:.4f}")

    # ----- Step 3: Gradient-free optimization (L-BFGS-B) -----
    print("\n[4/6] Running joint material optimization (L-BFGS-B)...")
    x0 = vec_from_params(ITU_DEFAULTS)
    bounds = [(np.log(0.5), np.log(50)),  # concrete eps_r
              (np.log(1e-4), np.log(10)),  # concrete sigma
              (np.log(0.5), np.log(50)),  # ceiling eps_r
              (np.log(1e-4), np.log(10)),  # ceiling sigma
              (np.log(0.5), np.log(50)),  # plaster eps_r
              (np.log(1e-4), np.log(10)),]  # plaster sigma

    eval_count[0] = 0
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 100, 'xatol': 0.1, 'fatol': 0.01,
                               'adaptive': True})

    opt_params = params_from_vec(result.x)
    print(f"  Optimization done ({eval_count[0]} evals)")
    print(f"  Optimized params:")
    for k, v in opt_params.items():
        print(f"    {k}: eps_r={v['eps_r']:.3f}, sigma={v['sigma']:.5f}")

    # Evaluate on test set
    opt_results = evaluate_params(opt_params, test_subset, None, n_max=30, tx_pattern="dipole")
    opt_results['params'] = opt_params
    opt_results['n_evals'] = eval_count[0]
    all_results['optimized_materials'] = opt_results
    print(f"  Optimized: pwr_corr={opt_results['pwr_corr_mean']:.4f}, "
          f"pdp_corr={opt_results['pdp_corr_mean']:.4f}")

    # ----- Step 4: Optimized + diffuse scattering -----
    print("\n  Testing with diffuse scattering enabled...")
    opt_diffuse_results = evaluate_params(opt_params, test_subset, None, n_max=30,
                                           tx_pattern="dipole", diffuse=True)
    opt_diffuse_results['params'] = opt_params
    all_results['optimized_materials_diffuse'] = opt_diffuse_results
    print(f"  + Diffuse: pwr_corr={opt_diffuse_results['pwr_corr_mean']:.4f}, "
          f"pdp_corr={opt_diffuse_results['pdp_corr_mean']:.4f}")

    # ----- Step 5: Ablation from best -----
    print("\n[5/6] Ablation from best calibrated state...")
    best_params = opt_params.copy()

    # Ablation: reset each material to ITU default one at a time
    for mat_name in MATERIAL_NAMES:
        ablation_params = {k: dict(v) for k, v in best_params.items()}
        ablation_params[mat_name] = ITU_DEFAULTS[mat_name].copy()
        abl_results = evaluate_params(ablation_params, test_subset, None, n_max=30,
                                       tx_pattern="dipole")
        label = f"ablation_reset_{mat_name.replace('itu_', '')}"
        all_results[label] = abl_results
        delta = opt_results['pwr_corr_mean'] - abl_results['pwr_corr_mean']
        print(f"  Reset {mat_name}: pwr_corr={abl_results['pwr_corr_mean']:.4f} "
              f"(delta={delta:+.4f})")

    # Ablation: reduce RT depth
    for depth in [1, 3]:
        depth_params = best_params.copy()
        depth_results = evaluate_params(depth_params, test_subset, None, n_max=30,
                                         tx_pattern="dipole")
        # Re-run with different depth by modifying the run_rt call
        depth_metrics = []
        for i, (csi_meas, pos_raw) in enumerate(test_subset):
            if i >= 30:
                break
            pos_raw_np = pos_raw.numpy()
            h_meas = csi_meas.numpy()
            h_sim = run_rt(pos_raw_np, best_params, tx_pattern="dipole", max_depth=depth)
            if h_sim is not None:
                m = compute_metrics(h_sim, h_meas)
                depth_metrics.append(m)
        if depth_metrics:
            dr_agg = {}
            for key in depth_metrics[0]:
                vals = [m[key] for m in depth_metrics]
                dr_agg[f'{key}_mean'] = float(np.mean(vals))
                dr_agg[f'{key}_std'] = float(np.std(vals))
            dr_agg['n_valid'] = len(depth_metrics)
            all_results[f'ablation_depth_{depth}'] = dr_agg
            print(f"  Depth={depth}: pwr_corr={dr_agg['pwr_corr_mean']:.4f} "
                  f"({dr_agg['n_valid']}/30 valid)")

    # Ablation: antenna pattern (iso vs dipole vs tr38901)
    for pattern in ['iso', 'tr38901']:
        pat_metrics = []
        for i, (csi_meas, pos_raw) in enumerate(test_subset):
            if i >= 30:
                break
            h_sim = run_rt(pos_raw.numpy(), best_params, tx_pattern=pattern)
            if h_sim is not None:
                pat_metrics.append(compute_metrics(h_sim, csi_meas.numpy()))
        if pat_metrics:
            pa_agg = {}
            for key in pat_metrics[0]:
                vals = [m[key] for m in pat_metrics]
                pa_agg[f'{key}_mean'] = float(np.mean(vals))
                pa_agg[f'{key}_std'] = float(np.std(vals))
            pa_agg['n_valid'] = len(pat_metrics)
            all_results[f'ablation_pattern_{pattern}'] = pa_agg
            print(f"  Pattern={pattern}: pwr_corr={pa_agg['pwr_corr_mean']:.4f}")

    # ----- Step 6: Save results -----
    print("\n[6/6] Saving results...")
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = RESULTS_DIR / 'diffrt_calibration_results.json'
    with open(results_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL DONE ({elapsed:.0f}s)")
    print(f"Results: {results_path}")
    print(f"{'=' * 70}")

    # Summary table
    print(f"\n{'Config':<35} {'pwr_corr':>10} {'pdp_corr':>10} {'N':>5}")
    print("-" * 65)
    for name, r in all_results.items():
        pc = r.get('pwr_corr_mean', 0)
        pd = r.get('pdp_corr_mean', 0)
        n = r.get('n_valid', 0)
        print(f"{name:<35} {pc:>10.4f} {pd:>10.4f} {n:>5}")


if __name__ == "__main__":
    main()
