#!/usr/bin/env python
"""
Fast calibration refinement + ablation for DICHASUS dc41.
Scene is created once and reused to avoid drjit resource leaks.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
import json
import gc
import time
import csv
import drjit as dr
from pathlib import Path
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, RadioMaterial

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "diffrt_calibration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCENE_PATH = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")

BANDWIDTH = 50e6
N_SUB = 1024
FREQ = 3.438e9

# Coordinate transform
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

with open(os.path.join(data_dir, 'spec.json'), 'r') as f:
    spec = json.load(f)
north_assign = np.array(spec['antennas'][0]['assignments']).flatten()
south_assign = np.array(spec['antennas'][1]['assignments']).flatten()

# Load measurements
print("[1] Loading measurements...", flush=True)
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
    sto = tf.tensordot(tf.constant(offsets['sto'], tf.float32),
        2 * np.pi * tf.range(n_sub, dtype=tf.float32) / tf.cast(n_sub, tf.float32), axes=0)
    cpo = tf.tensordot(tf.constant(offsets['cpo'], tf.float32),
        tf.ones(n_sub, dtype=tf.float32), axes=0)
    csi = csi * tf.exp(tf.complex(0.0, sto + cpo))
    pos = tf.ensure_shape(tf.io.parse_tensor(record['pos-tachy'], out_type=tf.float64), (3,))
    return csi, pos

ds = tf.data.TFRecordDataset([tfrecord_path])
all_samples = []
for i, item in enumerate(ds.map(parse_fn)):
    all_samples.append(item)
    if i >= 3281:
        break
print(f"  {len(all_samples)} samples loaded", flush=True)

freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())
time_ns = np.arange(N_SUB) / BANDWIDTH * 1e9

rng = np.random.RandomState(42)
TRAIN_IDX = sorted(rng.choice(2000, size=20, replace=False).tolist())
TEST_IDX = sorted(rng.choice(range(2000, len(all_samples)), size=30, replace=False).tolist())
train_set = [all_samples[i] for i in TRAIN_IDX]
test_set = [all_samples[i] for i in TEST_IDX]

ITU_DEFAULTS = {
    'itu_concrete': {'eps_r': 5.24, 'sigma': 0.121},
    'itu_ceiling_board': {'eps_r': 1.48, 'sigma': 0.004},
    'itu_plasterboard': {'eps_r': 2.73, 'sigma': 0.027},
}
COORD_DESCENT = {
    'itu_concrete': {'eps_r': 1.5, 'sigma': 0.05},
    'itu_ceiling_board': {'eps_r': 25.0, 'sigma': 0.001},
    'itu_plasterboard': {'eps_r': 4.0, 'sigma': 0.027},
}
OBJ_MAT_MAP = {'floor': 'itu_concrete', 'ceiling': 'itu_ceiling_board',
                'walls': 'itu_plasterboard'}


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
        p = pdp / (pdp.sum() + 1e-30); mt = np.sum(t * p)
        return np.sqrt(np.sum((t - mt) ** 2 * p))
    rms_err = abs(rms_delay(pdp_sim[:200], time_ns[:200]) - rms_delay(pdp_meas[:200], time_ns[:200]))
    return {'pwr_corr': float(pwr_corr) if not np.isnan(pwr_corr) else 0.0,
            'pdp_corr': float(pdp_corr) if not np.isnan(pdp_corr) else 0.0,
            'rms_delay_err_ns': float(rms_err)}


def transform_simple(pos_raw):
    return rot_mat @ (pos_raw - center)


def setup_scene(tx_pattern="dipole", max_depth=5, diffuse=False):
    """Create scene with TX/RX arrays and solver. Reusable."""
    scene = load_scene(SCENE_PATH)
    scene.frequency = FREQ
    scene.tx_array = PlanarArray(num_rows=4, num_cols=8, vertical_spacing=0.5,
        horizontal_spacing=0.5, pattern=tx_pattern, polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
        horizontal_spacing=0.5, pattern="dipole", polarization="V")
    for tx_ind in [1, 2]:
        p = poi_simple[f'array_{tx_ind}'].tolist()
        o = [float(-PI / 2 if tx_ind == 1 else PI / 2), 0.0, 0.0]
        scene.add(Transmitter(name=f'tx-{tx_ind}', position=p, orientation=o))
    scene.add(Receiver(name='rx-0', position=[0, 0, 0]))
    solver = PathSolver()
    return scene, solver


def set_materials(scene, material_params):
    """Set material parameters on existing scene."""
    for obj_name, mat_name in OBJ_MAT_MAP.items():
        if mat_name in material_params:
            mp = material_params[mat_name]
            scene.objects[obj_name].radio_material = RadioMaterial(
                f"cal_{mat_name}", relative_permittivity=max(1.0, float(mp['eps_r'])),
                conductivity=max(1e-6, float(mp['sigma'])))


def run_single(scene, solver, pos_raw_np, max_depth=5, samples=500000, diffuse=False):
    """Run RT for one position using existing scene."""
    pos_np = transform_simple(pos_raw_np)
    scene.receivers['rx-0'].position = pos_np.tolist()

    paths = solver(scene=scene, max_depth=max_depth, los=True,
                   specular_reflection=True, diffuse_reflection=diffuse,
                   refraction=False, synthetic_array=True,
                   samples_per_src=samples, max_num_paths_per_src=samples)
    if paths.tau.shape[-1] == 0:
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
    return h_sim


def eval_config(material_params, positions, tx_pattern="dipole",
                max_depth=5, diffuse=False):
    """Evaluate material params on positions."""
    scene, solver = setup_scene(tx_pattern=tx_pattern)
    set_materials(scene, material_params)

    metrics = []
    for csi_meas, pos_raw in positions:
        h_sim = run_single(scene, solver, pos_raw.numpy(), max_depth=max_depth,
                            diffuse=diffuse)
        if h_sim is not None:
            metrics.append(compute_metrics(h_sim, csi_meas.numpy()))

    del scene, solver
    dr.flush_malloc_cache()
    gc.collect()

    if not metrics:
        return {'pwr_corr_mean': -1, 'n_valid': 0}
    result = {f'{k}_mean': float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    result.update({f'{k}_std': float(np.std([m[k] for m in metrics])) for k in metrics[0]})
    result['n_valid'] = len(metrics)
    return result


def refine_params(base_params, positions):
    """Coordinate refinement with finer grid around base_params."""
    best_params = {k: dict(v) for k, v in base_params.items()}
    best_corr = eval_config(best_params, positions)['pwr_corr_mean']
    print(f"  Starting pwr_corr: {best_corr:.4f}", flush=True)

    for mat_name in ['itu_concrete', 'itu_ceiling_board', 'itu_plasterboard']:
        short = mat_name.replace('itu_', '')
        for param in ['eps_r', 'sigma']:
            cur = best_params[mat_name][param]
            if param == 'eps_r':
                candidates = sorted(set([max(1.0, cur * f) for f in
                    [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]]))
            else:
                candidates = sorted(set([max(1e-5, cur * f) for f in
                    [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]]))

            best_val = cur
            for val in candidates:
                trial = {k: dict(v) for k, v in best_params.items()}
                trial[mat_name][param] = val
                r = eval_config(trial, positions)
                if r['pwr_corr_mean'] > best_corr:
                    best_corr = r['pwr_corr_mean']
                    best_val = val
            best_params[mat_name][param] = best_val
            print(f"  {short}.{param}: {cur:.4f} -> {best_val:.4f} (corr={best_corr:.4f})", flush=True)
    return best_params, best_corr


def main():
    all_results = {}
    t0 = time.time()
    print("\n" + "=" * 70, flush=True)
    print("CALIBRATION + ABLATION (scene-reuse)", flush=True)
    print("=" * 70, flush=True)

    # Step 1: ITU baseline
    print("\n[2] ITU baseline (dipole)...", flush=True)
    r = eval_config(ITU_DEFAULTS, test_set)
    all_results['itu_baseline'] = r
    print(f"  pwr_corr={r['pwr_corr_mean']:.4f} pdp={r['pdp_corr_mean']:.4f} ({r['n_valid']}/30)", flush=True)

    # Step 2: Coord descent
    print("\n[3] Coord descent...", flush=True)
    r = eval_config(COORD_DESCENT, test_set)
    all_results['coord_descent'] = r
    print(f"  pwr_corr={r['pwr_corr_mean']:.4f} pdp={r['pdp_corr_mean']:.4f}", flush=True)

    # Step 3: Refinement on train set
    print("\n[4] Refinement from coord descent (on train set)...", flush=True)
    refined_params, train_corr = refine_params(COORD_DESCENT, train_set)

    # Evaluate on test
    print("\n  Evaluating on test set...", flush=True)
    r = eval_config(refined_params, test_set)
    all_results['refined'] = r
    all_results['refined']['params'] = {k: dict(v) for k, v in refined_params.items()}
    print(f"  Refined: pwr_corr={r['pwr_corr_mean']:.4f} pdp={r['pdp_corr_mean']:.4f}", flush=True)

    # Step 4: Ablation
    print("\n[5] Ablation study...", flush=True)

    # 4a: Reset each material
    for mat_name in OBJ_MAT_MAP.values():
        abl = {k: dict(v) for k, v in refined_params.items()}
        abl[mat_name] = dict(ITU_DEFAULTS[mat_name])
        r = eval_config(abl, test_set)
        short = mat_name.replace('itu_', '')
        all_results[f'ablation_reset_{short}'] = r
        delta = all_results['refined']['pwr_corr_mean'] - r['pwr_corr_mean']
        print(f"  reset {short}: pwr_corr={r['pwr_corr_mean']:.4f} (delta={delta:+.4f})", flush=True)

    # 4b: Depth ablation
    for depth in [1, 3]:
        r = eval_config(refined_params, test_set, max_depth=depth)
        all_results[f'ablation_depth_{depth}'] = r
        print(f"  depth={depth}: pwr_corr={r['pwr_corr_mean']:.4f} ({r['n_valid']}/30)", flush=True)

    # 4c: Antenna pattern with calibrated materials
    for pat in ['iso', 'tr38901']:
        r = eval_config(refined_params, test_set, tx_pattern=pat)
        all_results[f'calibrated_{pat}'] = r
        print(f"  cal+{pat}: pwr_corr={r['pwr_corr_mean']:.4f}", flush=True)

    # 4d: Diffuse
    r = eval_config(refined_params, test_set, diffuse=True)
    all_results['calibrated_diffuse'] = r
    print(f"  cal+diffuse: pwr_corr={r['pwr_corr_mean']:.4f}", flush=True)

    # Save
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(RESULTS_DIR / 'calibration_results.json', 'w') as f:
        json.dump(serialize(all_results), f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}", flush=True)
    print(f"DONE ({elapsed / 60:.1f} min)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    print(f"{'Config':<35} {'pwr_corr':>10} {'pdp_corr':>10} {'N':>5}")
    print("-" * 65)
    for name, r in all_results.items():
        if 'pwr_corr_mean' in r:
            print(f"{name:<35} {r['pwr_corr_mean']:>10.4f} {r.get('pdp_corr_mean', 0):>10.4f} {r.get('n_valid', 0):>5}")


if __name__ == "__main__":
    main()
