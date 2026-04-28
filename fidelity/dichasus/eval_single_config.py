#!/usr/bin/env python
"""
Evaluate a single material+scattering+antenna configuration.
Designed to be called as subprocess to avoid drjit memory leaks.

Usage:
  python eval_single_config.py <config.json> <output.json>

Config JSON fields:
  materials: {mat_name: {eps_r, sigma, scatt_coeff}}
  tx_pattern: str
  rx_pattern: str
  diffuse: bool
  scatt_pattern: str (lambertian, backscatter_5_5, etc.)
  max_depth: int
  samples: int
  position_indices: [int] — which positions to evaluate
"""
import os, sys, json, gc, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import drjit as dr
from pathlib import Path
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       PathSolver, RadioMaterial, BackscatteringPattern,
                       DirectivePattern, LambertianPattern)

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent
SCENE_PATH = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")
BANDWIDTH = 50e6; N_SUB = 1024; FREQ = 3.438e9

# Coordinate transform
data_dir = str(BASE_DIR / "diff-rt-calibration/data")
data_dict = {}
with open(os.path.join(data_dir, 'coordinates.csv'), mode='r') as f:
    for row in csv.DictReader(f): data_dict[row['Name']] = {k: v for k, v in row.items() if k != 'Name'}
poi_raw = {}
for name, pos in data_dict.items():
    if pos['South'] != 'noLoS':
        poi_raw[name] = np.array([float(pos['West']), float(pos['South']), float(pos['Height'])], dtype=np.float64)
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

with open(os.path.join(data_dir, 'spec.json'), 'r') as f: spec = json.load(f)
north_assign = np.array(spec['antennas'][0]['assignments']).flatten()
south_assign = np.array(spec['antennas'][1]['assignments']).flatten()

# Load measurements
tfrecord_path = str(BASE_DIR / "dcxx/dichasus-dc41.tfrecords")
with open(str(BASE_DIR / "dcxx/reftx-offsets-dichasus-dc41.json"), 'r') as f: offsets = json.load(f)
feature_desc = {'csi': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'pos-tachy': tf.io.FixedLenFeature([], tf.string, default_value='')}
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
all_samples = list(ds.map(parse_fn))

freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())
time_ns = np.arange(N_SUB) / BANDWIDTH * 1e9

OBJ_MAT_MAP = {'floor': 'itu_concrete', 'ceiling': 'itu_ceiling_board', 'walls': 'itu_plasterboard'}

SCATT_PATTERNS = {
    'lambertian': LambertianPattern(),
    'directive_5': DirectivePattern(alpha_r=5),
    'directive_10': DirectivePattern(alpha_r=10),
    'backscatter_3_3': BackscatteringPattern(alpha_r=3, alpha_i=3),
    'backscatter_5_5': BackscatteringPattern(alpha_r=5, alpha_i=5),
    'backscatter_10_3': BackscatteringPattern(alpha_r=10, alpha_i=3),
    'backscatter_3_10': BackscatteringPattern(alpha_r=3, alpha_i=10),
}

def compute_metrics(h_sim, h_meas):
    sim_pwr = np.array([np.mean(np.abs(h_sim[i])**2) for i in range(64)])
    meas_pwr = np.array([np.mean(np.abs(h_meas[i])**2) for i in range(64)])
    sp, mp = sim_pwr.sum(), meas_pwr.sum()
    pwr_corr = np.corrcoef(sim_pwr/(sp+1e-30), meas_pwr/(mp+1e-30))[0,1] if sp > 0 and mp > 0 else 0.0
    pdp_sim = np.mean(np.abs(np.fft.ifft(h_sim, axis=1))**2, axis=0)
    pdp_meas = np.mean(np.abs(np.fft.ifft(h_meas, axis=1))**2, axis=0)
    pdp_corr = np.corrcoef(pdp_sim[:200]/(pdp_sim[:200].max()+1e-30), pdp_meas[:200]/(pdp_meas[:200].max()+1e-30))[0,1]
    def rms_delay(pdp, t):
        p = pdp/(pdp.sum()+1e-30); mt = np.sum(t*p); return np.sqrt(np.sum((t-mt)**2*p))
    rms_err = abs(rms_delay(pdp_sim[:200], time_ns[:200]) - rms_delay(pdp_meas[:200], time_ns[:200]))
    ant_corrs = []
    for i in range(64):
        ns, nm = np.linalg.norm(h_sim[i]), np.linalg.norm(h_meas[i])
        if ns > 0 and nm > 0: ant_corrs.append(np.abs(np.vdot(h_sim[i], h_meas[i]))/(ns*nm))
    return {
        'pwr_corr': float(pwr_corr) if not np.isnan(pwr_corr) else 0.0,
        'pdp_corr': float(pdp_corr) if not np.isnan(pdp_corr) else 0.0,
        'rms_delay_err_ns': float(rms_err),
        'mean_ant_corr': float(np.mean(ant_corrs)) if ant_corrs else 0.0,
    }

def main():
    cfg_path, out_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f: cfg = json.load(f)

    materials = cfg['materials']
    position_indices = cfg.get('position_indices', list(range(0, 100, 5)))
    tx_pattern = cfg.get('tx_pattern', 'dipole')
    rx_pattern = cfg.get('rx_pattern', 'dipole')
    diffuse = cfg.get('diffuse', True)
    max_depth = cfg.get('max_depth', 5)
    samples = cfg.get('samples', 50000)
    scatt_pat_name = cfg.get('scatt_pattern', 'lambertian')
    scatt_pattern = SCATT_PATTERNS.get(scatt_pat_name, LambertianPattern())

    metrics_list = []
    for idx in position_indices:
        if idx >= len(all_samples): continue
        csi_meas, pos_raw = all_samples[idx]
        pos_np = rot_mat @ (pos_raw.numpy() - center)

        scene = load_scene(SCENE_PATH)
        scene.frequency = FREQ
        scene.tx_array = PlanarArray(num_rows=4, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5,
                                      pattern=tx_pattern, polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                      pattern=rx_pattern, polarization="V")
        for obj_name, mat_name in OBJ_MAT_MAP.items():
            if mat_name in materials:
                mp = materials[mat_name]
                rm = RadioMaterial(f"cal_{mat_name}", relative_permittivity=float(mp['eps_r']),
                                   conductivity=float(mp['sigma']))
                sc = mp.get('scatt_coeff', 0.0)
                if sc > 0.01:
                    rm.scattering_coefficient = float(sc)
                    rm.xpd_coefficient = float(mp.get('xpd_coeff', 0.0))
                    rm.scattering_pattern = scatt_pattern
                scene.objects[obj_name].radio_material = rm

        for tx_ind in [1, 2]:
            p = poi[f'array_{tx_ind}'].tolist()
            o = [float(-PI/2 if tx_ind == 1 else PI/2), 0.0, 0.0]
            scene.add(Transmitter(name=f'tx-{tx_ind}', position=p, orientation=o))
        scene.add(Receiver(name='rx-0', position=pos_np.tolist()))

        solver = PathSolver()
        try:
            paths = solver(scene=scene, max_depth=max_depth, los=True, specular_reflection=True,
                          diffuse_reflection=diffuse, refraction=False, synthetic_array=True,
                          samples_per_src=samples, max_num_paths_per_src=samples)
        except:
            del scene, solver; dr.flush_malloc_cache(); gc.collect(); continue

        if paths.tau.shape[-1] == 0:
            del scene, solver, paths; dr.flush_malloc_cache(); gc.collect(); continue

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

        m = compute_metrics(h_sim, csi_meas.numpy())
        metrics_list.append(m)
        del scene, solver, paths, cfr, cfr_np
        dr.flush_malloc_cache(); gc.collect()

    if not metrics_list:
        result = {'pwr_corr_mean': -1, 'pdp_corr_mean': -1, 'n_valid': 0}
    else:
        result = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list]
            result[f'{key}_mean'] = float(np.mean(vals))
            result[f'{key}_std'] = float(np.std(vals))
        result['n_valid'] = len(metrics_list)

    with open(out_path, 'w') as f: json.dump(result, f, indent=2)
    print(f"pwr={result.get('pwr_corr_mean',0):.4f} pdp={result.get('pdp_corr_mean',0):.4f} "
          f"n={result.get('n_valid',0)}", flush=True)

if __name__ == "__main__":
    main()
