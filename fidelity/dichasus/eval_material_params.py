"""
Evaluate a set of material parameters against DICHASUS measurements.
Run as subprocess to avoid DrJIT state accumulation.

Usage: python eval_material_params.py <params_json> <output_json>

params_json has: materials (dict of {mat_name: {rel_perm, cond}}),
                 position_indices (list of int),
                 scene_path, ...
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import csv
import gc
import time
import numpy as np
import tensorflow as tf
import drjit as dr
from pathlib import Path
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver, RadioMaterial

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent
BANDWIDTH = 50e6
N_SUB = 1024

# Load params
with open(sys.argv[1]) as f:
    params = json.load(f)
with open(sys.argv[2], 'w') as f:
    pass  # touch output

scene_path = params["scene_path"]
materials = params["materials"]
position_indices = params["position_indices"]

# Antenna assignments
data_dir = str(BASE_DIR / "diff-rt-calibration/data")
with open(os.path.join(data_dir, 'spec.json'), 'r') as f:
    spec = json.load(f)
north_assign = np.array(spec['antennas'][0]['assignments']).flatten()
south_assign = np.array(spec['antennas'][1]['assignments']).flatten()

# Load measurements
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
all_samples = list(ds.map(parse_fn).take(3282))

freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())

pwr_corrs = []
pdp_corrs = []
t0 = time.time()

for si, idx in enumerate(position_indices):
    csi_meas, pos_raw = all_samples[idx]
    pos_np = pos_raw.numpy()
    h_meas_np = csi_meas.numpy()

    try:
        scene = load_scene(scene_path)
        scene.frequency = 3.438e9

        # Apply material overrides
        for oname in scene.objects:
            rm = scene.objects[oname].radio_material
            if rm is not None and rm.name in materials:
                mp = materials[rm.name]
                new_rm = RadioMaterial(f"cal_{oname}",
                                       relative_permittivity=mp["relative_permittivity"],
                                       conductivity=mp["conductivity"])
                scene.objects[oname].radio_material = new_rm

        scene.tx_array = PlanarArray(num_rows=4, num_cols=8,
                                      vertical_spacing=0.5, horizontal_spacing=0.5,
                                      pattern="tr38901", polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                      vertical_spacing=0.5, horizontal_spacing=0.5,
                                      pattern="dipole", polarization="V")

        scene.add(Transmitter('tx-1', position=[7.48, -20.98, 1.39],
                               orientation=[float(-PI / 2), 0.0, 0.0]))
        scene.add(Transmitter('tx-2', position=[-6.39, 24.44, 1.42],
                               orientation=[float(PI / 2), 0.0, 0.0]))
        scene.add(Receiver(name='rx-0', position=pos_np.tolist()))

        solver = PathSolver()
        paths = solver(scene=scene, max_depth=5, los=True, specular_reflection=True,
                       diffuse_reflection=False, refraction=False, synthetic_array=True,
                       samples_per_src=500000, max_num_paths_per_src=500000)

        n_paths = paths.tau.shape[-1]
        if n_paths == 0:
            del scene, solver, paths
            dr.flush_malloc_cache(); gc.collect()
            continue

        cfr = paths.cfr(freq_dr, normalize_delays=False, out_type='tf')
        cfr_np = cfr.numpy().squeeze()

        h_sim = np.zeros((64, N_SUB), dtype=np.complex64)
        if cfr_np.ndim == 3:
            for ant in range(32):
                h_sim[north_assign[ant]] = cfr_np[0, ant]
                h_sim[south_assign[ant]] = cfr_np[1, ant]

        # pwr_corr
        sim_pwr = np.array([np.mean(np.abs(h_sim[i]) ** 2) for i in range(64)])
        meas_pwr = np.array([np.mean(np.abs(h_meas_np[i]) ** 2) for i in range(64)])
        pwr_corr = float(np.corrcoef(sim_pwr / (sim_pwr.sum() + 1e-30), meas_pwr / meas_pwr.sum())[0, 1])
        pwr_corrs.append(pwr_corr)

        # pdp_corr
        pdp_sim = np.mean(np.abs(np.fft.ifft(h_sim, axis=1)) ** 2, axis=0)
        pdp_meas = np.mean(np.abs(np.fft.ifft(h_meas_np, axis=1)) ** 2, axis=0)
        pdp_corr = float(np.corrcoef(pdp_sim[:200] / (pdp_sim[:200].max() + 1e-30),
                                      pdp_meas[:200] / pdp_meas[:200].max())[0, 1])
        pdp_corrs.append(pdp_corr)

        del scene, solver, paths, cfr, cfr_np
        dr.flush_malloc_cache(); gc.collect()

    except Exception as e:
        print(f"  pos {si}: error ({e})", flush=True)
        dr.flush_malloc_cache(); gc.collect()

elapsed = time.time() - t0
result = {
    "n_valid": len(pwr_corrs),
    "n_total": len(position_indices),
    "pwr_corr_mean": float(np.mean(pwr_corrs)) if pwr_corrs else 0,
    "pwr_corr_std": float(np.std(pwr_corrs)) if pwr_corrs else 0,
    "pdp_corr_mean": float(np.mean(pdp_corrs)) if pdp_corrs else 0,
    "elapsed_s": elapsed,
}
with open(sys.argv[2], 'w') as f:
    json.dump(result, f, indent=2)
print(f"  {len(pwr_corrs)} valid, pwr_corr={result['pwr_corr_mean']:.4f}, {elapsed:.0f}s", flush=True)
