"""
Run a single fidelity config. Called by the main experiment runner.
Usage: python run_single_config.py <config_json_path> <output_json_path>
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
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent

BANDWIDTH = 50e6
N_SUB = 1024

# Load config
config_path = sys.argv[1]
output_path = sys.argv[2]
with open(config_path) as f:
    config = json.load(f)

print(f"Config: {config['name']} (tag={config['tag']})", flush=True)

# Coordinate systems
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


def transform_raw(pos_raw):
    return pos_raw


# Antenna assignments
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
    csi = tf.ensure_shape(tf.io.parse_tensor(record['csi'], out_type=tf.float32),
                           (64, 1024, 2))
    csi = tf.signal.fftshift(csi, axes=1)
    csi = tf.complex(csi[..., 0], csi[..., 1])
    n_sub = tf.shape(csi)[1]
    sto = tf.tensordot(
        tf.constant(offsets['sto'], tf.float32),
        2 * np.pi * tf.range(n_sub, dtype=tf.float32) / tf.cast(n_sub, tf.float32),
        axes=0)
    cpo = tf.tensordot(
        tf.constant(offsets['cpo'], tf.float32),
        tf.ones(n_sub, dtype=tf.float32), axes=0)
    csi = csi * tf.exp(tf.complex(0.0, sto + cpo))
    pos = tf.ensure_shape(tf.io.parse_tensor(record['pos-tachy'], out_type=tf.float64), (3,))
    return csi, pos


N_POSITIONS = config.get('n_positions', 30)
ds = tf.data.TFRecordDataset([tfrecord_path])
all_samples = list(ds.map(parse_fn).take(3282))
total_samples = len(all_samples)

step = max(1, total_samples // N_POSITIONS)
sample_indices = list(range(0, total_samples, step))[:N_POSITIONS]
print(f"  Samples: {total_samples}, using {len(sample_indices)} positions", flush=True)

# Frequencies
freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())
time_ns = np.arange(N_SUB) / BANDWIDTH * 1e9


def compute_metrics(h_sim, h_meas):
    h_sf = h_sim.flatten()
    h_mf = h_meas.flatten()
    alpha = np.vdot(h_sf, h_mf) / (np.vdot(h_sf, h_sf) + 1e-30)
    scaled_nmse = np.sum(np.abs(alpha * h_sim - h_meas) ** 2) / np.sum(np.abs(h_meas) ** 2)
    scaled_nmse_db = 10 * np.log10(max(scaled_nmse, 1e-30))

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
        'scaled_nmse_db': float(scaled_nmse_db),
        'pwr_corr': float(pwr_corr),
        'pdp_corr': float(pdp_corr),
        'rms_delay_err_ns': float(rms_err),
        'mean_ant_corr': float(mean_ant_corr),
        'rms_sim_ns': float(rms_sim),
        'rms_meas_ns': float(rms_meas),
    }


# Run all positions
config_results = []
t_start = time.time()

for si, idx in enumerate(sample_indices):
    csi_meas, pos_raw = all_samples[idx]
    pos_raw_np = pos_raw.numpy()

    if config["coord_system"] == "simple":
        pos_np = transform_simple(pos_raw_np)
    else:
        pos_np = transform_raw(pos_raw_np)

    try:
        scene = load_scene(config["scene_path"])
        scene.frequency = 3.438e9

        scene.tx_array = PlanarArray(
            num_rows=4, num_cols=8,
            vertical_spacing=config["tx_spacing"],
            horizontal_spacing=config["tx_spacing"],
            pattern=config["tx_pattern"],
            polarization=config["polarization"])

        scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern=config["rx_pattern"],
            polarization=config["polarization"])

        if config.get("material_override"):
            for obj_name, obj in scene.objects.items():
                try:
                    obj.radio_material = config["material_override"]
                except Exception:
                    pass

        if config["coord_system"] == "simple":
            for tx_ind in [1, 2]:
                p = poi_simple[f'array_{tx_ind}'].tolist()
                o = [float(-PI / 2 if tx_ind == 1 else PI / 2), 0.0, 0.0]
                scene.add(Transmitter(name=f'tx-{tx_ind}', position=p, orientation=o))
        else:
            scene.add(Transmitter('tx-1', position=[7.48, -20.98, 1.39],
                                   orientation=[float(-PI / 2), 0.0, 0.0]))
            scene.add(Transmitter('tx-2', position=[-6.39, 24.44, 1.42],
                                   orientation=[float(PI / 2), 0.0, 0.0]))

        scene.add(Receiver(name='rx-0', position=pos_np.tolist()))

        solver = PathSolver()
        paths = solver(scene=scene,
                       max_depth=config["max_depth"],
                       los=True,
                       specular_reflection=True,
                       diffuse_reflection=config["diffuse_reflection"],
                       refraction=False,
                       synthetic_array=True,
                       samples_per_src=config["samples_per_src"],
                       max_num_paths_per_src=config["samples_per_src"])

        n_paths = paths.tau.shape[-1]
        if n_paths == 0:
            del scene, solver, paths
            dr.flush_malloc_cache()
            gc.collect()
            continue

        cfr = paths.cfr(freq_dr, normalize_delays=False, out_type='tf')
        cfr_np = cfr.numpy().squeeze()

        h_sim = np.zeros((64, N_SUB), dtype=complex)
        if cfr_np.ndim == 3:
            for ant in range(32):
                h_sim[north_assign[ant]] = cfr_np[0, ant]
                h_sim[south_assign[ant]] = cfr_np[1, ant]
        elif cfr_np.ndim == 2:
            for ant in range(min(32, cfr_np.shape[0])):
                h_sim[north_assign[ant]] = cfr_np[ant]

        h_meas = csi_meas.numpy()
        metrics = compute_metrics(h_sim, h_meas)
        metrics['n_paths'] = int(n_paths)
        config_results.append(metrics)

        del scene, solver, paths, cfr, cfr_np
        dr.flush_malloc_cache()
        gc.collect()

    except Exception as e:
        print(f"  pos {si}: error ({e})", flush=True)
        dr.flush_malloc_cache()
        gc.collect()

    if (si + 1) % 10 == 0:
        elapsed = time.time() - t_start
        if config_results:
            avg_pwr = np.mean([r['pwr_corr'] for r in config_results])
            print(f"  {si+1}/{len(sample_indices)} done ({elapsed:.0f}s), "
                  f"avg pwr_corr={avg_pwr:.3f}", flush=True)
        else:
            print(f"  {si+1}/{len(sample_indices)} done ({elapsed:.0f}s)", flush=True)

# Aggregate and save
result = {"config": config, "tag": config["tag"]}
if config_results:
    agg = {}
    for key in config_results[0]:
        vals = [r[key] for r in config_results if isinstance(r[key], (int, float))]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
            agg[f"{key}_median"] = float(np.median(vals))

    result["n_valid"] = len(config_results)
    result["aggregate"] = agg
    result["positions"] = config_results

    elapsed = time.time() - t_start
    print(f"  Done: {len(config_results)}/{len(sample_indices)} valid, "
          f"{elapsed:.0f}s total", flush=True)
    print(f"  pwr_corr={agg.get('pwr_corr_mean', 0):.3f} +/- "
          f"{agg.get('pwr_corr_std', 0):.3f}", flush=True)
else:
    result["n_valid"] = 0
    result["aggregate"] = {}
    result["positions"] = []
    print("  No valid positions!", flush=True)

with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved: {output_path}", flush=True)
