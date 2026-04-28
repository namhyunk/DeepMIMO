"""
Capture raw sim and measured channels for ML beam prediction.
Same RT pipeline as run_single_config.py but saves channels + positions to .npz.

Usage: python run_beam_pred_config.py <config_json> <output_npz>
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


N_POSITIONS = config.get('n_positions', 300)
# Optional chunked execution: slice = [start, end) of the uniform-subsample index list
CHUNK_START = config.get('chunk_start', 0)
CHUNK_END = config.get('chunk_end', N_POSITIONS)
ds = tf.data.TFRecordDataset([tfrecord_path])
all_samples = list(ds.map(parse_fn).take(3282))
total_samples = len(all_samples)

step = max(1, total_samples // N_POSITIONS)
full_indices = list(range(0, total_samples, step))[:N_POSITIONS]
sample_indices = full_indices[CHUNK_START:CHUNK_END]
print(f"  Samples: {total_samples}, total target {N_POSITIONS}, "
      f"this chunk [{CHUNK_START}:{CHUNK_END}] = {len(sample_indices)} positions", flush=True)

# Frequencies
freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())

sim_channels_list = []
meas_channels_list = []
positions_list = []
pos_indices_list = []  # original index into DICHASUS tfrecord
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

        # Per-material calibration overrides
        if config.get("calibrated_materials"):
            from sionna.rt import RadioMaterial as _RM
            for obj_name, obj in scene.objects.items():
                rm = obj.radio_material
                if rm is not None and rm.name in config["calibrated_materials"]:
                    mp = config["calibrated_materials"][rm.name]
                    obj.radio_material = _RM(
                        f"cal_{obj_name}",
                        relative_permittivity=mp["relative_permittivity"],
                        conductivity=mp["conductivity"])

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

        h_sim = np.zeros((64, N_SUB), dtype=np.complex64)
        if cfr_np.ndim == 3:
            for ant in range(32):
                h_sim[north_assign[ant]] = cfr_np[0, ant]
                h_sim[south_assign[ant]] = cfr_np[1, ant]
        elif cfr_np.ndim == 2:
            for ant in range(min(32, cfr_np.shape[0])):
                h_sim[north_assign[ant]] = cfr_np[ant]

        h_meas = csi_meas.numpy().astype(np.complex64)

        sim_channels_list.append(h_sim)
        meas_channels_list.append(h_meas)
        positions_list.append(pos_raw_np.astype(np.float32))
        pos_indices_list.append(int(idx))

        del scene, solver, paths, cfr, cfr_np
        dr.flush_malloc_cache()
        gc.collect()

    except Exception as e:
        print(f"  pos {si}: error ({e})", flush=True)
        dr.flush_malloc_cache()
        gc.collect()

    if (si + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"  {si+1}/{len(sample_indices)} done ({elapsed:.0f}s), "
              f"{len(sim_channels_list)} valid", flush=True)

elapsed = time.time() - t_start
print(f"  Done: {len(sim_channels_list)}/{len(sample_indices)} valid, "
      f"{elapsed:.0f}s total", flush=True)

if sim_channels_list:
    np.savez_compressed(
        output_path,
        sim_channels=np.stack(sim_channels_list),   # (N, 64, 1024) complex64
        meas_channels=np.stack(meas_channels_list), # (N, 64, 1024) complex64
        positions=np.stack(positions_list),         # (N, 3) float32
        pos_indices=np.array(pos_indices_list, dtype=np.int32),
        north_assign=north_assign,
        south_assign=south_assign,
        config_name=config['name'],
        config_tag=config['tag'],
    )
    print(f"  Saved: {output_path}", flush=True)
else:
    print(f"  No valid positions, nothing saved", flush=True)
    sys.exit(1)
