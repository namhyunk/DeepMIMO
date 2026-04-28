"""
DICHASUS INUE Scene - Sionna 2.0 RT vs Measured CSI comparison.
Loads the inue_simple scene, runs RT at measured positions, computes NMSE.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import json
import csv
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

PI = np.float64(np.pi)

print("=" * 60)
print("DICHASUS INUE - Sionna 2.0 RT vs Measured CSI")
print("=" * 60)

# ============================================================
# 1. Coordinate system (replicate NVlabs transformation)
# ============================================================
print("\n[1/5] Setting up coordinate system...")

data_dir = os.path.join(os.path.dirname(__file__), "diff-rt-calibration/data")
coord_file = os.path.join(data_dir, "coordinates.csv")

data_dict = {}
with open(coord_file, mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data_dict[row['Name']] = {k: v for k, v in row.items() if k != 'Name'}

# Load POIs as [West, South, Height] → [x, y, z]
poi_raw = {}
for name, pos in data_dict.items():
    if pos["South"] != "noLoS":
        poi_raw[name] = np.array([float(pos["West"]),
                                   float(pos["South"]),
                                   float(pos["Height"])], dtype=np.float64)

# Add antenna array positions (from spec.json)
poi_raw["array_1"] = np.array([7.480775, -20.9824, 1.39335], dtype=np.float64)
poi_raw["array_2"] = np.array([-6.390425, 24.440075, 1.4197], dtype=np.float64)

# Step 1: Center at AU
center = poi_raw["AU"].copy()
poi = {k: v - center for k, v in poi_raw.items()}

# Step 2: Rotate to align NWU with +Y direction
nwu = poi["NWU"]
phi_nwu = np.arctan2(nwu[1], nwu[0])
rot_angle = PI / 2 - phi_nwu
cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
rot_mat = np.array([[cos_a, -sin_a, 0],
                     [sin_a,  cos_a, 0],
                     [0,      0,     1]], dtype=np.float64)

for p in poi:
    poi[p] = rot_mat @ poi[p]

print(f"  Origin (AU): {center}")
print(f"  Array 1 (North): {poi['array_1']}")
print(f"  Array 2 (South): {poi['array_2']}")
print(f"  NWU (should be ~[0,Y,0]): {poi['NWU']}")


def transform_position(pos_raw):
    """Transform raw tachymeter position to scene coordinates."""
    return rot_mat @ (pos_raw - center)


# ============================================================
# 2. Load scene + setup TX/RX
# ============================================================
print("\n[2/5] Loading INUE scene...")

scene_path = os.path.join(os.path.dirname(__file__),
                          "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")
scene = load_scene(scene_path)
scene.frequency = 3.438e9

# TX = fixed antenna arrays (4×8 each), RX = mobile robot (1 antenna)
scene.tx_array = PlanarArray(num_rows=4, num_cols=8,
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="tr38901",
                              polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="dipole",
                              polarization="V")

# Place TX arrays with orientation
for tx_ind in [1, 2]:
    pos = poi[f"array_{tx_ind}"].tolist()
    if tx_ind == 1:
        orientation = [float(-PI / 2), 0.0, 0.0]
    else:
        orientation = [float(PI / 2), 0.0, 0.0]
    scene.add(Transmitter(name=f"tx-{tx_ind}",
                           position=pos,
                           orientation=orientation))

# Single RX (repositioned per measurement)
scene.add(Receiver(name="rx-0", position=[0.0, 5.0, 1.0]))

print(f"  Scene objects: {list(scene.objects.keys())}")
print(f"  Transmitters: {list(scene.transmitters.keys())}")

# ============================================================
# 3. Quick RT test
# ============================================================
print("\n[3/5] Quick RT test...")

solver = PathSolver()
paths = solver(scene=scene, max_depth=5, los=True, specular_reflection=True,
               diffuse_reflection=False, refraction=False,
               synthetic_array=True, samples_per_src=1000000,
               max_num_paths_per_src=1000000)

n_paths = paths.tau.shape[-1]  # (batch, num_tx, num_paths)
print(f"  Paths found: {n_paths}")
print(f"  paths.a: tuple of {len(paths.a)} elements, shape={paths.a[0].shape}")
print(f"  paths.tau: shape={paths.tau.shape}")

# Test CFR generation
bandwidth = 50e6
num_subcarriers = 1024
subcarrier_spacing = bandwidth / num_subcarriers
# Sionna expects DrJIT Float for frequencies
import drjit as dr
frequencies_np = np.arange(num_subcarriers) * subcarrier_spacing - bandwidth / 2
frequencies_dr = dr.cuda.ad.Float(frequencies_np.tolist())
print(f"  Computing CFR at {num_subcarriers} subcarriers...")

cfr = paths.cfr(frequencies_dr, normalize_delays=False, out_type='tf')
print(f"  CFR type: {type(cfr)}, shape: {cfr.shape}")

# ============================================================
# 4. Load measured CSI from dc41
# ============================================================
print("\n[4/5] Loading measured CSI from dc41...")

tfrecord_path = os.path.join(os.path.dirname(__file__),
                              "dcxx/dichasus-dc41.tfrecords")

feature_description = {
    "cfo": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "csi": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "gt-interp-age-tachy": tf.io.FixedLenFeature([], tf.float32, default_value=0),
    "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "snr": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
}

# Load calibration offsets
offset_file = os.path.join(os.path.dirname(__file__),
                            "dcxx/reftx-offsets-dichasus-dc41.json")
with open(offset_file, 'r') as f:
    offsets = json.load(f)

def parse_and_calibrate(proto):
    record = tf.io.parse_single_example(proto, feature_description)
    csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (64, 1024, 2))
    csi = tf.signal.fftshift(csi, axes=1)
    csi = tf.complex(csi[..., 0], csi[..., 1])

    # Apply STO and CPO calibration
    n_sub = tf.shape(csi)[1]
    sto_offset = tf.tensordot(
        tf.constant(offsets["sto"], dtype=tf.float32),
        2 * np.pi * tf.range(n_sub, dtype=tf.float32) / tf.cast(n_sub, tf.float32),
        axes=0)
    cpo_offset = tf.tensordot(
        tf.constant(offsets["cpo"], dtype=tf.float32),
        tf.ones(n_sub, dtype=tf.float32),
        axes=0)
    csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

    pos = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3,))
    return csi, pos

raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
dataset = raw_dataset.map(parse_and_calibrate).cache()

# Load a batch of samples
n_samples = 50
samples = list(dataset.take(n_samples))
print(f"  Loaded {n_samples} samples")

# ============================================================
# 5. RT simulation at measured positions → NMSE
# ============================================================
print("\n[5/5] Simulating RT at measured positions...")

# DICHASUS antenna assignment matrices (which physical antenna index → which CSI column)
# spec.json: North array assignments (4×8), South array assignments (4×8)
with open(os.path.join(os.path.dirname(__file__), "diff-rt-calibration/data/spec.json"), 'r') as f:
    spec = json.load(f)

north_assignments = np.array(spec["antennas"][0]["assignments"]).flatten()  # 32 elements
south_assignments = np.array(spec["antennas"][1]["assignments"]).flatten()  # 32 elements

nmse_list = []
for idx in range(min(10, len(samples))):
    csi_meas, pos_raw = samples[idx]

    # Transform position to scene coordinates
    pos_np = transform_position(pos_raw.numpy())
    pos_list = pos_np.tolist()

    # Set RX position
    scene.receivers["rx-0"].position = pos_list

    # Run RT
    paths = solver(scene=scene, max_depth=5, los=True, specular_reflection=True,
                   diffuse_reflection=False, refraction=False,
                   synthetic_array=True, samples_per_src=1000000,
                   max_num_paths_per_src=1000000)

    n_paths = paths.tau.shape[-1]
    if n_paths == 0:
        print(f"  Sample {idx}: pos={pos_np}, 0 paths found, skipping")
        continue

    # Compute CFR: shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_subcarriers]
    cfr = paths.cfr(frequencies_dr, normalize_delays=False, out_type='tf')
    # cfr shape: [batch?, rx, rx_ant, tx, tx_ant, subcarriers] or similar
    # Squeeze batch dimensions
    cfr_np = cfr.numpy().squeeze()
    print(f"  Sample {idx}: pos=[{pos_np[0]:.2f},{pos_np[1]:.2f},{pos_np[2]:.2f}], "
          f"paths={n_paths}, cfr shape={cfr_np.shape}")

    # Map simulated CFR to DICHASUS antenna ordering
    # cfr_np should be [rx_ant=1, num_tx=2, tx_ant=32, subcarriers=1024]
    # After squeeze: [num_tx=2, tx_ant=32, subcarriers=1024] or [tx_ant, subcarriers]
    # We need to map to [64, 1024] matching DICHASUS ordering

    if cfr_np.ndim == 3:
        # Shape: [num_tx, tx_ant, subcarriers]
        h_sim = np.zeros((64, num_subcarriers), dtype=complex)
        # TX-1 = North array: 32 antennas → map via north_assignments
        for ant_idx in range(32):
            dichasus_idx = north_assignments[ant_idx]
            h_sim[dichasus_idx] = cfr_np[0, ant_idx]
        # TX-2 = South array: 32 antennas → map via south_assignments
        for ant_idx in range(32):
            dichasus_idx = south_assignments[ant_idx]
            h_sim[dichasus_idx] = cfr_np[1, ant_idx]
    elif cfr_np.ndim == 2:
        print(f"    Unexpected CFR shape: {cfr_np.shape}")
        continue
    else:
        print(f"    Unexpected CFR ndim: {cfr_np.ndim}, shape: {cfr_np.shape}")
        continue

    # Measured CSI: [64, 1024]
    h_meas = csi_meas.numpy()

    # Power levels
    sim_power = np.mean(np.abs(h_sim) ** 2)
    meas_power = np.mean(np.abs(h_meas) ** 2)

    # Raw NMSE (without scaling — will be ~0 dB due to power mismatch)
    raw_nmse = np.sum(np.abs(h_sim - h_meas) ** 2) / np.sum(np.abs(h_meas) ** 2)

    # Scaled NMSE: find optimal complex scalar alpha = <h_meas, h_sim> / ||h_sim||^2
    # This removes global power/phase offset
    h_sim_flat = h_sim.flatten()
    h_meas_flat = h_meas.flatten()
    alpha = np.vdot(h_sim_flat, h_meas_flat) / np.vdot(h_sim_flat, h_sim_flat)
    scaled_nmse = np.sum(np.abs(alpha * h_sim - h_meas) ** 2) / np.sum(np.abs(h_meas) ** 2)
    scaled_nmse_db = 10 * np.log10(max(scaled_nmse, 1e-30))

    # Per-antenna correlation (average cosine similarity)
    corr_per_ant = []
    for ant in range(64):
        s = h_sim[ant]
        m = h_meas[ant]
        if np.linalg.norm(s) > 0 and np.linalg.norm(m) > 0:
            c = np.abs(np.vdot(s, m)) / (np.linalg.norm(s) * np.linalg.norm(m))
            corr_per_ant.append(c)
    avg_corr = np.mean(corr_per_ant) if corr_per_ant else 0.0

    print(f"    scaled_NMSE={scaled_nmse_db:.1f} dB, corr={avg_corr:.3f}, "
          f"power_ratio={10*np.log10(sim_power/meas_power):.1f} dB")
    nmse_list.append(scaled_nmse_db)

if nmse_list:
    print(f"\n  Average NMSE: {np.mean(nmse_list):.1f} dB (over {len(nmse_list)} samples)")
    print(f"  Min NMSE: {np.min(nmse_list):.1f} dB, Max: {np.max(nmse_list):.1f} dB")
else:
    print("\n  No valid samples for NMSE computation")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
