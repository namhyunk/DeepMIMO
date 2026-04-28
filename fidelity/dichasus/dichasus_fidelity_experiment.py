"""
DICHASUS Fidelity Degradation Experiment
Replicate Canyon-style 4-axis degradation study with REAL measurements as ground truth.

Axes: Geometry (3), Material (6), RT params (9), Hardware (6) = 24 configs
Ground truth: DICHASUS dc41 measured CSI (3.438 GHz, 64 antennas, 1024 subcarriers)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import json
import csv
import gc
import time
import drjit as dr
from pathlib import Path
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

PI = np.float64(np.pi)
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "dichasus_fidelity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Scene paths
SCENE_SIMPLE = str(BASE_DIR / "diff-rt-calibration/scenes/inue_simple/inue_simple.xml")
SCENE_DETAILED_NB = str(BASE_DIR / "dcxx/inue_detailed/inue_no_blockers.xml")
SCENE_DETAILED_FULL = str(BASE_DIR / "dcxx/inue_detailed/inue_detailed.xml")

BANDWIDTH = 50e6
N_SUB = 1024
N_POSITIONS = 30  # positions per config (balance accuracy vs time)

# ============================================================
# Coordinate systems
# ============================================================
# Simple scene uses NVlabs transform (center+rotate)
# Detailed scenes use raw (West, South, Height)

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
    return pos_raw  # detailed scenes use Blender native coords


# Antenna assignments
with open(os.path.join(data_dir, 'spec.json'), 'r') as f:
    spec = json.load(f)
north_assign = np.array(spec['antennas'][0]['assignments']).flatten()
south_assign = np.array(spec['antennas'][1]['assignments']).flatten()

# ============================================================
# Configuration definitions
# ============================================================

def make_config(name, tag, scene_path, coord_system="raw",
                max_depth=5, samples=500000, diffuse=False,
                tx_pattern="tr38901", rx_pattern="dipole",
                tx_spacing=0.5, polarization="V",
                material_override=None):
    return {
        "name": name, "tag": tag, "scene_path": scene_path,
        "coord_system": coord_system,
        "max_depth": max_depth, "samples_per_src": samples,
        "diffuse_reflection": diffuse,
        "tx_pattern": tx_pattern, "rx_pattern": rx_pattern,
        "tx_spacing": tx_spacing, "polarization": polarization,
        "material_override": material_override,
    }

CONFIGS = [
    # --- Geometry (3 configs) ---
    make_config("geo_simple", "geometry", SCENE_SIMPLE, coord_system="simple"),
    make_config("geo_detailed", "geometry", SCENE_DETAILED_NB),
    make_config("geo_full", "geometry", SCENE_DETAILED_FULL),

    # --- Material (6 configs) - use detailed scene, vary materials ---
    make_config("mat_baseline", "material", SCENE_DETAILED_NB),
    make_config("mat_all_concrete", "material", SCENE_DETAILED_NB,
                material_override="itu_concrete"),
    make_config("mat_all_metal", "material", SCENE_DETAILED_NB,
                material_override="itu_metal"),
    make_config("mat_all_glass", "material", SCENE_DETAILED_NB,
                material_override="itu_glass"),
    make_config("mat_all_plasterboard", "material", SCENE_DETAILED_NB,
                material_override="itu_plasterboard"),
    make_config("mat_all_wood", "material", SCENE_DETAILED_NB,
                material_override="itu_wood"),

    # --- RT Params (9 configs) ---
    make_config("rt_depth_1", "rt", SCENE_DETAILED_NB, max_depth=1),
    make_config("rt_depth_2", "rt", SCENE_DETAILED_NB, max_depth=2),
    make_config("rt_depth_3", "rt", SCENE_DETAILED_NB, max_depth=3),
    make_config("rt_depth_5", "rt", SCENE_DETAILED_NB, max_depth=5),
    make_config("rt_depth_7", "rt", SCENE_DETAILED_NB, max_depth=7),
    make_config("rt_samples_5k", "rt", SCENE_DETAILED_NB, samples=5000),
    make_config("rt_samples_50k", "rt", SCENE_DETAILED_NB, samples=50000),
    make_config("rt_samples_500k", "rt", SCENE_DETAILED_NB, samples=500000),
    make_config("rt_diffuse_on", "rt", SCENE_DETAILED_NB, diffuse=True),

    # --- Hardware (6 configs) ---
    make_config("hw_baseline", "hardware", SCENE_DETAILED_NB),
    make_config("hw_tx_dipole", "hardware", SCENE_DETAILED_NB, tx_pattern="dipole"),
    make_config("hw_tx_iso", "hardware", SCENE_DETAILED_NB, tx_pattern="iso"),
    make_config("hw_spacing_04", "hardware", SCENE_DETAILED_NB, tx_spacing=0.4),
    make_config("hw_pol_H", "hardware", SCENE_DETAILED_NB, polarization="H"),
    make_config("hw_rx_iso", "hardware", SCENE_DETAILED_NB, rx_pattern="iso"),
]

# ============================================================
# Load measurements
# ============================================================
print("[1/4] Loading DICHASUS dc41 measurements...")

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


ds = tf.data.TFRecordDataset([tfrecord_path])
all_samples = list(ds.map(parse_fn).take(3282))
total_samples = len(all_samples)
print(f"  Total samples: {total_samples}")

# Subsample uniformly
step = max(1, total_samples // N_POSITIONS)
sample_indices = list(range(0, total_samples, step))[:N_POSITIONS]
print(f"  Using {len(sample_indices)} positions (step={step})")

# Precompute frequencies
freq_np = np.arange(N_SUB) * (BANDWIDTH / N_SUB) - BANDWIDTH / 2
freq_dr = dr.cuda.ad.Float(freq_np.tolist())
time_ns = np.arange(N_SUB) / BANDWIDTH * 1e9


# ============================================================
# Metrics computation
# ============================================================

def compute_metrics(h_sim, h_meas):
    """Compute all comparison metrics between simulated and measured channels."""
    # 1. Scaled NMSE
    h_sf = h_sim.flatten()
    h_mf = h_meas.flatten()
    alpha = np.vdot(h_sf, h_mf) / (np.vdot(h_sf, h_sf) + 1e-30)
    scaled_nmse = np.sum(np.abs(alpha * h_sim - h_meas) ** 2) / np.sum(np.abs(h_meas) ** 2)
    scaled_nmse_db = 10 * np.log10(max(scaled_nmse, 1e-30))

    # 2. Spatial power correlation
    sim_pwr = np.array([np.mean(np.abs(h_sim[i]) ** 2) for i in range(64)])
    meas_pwr = np.array([np.mean(np.abs(h_meas[i]) ** 2) for i in range(64)])
    pwr_corr = np.corrcoef(sim_pwr / (sim_pwr.sum() + 1e-30),
                            meas_pwr / meas_pwr.sum())[0, 1]

    # 3. PDP correlation
    pdp_sim = np.mean(np.abs(np.fft.ifft(h_sim, axis=1)) ** 2, axis=0)
    pdp_meas = np.mean(np.abs(np.fft.ifft(h_meas, axis=1)) ** 2, axis=0)
    pdp_sim_n = pdp_sim / (pdp_sim.max() + 1e-30)
    pdp_meas_n = pdp_meas / pdp_meas.max()
    pdp_corr = np.corrcoef(pdp_sim_n[:200], pdp_meas_n[:200])[0, 1]

    # 4. RMS delay spread error
    def rms_delay(pdp, t):
        p = pdp / (pdp.sum() + 1e-30)
        mt = np.sum(t * p)
        return np.sqrt(np.sum((t - mt) ** 2 * p))

    rms_sim = rms_delay(pdp_sim[:200], time_ns[:200])
    rms_meas = rms_delay(pdp_meas[:200], time_ns[:200])
    rms_err = abs(rms_sim - rms_meas)

    # 5. Per-antenna frequency correlation
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


# ============================================================
# Run simulation for a single (config, position) pair
# ============================================================

def run_single(config, csi_meas, pos_raw_np):
    """Run RT for one config at one position, return metrics."""
    # Transform position
    if config["coord_system"] == "simple":
        pos_np = transform_simple(pos_raw_np)
    else:
        pos_np = transform_raw(pos_raw_np)

    # Load scene
    scene = load_scene(config["scene_path"])
    scene.frequency = 3.438e9

    # TX array
    scene.tx_array = PlanarArray(
        num_rows=4, num_cols=8,
        vertical_spacing=config["tx_spacing"],
        horizontal_spacing=config["tx_spacing"],
        pattern=config["tx_pattern"],
        polarization=config["polarization"])

    # RX array
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern=config["rx_pattern"],
        polarization=config["polarization"])

    # Material override
    if config["material_override"]:
        for obj_name, obj in scene.objects.items():
            try:
                obj.radio_material = config["material_override"]
            except Exception:
                pass

    # Place TX arrays
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

    # Run PathSolver
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
        return None

    # Compute CFR
    cfr = paths.cfr(freq_dr, normalize_delays=False, out_type='tf')
    cfr_np = cfr.numpy().squeeze()

    # Map to DICHASUS antenna ordering [64, 1024]
    h_sim = np.zeros((64, N_SUB), dtype=complex)
    if cfr_np.ndim == 3:
        for ant in range(32):
            h_sim[north_assign[ant]] = cfr_np[0, ant]
            h_sim[south_assign[ant]] = cfr_np[1, ant]
    elif cfr_np.ndim == 2:
        for ant in range(min(32, cfr_np.shape[0])):
            h_sim[north_assign[ant]] = cfr_np[ant]

    h_meas = csi_meas.numpy()

    # Compute metrics
    metrics = compute_metrics(h_sim, h_meas)
    metrics['n_paths'] = int(n_paths)

    # Cleanup
    del scene, solver, paths, cfr, cfr_np
    dr.flush_malloc_cache()
    gc.collect()

    return metrics


# ============================================================
# Main experiment loop
# ============================================================
print(f"\n[2/4] Running {len(CONFIGS)} configs × {len(sample_indices)} positions...")

# Check for existing results (resume support)
results_file = RESULTS_DIR / "all_results.json"
if results_file.exists():
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    print(f"  Loaded existing results: {len(all_results)} configs")
else:
    all_results = {}

for ci, config in enumerate(CONFIGS):
    cname = config["name"]
    if cname in all_results and len(all_results[cname].get("positions", [])) >= len(sample_indices):
        print(f"  [{ci+1}/{len(CONFIGS)}] {cname}: already done, skipping")
        continue

    print(f"\n  [{ci+1}/{len(CONFIGS)}] {cname} ({config['tag']})")
    config_results = []
    t_start = time.time()

    for si, idx in enumerate(sample_indices):
        csi_meas, pos_raw = all_samples[idx]
        pos_raw_np = pos_raw.numpy()

        try:
            metrics = run_single(config, csi_meas, pos_raw_np)
        except Exception as e:
            print(f"    pos {si}: error ({e})")
            dr.flush_malloc_cache()
            gc.collect()
            metrics = None

        if metrics is not None:
            config_results.append(metrics)
            if (si + 1) % 10 == 0:
                elapsed = time.time() - t_start
                avg_pwr = np.mean([r['pwr_corr'] for r in config_results])
                print(f"    {si+1}/{len(sample_indices)} done ({elapsed:.0f}s), "
                      f"avg pwr_corr={avg_pwr:.3f}")
        else:
            if (si + 1) % 10 == 0:
                print(f"    {si+1}/{len(sample_indices)} done (some positions failed)")

    # Aggregate
    if config_results:
        agg = {}
        for key in config_results[0]:
            vals = [r[key] for r in config_results if isinstance(r[key], (int, float))]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
                agg[f"{key}_median"] = float(np.median(vals))

        all_results[cname] = {
            "config": {k: v for k, v in config.items()
                       if k not in ('material_override',)},
            "tag": config["tag"],
            "n_valid": len(config_results),
            "aggregate": agg,
            "positions": config_results,
        }

        elapsed = time.time() - t_start
        print(f"    Done: {len(config_results)}/{len(sample_indices)} valid, "
              f"{elapsed:.0f}s total")
        print(f"    pwr_corr={agg.get('pwr_corr_mean', 0):.3f} ± "
              f"{agg.get('pwr_corr_std', 0):.3f}, "
              f"pdp_corr={agg.get('pdp_corr_mean', 0):.4f}")

    # Save after each config (checkpoint)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

# ============================================================
# Analysis and plotting
# ============================================================
print("\n[3/4] Generating analysis and plots...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Prepare axis-specific data
axis_data = {"geometry": [], "material": [], "rt": [], "hardware": []}
for cname, data in all_results.items():
    tag = data["tag"]
    agg = data["aggregate"]
    axis_data[tag].append({
        "name": cname,
        "pwr_corr": agg.get("pwr_corr_mean", 0),
        "pdp_corr": agg.get("pdp_corr_mean", 0),
        "rms_err": agg.get("rms_delay_err_ns_mean", 0),
        "ant_corr": agg.get("mean_ant_corr_mean", 0),
        "n_paths": agg.get("n_paths_mean", 0),
    })

# Summary table
print("\n" + "=" * 80)
print(f"{'Config':<25s} {'pwr_corr':>10s} {'pdp_corr':>10s} {'rms_err':>10s} {'ant_corr':>10s} {'paths':>8s}")
print("=" * 80)
for tag in ["geometry", "material", "rt", "hardware"]:
    for d in axis_data[tag]:
        print(f"  {d['name']:<23s} {d['pwr_corr']:>10.3f} {d['pdp_corr']:>10.4f} "
              f"{d['rms_err']:>10.1f} {d['ant_corr']:>10.4f} {d['n_paths']:>8.0f}")
    print("-" * 80)

# Plot: 4 axes × bar charts for key metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metrics_to_plot = [
    ("pwr_corr", "Spatial Power Correlation", (-0.5, 1.0)),
    ("pdp_corr", "PDP Correlation", (-0.05, 0.15)),
    ("rms_err", "RMS Delay Error (ns)", (0, 100)),
    ("ant_corr", "Per-Antenna CFR Correlation", (0, 0.15)),
]

colors = {"geometry": "#2196F3", "material": "#4CAF50", "rt": "#FF9800", "hardware": "#F44336"}

for ax, (metric, title, ylim) in zip(axes, metrics_to_plot):
    all_names = []
    all_vals = []
    all_colors = []
    for tag in ["geometry", "material", "rt", "hardware"]:
        for d in axis_data[tag]:
            short_name = d['name'].replace('geo_', '').replace('mat_', '').replace('rt_', '').replace('hw_', '')
            all_names.append(short_name)
            all_vals.append(d[metric])
            all_colors.append(colors[tag])

    x = np.arange(len(all_names))
    bars = ax.bar(x, all_vals, color=all_colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(title)
    ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(title)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t.title()) for t, c in colors.items()]
fig.legend(handles=legend_elements, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.02), fontsize=10)

plt.suptitle("DICHASUS Real-World Fidelity Degradation Study\n"
             "Simulated Channel vs Measured CSI (dc41, 3.438 GHz)",
             fontsize=13, fontweight='bold', y=1.06)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "fidelity_degradation.png"), dpi=150, bbox_inches='tight')
print(f"  Plot saved: {RESULTS_DIR}/fidelity_degradation.png")

# Per-axis detail plots
for tag in ["geometry", "material", "rt", "hardware"]:
    data = axis_data[tag]
    if not data:
        continue

    fig, axes2 = plt.subplots(1, 4, figsize=(16, 3.5))
    names = [d['name'].replace(f'{tag[:3]}_', '').replace('all_', '') for d in data]

    for ax, (metric, title, _) in zip(axes2, metrics_to_plot):
        vals = [d[metric] for d in data]
        ax.bar(range(len(vals)), vals, color=colors[tag], alpha=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f"{tag.title()} Axis Degradation", fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / f"{tag}_detail.png"), dpi=150, bbox_inches='tight')

# ============================================================
# Save summary
# ============================================================
print("\n[4/4] Saving summary...")

summary = {
    "experiment": "DICHASUS Fidelity Degradation Study",
    "dataset": "dichasus-dc41",
    "frequency_ghz": 3.438,
    "n_positions": len(sample_indices),
    "n_configs": len(CONFIGS),
    "axes": {}
}

for tag in ["geometry", "material", "rt", "hardware"]:
    summary["axes"][tag] = axis_data[tag]

with open(RESULTS_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"Results: {RESULTS_DIR}")
print(f"Configs: {len(all_results)}/{len(CONFIGS)}")
