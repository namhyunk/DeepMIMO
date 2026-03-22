# Fidelity Experiment Framework

Generate digital twin channel datasets at varying fidelity levels to quantify
how each component (geometry, material, ray-tracing, hardware) impacts channel
accuracy and downstream DL performance.

## Quick Start

```bash
# List all available fidelity configs
python -m fidelity.run_fidelity --list

# Run baseline on Sionna built-in scene
python -m fidelity.run_fidelity --scene builtin:simple_street_canyon --configs baseline

# Run all geometry degradation configs
python -m fidelity.run_fidelity --scene builtin:simple_street_canyon --tag geometry

# Run specific configs
python -m fidelity.run_fidelity --configs baseline geo_noise_5m mat_all_concrete rt_depth_1

# Run on an OSM scene folder
python -m fidelity.run_fidelity --scene /path/to/osm_folder --all

# Compare degraded datasets against baseline
python -m fidelity.analyze_channels --baseline baseline --degraded geo_noise_5m mat_all_concrete
```

## Architecture

```
fidelity/
├── config.py              # FidelityConfig dataclass + 17 preset levels
├── scene_editors.py       # Geometry degradation (noise, removal)
├── material_editors.py    # Material simplification (uniform, no scattering)
├── run_fidelity.py        # Main runner CLI
├── analyze_channels.py    # Channel comparison metrics
└── results/               # Generated datasets and metadata
```

## Fidelity Components

| Component | Config Names | What Changes |
|---|---|---|
| **Baseline** | `baseline` | Full fidelity reference |
| **Geometry** | `geo_noise_1m/5m/10m`, `geo_height_noise_3m`, `geo_remove_small`, `geo_remove_30pct` | Position/height noise, building removal |
| **Material** | `mat_all_concrete`, `mat_no_scattering` | Uniform material, disable scattering |
| **Ray Tracing** | `rt_depth_1/3`, `rt_low_rays`, `rt_very_low_rays`, `rt_with_diffraction` | Reflection depth, ray count |
| **Hardware** | `hw_dipole`, `hw_tr38901`, `hw_4x4_array` | Antenna patterns, array size |

## Experimental Workflow

1. **Step 1 (Baseline)**: Run `baseline` config → reference CSI dataset
2. **Step 2 (Degradation)**: Run each degraded config → degraded CSI datasets
3. **Step 3 (Analysis)**: Run `analyze_channels.py` → channel metrics (LoS accuracy, path loss RMSE, NMSE, delay spread)
4. **Steps 4–7**: Correlation analysis, DL training, component ranking (future scripts)
