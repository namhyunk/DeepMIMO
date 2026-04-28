#!/usr/bin/env bash
# The headline figure: scene complexity -> fidelity sensitivity slope.
#
# X-axis: a chosen scene complexity descriptor from scene_complexity.json
#         (default: building_count; also produce variants for los_probability,
#         building_volume_fraction, angular_spread_proxy).
# Y-axis: fidelity sensitivity slope, defined per axis as the NMSE rise per
#         unit perturbation. Examples:
#           - geometry slope = NMSE(1m) - NMSE(0m), in dB per meter
#           - rt_depth slope = max_reflections needed to reach within 0.5 dB
#                              of the depth=10 reference
#           - rt_rays slope  = n_samples_per_src needed for elbow
#           - hardware slope = max NMSE delta across pattern variants
#
# We expect the slope to *increase* with scene complexity for geometry,
# material, and rt_depth (more multipath -> more sensitive to perturbations).
# Hardware sensitivity is mostly antenna-pattern driven and should be
# relatively scene-invariant -- which is itself a useful negative result.
#
# This is the figure that argues hand-crafted scene complexity already
# explains most of the cross-scene variation in required fidelity (or
# motivates Geo-LWM if it doesn't).
#
# TODO: implement fidelity.plot_complexity_vs_sensitivity (new module).
#       Inputs:
#         fidelity/results/_aggregate/scene_complexity.json
#         fidelity/results/_aggregate/threshold_table.json
#
# Output: fidelity/results/_figures/complexity_vs_sensitivity_<descriptor>.{png,pdf}

set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p fidelity/results/_figures

for descriptor in building_count los_probability building_volume_fraction angular_spread_proxy; do
    python -m fidelity.plot_complexity_vs_sensitivity \
        --complexity  fidelity/results/_aggregate/scene_complexity.json \
        --thresholds  fidelity/results/_aggregate/threshold_table.json \
        --descriptor  "$descriptor" \
        --output      "fidelity/results/_figures/complexity_vs_sensitivity_${descriptor}"
done
