#!/usr/bin/env bash
# Aggregate canyon vs munich vs DICHASUS into one comparison table.
#
# For each scene, compute the "sufficient fidelity threshold" along each axis:
#
#   geometry: smallest position_noise_std at which channel NMSE > 0.1 dB
#             (and same for beam_top1_drop > 1%).
#   material: max delta-NMSE across uniform-material variants.
#   rt_depth: smallest max_reflections at which NMSE elbow flattens.
#   rt_rays:  smallest n_samples_per_src at which NMSE elbow flattens.
#   hardware: pattern-induced NMSE delta vs tr38901 baseline.
#
# Threshold detection is heuristic (knee detection on the sweep curve).
# All thresholds are produced for BOTH channel-NMSE and downstream-ML-task
# metrics so we can show ML is more tolerant than channel reconstruction
# (the canyon result we want to test on harder scenes).
#
# TODO: implement fidelity.aggregate (new module). Reads:
#         fidelity/results/<scene>/<config>/metadata.json
#         fidelity/results/<scene>/<config>/channel_nmse.json (from quantify_fidelity)
#         fidelity/results/<scene>/<config>/ml_task.json      (from ml_task_validation)
#       Produces:
#         fidelity/results/_aggregate/threshold_table.csv
#         fidelity/results/_aggregate/threshold_table.json
#
# Output: fidelity/results/_aggregate/threshold_table.{csv,json}

set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p fidelity/results/_aggregate

python -m fidelity.aggregate \
    --scenes simple_street_canyon munich dichasus \
    --metrics channel_nmse beam_top1 localization_rmse \
    --output fidelity/results/_aggregate/threshold_table
