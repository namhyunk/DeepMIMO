#!/usr/bin/env bash
# Appendix: per-UE feature correlation.
#
# This was the original "April folder" direction -- correlate per-UE
# ray-derived features (tx_rx_dist, n_paths, mean_bounces, shortest_path)
# against per-UE channel statistics. It is now demoted to a diagnostic
# appendix because:
#   (a) n_paths / mean_bounces / shortest_path are RT outputs, not pure
#       geometry -- they bake in propagation and weaken the "geometry
#       feature" framing.
#   (b) The main story is scene-level complexity vs cross-scene fidelity
#       sensitivity, which is more decision-relevant than per-UE r values.
#
# Kept here because the per-UE distribution plots are still useful as a
# sanity check that channels behave as physical intuition predicts within
# any single scene.
#
# Reads scenarios from fidelity/results/<scene>/baseline/ and produces
# correlation_scatter.png, component_scores_vs_nmse.png style figures.
#
# Output: fidelity/results/<scene>/_appendix/per_ue_correlation/

set -euo pipefail

cd "$(dirname "$0")/../.."

# Existing entry point (already implemented)
python -m fidelity.correlation_analysis --scene simple_street_canyon
python -m fidelity.correlation_analysis --scene munich
python -m fidelity.correlation_analysis --scene dichasus
