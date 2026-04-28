#!/usr/bin/env bash
# Extract scene-level geometry complexity descriptors.
#
# These are SCENE-level (one row per scene), NOT per-UE. The point is to make
# canyon, munich, and DICHASUS comparable on a single complexity axis so we
# can plot fidelity-sensitivity-vs-complexity later.
#
# Descriptors to compute (all from the .xml / mesh, no RT outputs):
#   - building_count                : number of objects classified as buildings
#   - mean_building_height           : mean height [m]
#   - height_std                     : std of building heights [m]
#   - total_footprint_area           : total horizontal footprint [m^2]
#   - total_surface_area             : total exterior wall + roof area [m^2]
#   - scene_volume_envelope          : axis-aligned bounding box volume [m^3]
#   - building_volume_fraction       : sum(building volume) / envelope volume
#   - los_probability                : sampled LoS probability from TX over UE grid
#   - mean_blockage_count            : average #buildings intersected by TX-UE line
#   - angular_spread_proxy           : std of TX-UE bearing angles to UE grid
#
# These should be deterministic given the scene XML + the UE grid that
# fidelity.run_fidelity uses. They do NOT depend on the fidelity sweep,
# only on the scene + UE geometry.
#
# TODO: implement fidelity.scene_complexity (new module).
#       Suggested entry point: fidelity/scene_complexity.py with
#       compute_scene_complexity(scene_xml, tx_pos, rx_pos) -> dict.
#
# Output: fidelity/results/_aggregate/scene_complexity.json

set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p fidelity/results/_aggregate

python -m fidelity.scene_complexity \
    --scenes \
        builtin:simple_street_canyon \
        builtin:munich \
        fidelity/dichasus/dcxx/inue_detailed/inue_detailed.xml \
    --output fidelity/results/_aggregate/scene_complexity.json
