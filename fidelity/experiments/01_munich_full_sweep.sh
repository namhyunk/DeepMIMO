#!/usr/bin/env bash
# Munich fidelity sweep -- the missing complex-geometry benchmark.
#
# Question (per scene): does the canyon threshold survive when geometry
# complexity goes up?
#   - Does the geometry perturbation threshold tighten (e.g. 1m -> 0.5m)?
#   - Does material impact grow with multipath richness?
#   - Does the sufficient RT depth shift higher (canyon: 3 -> munich: 5+)?
#   - Does the ray-count diminishing-return point hold?
#   - Does ML task remain more tolerant than channel NMSE?
#
# Cost: Munich on a built-in scene is ~45-50 s/batch; full sweep is multi-hour.
# Run overnight. Each tag is split so they can be resumed independently.
#
# Output: fidelity/results/munich/<config_name>/

set -euo pipefail

cd "$(dirname "$0")/../.."

SCENE="builtin:munich"
OUTPUT="fidelity/results"

python -m fidelity.run_fidelity --scene "$SCENE" --tag geometry    --output "$OUTPUT"
python -m fidelity.run_fidelity --scene "$SCENE" --tag material    --output "$OUTPUT"
python -m fidelity.run_fidelity --scene "$SCENE" --tag ray_tracing --output "$OUTPUT"
python -m fidelity.run_fidelity --scene "$SCENE" --tag hardware    --output "$OUTPUT"

# Once the sweep finishes, recompute fidelity metrics (channel NMSE, beam
# accuracy, localization) so the JSON summaries land alongside canyon's:
python -m fidelity.quantify_fidelity --scene munich --output "$OUTPUT"
python -m fidelity.ml_task_validation --scene munich --output "$OUTPUT"
