#!/usr/bin/env bash
# Reproduce the canyon fidelity sweep.
#
# Purpose: sanity-check that the existing canyon thresholds (geometry breaks
# at 1m, beam prediction tolerates 2m, RT depth >= 3 sufficient, RT rays >= 5K
# sufficient, hardware pattern most critical) still reproduce on the current
# code. This is the controlled toy reference that anchors all cross-scene
# comparisons.
#
# Output: fidelity/results/simple_street_canyon/<config_name>/

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

SCENE="builtin:simple_street_canyon"
OUTPUT="fidelity/results"

# Geometry sweep (10 perturbation levels: 0.1 .. 10 m + height + removals)
python -m fidelity.run_fidelity --scene "$SCENE" --tag geometry --output "$OUTPUT"

# Material sweep (uniform-material variants + scattering off, all w/ ds_enable=True)
python -m fidelity.run_fidelity --scene "$SCENE" --tag material --output "$OUTPUT"

# Ray-tracing sweep (depth 0..10, rays 1K..500K, diffraction)
python -m fidelity.run_fidelity --scene "$SCENE" --tag ray_tracing --output "$OUTPUT"

# Hardware sweep (4x4 UPA: tr38901 baseline vs dipole, iso, spacing, polarization)
python -m fidelity.run_fidelity --scene "$SCENE" --tag hardware --output "$OUTPUT"
