#!/usr/bin/env bash
# DICHASUS dc41 fidelity sweep -- the sim-to-real anchor.
#
# Unlike the Sionna built-in scenes, DICHASUS uses a real measured channel
# (DICHASUS dc41) plus a hand-crafted scene exported from Blender. The
# meaningful question here is not just "what fidelity is sufficient" but
# "does the canyon-derived ranking survive against measured ground truth?"
#
# Notable canyon-vs-DICHASUS difference already observed: TR38.901 antenna
# pattern is the high-fidelity choice in canyon, but isotropic/dipole match
# DICHASUS measurements better. That asymmetry IS the result -- fidelity
# requirements are conditional on deployment realism, not absolute.
#
# Output: fidelity/dichasus/results/dichasus_fidelity/<config_name>/

set -euo pipefail

cd "$(dirname "$0")/../.."

# The DICHASUS sweep uses its own subprocess orchestrator (DrJIT crash
# isolation) and its own scene XMLs, so it does NOT go through
# fidelity.run_fidelity. It is checkpointed -- safe to re-run.
python -m fidelity.dichasus.run_experiment

# Real-world validation: how do simulated configs compare against the
# measured DICHASUS channel for downstream beam prediction?
python -m fidelity.dichasus.real_world_validation

# Beam-prediction sweep against measured ground truth
python -m fidelity.dichasus.run_beam_pred_experiment

# Generate the DICHASUS-specific report (PDP, beam-prediction, calibration)
python -m fidelity.dichasus.generate_report
