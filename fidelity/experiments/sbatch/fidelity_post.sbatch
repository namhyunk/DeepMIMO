#!/usr/bin/env bash
#SBATCH --job-name=fidelity_post
#SBATCH --output=fidelity/experiments/sbatch/logs/%x.%j.out
#SBATCH --error=fidelity/experiments/sbatch/logs/%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

# Post-sweep aggregation: re-runs quantify_fidelity + ml_task_validation
# against an already-completed sweep. Submit with `afterok` dependency on the
# sweep array job.
#
# Override via env vars (mirror fidelity_sweep.sbatch where applicable):
#   SCENE_NAME   short scene name used as subdir under $OUTPUT
#                (default: derived from $SCENE if set, else "munich")
#   OUTPUT       output root (default: fidelity/results)
#   REPO_ROOT    auto-detected
#   VENV         auto-detected
#   SKIP_ML      if non-empty, skip ml_task_validation
#   ML_DEVICE    torch device for ml_task_validation (default: cuda)
#
# Submit examples:
#   SCENE_NAME=munich sbatch \
#     --dependency=afterok:$SWEEP_JOBID \
#     fidelity/experiments/sbatch/fidelity_post.sbatch

set -euo pipefail

# See fidelity_sweep.sbatch for why we trust SLURM_SUBMIT_DIR over BASH_SOURCE.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
if [[ ! -d "fidelity" || ! -f "pyproject.toml" ]]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the DeepMIMO repo root." >&2
    exit 4
fi

if [[ -n "${SCENE:-}" ]]; then
    # Strip "builtin:" prefix or path components for the result subdir name.
    DEFAULT_SCENE_NAME="$(echo "$SCENE" | sed -E 's#^builtin:##; s#.*/##')"
else
    DEFAULT_SCENE_NAME="munich"
fi
SCENE_NAME="${SCENE_NAME:-$DEFAULT_SCENE_NAME}"
OUTPUT="${OUTPUT:-fidelity/results}"
VENV="${VENV:-$REPO_ROOT/.venv}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==================== fidelity_post ====================="
echo "host       : $(hostname)"
echo "scene_name : $SCENE_NAME"
echo "output     : $OUTPUT"
echo "venv       : $VENV"
echo "========================================================"

srun --unbuffered python -m fidelity.quantify_fidelity \
    --scenario "$SCENE_NAME" \
    --output   "$OUTPUT"

# Note: ml_task_validation does NOT accept --scene/--scenario; it iterates
# over SELECTED_CONFIGS and reads from --output (default fidelity/results).
# When the upstream sweep wrote to fidelity/results/<scene>/<config>, point
# --output at that scene subdir so dm.load() resolves the per-config dirs.
if [[ -z "${SKIP_ML:-}" ]]; then
    srun --unbuffered python -m fidelity.ml_task_validation \
        --output   "$OUTPUT/$SCENE_NAME" \
        --scenario "$SCENE_NAME" \
        --device   "${ML_DEVICE:-cuda}"
fi

echo "==================== done ============================="
