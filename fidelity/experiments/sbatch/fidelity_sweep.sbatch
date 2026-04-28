#!/usr/bin/env bash
#SBATCH --job-name=fidelity_sweep
#SBATCH --output=fidelity/experiments/sbatch/logs/%x.%A_%a.out
#SBATCH --error=fidelity/experiments/sbatch/logs/%x.%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Generic per-tag fidelity sweep launcher.
#
# Submits one SLURM task per fidelity tag (geometry, material, ray_tracing,
# hardware) so a DrJIT/Sionna crash in one tag does not poison the others.
# Designed to be reusable across scenes (canyon, munich, custom OSM, ...).
#
# Override anything via env vars at submit time:
#   SCENE        ray-tracing scene id, e.g. "builtin:munich" or "/path/to/osm"
#                (default: builtin:munich)
#   OUTPUT       output root passed to run_fidelity --output
#                (default: fidelity/results)
#   PARTITION    SLURM partition (default: htc)
#   GPU_TYPE     SLURM GRES gpu spec (default: v100:1) — see `sinfo -O gres`
#   WALLTIME     per-task time limit (default: 04:00:00 — htc default QOS caps
#                at 240 min; switch to PARTITION=general for longer)
#   MEM          per-task RAM (default: 24G)
#   CPUS         cpus-per-task (default: 4)
#   TAGS         space-separated subset, default "geometry material ray_tracing hardware".
#                Adjust --array=0-N accordingly when you submit, e.g.:
#                  TAGS="geometry hardware"  sbatch --array=0-1 ...
#   EXTRA_FLAGS  extra args appended to run_fidelity, e.g. EXTRA_FLAGS="--num-rx 64"
#   REPO_ROOT    DeepMIMO repo root (auto-detected from script location)
#   VENV         path to .venv to activate (default: $REPO_ROOT/.venv)
#
# Submit examples:
#   # Munich, all four tags, defaults:
#   sbatch fidelity/experiments/sbatch/fidelity_sweep.sbatch
#
#   # Canyon, only geometry + ray_tracing on a 1080:
#   SCENE=builtin:simple_street_canyon \
#   TAGS="geometry ray_tracing" \
#   GPU_TYPE=gtx1080:1 \
#   sbatch --array=0-1 fidelity/experiments/sbatch/fidelity_sweep.sbatch
#
#   # Smoke run (4 RX) for any scene:
#   EXTRA_FLAGS="--num-rx 4" \
#   sbatch --array=0-0 fidelity/experiments/sbatch/fidelity_sweep.sbatch

set -euo pipefail

# ---- resolve repo root ------------------------------------------------------
# SLURM copies the submitted script to a spool dir and re-execs it from there,
# so BASH_SOURCE[0] points into /var/spool/slurm/... — useless for relative
# path math. Use SLURM_SUBMIT_DIR (the directory from which sbatch was called)
# as the canonical repo root. submit_munich.sh enforces cwd==repo-root before
# submitting, and direct sbatch invocations are expected to follow the same.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
if [[ ! -d "fidelity" || ! -f "pyproject.toml" ]]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT does not look like the DeepMIMO repo root (no fidelity/ + pyproject.toml). Set REPO_ROOT explicitly or sbatch from the repo root." >&2
    exit 4
fi

# ---- env defaults -----------------------------------------------------------
SCENE="${SCENE:-builtin:munich}"
OUTPUT="${OUTPUT:-fidelity/results}"
TAGS_DEFAULT="geometry material ray_tracing hardware"
TAGS_STR="${TAGS:-$TAGS_DEFAULT}"
read -r -a TAGS_ARR <<< "$TAGS_STR"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
VENV="${VENV:-$REPO_ROOT/.venv}"

# ---- pick the tag for this array task ---------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= ${#TAGS_ARR[@]} )); then
    echo "FATAL: SLURM_ARRAY_TASK_ID=$TASK_ID exceeds TAGS list (${#TAGS_ARR[@]} entries: ${TAGS_ARR[*]})." >&2
    exit 2
fi
TAG="${TAGS_ARR[$TASK_ID]}"

# ---- activate Python env ----------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "FATAL: venv not found at $VENV/bin/python (run \`uv sync --extra sionna\` first)." >&2
    exit 3
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Per-task OptiX disk cache. With $HOME on NFS, parallel array tasks otherwise
# collide on a single SQLite cache and abort with
#   OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR ("Disk cache database error").
# Give each task its own cache dir under node-local /tmp.
OPTIX_CACHE_BASE="${OPTIX_CACHE_BASE:-/tmp/$USER/optix_cache}"
export OPTIX_CACHE_PATH="$OPTIX_CACHE_BASE/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$OPTIX_CACHE_PATH"

# ---- diagnostic banner ------------------------------------------------------
echo "==================== fidelity_sweep ===================="
echo "host        : $(hostname)"
echo "date        : $(date -Iseconds)"
echo "job         : ${SLURM_JOB_ID:-?}  array=${SLURM_ARRAY_JOB_ID:-?}_${TASK_ID}"
echo "scene       : $SCENE"
echo "tag         : $TAG  (task $TASK_ID of ${#TAGS_ARR[@]})"
echo "output      : $OUTPUT"
echo "extra_flags : $EXTRA_FLAGS"
echo "repo        : $REPO_ROOT"
echo "venv python : $(command -v python)  ($(python --version))"
echo "optix cache : $OPTIX_CACHE_PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python - <<'PY'
import sionna.rt, mitsuba, drjit, deepmimo
print(f"sionna.rt={sionna.rt.__version__}  mitsuba={mitsuba.__version__}  "
      f"drjit={drjit.__version__}  deepmimo={deepmimo.__version__}")
PY
echo "========================================================"

# ---- run the sweep ----------------------------------------------------------
# shellcheck disable=SC2086
srun --unbuffered python -m fidelity.run_fidelity \
    --scene  "$SCENE" \
    --tag    "$TAG" \
    --output "$OUTPUT" \
    $EXTRA_FLAGS

echo "==================== done $TAG ========================="
