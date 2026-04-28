#!/usr/bin/env bash
# Convenience submitter: queue the Munich sweep + post-aggregation, chained
# with `afterok`. Mirrors 01_munich_full_sweep.sh but as SLURM jobs.
#
# Usage:
#   bash fidelity/experiments/sbatch/submit_munich.sh                 # all 4 tags
#   TAGS="geometry"          bash submit_munich.sh                    # subset
#   GPU_TYPE=a100:1 WALLTIME=04:00:00 bash submit_munich.sh           # bigger GPU
#
# All env vars from fidelity_sweep.sbatch are honored.

set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root

SCENE="${SCENE:-builtin:munich}"
OUTPUT="${OUTPUT:-fidelity/results}"
TAGS_DEFAULT="geometry material ray_tracing hardware"
TAGS_STR="${TAGS:-$TAGS_DEFAULT}"
read -r -a TAGS_ARR <<< "$TAGS_STR"
N=${#TAGS_ARR[@]}
ARRAY_RANGE="0-$((N-1))"

PARTITION="${PARTITION:-htc}"
GPU_TYPE="${GPU_TYPE:-v100:1}"
WALLTIME="${WALLTIME:-04:00:00}"   # htc default QOS caps at 240 min; bump on `general`
MEM="${MEM:-24G}"
CPUS="${CPUS:-4}"

echo ">> sweep: scene=$SCENE  tags=(${TAGS_ARR[*]})  partition=$PARTITION  gpu=$GPU_TYPE"
SWEEP=$(SCENE="$SCENE" OUTPUT="$OUTPUT" TAGS="$TAGS_STR" \
    sbatch --parsable \
        --partition="$PARTITION" \
        --gres="gpu:$GPU_TYPE" \
        --time="$WALLTIME" \
        --mem="$MEM" \
        --cpus-per-task="$CPUS" \
        --array="$ARRAY_RANGE" \
        fidelity/experiments/sbatch/fidelity_sweep.sbatch)
echo ">> sweep job id: $SWEEP (array $ARRAY_RANGE)"

echo ">> post: chained with afterok:$SWEEP"
POST=$(SCENE="$SCENE" OUTPUT="$OUTPUT" \
    sbatch --parsable \
        --partition="$PARTITION" \
        --gres="gpu:$GPU_TYPE" \
        --dependency="afterok:$SWEEP" \
        --kill-on-invalid-dep=yes \
        fidelity/experiments/sbatch/fidelity_post.sbatch)
echo ">> post job id : $POST"

echo
echo "queued. monitor:"
echo "  squeue -j $SWEEP,$POST"
echo "  tail -f fidelity/experiments/sbatch/logs/fidelity_sweep.${SWEEP}_*.{out,err}"
