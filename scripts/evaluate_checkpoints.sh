#!/usr/bin/env bash
# Evaluate the checkpoints saved every 5,000 steps of a run on BoolQ, HellaSwag and OpenBookQA (class-label predictive).
# usage: bash scripts/evaluate_checkpoints.sh <backbone> <run_dir> [m=8] [gpus=0,1,2,3]
# Reads <run_dir>/output/checkpoint-<s>/ for s = 5000, 10000 ... 30000 and writes
# <run_dir>/checkpoints/checkpoint-<s>/<ds>/. The run was trained by scripts/train.sh. DRY_RUN=1 prints the commands.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BB=${1:?backbone}; RUN=${2:?run_dir}; M=${3:-8}; GPUS=${4:-0,1,2,3}
for s in $(seq 5000 5000 30000); do
  DATASETS="boolq hellaswag openbookqa" SCORING=class SUMMARY=0 \
    bash scripts/evaluate.sh "$BB" "$RUN/output/checkpoint-$s" "$M" "$RUN/checkpoints/checkpoint-$s" "$GPUS"
done
