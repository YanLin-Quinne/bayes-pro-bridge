#!/usr/bin/env bash
# Select the checkpoint with the lowest validation NLL and evaluate it on the eight test sets.
# usage: bash scripts/select_checkpoint.sh <backbone> <run_dir> [m=8] [gpus=0,1,2,3]
# Candidates are the checkpoints 5000 ... 30000 of <run_dir>/output. The validation NLL (120 held-out examples) is
# read from <run_dir>/train.log. Writes <run_dir>/selected/selected_checkpoint.json and the test-set results of the
# selected checkpoint to <run_dir>/selected/checkpoint-<s>/<ds>/. DRY_RUN=1 prints the commands.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BB=${1:?backbone}; RUN=${2:?run_dir}; M=${3:-8}; GPUS=${4:-0,1,2,3}
SEL=$RUN/selected
if [ -n "${DRY_RUN:-}" ]; then
  echo "python scripts/select_checkpoint.py $RUN/train.log $SEL/selected_checkpoint.json"
  S=$(python scripts/select_checkpoint.py "$RUN/train.log" /dev/null 2>/dev/null) || S='<S>'
else
  mkdir -p "$SEL"
  S=$(python scripts/select_checkpoint.py "$RUN/train.log" "$SEL/selected_checkpoint.json")
fi
SCORING=class bash scripts/evaluate.sh "$BB" "$RUN/output/checkpoint-$S" "$M" "$SEL/checkpoint-$S" "$GPUS"
