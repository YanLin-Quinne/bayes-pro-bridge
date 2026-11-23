#!/usr/bin/env bash
# Zero-shot accuracy and NLL of a frozen backbone (no adaptation) on the eight test sets, same prompt and scoring.
# usage: bash scripts/zero_shot.sh <backbone> [gpu=0] [out=runs/<backbone>_zero_shot]
# Writes <out>/<dataset>/summary.json for each test set. DRY_RUN=1 prints the commands instead of running them.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BB=${1:?backbone}; GPU=${2:-0}; OUT=${3:-runs/${BB}_zero_shot}
[ -f "configs/frozen_backbones/$BB.json" ] || { echo "unknown backbone $BB" >&2; exit 2; }
for ds in ARC-Challenge ARC-Easy boolq hellaswag openbookqa piqa social_i_qa winogrande; do
  CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python src/frozen_backbone_eval.py --test-set
    --backbone-config "configs/frozen_backbones/$BB.json" --data "dataset/$ds/test.json"
    --output-dir "$OUT/$ds" --batch-size 8 --device cuda:0 --dtype bf16 --progress-every 1000)
  if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
done
