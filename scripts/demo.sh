#!/usr/bin/env bash
# One-command check of the installation: 20 training steps of an 8-particle Qwen2-1.5B-Instruct system at beta = 0.1
# on four GPUs, then the mixture predictive on 32 BoolQ test examples. Needs the data from prepare_data.sh.
# usage: bash scripts/demo.sh [gpus=0,1,2,3]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
GPUS=${1:-0,1,2,3}
MAX_STEPS=20 OUT=runs/demo bash scripts/train.sh qwen2-1.5b-instruct 0.1 8 "$GPUS"
DATASETS=boolq EVAL_EXTRA="--limit 32" bash scripts/evaluate.sh qwen2-1.5b-instruct runs/demo/output 8 runs/demo/final "${GPUS%%,*}"
