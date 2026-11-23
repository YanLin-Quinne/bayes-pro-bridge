#!/usr/bin/env bash
# Pre-adaptation competence of a frozen backbone (no adapters): the probability it assigns to the correct label of the
# 170,300 training examples, with the prompt and the label-token readout of the training scripts.
# usage: bash scripts/pre_adaptation.sh <backbone> [gpu=0] [out=runs/pre_adaptation/<backbone>]
# Writes summary.json (with the pooled and per-dataset statistics) and per_example.npz to <out>.
# DRY_RUN=1 prints the command instead of running it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BB=${1:?backbone}; GPU=${2:-0}; OUT=${3:-runs/pre_adaptation/$BB}
[ -f "configs/frozen_backbones/$BB.json" ] || { echo "unknown backbone $BB" >&2; exit 2; }
CMD=(env CUDA_VISIBLE_DEVICES="$GPU" python src/frozen_backbone_eval.py
  --backbone-config "configs/frozen_backbones/$BB.json" --data data/commonsense_170k.json --output-dir "$OUT"
  --batch-size 8 --device cuda:0 --dtype bf16)
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
