#!/usr/bin/env bash
# Training-set audit of a trained Qwen3-4B-Base particle system. Scores all 170,420 rows of the training file
# (170,300 optimisation rows and the 120 held-out validation rows) with the class-label predictive of the trainer
# (batch 8, cutoff 256, bf16 backbone) and stores the gold-label log probability of every particle.
# usage: bash scripts/train_set_audit.sh <beta> [gpu=0]       beta is one of 0 0.01 0.1 0.2 0.3 1
# Reads runs/qwen3-4b-base_beta<beta>_m8/output (trained by scripts/train.sh) and writes train_set_audit/ next to it:
# train_nll_artifact.npz, summary.json, structural_audit.json and audit.log. One beta takes about three hours on one
# GPU. The Qwen3-4B-Base weights must be in the local Hugging Face cache, which any training run fills.
# DRY_RUN=1 prints the commands instead of running them.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BETA=${1:?beta}; GPU=${2:-0}
case "$BETA" in 0|0.01|0.1|0.2|0.3|1) ;; *) echo "beta must be one of 0 0.01 0.1 0.2 0.3 1" >&2; exit 2 ;; esac
RUN=${RUN:-runs/qwen3-4b-base_beta${BETA}_m8}; OUT=$RUN/train_set_audit
AUDIT=$ROOT/src/train_set_audit
SCORE=(env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  WANDB_MODE=disabled WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH="$AUDIT:$ROOT/src"
  python "$AUDIT/qwen_train_nll_eval.py" --method "beta$BETA" --run-root "$RUN/output" --output-dir "$OUT"
  --trainer src/train_qwen3.py --data data/commonsense_170k.json
  --batch-size 8 --cutoff 256 --limit 0 --device cuda --expected-global-step 31932)
CHECK=(env PYTHONDONTWRITEBYTECODE=1 python "$AUDIT/audit_results.py" --verify-sources
  --summary "$OUT/summary.json" --artifact "$OUT/train_nll_artifact.npz" --single-output "$OUT/structural_audit.json")
if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${SCORE[@]}"; echo
  printf '%q ' "${CHECK[@]}"; echo
  exit 0
fi
[ -s "$RUN/output/branch_0/adapter_model.safetensors" ] \
  || { echo "no trained run in $RUN/output, run scripts/train.sh qwen3-4b-base $BETA first" >&2; exit 2; }
mkdir -p "$OUT"
"${SCORE[@]}" 2>&1 | tee "$OUT/audit.log"
"${CHECK[@]}" 2>&1 | tee -a "$OUT/audit.log"
