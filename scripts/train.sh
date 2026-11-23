#!/usr/bin/env bash
# Train one particle system on the bridge objective (1 - beta) R_mix + beta R_ind.
# usage: bash scripts/train.sh <backbone> <beta> [m=8] [gpus=0,1,2,3]
# The reported runs used four GPUs for m = 4 and m = 8 (two particles per GPU for m = 8) and eight GPUs for m = 16.
# Writes <out>/output (adapters and checkpoints), <out>/train.log and <out>/resources.json (wall time, GPU hours and
# the per-GPU peak memory sampled every 15 s, see scripts/resources.sh).
# Set WANDB_PROJECT to log to Weights & Biases. MAX_STEPS > 0 stops early (the demo uses 20).
# DRY_RUN=1 prints the training command instead of running it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BB=${1:?backbone}; BETA=${2:?beta}; M=${3:-8}; GPUS=${4:-0,1,2,3}
source scripts/backbones.sh "$BB"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT=${OUT:-runs/${BB}_beta${BETA}_m${M}}
WB=(); [ -n "${WANDB_PROJECT:-}" ] && WB=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "${BB}_beta${BETA}_m${M}")
CMD=(env CUDA_VISIBLE_DEVICES="$GPUS" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/src"
  python -m torch.distributed.run --nproc_per_node="$NPROC" "src/$TRAINER" --base_model "$BASE" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
  --data_path data/commonsense_170k.json --output_dir "$OUT/output"
  --batch_size 16 --micro_batch_size 8 --num_epochs 3 --learning_rate 1e-4 --cutoff_len 256
  --val_set_size 120 --eval_step 500 --save_step 5000 --logging_steps 25
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]'
  --num_branches "$M" --branch_seeds "[$(seq 42 $((41 + M)) | paste -sd, -)]"
  --temperature_mode auto --weight_decay 0.01 --objective_mode pro_ce_convex --beta_ce "$BETA"
  --model_dtype bf16 --max_steps "${MAX_STEPS:-0}" --warmup_steps 50 --max_eval_batches 20 --diagnostic_batches 1
  ${WB[@]+"${WB[@]}"})
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$OUT"
source scripts/resources.sh
run_with_resources "$OUT" "$GPUS" "$OUT/train.log" "${CMD[@]}"
