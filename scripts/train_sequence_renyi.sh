#!/usr/bin/env bash
# Sequence-level alpha-Renyi particle training (alpha = 0.8, Llama 3.2-3B, m = 8) and its two evaluations.
# usage: bash scripts/train_sequence_renyi.sh <auto|zero> [gpus=0,1,2,3]
# auto: Langevin noise from the trainer's automatic temperature. zero: no injected noise.
# Writes runs/llama3.2-3b_renyi0.8_<auto|zero>_m8/ with output/, train.log, resources.json, and the test-set results
# of the eight datasets under the sequence-level predictive (final_qseq/) and the class-label predictive (final_qcl/).
# Set WANDB_PROJECT to log to Weights & Biases. DRY_RUN=1 prints the commands instead of running them.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
T=${1:?auto or zero}; GPUS=${2:-0,1,2,3}
case "$T" in auto|zero) ;; *) echo "temperature mode must be auto or zero" >&2; exit 2 ;; esac
BB=llama3.2-3b; M=8
source scripts/backbones.sh "$BB"
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
NAME=${BB}_renyi0.8_${T}_m${M}
OUT=${OUT:-runs/$NAME}
SEQ8="boolq piqa social_i_qa hellaswag winogrande ARC-Easy ARC-Challenge openbookqa"
WB=(); [ -n "${WANDB_PROJECT:-}" ] && WB=(--wandb_project "$WANDB_PROJECT" --wandb_run_name "$NAME")
CMD=(env CUDA_VISIBLE_DEVICES="$GPUS" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/src"
  python -m torch.distributed.run --nproc_per_node="$NPROC" src/train_sequence_renyi.py --base_model "$BASE"
  --data_path data/commonsense_170k.json --output_dir "$OUT/output"
  --batch_size 16 --micro_batch_size 8 --num_epochs 3 --learning_rate 1e-4 --cutoff_len 256
  --val_set_size 120 --eval_step 500 --save_step 5000 --logging_steps 25
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]'
  --num_branches "$M" --branch_seeds "[$(seq 42 $((41 + M)) | paste -sd, -)]"
  --temperature_mode "$T" --sequence_alpha 0.8 --weight_decay 0.01
  --model_dtype bf16 --max_steps 0 --warmup_steps 50 --max_eval_batches 20
  ${WB[@]+"${WB[@]}"})
if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${CMD[@]}"; echo
else
  mkdir -p "$OUT"
  source scripts/resources.sh
  run_with_resources "$OUT" "$GPUS" "$OUT/train.log" "${CMD[@]}"
fi
DATASETS=$SEQ8 SCORING=sequence bash scripts/evaluate.sh "$BB" "$OUT/output" "$M" "$OUT/final_qseq" "$GPUS"
DATASETS=$SEQ8 SCORING=class bash scripts/evaluate.sh "$BB" "$OUT/output" "$M" "$OUT/final_qcl" "$GPUS"
