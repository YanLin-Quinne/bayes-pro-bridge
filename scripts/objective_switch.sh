#!/usr/bin/env bash
# Objective switch experiment (Figures 5c, 5d and 11): Llama 3.2-3B with eight LoRA particles.
# A lead-in trains pure mixture (beta 0) for one epoch of a two epoch schedule. From its epoch 1 checkpoint the
# control arm resumes with beta 0 and the rescue arm switches the objective to the bridge with beta 0.1.
# Diagnostics are logged every 100 steps on a fixed held-out set of 120 examples. Both arms are then scored at
# epoch 2 on that set and on the eight test sets. The fork audit checks that only beta differs between the arms.
# usage: bash scripts/objective_switch.sh [--smoke] [gpus=0,1,2,3]
# Four GPUs are required (two particles per GPU). Finished stages are skipped when the script is started
# again. --smoke runs a few optimiser steps on a 64 example subset into runs/objective_switch_smoke.
# Set WANDB_PROJECT to log the full run to Weights & Biases. DRY_RUN=1 prints every command instead of running it.
# Needs scripts/prepare_data.sh first. The Llama models are gated on Hugging Face and need an accepted licence.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
MODE=full; if [ "${1:-}" = "--smoke" ]; then MODE=smoke; shift; fi
GPUS=${1:-0,1,2,3}
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
[ "$NPROC" = 4 ] || { echo "the protocol runs on exactly four GPUs, got: $GPUS" >&2; exit 2; }
CANON=data/canonical/train_split; FINAL8=data/canonical/final8; CFG=configs/objective_switch
if [ "$MODE" = full ]; then
  OUT=runs/objective_switch
  TRAIN_DATA=$CANON/train.jsonl; DIAG_DATA=$CANON/validation.jsonl; DIAG_STEPS=100; DIAG_MAX=120
  CFG_LEADIN=$CFG/llama3_2_3b_leadin_full.toml
  CFG_CONTROL=$CFG/llama3_2_3b_control_full.toml
  CFG_RESCUE=$CFG/llama3_2_3b_rescue_full.toml
else
  OUT=runs/objective_switch_smoke
  TRAIN_DATA=$CANON/smoke_train.jsonl; DIAG_DATA=$CANON/smoke_validation.jsonl; DIAG_STEPS=1; DIAG_MAX=16
  CFG_LEADIN=$CFG/llama3_2_3b_leadin_smoke.toml
  CFG_CONTROL=$CFG/llama3_2_3b_leadin_smoke.toml
  CFG_RESCUE=$CFG/llama3_2_3b_fork_smoke.toml
fi
SOURCE_CKPT=$OUT/leadin/checkpoints/epoch_0001
BATCH=8
EXTRA=(--extra-diagnostics --dense-diagnostics-window 300 --dense-diagnostics-every 10)

export CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src"
if [ -n "${DRY_RUN:-}" ]; then
  echo "# environment: CUDA_VISIBLE_DEVICES=$GPUS TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/src"
fi

# Print the command in dry run mode, run it otherwise.
run() { if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
dist_run() { run python -m torch.distributed.run --standalone --nproc-per-node="$NPROC" "$@"; }

# Skip a finished stage and refuse a half finished one, since the trainer appends to diagnostics.jsonl.
stage_todo() {  # name, run folder, files that exist once the stage has finished
  local name=$1 dir=$2 file missing=0; shift 2
  if [ -n "${DRY_RUN:-}" ]; then return 0; fi
  for file in "$@"; do [ -e "$file" ] || missing=1; done
  if [ "$missing" = 0 ]; then echo "$name is finished, skipping"; return 1; fi
  if [ -e "$dir" ]; then echo "$dir exists but $name did not finish, remove it to start the stage again" >&2; exit 3; fi
  return 0
}

train_segment() {  # name, extra trainer arguments
  local name=$1; shift
  local WB=(--wandb-mode disabled)
  if [ "$MODE" = full ] && [ -n "${WANDB_PROJECT:-}" ]; then
    WB=(--wandb-mode online --wandb-project "$WANDB_PROJECT" --wandb-run-name "objective_switch_$name")
  fi
  dist_run -m joint_pro.train \
    --data "$TRAIN_DATA" --diagnostics-data "$DIAG_DATA" \
    --diagnostics-steps "$DIAG_STEPS" --diagnostics-max-examples "$DIAG_MAX" \
    --output-dir "$OUT/$name" "${WB[@]}" "${EXTRA[@]}" "$@"
}

endpoints() {
  run python -m joint_pro.audit_forks --control "$OUT/control/fork_manifest.json" \
    --rescue "$OUT/rescue/fork_manifest.json" --output-json "$OUT/endpoints/fork_audit.json"
  local LIMIT=(); [ "$MODE" = smoke ] && LIMIT=(--max-examples-per-split 16)
  local pids=() role config status=0 pid
  for role in control rescue; do
    config=$CFG_CONTROL; [ "$role" = rescue ] && config=$CFG_RESCUE
    # The two arms are scored at the same time on the same GPUs.
    local cmd=(python -m torch.distributed.run --standalone --nproc-per-node="$NPROC"
      -m joint_pro.endpoints_parallel --config "$config" --role "$role"
      --checkpoint-dir "$OUT/$role/checkpoints/epoch_0002"
      --validation "$CANON/validation.jsonl" --final8-root "$FINAL8"
      --output-dir "$OUT/endpoints/$role" --batch-size "$BATCH" ${LIMIT[@]+"${LIMIT[@]}"})
    if [ -n "${DRY_RUN:-}" ]; then
      printf '%q ' "${cmd[@]}"; echo "> $OUT/endpoints/$role.log 2>&1 &"
    else
      "${cmd[@]}" > "$OUT/endpoints/$role.log" 2>&1 &
      pids+=($!)
    fi
  done
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || status=1; done
  return "$status"
}

if [ -z "${DRY_RUN:-}" ] && { [ ! -f "$CANON/manifest.json" ] || [ ! -f "$FINAL8/manifest.json" ]; }; then
  bash scripts/prepare_canonical_data.sh
fi

if stage_todo lead-in "$OUT/leadin" "$SOURCE_CKPT"; then
  train_segment leadin --config "$CFG_LEADIN" --stop-after-epoch 1
fi
if stage_todo "control arm" "$OUT/control" "$OUT/control/checkpoints/epoch_0002"; then
  train_segment control --config "$CFG_CONTROL" --resume "$SOURCE_CKPT"
fi
if stage_todo "rescue arm" "$OUT/rescue" "$OUT/rescue/checkpoints/epoch_0002"; then
  train_segment rescue --config "$CFG_RESCUE" --objective-switch-resume "$SOURCE_CKPT" --source-beta 0
fi
if stage_todo endpoints "$OUT/endpoints" "$OUT/endpoints/control/result.json" "$OUT/endpoints/rescue/result.json"; then
  [ -n "${DRY_RUN:-}" ] || mkdir -p "$OUT/endpoints"
  endpoints
fi
