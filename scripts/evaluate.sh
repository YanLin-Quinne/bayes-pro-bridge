#!/usr/bin/env bash
# Evaluate a trained (or downloaded) particle system on the eight test sets, then write macro and pooled averages
# to <out>/eval_summary.json. The default predictive is the class-label mixture (each particle is normalised over
# the candidate labels, then the particles are averaged). SCORING=sequence selects the sequence-level predictive of
# the alpha-Renyi runs (complete answer likelihoods averaged over particles, then normalised over the candidates).
# usage: bash scripts/evaluate.sh <backbone> <adapter_dir> [m=8] [out=<adapter_dir>/../final] [gpus=0,1,2,3]
# <adapter_dir> holds branch_0 ... branch_{m-1}. Relative paths are relative to the repository root.
# Environment: DATASETS (default: all eight test sets), SCORING (class or sequence, default class),
# SUMMARY=0 (skip eval_summary.json), EVAL_EXTRA (extra evaluator flags, e.g. "--limit 64"), DRY_RUN=1 (print the
# commands instead of running them).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
BB=${1:?backbone}; M=${3:-8}; GPUS=${5:-0,1,2,3}
if [ -n "${DRY_RUN:-}" ]; then ENS=${2:?adapter_dir}; else ENS=$(cd "${2:?adapter_dir}" && pwd); fi
OUT=${4:-$ENS/../final}
read -r -a DSETS <<< "${DATASETS:-ARC-Challenge ARC-Easy boolq hellaswag openbookqa piqa social_i_qa winogrande}"
read -r -a XA <<< "${EVAL_EXTRA:-}"
source scripts/backbones.sh "$BB"
case "${SCORING:-class}" in
  class) SCORE=(--batch_size 8 --max_length 512 --scoring_mode label_token
    --posterior_predictive_mode normalize_then_mean --choice_score_normalization length) ;;
  sequence) SCORE=(--batch_size 4 --max_length 512 --scoring_mode response_string
    --posterior_predictive_mode mean_then_normalize --choice_score_normalization none) ;;
  *) echo "SCORING must be class or sequence" >&2; exit 2 ;;
esac
IFS=, read -r -a G <<< "$GPUS"
i=0; pids=(); failed=0
for ds in "${DSETS[@]}"; do
  gpu=${G[$((i % ${#G[@]}))]}
  EV=(env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 python "src/$EVALUATOR"
    --base_model "$BASE" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} --ensemble_dir "$ENS" --output_dir "$OUT/$ds" --dataset "$ds"
    --num_branches "$M" --target_modules q_proj,k_proj,v_proj,up_proj,down_proj
    "${SCORE[@]}" --gpu_id 0 --log_every 20 ${XA[@]+"${XA[@]}"})
  i=$((i + 1))
  if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${EV[@]}"; echo; continue; fi
  mkdir -p "$OUT/$ds"
  ( printf '%q ' "${EV[@]}"; echo; "${EV[@]}" ) > "$OUT/$ds/eval.log" 2>&1 &
  pids+=($!)
  if [ ${#pids[@]} -ge ${#G[@]} ] || [ $i -eq ${#DSETS[@]} ]; then
    for p in "${pids[@]}"; do wait "$p" || failed=1; done; pids=()
  fi
done
[ "$failed" = 0 ] || { echo "an evaluation failed, see $OUT/*/eval.log" >&2; exit 1; }
[ "${SUMMARY:-1}" = 0 ] && exit 0
if [ -n "${DRY_RUN:-}" ]; then echo "python scripts/summarize.py $OUT"; else python scripts/summarize.py "$OUT"; fi
