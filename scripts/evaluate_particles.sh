#!/usr/bin/env bash
# Evaluate the released particles of the paper on the eight test sets, into the run layout of this repository.
# usage: bash scripts/evaluate_particles.sh [run ...]      default: the 32 runs whose final particles are released
# A run is <backbone>_beta<beta>_m<m>, for example llama3.2-3b_beta0.1_m8. Its particles are downloaded from
# Hugging Face to particles/<run>, converted with scripts/convert_adapters.py and evaluated to runs/<run>/final
# on the GPUs in GPUS (default 0,1,2,3). The Llama base models are gated: accept their licence on Hugging Face
# and log in with `hf auth login` first. DRY_RUN=1 prints the commands.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
LLAMA_REPO=Quinnnnnne/Llama-3.2-particle-mixtures
QWEN_REPO=Quinnnnnne/particle-mixtures-qwen
GPUS=${GPUS:-0,1,2,3}
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  for bb in llama3.2-1b llama3.2-3b qwen2-1.5b-instruct qwen3-4b-base; do
    for b in 0 0.01 0.1 0.2 0.3 1; do RUNS+=("${bb}_beta${b}_m8"); done
  done
  for bb in llama3.2-3b qwen3-4b-base; do
    for m in 4 16; do for b in 0 0.1; do RUNS+=("${bb}_beta${b}_m${m}"); done; done
  done
fi
for run in "${RUNS[@]}"; do
  bb=${run%%_beta*}; m=${run##*_m}
  case "$bb" in
    llama3.2-1b|llama3.2-3b) repo=$LLAMA_REPO ;;
    qwen2-1.5b-instruct|qwen3-4b-base) repo=$QWEN_REPO ;;
    *) echo "unknown run $run" >&2; exit 2 ;;
  esac
  if [ -n "${DRY_RUN:-}" ]; then
    echo "hf download $repo --include '$run/*' --local-dir particles"
    echo "python scripts/convert_adapters.py particles/$run"
    DRY_RUN=1 bash scripts/evaluate.sh "$bb" "particles/$run" "$m" "runs/$run/final" "$GPUS"
    continue
  fi
  hf download "$repo" --include "$run/*" --local-dir particles
  python scripts/convert_adapters.py "particles/$run"
  bash scripts/evaluate.sh "$bb" "particles/$run" "$m" "runs/$run/final" "$GPUS"
done
