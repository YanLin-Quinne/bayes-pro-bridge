#!/usr/bin/env bash
# Every run of the paper in the release layout, in an order that satisfies the dependencies.
# usage: bash scripts/run_all.sh          DRY_RUN=1 prints every command instead of running it.
# docs/REPRODUCIBILITY.md lists the runs behind each table and figure, and scripts/make_outputs.sh then
# regenerates all of them from runs/. One training run takes about 3 h (Llama 3.2-1B), 6 h (Llama 3.2-3B),
# 4.7 h (Qwen2-1.5B-Instruct) or 8.7 h (Qwen3-4B) on four H100 80GB GPUs. The paper runs took about 1,080 GPU
# hours, 886 of them for training. Commands use GPUs 0,1,2,3, except the m = 16 runs, which use eight GPUs.
# With eight GPUs, two four-GPU commands can run side by side with the GPU lists 0,1,2,3 and 4,5,6,7.
# Case Study 1 (step 9) needs no GPU and took about 1,200 core hours.
set -euo pipefail
cd "$(dirname "$0")/.."
BACKBONES="llama3.2-1b llama3.2-3b qwen2-1.5b-instruct qwen3-4b-base"
BETAS="0 0.01 0.1 0.2 0.3 1"

# 0. Data. The gated Llama models also need an accepted licence on Hugging Face (see the README).
if [ -z "${DRY_RUN:-}" ] && [ ! -f data/commonsense_170k.json ]; then bash scripts/prepare_data.sh; fi

# 1. Main matrix: four backbones, six values of beta, m = 8 particles, then the eight test sets.
for bb in $BACKBONES; do
  for b in $BETAS; do
    bash scripts/train.sh "$bb" "$b"
    bash scripts/evaluate.sh "$bb" "runs/${bb}_beta${b}_m8/output"
  done
done

# 2. Particle count: m = 4 on four GPUs and m = 16 on eight GPUs.
for bb in llama3.2-3b qwen3-4b-base; do
  for b in 0 0.1; do
    bash scripts/train.sh "$bb" "$b" 4
    bash scripts/evaluate.sh "$bb" "runs/${bb}_beta${b}_m4/output" 4
    bash scripts/train.sh "$bb" "$b" 16 0,1,2,3,4,5,6,7
    bash scripts/evaluate.sh "$bb" "runs/${bb}_beta${b}_m16/output" 16
  done
done

# 3. Checkpoint trajectories: the checkpoints saved every 5,000 steps on BoolQ, HellaSwag and OpenBookQA.
for b in 0 0.1 0.3 1; do bash scripts/evaluate_checkpoints.sh llama3.2-3b "runs/llama3.2-3b_beta${b}_m8"; done
for b in 0 0.1; do bash scripts/evaluate_checkpoints.sh qwen3-4b-base "runs/qwen3-4b-base_beta${b}_m8"; done

# 4. Validation-selected checkpoints of Llama 3.2-3B, then the eight test sets.
for b in 0 0.1 0.3 1; do bash scripts/select_checkpoint.sh llama3.2-3b "runs/llama3.2-3b_beta${b}_m8"; done

# 5. Training-set audit of the Qwen3-4B runs (one GPU each, about three hours).
for b in $BETAS; do bash scripts/train_set_audit.sh "$b"; done

# 6. Sequence-level alpha-Renyi runs (Llama 3.2-3B, alpha = 0.8) with both predictives on the eight test sets.
for t in zero auto; do bash scripts/train_sequence_renyi.sh "$t"; done

# 7. Objective switch (Llama 3.2-3B): lead-in, control and rescue arms, endpoints. It prepares its own data.
bash scripts/objective_switch.sh

# 8. Frozen backbones: correct-label probability on the training set and zero-shot results on the test sets.
for bb in $BACKBONES; do
  bash scripts/pre_adaptation.sh "$bb"
  bash scripts/zero_shot.sh "$bb"
done

# 9. Case Study 1: mean-field Langevin chains on the misspecified logistic regression (CPU, JOBS processes).
bash scripts/case_study1.sh
