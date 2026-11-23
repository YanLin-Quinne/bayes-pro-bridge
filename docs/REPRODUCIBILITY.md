# Reproducibility

Numbers of tables and figures refer to the arXiv version of the paper. The training code names the mixture risk
`pro` and the individual risk `ce`: `--objective_mode pro_ce_convex --beta_ce B` is the bridge risk
(1 - B) R_mix + B R_ind, where B = 0 is the pure mixture and B = 1 is independent training.

## Environment

All runs of Case Study 2 were trained and evaluated on NVIDIA H100 80GB SXM GPUs with Python 3.10, PyTorch 2.5.1
(CUDA 12.1) and Transformers 5.9.0. `requirements-lock.txt` is the complete package list of that environment.
LoRA is implemented with forward hooks in the training scripts, so no adapter library is needed. The runs of
Case Study 2 took about 1,080 GPU hours, 886 of them for training. The chains of Case Study 1 ran on CPUs and took
about 1,200 core hours.

Tables and figures need NumPy, SciPy and Matplotlib (`requirements-analysis.txt`). With these versions from the
Anaconda defaults channel (Python 3.12, FreeType 2.13.3) and the Arial font, the scripts in `analysis/` redraw the
figures of the paper pixel for pixel. The exception is Figure 9, which was drawn with the Matplotlib 3.11.1 wheel
from PyPI and DejaVu Sans. The Matplotlib wheels from PyPI bundle an older FreeType, so with them, or with other
versions or fonts, some text moves by a fraction of a point. The numbers and the tables do not change.

## Case Study 1

`src/case_study1_mfld.py` simulates the mean-field Langevin dynamics of the bridge with the Euler-Maruyama scheme,
on the logistic regression task of McLatchie et al. (2025, Section 5.2) with prior N(0, 100 I), 10,000 held-out
examples and a step size eps that satisfies eps n = 0.1. One process runs all chains of one setting and writes
their traces (`.json`) and the kept particles (`.npz`). `scripts/case_study1.sh` holds the 150 processes of the
paper in ten groups, and `scripts/case_study1_check.py` repeats the two checks of the implementation (finite
differences, and the Laplace approximation at beta = 1).

| Group in `runs/case_study1/` | Chains |
|---|---|
| `main` | n = 1,000, m = 8 and 32, 2e6 iterations (t = 200), six values of beta, three initialisations |
| `long` | 1e7 iterations (t = 1,000): m = 32 at beta = 0 and 0.01, and m = 8 at beta = 0 |
| `m128` | m = 128, 5e5 iterations (t = 50), six values of beta, three initialisations |
| `m128_long` | m = 128, 2e6 iterations, four chains: beta = 0 from the three initialisations, beta > 0 from the prior |
| `n200` | n = 200, 4e5 iterations (t = 200), six values of beta, three initialisations |
| `n5000` | n = 5,000, 1e6 iterations (t = 20), beta = 0, 0.01 and 0.1 |
| `half_step` | half the step size, 4e5 iterations (t = 20), beta = 0, 0.01 and 0.1 |
| `phase_n200`, `phase_n1000`, `phase_n5000` | the beta sweep of Figure 3, m = 32, chains started from the prior |

`analysis/case_study1_summary.py` reduces the chains to three small files, and `analysis/case_study1.py` draws
Table 7 and Figures 2 and 3 from them. `results/case_study1` holds these files for the chains of the paper.

| File | Content |
|---|---|
| `summary.json` | the numbers of Case Study 1 quoted in the paper, for every setting |
| `traces.npz` | curves of Figure 2: training-set mixture risk and minimum responsibility of the long chains, and whether each m = 8 chain is in the intended decomposition |
| `phase_transition.json` | the estimated threshold beta_crit for n = 200, 1,000, 5,000 and 200,000, and the beta sweep of Figure 3 |

The seeds are fixed. Repeating two groups of the beta sweep (n = 200, beta = 0 and beta = 0.5) on other hardware
with another NumPy version reproduced the kept particles to within 3e-12 and the held-out NLL of every chain to
four decimals. Differences from floating-point arithmetic grow slowly along a chain at beta = 0, so the longest
chains may agree less closely.

## Case Study 2: settings shared by every run

| Setting | Value |
|---|---|
| Particles | m = 8 (m = 4 and 16 in the particle-count runs), seeds 42, 43, ... |
| LoRA | rank 32, alpha 64, dropout 0.05, on q, k, v, up and down projections of every layer |
| Data | 170,300 training examples, 120 held out for validation, inputs truncated to 256 tokens |
| Optimisation | 3 epochs (31,932 steps), batch 16 (micro batch 8), learning rate 1e-4, 50 warmup steps then linear decay |
| Noisy AdamW | AdamW step with weight decay off, Gaussian noise of standard deviation sqrt(2 lr_t / (m n)) |
| Ridge penalty | lambda_0 = 0.01 on the LoRA parameters (the `--weight_decay 0.01` flag sets lambda_0) |
| Objective | `pro_ce_convex`: the bridge risk (1 - beta) R_mix + beta R_ind, beta in {0, 0.01, 0.1, 0.2, 0.3, 1} |
| Evaluation | class-label mixture predictive (each particle normalised over the candidates, then averaged), bf16 |

Pinned model revisions are in `scripts/backbones.sh`. Every launcher prints its commands with `DRY_RUN=1`. These
commands were checked argument by argument against the commands of the reported runs and differ only in paths,
W&B arguments, the master port, the run label `--experiment_family` and `--local_files_only`, which is dropped so
that the base models can be downloaded. The objective switch experiment uses the package in `src/joint_pro` with
the configurations in `configs/objective_switch`: the same particles, LoRA and penalty, a two-epoch schedule with
100 warmup steps, and the switch from beta = 0 to beta = 0.1 after the first epoch.
Macro averages weight the eight test sets equally, and pooled averages weight the 22,419 test examples equally.

## Released particles

The final particles of the 32 runs (the main matrix and the particle count runs) are on Hugging Face, as
`adapter_model.safetensors` and `config.json` per particle and the `run_config.json` of each run.
`scripts/evaluate_particles.sh` downloads them, writes the `lora_params.bin` files that the evaluators read with
`scripts/convert_adapters.py` (tensors bitwise identical to the safetensors files) and evaluates them into
`runs/<run>/final`. The released files are the ones written by training, and their tensors equal those that the
evaluations of the paper read. Two separate evaluations of the same particles on H100 GPUs gave byte-identical
predictions. Other GPUs or library versions can change the probabilities in the last digits.

| From | Tables | Figures |
|---|---|---|
| the final particles alone | 8, 9, 10, 11, 13 | 4, 6, 7, 8 |
| and the frozen backbones (`zero_shot.sh`, `pre_adaptation.sh`) | 3, 14 | 13 |
| the shipped summaries of Case Study 1 | 7 | 2, 3 |
| the training runs only (logs, particles saved during training, objective switch, training-set audit) | 1, 2, 4, 5, 6, 12 | 1, 5, 9, 10, 11, 12 |

## Run layout

`scripts/run_all.sh` runs everything below in order. Paths are relative to the repository root, `bb` is a backbone
and `ds` a test set.

```
runs/<bb>_beta<b>_m<m>/                    scripts/train.sh <bb> <b> [m]
    output/                                LoRA adapters of every particle, checkpoints every 5,000 steps
    train.log, resources.json              trainer log, wall time, GPU hours and peak GPU memory
    final/<ds>/                            scripts/evaluate.sh (eight test sets)
    checkpoints/checkpoint-<s>/<ds>/       scripts/evaluate_checkpoints.sh (BoolQ, HellaSwag, OpenBookQA)
    selected/                              scripts/select_checkpoint.sh (lowest validation NLL, eight test sets)
    train_set_audit/                       scripts/train_set_audit.sh (Qwen3-4B)
runs/llama3.2-3b_renyi0.8_<zero|auto>_m8/  scripts/train_sequence_renyi.sh <zero|auto>
    output/, train.log, resources.json, final_qseq/<ds>/, final_qcl/<ds>/
runs/objective_switch/                     scripts/objective_switch.sh
    leadin/, control/, rescue/, endpoints/
runs/pre_adaptation/<bb>/                  scripts/pre_adaptation.sh <bb> (training set, frozen backbone)
runs/<bb>_zero_shot/<ds>/                  scripts/zero_shot.sh <bb> (test sets, frozen backbone)
runs/case_study1/<group>/                  scripts/case_study1.sh [group ...]
```

## Tables and figures

`bash scripts/make_outputs.sh` runs every script below and writes the table bodies as printed in the paper to
`outputs/tables`, the figures under their file names in the paper to `outputs/figures`, and the numbers behind
them to `outputs/data`. Each script can also be run alone. The main matrix is the 24 runs with m = 8 of the four
backbones at the six values of beta, and the particle count runs are the m = 4 and m = 16 runs of Llama 3.2-3B and
Qwen3-4B at beta = 0 and 0.1.

| Script in `analysis/` | Paper output | Input |
|---|---|---|
| `case_study1.py` | Table 7, Figures 2 and 3 | `results/case_study1`, or the output of `case_study1_summary.py` on `runs/case_study1` |
| `make_tables.py` | Tables 1, 3 to 5, 8 to 10 and 12 to 14 | main matrix (`final`, `resources.json`), particle count (`final`), Llama 3.2-3B at beta = 0, 0.1, 0.3, 1 (`selected`, `train.log`), Qwen3-4B (`train_set_audit`), alpha-Renyi runs, pre-adaptation and zero shot |
| `particle_tables.py` | Tables 6 and 11 | main matrix (`final`), particle count at beta = 0.1 (`final`, `resources.json`) |
| `starvation.py` | Table 2, Figures 5a, 5b and 10 | Llama 3.2-3B at beta = 0 and 0.1 (`checkpoints`, `final`), Qwen3-4B at beta = 0 and 0.1 (`final`) |
| `starvation_qwen3.py` | Figure 12 | Qwen3-4B at beta = 0 and 0.1 (`checkpoints`, `final`) |
| `objective_switch_plots.py` | Figures 5c, 5d and 11 | objective switch |
| `fig_mechanism_puzzle.py` | Figure 1 | Llama 3.2-3B at beta = 0 and 0.1, Qwen3-4B at the six values of beta (`final`, `train_set_audit`) |
| `fig_beta_sweep.py` | Figure 4 | main matrix (`final`) |
| `fig_prop4_test.py` | Figure 6 | Llama 3.2-3B and Qwen3-4B at beta = 0 and 0.1 (`final`) |
| `fig_m_ablation.py` | Figure 7 | particle count and the m = 8 runs at beta = 0 and 0.1 (`final`) |
| `fig_reliability.py` | Figure 8 | Llama 3.2-3B and Qwen3-4B at beta = 0, 0.1 and 1 (`final`) |
| `fig_checkpoint_selection.py` | Figure 9 | Llama 3.2-3B at beta = 0, 0.1, 0.3, 1 (`checkpoints`, `selected`, `train.log`) |
| `fig_competence.py` | Figure 13 | pre-adaptation, the four backbones at beta = 0 and 1 (`final`) |

On the outputs of the reported runs, `make_outputs.sh` reproduces every row of the 14 tables as printed and the
13 figures of the paper.

## Source files

The training and evaluation scripts in `src/` are byte-identical copies of the files used for the runs of
Case Study 2 reported in the paper.

| Released file | Original name | sha256 (prefix) |
|---|---|---|
| `src/train_llama.py` | `finetune_joint_ensemble_PRO_branch_parallel_classlabel_convex_bf16_failfast.py` | `348c765ecec1035e` |
| `src/train_qwen2_instruct.py` | `finetune_joint_ensemble_PRO_branch_parallel_classlabel_convex_qwen2_instruct.py` | `4429ecc52d96b057` |
| `src/train_qwen3.py` | `finetune_joint_ensemble_PRO_branch_parallel_classlabel_convex_qwen3_20260719.py` | `867a4ae38e5c9c2a` |
| `src/NoisyAdamW_our.py` | `NoisyAdamW_our.py` | `46e0a20ef9e9416e` |
| `src/evaluate_llama.py` | `commonsense_evaluate_joint_pro_fullmetrics.py` | `0c10b43b125ade8a` |
| `src/evaluate_qwen2_instruct.py` | `commonsense_evaluate_joint_pro_fullmetrics_qwen2_instruct.py` | `689d28c85b18a153` |
| `src/evaluate_qwen3.py` | `commonsense_evaluate_joint_pro_fullmetrics_qwen3_bf16_recovery_20260811_anym.py` | `fef3ee9f350c63d7` |
| `src/train_sequence_renyi.py` | `finetune_joint_ensemble_PRO_branch_parallel_sequence_level.py` | `267a32190976885d` |

Notes on the other code that ran.

* The Llama 3.2-1B runs used a copy of the Llama trainer that differs from `train_llama.py` only in run metadata
  (an experiment label and the W&B grouping) and in lacking an immediate stop on a non-finite loss, so the
  computation is the same.
* The Qwen3 evaluator accepts any particle count. The m = 8 Qwen3 evaluations used a copy that required exactly
  eight particles and is otherwise identical.
* `src/frozen_backbone_eval.py` is the module that computed the frozen-backbone results. It differs only in its
  docstring, a comment, an output label, an error message and one helper that it imported from the package and
  now defines inline.
* `src/joint_pro`: 13 of 19 files are byte-identical with the package that ran the objective
  switch. The other 6 (`canonicalize_final8.py`, `endpoints_parallel.py`, `optimizer_diagnostics.py`, `prepare_final8_node.py`, `reviewer_evaluate.py`, `train.py`) lost the code that checked the provenance of
  internal copies of the data and checkpoints, and a few comments, docstrings and help texts that named internal
  resources were reworded. No line that computes anything changed.
* `src/train_set_audit` is the scoring code that ran, without the checks that tied its output to internal folders
  and with comments that named internal jobs reworded.
  The audits at beta = 0, 0.1, 0.3 and 1 ran an earlier copy that accepted only these four values of beta and is
  otherwise identical.
* `src/case_study1_mfld.py` is the simulator that ran Case Study 1, without a second task that the paper does not
  use and with a reworded docstring. No line that computes anything changed.

## Run notes

The Llama 3.2-1B runs start on a near-uniform plateau and leave it only in some runs (beta = 0.1, 0.2 and 1 did,
beta = 0, 0.01 and 0.3 did not). Repeating a 1B run can therefore change its result even with the same seeds,
since small floating-point differences between runs can decide whether the plateau is left. The other backbones
reproduced earlier runs of the same configuration to within 0.001 in macro NLL.
