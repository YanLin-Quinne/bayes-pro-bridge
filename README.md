# Improving the Optimisation Landscape for Particle Mixtures

Code for the paper *Improving the Optimisation Landscape for Particle Mixtures*
(Yan Lin, David T. Frazier, Paula Cordero Encinar and Chris J. Oates).

A predictive distribution is represented as an equally weighted mixture of m particles and trained through the
mixture risk R_mix, the log loss of the mixture. Gradient-based optimisation of R_mix can starve particles of
training signal. The bridge risk R_beta = (1 - beta) R_mix + beta R_ind adds the individual risk R_ind, the average
log loss of the particles, so that beta = 0 is the pure mixture and beta = 1 is independent training. A small
positive beta repairs the optimisation. The paper has two case studies.

* Case Study 1 computes a predictively oriented (PrO) posterior for a misspecified logistic regression by
  mean-field Langevin dynamics. It runs on CPUs with NumPy.
* Case Study 2 jointly fine-tunes m LoRA particles on a frozen language model with noisy AdamW, on four backbones
  and the eight multiple-choice commonsense benchmarks of LLM-Adapters. It needs GPUs.

## Install

```bash
pip install -r requirements-analysis.txt   # tables, figures and Case Study 1, CPU only
pip install -r requirements.txt            # Case Study 2 (exact environment: requirements-lock.txt, Python 3.10)
bash scripts/prepare_data.sh               # downloads and checks the LLM-Adapters data
bash scripts/demo.sh                       # 20 training steps and a 32-example evaluation on four GPUs
```

## Run

```bash
python analysis/case_study1.py                                                  # Table 7, Figures 2 and 3 from results/case_study1
bash scripts/case_study1.sh                                                     # every chain of Case Study 1
bash scripts/train.sh qwen2-1.5b-instruct 0.1                                   # m = 8 particles on GPUs 0 to 3
bash scripts/evaluate.sh qwen2-1.5b-instruct runs/qwen2-1.5b-instruct_beta0.1_m8/output
bash scripts/run_all.sh                                                         # every run of the paper
bash scripts/make_outputs.sh                                                    # every table and figure
```

`DRY_RUN=1 bash scripts/run_all.sh` lists the commands of all runs (about 1,080 H100 GPU hours for Case Study 2 and
1,200 core hours for Case Study 1). `docs/REPRODUCIBILITY.md` gives the runs and the script behind each table and
figure. The repository ships the summaries of the Case Study 1 chains of the paper in `results/case_study1`, so
its table and figures can be redrawn at once. The run outputs of Case Study 2 are not included, and its tables and
figures are computed from `runs/` after the runs have been repeated.

## Released particles

The final LoRA particles of the 32 runs of Case Study 2 are on Hugging Face, in
[Quinnnnnne/Llama-3.2-particle-mixtures](https://huggingface.co/Quinnnnnne/Llama-3.2-particle-mixtures) (Llama 3.2
Community License) and [Quinnnnnne/particle-mixtures-qwen](https://huggingface.co/Quinnnnnne/particle-mixtures-qwen)
(Apache-2.0). They reproduce the evaluations of the paper without any training:

```bash
bash scripts/evaluate_particles.sh            # downloads, converts and evaluates the 32 runs (GPU), or name some runs
bash scripts/zero_shot.sh qwen3-4b-base       # frozen backbones, likewise for the other three
bash scripts/pre_adaptation.sh qwen3-4b-base
bash scripts/make_outputs.sh                  # the tables and figures whose runs are present
```

With the environment of `requirements-lock.txt` on H100 GPUs the evaluation is deterministic, so the predictions
equal the ones behind the paper. From these runs `make_outputs.sh` writes Tables 3, 7 to 11, 13 and 14 and
Figures 2, 3, 4, 6, 7, 8 and 13. The other tables and figures need the training runs (training logs, the particles
saved during training and the objective switch experiment).

Backbones: `llama3.2-1b`, `llama3.2-3b`, `qwen2-1.5b-instruct`, `qwen3-4b-base`. The Llama models are gated on
Hugging Face and need an accepted licence. The training code names the mixture risk `pro` and the individual risk
`ce`, so `--objective_mode pro_ce_convex --beta_ce 0.1` is the bridge with beta = 0.1.

## Layout

| Path | Content |
|---|---|
| `src/` | training and evaluation scripts used for every reported run of Case Study 2 |
| `src/case_study1_mfld.py` | simulator of Case Study 1 |
| `scripts/evaluate_particles.sh` | evaluation of the particles released on Hugging Face |
| `src/joint_pro/` | package of the objective switch experiment |
| `src/train_set_audit/` | scoring of the training set for the Qwen3-4B runs |
| `scripts/` | data preparation and one launcher per experiment |
| `analysis/` | tables and figures of the paper |
| `results/case_study1/` | summaries of the Case Study 1 chains of the paper |
| `configs/` | frozen backbones and objective switch configurations |
| `docs/REPRODUCIBILITY.md` | environment, settings, run layout, outputs, file provenance and run notes |

## Licence

MIT for the code. The base models and datasets keep their own licences (Llama 3.2 Community License, Apache-2.0
for Qwen2 and Qwen3, Apache-2.0 for the LLM-Adapters data, and the licences of the original benchmarks).
