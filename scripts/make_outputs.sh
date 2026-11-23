#!/usr/bin/env bash
# Regenerate every table and figure of the paper (CPU only, about five minutes).
# usage: bash scripts/make_outputs.sh [runs=runs] [out=outputs]
# Writes <out>/tables/<label>.tex (table bodies as printed), <out>/figures/<name>.pdf (file names of the paper)
# and <out>/data/ (the numbers behind each table and figure). Needs requirements-analysis.txt.
# Case Study 2 reads the run outputs under <runs>. Case Study 1 reads the summaries of its chains: those of your
# own chains when <runs>/case_study1 exists, otherwise the ones shipped in results/case_study1.
# A script whose runs are missing (for example after scripts/evaluate_particles.sh alone) is reported at the end
# and the others still run. docs/REPRODUCIBILITY.md lists the runs behind each table and figure.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
RUNS=${1:-runs}; OUT=${2:-outputs}
A=(--runs "$RUNS" --out-dir "$OUT")
FAILED=()
run() { python "$@" || FAILED+=("$1"); }
CS1=results/case_study1
if [ -d "$RUNS/case_study1" ]; then
  CS1="$OUT/data/case_study1"
  run analysis/case_study1_summary.py --runs "$RUNS" --out-dir "$CS1"
fi
run analysis/case_study1.py --results "$CS1" --out-dir "$OUT"   # Table 7, Figures 2 and 3
run analysis/make_tables.py "${A[@]}" --cache "$OUT/cache"      # Tables 1, 3 to 5, 8 to 10 and 12 to 14
run analysis/particle_tables.py "${A[@]}"                       # Tables 6 and 11
run analysis/starvation.py "${A[@]}"                            # Table 2, Figures 5a, 5b and 10
run analysis/starvation_qwen3.py "${A[@]}"                      # Figure 12
run analysis/objective_switch_plots.py "${A[@]}"                # Figures 5c, 5d and 11
run analysis/fig_mechanism_puzzle.py "${A[@]}"                  # Figure 1
run analysis/fig_beta_sweep.py "${A[@]}"                        # Figure 4
run analysis/fig_prop4_test.py "${A[@]}"                        # Figure 6
run analysis/fig_m_ablation.py "${A[@]}"                        # Figure 7
run analysis/fig_reliability.py "${A[@]}"                       # Figure 8
run analysis/fig_checkpoint_selection.py "${A[@]}"              # Figure 9
run analysis/fig_competence.py "${A[@]}"                        # Figure 13
echo "tables in $OUT/tables, figures in $OUT/figures, numbers in $OUT/data"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "not written, see the messages above (usually runs that are missing): ${FAILED[*]}" >&2
  exit 1
fi
