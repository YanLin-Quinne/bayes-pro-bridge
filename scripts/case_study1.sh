#!/usr/bin/env bash
# Case Study 1: every chain of the paper (CPU only, numpy). One process simulates all chains of one setting.
# usage: bash scripts/case_study1.sh [group ...]     JOBS sets the number of parallel processes (default 8).
# DRY_RUN=1 prints the commands instead of running them. Groups, written to runs/case_study1/<group>:
#   main         n = 1,000, m = 8 and 32, 2e6 iterations (t = 200), six values of beta, three initialisations
#   long         1e7 iterations (t = 1,000): m = 32 at beta = 0 and 0.01, and m = 8 at beta = 0
#   m128         m = 128, 5e5 iterations (t = 50), six values of beta, three initialisations
#   m128_long    m = 128, 2e6 iterations, four chains: beta = 0 from the three initialisations, beta > 0 from the prior
#   n200         n = 200, 4e5 iterations (t = 200), six values of beta, three initialisations
#   n5000        n = 5,000, 1e6 iterations (t = 20), beta = 0, 0.01 and 0.1
#   half_step    half the step size, 4e5 iterations (t = 20), beta = 0, 0.01 and 0.1
#   phase_n200, phase_n1000, phase_n5000    the beta sweep of Figure 3, m = 32, chains started from the prior
# The step size satisfies eps n = 0.1 except in half_step. Eight chains per setting unless stated.
# The 150 processes of the paper took about 1,200 core hours in total, 130 of them for the beta sweep.
# Then: python analysis/case_study1_summary.py   (rebuilds results/case_study1 from the chains)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=runs/case_study1
BETAS="0 0.01 0.1 0.2 0.3 1"
SWEEP="0 0.05 0.1 0.15 0.175 0.2 0.225 0.24 0.25 0.26 0.275 0.3 0.35 0.4 0.5 0.7 1"
INITS=("--init prior" "--init slot --margin 8" "--init slot --margin 16")
GROUPS_ALL="main long m128 m128_long n200 n5000 half_step phase_n200 phase_n1000 phase_n5000"

grid() {  # n m eps steps chains log test keep "betas"  ->  one line per beta and initialisation
  local b i
  for b in $9; do for i in "${INITS[@]}"; do
    echo "--n $1 --m $2 --beta $b $i --eps $3 --steps $4 --chains $5 --log-every $6 --test-every $7 --keep-every $8"
  done; done
}

sweep() {  # n eps steps "betas"  ->  one line per beta, chains started from the prior
  local b
  for b in $4; do
    echo "--n $1 --m 32 --beta $b --init prior --eps $2 --steps $3 --chains 8 --log-every 2000 --test-every 2000 --keep-every 2000"
  done
}

jobs_of() {
  case "$1" in
    main) grid 1000 8 1e-4 2000000 8 500 5000 10000 "$BETAS"; grid 1000 32 1e-4 2000000 8 500 5000 10000 "$BETAS" ;;
    long) grid 1000 32 1e-4 10000000 8 2000 20000 50000 "0 0.01"; grid 1000 8 1e-4 10000000 8 2000 20000 50000 "0" ;;
    m128) grid 1000 128 1e-4 500000 8 500 5000 5000 "$BETAS" ;;
    m128_long) grid 1000 128 1e-4 2000000 4 1000 10000 10000 "0"
      grid 1000 128 1e-4 2000000 4 1000 10000 10000 "0.01 0.1 0.2 0.3 1" | grep -e "--init prior" ;;
    n200) grid 200 32 5e-4 400000 8 500 5000 10000 "$BETAS" ;;
    n5000) grid 5000 32 2e-5 1000000 8 500 10000 10000 "0 0.01 0.1" ;;
    half_step) grid 1000 32 5e-5 400000 8 200 2000 4000 "0 0.01 0.1" ;;
    phase_n200) sweep 200 5e-4 80000 "$SWEEP" ;;
    phase_n1000) sweep 1000 1e-4 400000 "$SWEEP" ;;
    phase_n5000) sweep 5000 2e-5 500000 "0.15 0.2 0.225 0.25 0.275 0.3 0.35 0.5 1" ;;
    *) echo "unknown group $1 ($GROUPS_ALL)" >&2; exit 2 ;;
  esac
}

commands() {
  local g
  for g in "$@"; do jobs_of "$g" | sed "s|^|python src/case_study1_mfld.py |; s|\$| --out $OUT/$g|"; done
}

[ "$#" -gt 0 ] || set -- $GROUPS_ALL
if [ -n "${DRY_RUN:-}" ]; then commands "$@"; exit 0; fi
commands "$@" | xargs -P "${JOBS:-8}" -I{} sh -c "{} || echo FAILED {}"
