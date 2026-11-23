#!/usr/bin/env bash
# Download the LLM-Adapters commonsense data (training set and the eight test sets) and check every file.
# usage: bash scripts/prepare_data.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
URL=https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/816657208af4db747803f87ba40a4c71383fed7a
mkdir -p data
curl -fsSL -o data/commonsense_170k.json "$URL/ft-training_set/commonsense_170k.json"
for ds in ARC-Challenge ARC-Easy boolq hellaswag openbookqa piqa social_i_qa winogrande; do
  mkdir -p "dataset/$ds"; curl -fsSL -o "dataset/$ds/test.json" "$URL/dataset/$ds/test.json"
done
sha256sum -c data/SHA256SUMS
