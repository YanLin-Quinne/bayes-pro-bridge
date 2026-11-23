#!/usr/bin/env bash
# Build the canonical class label data of the objective switch experiment from the files that
# scripts/prepare_data.sh downloads (data/commonsense_170k.json and dataset/<name>/test.json).
# Writes data/canonical/train_split (train and validation JSONL plus a small smoke subset) and
# data/canonical/final8 (the eight test sets), audits the training split and records sha256 sums.
# usage: bash scripts/prepare_canonical_data.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
OUT=data/canonical; CANON=$OUT/train_split; FINAL8=$OUT/final8
test -f data/commonsense_170k.json || { echo "run scripts/prepare_data.sh first" >&2; exit 5; }
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"
if [ ! -f "$CANON/manifest.json" ]; then
  python -m joint_pro.canonicalize_data --source data/commonsense_170k.json --output-root "$CANON"
fi
python -m joint_pro.audit_canonical_data --root "$CANON" --output-json "$OUT/train_split_audit.json.tmp"
mv "$OUT/train_split_audit.json.tmp" "$OUT/train_split_audit.json"
if [ ! -f "$FINAL8/manifest.json" ]; then
  python -m joint_pro.prepare_final8_node --source-root dataset --output-root "$FINAL8"
fi
sha256sum "$CANON"/*.jsonl "$CANON"/manifest.json "$FINAL8"/*.jsonl "$FINAL8"/manifest.json > "$OUT/canonical_sha256.txt"
echo "prepared $CANON and $FINAL8"
