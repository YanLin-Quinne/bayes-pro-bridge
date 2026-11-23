"""Canonicalize the eight historical test sets (CPU only).

Uses the same per-record transformation and roundtrip checks as
``canonicalize_final8`` (``canonicalize_historical_record``), without the
provenance manifests.  Writes ``<dataset>.jsonl`` and ``manifest.json``
with source and output sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Sequence
import uuid

from .canonicalize_final8 import (
    FINAL8_DATASET_COUNTS,
    FINAL8_DATASET_NAMES,
    _canonicalize_legacy_dataset,
    _write_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_final8(source_root: str | Path, output_root: str | Path) -> dict:
    source_root = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"output root already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        datasets = {}
        for name in FINAL8_DATASET_NAMES:
            source = source_root / name / "test.json"
            records, gold_counts, choice_counts = _canonicalize_legacy_dataset(
                source, dataset_name=name, expected_count=FINAL8_DATASET_COUNTS[name])
            _write_jsonl(temporary / f"{name}.jsonl", records)
            datasets[name] = {
                "source_path": str(source),
                "source_sha256": _sha256(source),
                "count": len(records),
                "output_sha256": _sha256(temporary / f"{name}.jsonl"),
                "gold_label_counts": gold_counts,
                "choice_count_counts": choice_counts,
            }
        manifest = {
            "schema": "joint_pro.final8_node.v1",
            "prompt_construction": "joint_pro.data.canonicalize_historical_record",
            "dataset_order": list(FINAL8_DATASET_NAMES),
            "total_count": sum(d["count"] for d in datasets.values()),
            "datasets": datasets,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", required=True, help="folder holding <dataset>/test.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    manifest = prepare_final8(args.source_root, args.output_root)
    print(json.dumps({k: manifest[k] for k in ("total_count", "dataset_order")}))


if __name__ == "__main__":
    main()
