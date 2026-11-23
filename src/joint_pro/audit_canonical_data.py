"""Fail closed audit for canonical historical data packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .canonicalize_data import (
    CANONICAL_DATA_SCHEMA,
    CANONICAL_DATA_SCHEMA_VERSION,
    CANONICAL_FILES,
)
from .config import HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE
from .data import validate_historical_canonical_record


REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "schema",
    "prompt_style",
    "source_path",
    "source_count",
    "train_count",
    "validation_count",
    "split_seed",
    "shuffle_seed",
    "validation_size",
    "smoke_train_count",
    "smoke_validation_count",
    "files",
}


def _audit_jsonl(path: Path, prefix_size: int) -> tuple[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical data file: {path.name}")
    count = 0
    prefix = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank canonical record at {path.name}:{line_number}")
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                validate_historical_canonical_record(record)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid canonical record at {path.name}:{line_number}: {error}"
                ) from error
            if count < prefix_size:
                prefix.append(record)
            count += 1
    return count, prefix


def audit_canonical_data(
    root: str | Path,
    *,
    expected_source_count: int = 170420,
    expected_validation_count: int = 120,
    expected_smoke_train_count: int = 64,
    expected_smoke_validation_count: int = 16,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing canonical data manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("canonical data manifest must be a JSON object")
    missing = REQUIRED_MANIFEST_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"canonical data manifest is missing fields: {sorted(missing)}")
    expected = {
        "schema_version": CANONICAL_DATA_SCHEMA_VERSION,
        "schema": CANONICAL_DATA_SCHEMA,
        "prompt_style": HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE,
        "source_count": expected_source_count,
        "train_count": expected_source_count - expected_validation_count,
        "validation_count": expected_validation_count,
        "split_seed": 42,
        "shuffle_seed": 42,
        "validation_size": expected_validation_count,
        "smoke_train_count": expected_smoke_train_count,
        "smoke_validation_count": expected_smoke_validation_count,
        "files": CANONICAL_FILES,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"canonical data manifest {key} mismatch")
    if manifest["train_count"] + manifest["validation_count"] != manifest["source_count"]:
        raise ValueError("canonical partition counts do not sum to the source count")

    train_count, train_prefix = _audit_jsonl(
        directory / CANONICAL_FILES["train"],
        expected_smoke_train_count,
    )
    validation_count, validation_prefix = _audit_jsonl(
        directory / CANONICAL_FILES["validation"],
        expected_smoke_validation_count,
    )
    smoke_train_count, smoke_train = _audit_jsonl(
        directory / CANONICAL_FILES["smoke_train"],
        expected_smoke_train_count,
    )
    smoke_validation_count, smoke_validation = _audit_jsonl(
        directory / CANONICAL_FILES["smoke_validation"],
        expected_smoke_validation_count,
    )
    counts = {
        "train_count": train_count,
        "validation_count": validation_count,
        "smoke_train_count": smoke_train_count,
        "smoke_validation_count": smoke_validation_count,
    }
    for key, value in counts.items():
        if manifest[key] != value:
            raise ValueError(f"canonical data file {key} mismatch")
    if smoke_train != train_prefix:
        raise ValueError("smoke train is not the deterministic train prefix")
    if smoke_validation != validation_prefix:
        raise ValueError("smoke validation is not the deterministic validation prefix")
    return {
        "status": "PASS",
        "root": str(directory),
        "prompt_style": manifest["prompt_style"],
        "source_count": manifest["source_count"],
        **counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit canonical historical data")
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-source-count", type=int, default=170420)
    parser.add_argument("--expected-validation-count", type=int, default=120)
    parser.add_argument("--expected-smoke-train-count", type=int, default=64)
    parser.add_argument("--expected-smoke-validation-count", type=int, default=16)
    parser.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = audit_canonical_data(
        args.root,
        expected_source_count=args.expected_source_count,
        expected_validation_count=args.expected_validation_count,
        expected_smoke_train_count=args.expected_smoke_train_count,
        expected_smoke_validation_count=args.expected_smoke_validation_count,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
