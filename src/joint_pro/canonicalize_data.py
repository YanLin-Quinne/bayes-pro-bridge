"""Canonicalize the historical JSON array for matched rescue experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence
import uuid

from .config import HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE
from .data import canonicalize_historical_record


CANONICAL_DATA_SCHEMA = "historical_alpaca_class_label_jsonl_v1"
CANONICAL_DATA_SCHEMA_VERSION = 1
CANONICAL_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "smoke_train": "smoke_train.jsonl",
    "smoke_validation": "smoke_validation.jsonl",
}


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    return count


def canonicalize_data(
    source_path: str | Path,
    output_root: str | Path,
    *,
    expected_source_count: int = 170420,
    validation_size: int = 120,
    split_seed: int = 42,
    shuffle_seed: int = 42,
    smoke_train_size: int = 64,
    smoke_validation_size: int = 16,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    destination = Path(output_root).resolve()
    numeric = (
        expected_source_count,
        validation_size,
        smoke_train_size,
        smoke_validation_size,
    )
    if not source.is_file():
        raise FileNotFoundError(f"missing historical source JSON: {source}")
    if any(value <= 0 for value in numeric):
        raise ValueError("canonical data counts must be positive")
    if validation_size >= expected_source_count:
        raise ValueError("validation size must be smaller than the source count")
    if smoke_train_size > expected_source_count - validation_size:
        raise ValueError("smoke train size exceeds the train partition")
    if smoke_validation_size > validation_size:
        raise ValueError("smoke validation size exceeds the validation partition")
    if destination.exists():
        raise FileExistsError("canonical data output root already exists")

    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(source), split="train")
    if len(dataset) != expected_source_count:
        raise ValueError(
            f"historical source count mismatch: expected {expected_source_count}, "
            f"found {len(dataset)}"
        )
    split = dataset.train_test_split(
        test_size=validation_size,
        shuffle=True,
        seed=split_seed,
    )
    train = split["train"].shuffle(seed=shuffle_seed)
    validation = split["test"].shuffle(seed=shuffle_seed)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        train_records = (
            canonicalize_historical_record(dict(record)) for record in train
        )
        validation_records = (
            canonicalize_historical_record(dict(record)) for record in validation
        )
        train_count = _write_jsonl(
            temporary / CANONICAL_FILES["train"],
            train_records,
        )
        validation_count = _write_jsonl(
            temporary / CANONICAL_FILES["validation"],
            validation_records,
        )
        smoke_train_records = (
            canonicalize_historical_record(dict(train[index]))
            for index in range(smoke_train_size)
        )
        smoke_validation_records = (
            canonicalize_historical_record(dict(validation[index]))
            for index in range(smoke_validation_size)
        )
        _write_jsonl(
            temporary / CANONICAL_FILES["smoke_train"],
            smoke_train_records,
        )
        _write_jsonl(
            temporary / CANONICAL_FILES["smoke_validation"],
            smoke_validation_records,
        )
        manifest = {
            "schema_version": CANONICAL_DATA_SCHEMA_VERSION,
            "schema": CANONICAL_DATA_SCHEMA,
            "prompt_style": HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE,
            "source_path": str(source),
            "source_count": len(dataset),
            "train_count": train_count,
            "validation_count": validation_count,
            "split_seed": int(split_seed),
            "shuffle_seed": int(shuffle_seed),
            "validation_size": int(validation_size),
            "smoke_train_count": int(smoke_train_size),
            "smoke_validation_count": int(smoke_validation_size),
            "files": dict(CANONICAL_FILES),
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonicalize historical class label data"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-source-count", type=int, default=170420)
    parser.add_argument("--validation-size", type=int, default=120)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--smoke-train-size", type=int, default=64)
    parser.add_argument("--smoke-validation-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = canonicalize_data(
        args.source,
        args.output_root,
        expected_source_count=args.expected_source_count,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
        shuffle_seed=args.shuffle_seed,
        smoke_train_size=args.smoke_train_size,
        smoke_validation_size=args.smoke_validation_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
