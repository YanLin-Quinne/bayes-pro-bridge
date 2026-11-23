"""Canonicalize and load the eight Final8 evaluation test sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .data import (
    Example,
    canonicalize_historical_record,
    decanonicalize_historical_record,
    historical_source_labels,
    parse_record,
    validate_historical_canonical_record,
)


FINAL8_DATASET_NAMES = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
)
FINAL8_DATASET_COUNTS = {
    "boolq": 3270,
    "piqa": 1838,
    "social_i_qa": 1954,
    "hellaswag": 10042,
    "winogrande": 1267,
    "ARC-Challenge": 1172,
    "ARC-Easy": 2376,
    "openbookqa": 500,
}


def _load_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {path}") from error


def _exact_keys(value: Any, expected: set[str], description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{description} schema mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _canonicalize_legacy_dataset(
    source: Path,
    *,
    dataset_name: str,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    raw_records = _load_json(source, f"Final8 source dataset {dataset_name}")
    if not isinstance(raw_records, list):
        raise ValueError(f"Final8 source dataset {dataset_name} must be a JSON array")
    if len(raw_records) != expected_count:
        raise ValueError(
            f"Final8 source dataset {dataset_name} count mismatch: "
            f"expected {expected_count}, found {len(raw_records)}"
        )
    canonical_records = []
    gold_counts: dict[str, int] = {}
    choice_counts: dict[str, int] = {}
    for index, raw in enumerate(raw_records):
        record = _exact_keys(
            raw,
            {"instruction", "input", "output", "answer"},
            f"Final8 source dataset {dataset_name} record {index}",
        )
        if any(not isinstance(record[key], str) for key in record):
            raise ValueError(
                f"Final8 source dataset {dataset_name} record {index} "
                "fields must be strings"
            )
        canonical_input = {
            "instruction": record["instruction"],
            "input": record["input"],
            "answer": record["answer"],
        }
        canonical = canonicalize_historical_record(canonical_input)
        canonical_example = validate_historical_canonical_record(canonical)
        neutral = decanonicalize_historical_record(canonical)
        source_labels = historical_source_labels(record["instruction"])
        expected = parse_record(
            {
                **canonical_input,
                "answer": record["answer"].lower(),
                "choice_labels": list(source_labels),
            }
        )
        if neutral != expected or canonical_example.answer_index != expected.answer_index:
            raise ValueError(
                f"Final8 source dataset {dataset_name} record {index} "
                "failed canonical roundtrip"
            )
        gold_label = source_labels[expected.answer_index]
        gold_counts[gold_label] = gold_counts.get(gold_label, 0) + 1
        choice_key = str(len(source_labels))
        choice_counts[choice_key] = choice_counts.get(choice_key, 0) + 1
        canonical_records.append(canonical)
    return canonical_records, gold_counts, choice_counts


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def load_strict_canonical_jsonl(
    path: str | Path,
    expected_count: int,
) -> tuple[Example, ...]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical Final8 JSONL: {path}")
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"canonical Final8 JSONL contains a blank line at {line_number}"
                )
            try:
                record = json.loads(line)
                _exact_keys(
                    record,
                    {"prompt", "choices", "answer_index"},
                    f"canonical Final8 record {line_number}",
                )
                example = validate_historical_canonical_record(record)
                neutral = decanonicalize_historical_record(record)
                if (
                    neutral.choices != example.choices
                    or neutral.answer_index != example.answer_index
                ):
                    raise ValueError("canonical record roundtrip mismatch")
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid canonical Final8 record on line {line_number}: {error}"
                ) from error
            examples.append(example)
    if len(examples) != expected_count:
        raise ValueError(
            f"canonical Final8 count mismatch: expected {expected_count}, "
            f"found {len(examples)}"
        )
    return tuple(examples)
