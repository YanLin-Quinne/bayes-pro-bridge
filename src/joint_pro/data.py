"""Validated JSONL schemas and class label batching."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
import unicodedata

import torch

from .config import (
    CANONICAL_LABELS,
    HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE,
    PLAIN_PROMPT_STYLE,
    QWEN2_CHATML_PROMPT_STYLE,
)


ANSWER_FORMAT = re.compile(r"Answer\s+format\s*:\s*(.+?)\s*$", re.IGNORECASE)
QWEN2_ASSISTANT_PROMPT_SUFFIX = "<|im_start|>assistant\n"
HISTORICAL_LABEL_PATTERN = re.compile(
    r"true|false|solution\d+|answer\d+|ending\d+|option\d+"
)
HISTORICAL_INPUT_HEADER = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
)
HISTORICAL_NO_INPUT_HEADER = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
)
HISTORICAL_RESPONSE = "### Response:\n"


@dataclass(frozen=True)
class Example:
    prompt: str
    choices: tuple[str, ...]
    answer_index: int

    @property
    def choice_keys(self) -> tuple[str, ...]:
        return CANONICAL_LABELS[: len(self.choices)]


def _validate_example(example: Example) -> Example:
    if not example.prompt.strip():
        raise ValueError("prompt must be nonempty")
    if not 2 <= len(example.choices) <= 5:
        raise ValueError("each example must contain two through five choices")
    if any(not value.strip() for value in example.choices):
        raise ValueError("choice values must be nonempty")
    if len(set(example.choices)) != len(example.choices):
        raise ValueError("choice values must be unique")
    if not 0 <= example.answer_index < len(example.choices):
        raise ValueError("answer_index is outside the choice list")
    return example


def _declared_historical_choices(record: dict[str, Any]) -> tuple[str, ...]:
    explicit = record.get("choice_labels")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("choice_labels must be a list")
        return tuple(str(item).strip() for item in explicit)
    match = ANSWER_FORMAT.search(str(record["instruction"]))
    if match is None:
        raise ValueError(
            "historical records require choice_labels or a trailing Answer format declaration"
        )
    values = re.split(r"\s*(?:/|\||,)\s*", match.group(1).strip())
    return tuple(value for value in values if value)


def historical_source_labels(instruction: str) -> tuple[str, ...]:
    match = re.search(
        r"Answer format:\s*([^\n]+)",
        instruction or "",
        flags=re.IGNORECASE,
    )
    labels = () if match is None else tuple(
        HISTORICAL_LABEL_PATTERN.findall(match.group(1).lower())
    )
    if not 2 <= len(labels) <= len(CANONICAL_LABELS):
        raise ValueError("historical Answer format must contain two through five labels")
    if len(set(labels)) != len(labels):
        raise ValueError("historical Answer format labels must be unique")
    return labels


def historical_label_mapping(source_labels: Sequence[str]) -> dict[str, str]:
    labels = tuple(str(label).lower() for label in source_labels)
    if not 2 <= len(labels) <= len(CANONICAL_LABELS):
        raise ValueError("historical label count must be two through five")
    if len(set(labels)) != len(labels):
        raise ValueError("historical labels must be unique")
    return {
        source_label: CANONICAL_LABELS[index]
        for index, source_label in enumerate(labels)
    }


def format_historical_alpaca_class_label_prompt(record: dict[str, Any]) -> str:
    instruction = str(record.get("instruction", ""))
    source_labels = historical_source_labels(instruction)
    label_map = historical_label_mapping(source_labels)
    mapping = ", ".join(
        f"{letter} = {source_label}"
        for source_label, letter in label_map.items()
    )
    mapped_instruction = (
        f"{instruction}\n\n"
        f"Use the following option-letter mapping: {mapping}.\n"
        "Answer with the single option letter only."
    )
    input_text = str(record.get("input", ""))
    if input_text:
        return (
            f"{HISTORICAL_INPUT_HEADER}"
            f"### Instruction:\n{mapped_instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"{HISTORICAL_RESPONSE}"
        )
    return (
        f"{HISTORICAL_NO_INPUT_HEADER}"
        f"### Instruction:\n{mapped_instruction}\n\n"
        f"{HISTORICAL_RESPONSE}"
    )


def canonicalize_historical_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("historical record must be a JSON object")
    required = {"instruction", "input", "answer"}
    missing = required - record.keys()
    missing.update(key for key in required if record.get(key) is None)
    if missing:
        raise ValueError(f"historical record is missing fields: {sorted(missing)}")
    source_labels = historical_source_labels(str(record["instruction"]))
    label_map = historical_label_mapping(source_labels)
    answer = str(record["answer"]).lower()
    if answer not in label_map:
        raise ValueError("historical answer is not in the declared Answer format")
    return {
        "prompt": format_historical_alpaca_class_label_prompt(record),
        "choices": list(source_labels),
        "answer_index": source_labels.index(answer),
    }


@dataclass(frozen=True)
class _HistoricalCanonicalParts:
    example: Example
    source_instruction: str
    input_text: str


def _parse_historical_canonical_record(
    record: dict[str, Any],
) -> _HistoricalCanonicalParts:
    if set(record) != {"prompt", "choices", "answer_index"}:
        raise ValueError("canonical historical record fields are invalid")
    if not isinstance(record["prompt"], str):
        raise ValueError("canonical historical prompt must be a string")
    if not isinstance(record["choices"], list) or any(
        not isinstance(choice, str) for choice in record["choices"]
    ):
        raise ValueError("canonical historical choices must be a list of strings")
    if type(record["answer_index"]) is not int:
        raise ValueError("canonical historical answer_index must be an integer")
    example = parse_record(record)
    prompt = example.prompt
    if prompt.startswith(HISTORICAL_INPUT_HEADER):
        body = prompt[len(HISTORICAL_INPUT_HEADER) :]
        instruction, remainder = body.split("\n\n### Input:\n", 1)
        input_text, response = remainder.rsplit("\n\n### Response:\n", 1)
    elif prompt.startswith(HISTORICAL_NO_INPUT_HEADER):
        body = prompt[len(HISTORICAL_NO_INPUT_HEADER) :]
        instruction, response = body.rsplit("\n\n### Response:\n", 1)
        input_text = ""
    else:
        raise ValueError("canonical historical prompt header is invalid")
    if response:
        raise ValueError("canonical historical prompt must end at the response marker")
    prefix = "### Instruction:\n"
    if not instruction.startswith(prefix):
        raise ValueError("canonical historical instruction marker is invalid")
    mapped_instruction = instruction[len(prefix) :]
    mapping = historical_label_mapping(example.choices)
    mapping_text = ", ".join(
        f"{letter} = {source_label}"
        for source_label, letter in mapping.items()
    )
    suffix = (
        f"\n\nUse the following option-letter mapping: {mapping_text}.\n"
        "Answer with the single option letter only."
    )
    if not mapped_instruction.endswith(suffix):
        raise ValueError("canonical historical option mapping is invalid")
    source_instruction = mapped_instruction[: -len(suffix)]
    if historical_source_labels(source_instruction) != example.choices:
        raise ValueError("canonical historical source labels do not match the prompt")
    expected = format_historical_alpaca_class_label_prompt(
        {"instruction": source_instruction, "input": input_text}
    )
    if prompt != expected:
        raise ValueError("canonical historical prompt does not match the formatter")
    return _HistoricalCanonicalParts(example, source_instruction, input_text)


def validate_historical_canonical_record(record: dict[str, Any]) -> Example:
    return _parse_historical_canonical_record(record).example


def decanonicalize_historical_record(record: dict[str, Any]) -> Example:
    """Recover the neutral example represented by a strict canonical prompt."""
    parsed = _parse_historical_canonical_record(record)
    example = parsed.example
    return parse_record(
        {
            "instruction": parsed.source_instruction,
            "input": parsed.input_text,
            "answer": example.choices[example.answer_index],
            "choice_labels": list(example.choices),
        }
    )


def rerender_historical_canonical_prompt(record: dict[str, Any]) -> str:
    """Strictly parse a canonical record and rebuild its historical prompt."""
    parsed = _parse_historical_canonical_record(record)
    return format_historical_alpaca_class_label_prompt(
        {
            "instruction": parsed.source_instruction,
            "input": parsed.input_text,
        }
    )


def parse_record(record: dict[str, Any]) -> Example:
    if {"prompt", "choices", "answer_index"} <= record.keys():
        choices = record["choices"]
        if not isinstance(choices, list):
            raise ValueError("choices must be a list")
        return _validate_example(
            Example(
                prompt=str(record["prompt"]),
                choices=tuple(str(item) for item in choices),
                answer_index=int(record["answer_index"]),
            )
        )
    if {"instruction", "input", "answer"} <= record.keys():
        choices = _declared_historical_choices(record)
        answer = str(record["answer"]).strip()
        canonical = CANONICAL_LABELS[: len(choices)]
        if answer in canonical:
            answer_index = canonical.index(answer)
        else:
            matches = [index for index, value in enumerate(choices) if value == answer]
            if len(matches) != 1:
                raise ValueError("historical answer does not match a declared choice")
            answer_index = matches[0]
        instruction = str(record["instruction"]).strip()
        input_text = str(record["input"]).strip()
        prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
        return _validate_example(Example(prompt, choices, answer_index))
    raise ValueError("record does not match a supported JSONL schema")


def load_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                examples.append(parse_record(record))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid JSONL record on line {line_number}: {error}") from error
    if not examples:
        raise ValueError("data file contains no examples")
    return examples


def as_hf_dataset(examples: Sequence[Example]):
    """Return a Datasets object without downloading or consulting a hub."""
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "prompt": example.prompt,
                "choices": list(example.choices),
                "answer_index": example.answer_index,
            }
            for example in examples
        ]
    )


def format_prompt(example: Example) -> str:
    options = "\n".join(
        f"{key}. {value}" for key, value in zip(example.choice_keys, example.choices)
    )
    return f"{example.prompt}\n\n{options}\n\nAnswer:"


def render_prompt(
    example: Example,
    *,
    prompt_style: str = PLAIN_PROMPT_STYLE,
    tokenizer=None,
) -> str:
    content = format_prompt(example)
    if prompt_style == PLAIN_PROMPT_STYLE:
        return content
    if prompt_style == HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE:
        return example.prompt
    if prompt_style == QWEN2_CHATML_PROMPT_STYLE:
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("Qwen2 ChatML rendering requires a chat-template tokenizer")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("chat template must render a nonempty string")
        if not rendered.endswith(QWEN2_ASSISTANT_PROMPT_SUFFIX):
            raise ValueError("Qwen2 ChatML must end with the assistant generation prompt")
        return rendered
    raise ValueError(f"unsupported prompt_style: {prompt_style}")


def normalize_decoded_label(value: str) -> str:
    """Normalize a decoded bare label for protocol validation."""
    return unicodedata.normalize("NFKC", value).strip()


def validate_label_tokenization(tokenizer, labels: Iterable[str] = CANONICAL_LABELS) -> dict[str, int]:
    """Validate the paper's bare canonical next-token readout.

    This intentionally tokenizes each label independently. Retokenizing the
    human-readable concatenation ``prompt + label`` is not part of the protocol.
    """
    token_ids: dict[str, int] = {}
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ValueError("canonical labels must be nonempty strings")
        encoded = tokenizer(label, add_special_tokens=False)
        values = encoded["input_ids"]
        if isinstance(values, torch.Tensor):
            values = values.tolist()
        if len(values) != 1:
            raise ValueError(f"label {label!r} must tokenize to exactly one token")
        token_id = int(values[0])
        if token_id in special_ids:
            raise ValueError(f"label {label!r} maps to a special token")
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if normalize_decoded_label(decoded) != normalize_decoded_label(label):
            raise ValueError(f"label {label!r} does not decode to the canonical label")
        token_ids[label] = token_id
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("canonical labels must map to distinct tokens")
    return token_ids


def last_valid_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2 or not torch.all(
        (attention_mask == 0) | (attention_mask == 1)
    ):
        raise ValueError("attention_mask must be a binary matrix")
    if torch.any(attention_mask.sum(dim=1) == 0):
        raise ValueError("every prompt must contain at least one token")
    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).unsqueeze(0)
    return positions.masked_fill(attention_mask == 0, -1).max(dim=1).values


class ClassLabelCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        label_token_ids: dict[str, int],
        prompt_style: str = PLAIN_PROMPT_STYLE,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_token_ids = dict(label_token_ids)
        self.prompt_style = prompt_style

    def __call__(self, examples: Sequence[Example]) -> dict[str, Any]:
        if self.prompt_style == HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE:
            return self._historical_batch(examples)
        prompts = [
            render_prompt(
                example,
                prompt_style=self.prompt_style,
                tokenizer=self.tokenizer,
            )
            for example in examples
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        choice_width = max(len(example.choices) for example in examples)
        choice_token_ids = torch.zeros((len(examples), choice_width), dtype=torch.long)
        choice_mask = torch.zeros((len(examples), choice_width), dtype=torch.bool)
        gold = torch.empty(len(examples), dtype=torch.long)
        choice_keys: list[tuple[str, ...]] = []
        for row, example in enumerate(examples):
            keys = example.choice_keys
            choice_keys.append(keys)
            for column, key in enumerate(keys):
                choice_token_ids[row, column] = self.label_token_ids[key]
                choice_mask[row, column] = True
            gold[row] = example.answer_index
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "choice_token_ids": choice_token_ids,
            "choice_mask": choice_mask,
            "gold_indices": gold,
            "choice_keys": choice_keys,
        }

    def _historical_batch(self, examples: Sequence[Example]) -> dict[str, Any]:
        if not examples:
            raise ValueError("historical class label batch must be nonempty")
        rows = []
        max_choices = 0
        for example in examples:
            validate_historical_canonical_record(
                {
                    "prompt": example.prompt,
                    "choices": list(example.choices),
                    "answer_index": example.answer_index,
                }
            )
            encoded = self.tokenizer(
                example.prompt,
                add_special_tokens=True,
            )["input_ids"]
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.tolist()
            input_ids = [int(value) for value in encoded]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length :]
            choice_keys = example.choice_keys
            rows.append(
                {
                    "input_ids": input_ids,
                    "choice_token_ids": [
                        self.label_token_ids[label] for label in choice_keys
                    ],
                    "choice_keys": choice_keys,
                    "gold_index": example.answer_index,
                }
            )
            max_choices = max(max_choices, len(choice_keys))
        max_length = max(len(row["input_ids"]) for row in rows)
        if max_length % 8:
            max_length = ((max_length // 8) + 1) * 8
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        input_ids = []
        attention_mask = []
        choice_token_ids = []
        choice_mask = []
        gold_indices = []
        choice_keys = []
        for row in rows:
            pad_length = max_length - len(row["input_ids"])
            input_ids.append([pad_id] * pad_length + row["input_ids"])
            attention_mask.append(
                [0] * pad_length + [1] * len(row["input_ids"])
            )
            choice_padding = max_choices - len(row["choice_token_ids"])
            choice_token_ids.append(
                row["choice_token_ids"] + [0] * choice_padding
            )
            choice_mask.append(
                [True] * len(row["choice_token_ids"])
                + [False] * choice_padding
            )
            gold_indices.append(row["gold_index"])
            choice_keys.append(row["choice_keys"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "choice_token_ids": torch.tensor(choice_token_ids, dtype=torch.long),
            "choice_mask": torch.tensor(choice_mask, dtype=torch.bool),
            "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
            "choice_keys": choice_keys,
        }


def make_class_label_collator(config, tokenizer, label_token_ids: dict[str, int]):
    return ClassLabelCollator(
        tokenizer,
        config.training.max_length,
        label_token_ids,
        config.model.prompt_style,
    )
