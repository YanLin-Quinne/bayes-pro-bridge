"""Frozen backbone gold class-label probability, with the prompt and readout of the training scripts.

The prompt builders, tokenizer setup, collator and class-label readout below
are copied from the historical trainers that produced the paper numbers:

* Llama-3.2-1B/3B: finetune_joint_ensemble_PRO_branch_parallel_classlabel{,_convex,_convex_1b_20260718}.py
  (sha256 c1cd2846..., 90251da2...; the prompt and collator code is identical
  in all three files)
* Qwen3-4B-Base: finetune_joint_ensemble_PRO_branch_parallel_classlabel_convex_qwen3_20260719.py
  (sha256 867a4ae3...; same Alpaca prompt, pad token = EOS = 151643)
* Qwen2-1.5B-Instruct: ..._classlabel_convex_qwen2_instruct.py (sha256 4429ecc5...;
  the model's own ChatML template with one user turn and the generation prompt)

Readout (all trainers, ``BranchParallelLoRAEnsemble.local_choice_log_probs``):
next-token logits at the last attended position, gathered at the single
canonical label tokens A..E, padded candidates masked, softmax over the
candidate set only.  Every label is exactly one token, so the evaluator's
``choice_score_normalization=length`` is the identity here.  No adapter is
loaded: this is the frozen backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = "joint_pro.h4_historical.v1"
CANONICAL_LABELS = ["A", "B", "C", "D", "E"]
EXPECTED_SOURCE_COUNT = 170_420
EXPECTED_TRAIN_COUNT = 170_300
VALIDATION_SIZE = 120
SPLIT_SEED = 42
CUTOFF_LEN = 256
PROMPT_FAMILIES = ("alpaca", "qwen2_chatml")
TOKENIZER_SETUPS = ("pad_zero", "pad_eos")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
FAMILY_BY_LABELS = {
    ("true", "false"): "boolq",
    ("solution1", "solution2"): "piqa",
    ("answer1", "answer2", "answer3"): "social_i_qa",
    ("ending1", "ending2", "ending3", "ending4"): "hellaswag",
    ("option1", "option2"): "winogrande",
    ("answer1", "answer2", "answer3", "answer4"): "arc_or_openbookqa",
    ("answer1", "answer2", "answer3", "answer4", "answer5"): "arc_or_openbookqa",
}
FAMILIES = (
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "arc_or_openbookqa",
)


# ---- historical prompt code (verbatim from the trainers) -------------------

def _candidate_labels(instruction: str) -> list[str]:
    match = re.search(r"Answer format:\s*([^\n]+)", instruction or "", flags=re.IGNORECASE)
    if match:
        labels = re.findall(
            r"true|false|solution\d+|answer\d+|ending\d+|option\d+",
            match.group(1).lower(),
        )
        if labels:
            return labels
    raise ValueError(f"Cannot infer answer labels from instruction: {instruction[:200]}")


def _canonicalise_labels(source_labels: list[str]) -> dict[str, str]:
    if len(source_labels) > len(CANONICAL_LABELS):
        raise ValueError(f"Too many labels for canonical mapping: {source_labels}")
    return {
        source_label: CANONICAL_LABELS[idx]
        for idx, source_label in enumerate(source_labels)
    }


def _label_token_instruction(instruction: str, source_labels: list[str]) -> str:
    label_map = _canonicalise_labels(source_labels)
    mapping = ", ".join(
        f"{letter} = {source_label}"
        for source_label, letter in label_map.items()
    )
    return (
        f"{instruction}\n\n"
        f"Use the following option-letter mapping: {mapping}.\n"
        "Answer with the single option letter only."
    )


def format_alpaca_label_prompt(data_point: dict) -> str:
    source_labels = _candidate_labels(data_point.get("instruction", ""))
    instruction = _label_token_instruction(data_point.get("instruction", ""), source_labels)
    if data_point.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{data_point['input']}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _format_user_content(instruction: str, input_text: str) -> str:
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def format_qwen2_label_prompt(tokenizer, data_point: dict) -> str:
    source_labels = _candidate_labels(data_point.get("instruction", ""))
    instruction = _label_token_instruction(data_point.get("instruction", ""), source_labels)
    content = _format_user_content(instruction, str(data_point.get("input") or ""))
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


class HistoricalClassLabelCollator:
    """The historical ``ClassLabelCollator`` with the prompt family selectable."""

    def __init__(self, tokenizer, cutoff_len: int, label_token_ids: dict[str, int],
                 prompt_family: str):
        if prompt_family not in PROMPT_FAMILIES:
            raise ValueError(f"unknown prompt family {prompt_family!r}")
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must be configured with a pad token before collation")
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.pad_id = int(tokenizer.pad_token_id)
        self.label_token_ids = dict(label_token_ids)
        self.prompt_family = prompt_family
        self.total_examples = 0
        self.truncated_examples = 0

    def _encode(self, example: dict) -> list[int]:
        if self.prompt_family == "alpaca":
            prompt = format_alpaca_label_prompt(example)
            return self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt = format_qwen2_label_prompt(self.tokenizer, example)
        return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        rows = []
        max_choices = 0
        for example in examples:
            source_labels = _candidate_labels(example.get("instruction", ""))
            label_map = _canonicalise_labels(source_labels)
            gold_source = str(example.get("answer", "")).lower()
            if gold_source not in label_map:
                raise ValueError(f"Gold answer {gold_source!r} not in labels {source_labels}")
            input_ids = list(self._encode(example))
            self.total_examples += 1
            if len(input_ids) > self.cutoff_len:
                self.truncated_examples += 1
                input_ids = input_ids[-self.cutoff_len:]
            choice_labels = [label_map[source_label] for source_label in source_labels]
            choice_token_ids = [self.label_token_ids[label] for label in choice_labels]
            gold_choice_index = choice_labels.index(label_map[gold_source])
            rows.append({
                "input_ids": input_ids,
                "choice_token_ids": choice_token_ids,
                "gold_choice_index": gold_choice_index,
            })
            max_choices = max(max_choices, len(choice_token_ids))

        max_len = max(len(row["input_ids"]) for row in rows)
        if max_len % 8:
            max_len = ((max_len // 8) + 1) * 8

        input_ids, attention_mask, choice_ids, choice_mask, gold = [], [], [], [], []
        for row in rows:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append([self.pad_id] * pad_len + row["input_ids"])
            attention_mask.append([0] * pad_len + [1] * len(row["input_ids"]))
            choice_pad = max_choices - len(row["choice_token_ids"])
            choice_ids.append(row["choice_token_ids"] + [0] * choice_pad)
            choice_mask.append([True] * len(row["choice_token_ids"]) + [False] * choice_pad)
            gold.append(row["gold_choice_index"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "choice_token_ids": torch.tensor(choice_ids, dtype=torch.long),
            "choice_mask": torch.tensor(choice_mask, dtype=torch.bool),
            "gold_choice_indices": torch.tensor(gold, dtype=torch.long),
        }


def choice_log_probs(model, input_ids, attention_mask, choice_token_ids, choice_mask):
    """``local_choice_log_probs`` of the historical trainers for the bare model."""
    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    last_token_indices = (
        attention_mask.size(1)
        - 1
        - attention_mask.long().flip(dims=[1]).argmax(dim=1)
    )
    safe_choice_ids = choice_token_ids.clamp_min(0)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    next_token_logits = outputs.logits[batch_indices, last_token_indices, :]
    logits = next_token_logits.gather(1, safe_choice_ids)
    logits = logits.masked_fill(~choice_mask, torch.finfo(logits.dtype).min)
    return F.log_softmax(logits.float(), dim=-1)


# ---- configuration, data, tokenizer ---------------------------------------

def load_backbone_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"key", "model_id", "revision", "prompt_family", "tokenizer_setup"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"backbone config misses {sorted(missing)}")
    if config["prompt_family"] not in PROMPT_FAMILIES:
        raise ValueError("unknown prompt_family")
    if config["tokenizer_setup"] not in TOKENIZER_SETUPS:
        raise ValueError("unknown tokenizer_setup")
    return config


def require_pinned_revision(config: dict[str, Any]) -> str:
    revision = str(config["revision"])
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(
            f"{config['key']}: revision {revision!r} is not a pinned 40 hex commit; "
            "fill it from scripts/backbones.sh before running"
        )
    return revision


def dataset_family(instruction: str) -> str:
    return FAMILY_BY_LABELS[tuple(_candidate_labels(instruction))]


def record_hash(record: dict) -> str:
    payload = {key: record.get(key) for key in ("instruction", "input", "output", "answer")}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_train_split(records: Sequence[dict], *, seed: int = SPLIT_SEED,
                           validation_size: int = VALIDATION_SIZE) -> list[dict]:
    """Same datasets calls as the trainers' ``_build_datasets``.

    ``train_test_split(test_size=120, shuffle=True, seed=42)`` then
    ``split["train"].shuffle(seed=42)``.  Position i of the result is line
    i + 1 of the canonical train.jsonl written by ``joint_pro.canonicalize_data``.
    """
    from datasets import Dataset

    dataset = Dataset.from_list(list(records))
    split = dataset.train_test_split(test_size=validation_size, shuffle=True, seed=seed)
    return split["train"].shuffle(seed=seed).to_list()


def configure_tokenizer(tokenizer, config: dict[str, Any]) -> dict[str, int]:
    if config["tokenizer_setup"] == "pad_zero":
        tokenizer.pad_token_id = 0
    elif tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    label_token_ids = {}
    for label in CANONICAL_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Canonical label {label!r} is not one token: {token_ids}")
        label_token_ids[label] = int(token_ids[0])
    expected_pad = config.get("expected_pad_token_id")
    if expected_pad is not None and int(tokenizer.pad_token_id) != int(expected_pad):
        raise ValueError(f"pad token id {tokenizer.pad_token_id} != {expected_pad}")
    expected_labels = config.get("expected_label_token_ids")
    if expected_labels is not None and label_token_ids != expected_labels:
        raise ValueError(f"label token ids {label_token_ids} != {expected_labels}")
    if config["prompt_family"] == "qwen2_chatml":
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with A only."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(probe, str) or not probe.endswith("<|im_start|>assistant\n"):
            raise ValueError(f"Unexpected ChatML generation boundary: {probe[-80:]!r}")
    return label_token_ids


# ---- statistics -----------------------------------------------------------

def summarize(gold_probabilities: np.ndarray, num_choices: np.ndarray) -> dict[str, Any]:
    p = np.asarray(gold_probabilities, dtype=np.float64)
    k = np.asarray(num_choices, dtype=np.int64)
    if p.ndim != 1 or p.shape != k.shape or p.size == 0:
        raise ValueError("summary needs equal length nonempty 1-D arrays")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("gold probabilities must be finite and in [0, 1]")
    if (k < 2).any() or (k > 5).any():
        raise ValueError("candidate counts must lie in 2..5")

    def quantile(q: float) -> float:
        return float(np.quantile(p, q, method="linear"))

    return {
        "count": int(p.size),
        "mean": float(p.mean()),
        "min": float(p.min()),
        "q01": quantile(0.01),
        "q05": quantile(0.05),
        "q10": quantile(0.10),
        "median": quantile(0.50),
        "q90": quantile(0.90),
        "fraction_below_0.01": float((p < 0.01).mean()),
        "fraction_below_1_over_K": float((p < 1.0 / k).mean()),
        "fraction_below_1_over_8": float((p < 1.0 / 8.0).mean()),
        "mean_gold_nll": float(-np.log(np.clip(p, 1e-300, None)).mean()),
        "accuracy_argmax": None,
    }


def summarize_with_breakdown(gold_probabilities, num_choices, families, correct) -> dict[str, Any]:
    families = np.asarray(families)
    correct = np.asarray(correct, dtype=bool)
    pooled = summarize(gold_probabilities, num_choices)
    pooled["accuracy_argmax"] = float(correct.mean())
    per_dataset = {}
    for name in FAMILIES:
        mask = families == name
        if not mask.any():
            continue
        row = summarize(np.asarray(gold_probabilities)[mask], np.asarray(num_choices)[mask])
        row["accuracy_argmax"] = float(correct[mask].mean())
        per_dataset[name] = row
    return {"pooled": pooled, "per_dataset": per_dataset}


# ---- main -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone-config", required=True)
    parser.add_argument("--data", required=True, help="commonsense_170k.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, help="smoke: first N training examples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--test-set", action="store_true",
                        help="score a test file as given (no 120-example split, no training-count check)")
    return parser


def _load_model(config: dict[str, Any], revision: str, dtype: torch.dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=revision,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": dtype}
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = config["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], **kwargs).to(device)
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _write_atomic(path: Path, writer) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    writer(temporary)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    def reset_peak_memory_after_cuda_init(device: torch.device) -> None:
        """Reset CUDA peak memory counters only after CUDA is initialised."""
        if device.type != "cuda":
            return
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.monotonic()
    if args.batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("batch size and progress interval must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max_examples must be positive")
    config = load_backbone_config(args.backbone_config)
    revision = require_pinned_revision(config)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_example.npz"
    if summary_path.exists() or rows_path.exists():
        raise FileExistsError(f"refusing to overwrite outputs in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_sha256 = sha256_file(args.data)
    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.test_set:
        # Zero-shot test-set evaluation of the frozen backbone: file order, no split.
        train = list(records)
    else:
        if len(records) != EXPECTED_SOURCE_COUNT:
            raise ValueError(f"expected {EXPECTED_SOURCE_COUNT} source records, got {len(records)}")
        train = historical_train_split(records)
        if len(train) != EXPECTED_TRAIN_COUNT:
            raise ValueError(f"expected {EXPECTED_TRAIN_COUNT} training records, got {len(train)}")
    if args.max_examples is not None:
        train = train[: args.max_examples]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reset_peak_memory_after_cuda_init(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tokenizer = _load_model(config, revision, dtype, device)
    label_token_ids = configure_tokenizer(tokenizer, config)
    collator = HistoricalClassLabelCollator(
        tokenizer, CUTOFF_LEN, label_token_ids, config["prompt_family"])

    n = len(train)
    gold_logp = np.empty(n, dtype=np.float64)
    candidate_probs = np.zeros((n, len(CANONICAL_LABELS)), dtype=np.float32)
    num_choices = np.empty(n, dtype=np.int8)
    gold_index = np.empty(n, dtype=np.int8)
    families = np.array([dataset_family(r["instruction"]) for r in train])
    hashes = np.array([record_hash(r) for r in train])
    inference_started = time.monotonic()
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, n, args.batch_size), 1):
            stop = min(start + args.batch_size, n)
            batch = collator(train[start:stop])
            batch = {key: value.to(device) for key, value in batch.items()}
            log_probs = choice_log_probs(
                model, batch["input_ids"], batch["attention_mask"],
                batch["choice_token_ids"], batch["choice_mask"])
            if not torch.isfinite(log_probs.masked_select(batch["choice_mask"])).all():
                raise ValueError(f"non finite candidate log probabilities in rows {start}..{stop}")
            rows = torch.arange(stop - start, device=device)
            gold = batch["gold_choice_indices"]
            gold_logp[start:stop] = log_probs[rows, gold].double().cpu().numpy()
            probs = log_probs.exp().masked_fill(~batch["choice_mask"], 0.0)
            candidate_probs[start:stop, : probs.shape[1]] = probs.cpu().numpy()
            num_choices[start:stop] = batch["choice_mask"].sum(dim=1).cpu().numpy()
            gold_index[start:stop] = gold.cpu().numpy()
            if batch_number % args.progress_every == 0:
                elapsed = time.monotonic() - inference_started
                print(json.dumps({"processed": stop, "of": n, "elapsed_seconds": round(elapsed, 1),
                                  "examples_per_second": round(stop / elapsed, 1)}), flush=True)
    inference_seconds = time.monotonic() - inference_started
    row_sums = candidate_probs.astype(np.float64).sum(axis=1)
    gold_prob = np.exp(gold_logp)
    correct = candidate_probs.argmax(axis=1) == gold_index
    statistics = summarize_with_breakdown(gold_prob, num_choices, families, correct)

    import transformers

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "pre_adaptation_gold_probability",
        "mode": "smoke" if args.max_examples is not None else "full",
        "backbone": config,
        "model_revision": revision,
        "resolved_attn_implementation": getattr(model.config, "_attn_implementation", None),
        "dtype": args.dtype,
        "device": str(device),
        "batch_size": args.batch_size,
        "data": {
            "path": str(Path(args.data).resolve()),
            "sha256": data_sha256,
            "source_count": len(records) if args.test_set else EXPECTED_SOURCE_COUNT,
            "split": ("none: test file scored as given" if args.test_set else
                      "datasets train_test_split(test_size=120, shuffle=True, seed=42); train.shuffle(seed=42)"),
            "train_count": len(train) if args.test_set else EXPECTED_TRAIN_COUNT,
            "processed_count": n,
            "row_order": ("row i = test file record i" if args.test_set else "row i = canonical train.jsonl line i+1"),
        },
        "protocol": {
            "prompt_family": config["prompt_family"],
            "tokenizer_setup": config["tokenizer_setup"],
            "pad_token_id": int(tokenizer.pad_token_id),
            "label_token_ids": label_token_ids,
            "cutoff_len": CUTOFF_LEN,
            "truncation": "keep last cutoff_len tokens",
            "padding": "left, padded length rounded up to a multiple of 8",
            "readout": "label_token: next-token logits at the last attended position, "
                       "softmax over the example's candidate labels",
            "length_normalisation": "identity (every label is one token)",
            "adapters_loaded": False,
            "truncated_examples": collator.truncated_examples,
        },
        "checks": {
            "max_candidate_row_sum_error": float(np.abs(row_sums - 1.0).max()),
            "all_gold_finite": bool(np.isfinite(gold_logp).all()),
        },
        "statistics": statistics,
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "inference_seconds": inference_seconds,
            "examples_per_second": n / inference_seconds,
            "peak_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        },
        "software": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "numpy": np.__version__},
    }
    arrays = dict(train_index=np.arange(n, dtype=np.int64), example_hash=hashes,
                  dataset_family=families, num_choices=num_choices, gold_index=gold_index,
                  gold_log_prob=gold_logp, gold_prob=gold_prob, candidate_probs=candidate_probs)
    _write_atomic(rows_path, lambda path: _save_npz(path, arrays))
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    _write_atomic(summary_path, lambda path: path.write_text(encoded + "\n", encoding="utf-8"))
    print(json.dumps(statistics["pooled"], sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
