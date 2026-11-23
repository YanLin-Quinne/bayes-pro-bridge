"""Evaluate a complete class label LoRA particle ensemble."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .checkpoint import (
    load_branch_checkpoint,
    validate_complete_checkpoint,
    validate_protocol_config,
)
from .config import (
    Config,
    HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE,
    load_config,
    validate_config,
)
from .data import (
    Example,
    as_hf_dataset,
    load_jsonl,
    make_class_label_collator,
    parse_record,
    validate_label_tokenization,
)
from .lora import LoRAEnsemble
from .metrics import (
    EvaluationMetrics,
    evaluate_probabilities,
    normalize_then_mean,
    validate_choice_key_consistency,
)
from .objective import masked_candidate_log_probs


PAPER_READOUT = "label_token_normalize_then_mean"


@dataclass(frozen=True)
class PreparedCheckpointEvaluator:
    """A protocol-validated native checkpoint ready for probability evaluation."""

    config: Config
    checkpoint_dir: Path
    manifest: dict[str, Any]
    device: torch.device
    ensemble: LoRAEnsemble
    tokenizer: Any
    label_token_ids: dict[str, int]


@dataclass(frozen=True)
class ProbabilityEvaluation:
    """Per-example tensors retained so pooled metrics remain exactly recomputable."""

    probabilities: torch.Tensor
    gold_indices: torch.Tensor
    candidate_mask: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.probabilities.shape[0])

    def metrics(self, *, ece_bins: int = 10) -> EvaluationMetrics:
        return evaluate_probabilities(
            self.probabilities,
            self.gold_indices,
            self.candidate_mask,
            ece_bins=ece_bins,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a complete class label PrO ensemble"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--output-json")
    parser.add_argument("--readout", choices=(PAPER_READOUT,), default=PAPER_READOUT)
    return parser


def validate_evaluation_args(
    args: argparse.Namespace,
    config: Config,
    *,
    checkpoint_world_size: int | None = None,
) -> None:
    validate_config(config, world_size=checkpoint_world_size)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.ece_bins <= 0:
        raise ValueError("ECE bin count must be positive")
    if args.readout != PAPER_READOUT:
        raise ValueError("only the paper label token readout is supported")
    if checkpoint_world_size is None and config.training.num_branches != 8:
        raise ValueError(
            "evaluation requires exactly eight branches when checkpoint world size "
            "is unavailable"
        )


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError("dtype must be bf16 or fp32")


def load_base_model_and_tokenizer(config: Config, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=False,
    )
    if config.model.prompt_style == HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE:
        tokenizer.pad_token_id = 0
    elif tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.training.padding_side
    tokenizer.truncation_side = "left"
    label_token_ids = validate_label_tokenization(tokenizer, config.model.labels)
    model_kwargs = {
        "revision": config.model.revision,
        "trust_remote_code": False,
        "torch_dtype": _torch_dtype(config.training.dtype),
    }
    if config.training.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.training.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        **model_kwargs,
    ).to(device)
    model.config.use_cache = False
    return model, tokenizer, label_token_ids


_load_model_and_tokenizer = load_base_model_and_tokenizer


def _pad_choice_axis(
    tensor: torch.Tensor,
    width: int,
    value: float | bool,
) -> torch.Tensor:
    if tensor.shape[1] > width:
        raise ValueError("choice tensor exceeds configured label count")
    if tensor.shape[1] == width:
        return tensor
    padding = torch.full(
        (tensor.shape[0], width - tensor.shape[1]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=1)


def prepare_checkpoint_evaluator(
    config: Config,
    checkpoint_dir: str | Path,
    *,
    batch_size: int,
    ece_bins: int,
    readout: str = PAPER_READOUT,
    device: torch.device | None = None,
) -> PreparedCheckpointEvaluator:
    """Validate and load a native immutable checkpoint for evaluation."""

    checkpoint_path = Path(checkpoint_dir).resolve()
    manifest = validate_complete_checkpoint(checkpoint_path)
    protocol = manifest["protocol"]
    checkpoint_world_size = int(protocol["training"]["world_size"])
    arguments = argparse.Namespace(
        batch_size=batch_size,
        ece_bins=ece_bins,
        readout=readout,
    )
    validate_evaluation_args(
        arguments,
        config,
        checkpoint_world_size=checkpoint_world_size,
    )
    validate_protocol_config(protocol, config)
    target_device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    base_model, tokenizer, label_token_ids = load_base_model_and_tokenizer(
        config,
        target_device,
    )
    branch_ids = tuple(range(config.training.num_branches))
    ensemble = LoRAEnsemble(
        base_model,
        branch_ids,
        config.training.branch_seeds,
        config.lora.target_modules,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
    ).to(target_device)
    fingerprint = manifest["protocol_fingerprint"]
    completed_epoch = manifest["completed_epoch"]
    optimizer_step = manifest["optimizer_step"]
    objective = protocol["objective"]
    for branch in ensemble.branches:
        load_branch_checkpoint(
            checkpoint_path / f"branch_{branch.branch_id}",
            branch,
            expected_model_id=config.model.id,
            expected_revision=config.model.revision,
            expected_label_readout=config.model.label_readout,
            expected_prompt_style=config.model.prompt_style,
            expected_seed=config.training.branch_seeds[branch.branch_id],
            expected_target_modules=config.lora.target_modules,
            expected_lora_rank=config.lora.rank,
            expected_lora_alpha=config.lora.alpha,
            expected_lora_dropout=config.lora.dropout,
            expected_objective=config.objective.name,
            expected_beta=config.objective.beta,
            expected_prior_lambda=config.objective.prior_lambda,
            expected_temperature=objective["temperature"],
            expected_fingerprint=fingerprint,
            expected_completed_epoch=completed_epoch,
            expected_optimizer_step=optimizer_step,
            expected_objective_alpha=config.objective.alpha,
        )
    ensemble.eval()
    return PreparedCheckpointEvaluator(
        config=config,
        checkpoint_dir=checkpoint_path,
        manifest=manifest,
        device=target_device,
        ensemble=ensemble,
        tokenizer=tokenizer,
        label_token_ids=label_token_ids,
    )


@torch.no_grad()
def evaluate_examples(
    prepared: PreparedCheckpointEvaluator,
    examples: Sequence[Example],
    *,
    batch_size: int,
) -> ProbabilityEvaluation:
    """Evaluate examples and retain the exact tensors needed for pooled metrics."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not examples:
        raise ValueError("evaluation examples must be nonempty")
    config = prepared.config
    collator = make_class_label_collator(
        config,
        prepared.tokenizer,
        prepared.label_token_ids,
    )
    dataset = as_hf_dataset(examples)

    def collate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return collator([parse_record(dict(record)) for record in records])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_records,
        num_workers=0,
    )
    probability_batches = []
    mask_batches = []
    gold_batches = []
    width = len(config.model.labels)
    for batch in loader:
        choice_keys = batch.pop("choice_keys")
        validate_choice_key_consistency(
            [choice_keys for _ in range(config.training.num_branches)]
        )
        batch = {
            key: value.to(prepared.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        logits = prepared.ensemble.local_candidate_logits(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            choice_token_ids=batch["choice_token_ids"],
        )
        branch_probabilities = masked_candidate_log_probs(
            logits,
            batch["choice_mask"],
        ).exp()
        ensemble_probabilities = normalize_then_mean(
            branch_probabilities,
            batch["choice_mask"],
        )
        probability_batches.append(
            _pad_choice_axis(ensemble_probabilities, width, 0.0).cpu()
        )
        mask_batches.append(
            _pad_choice_axis(batch["choice_mask"], width, False).cpu()
        )
        gold_batches.append(batch["gold_indices"].cpu())
    return ProbabilityEvaluation(
        probabilities=torch.cat(probability_batches),
        gold_indices=torch.cat(gold_batches),
        candidate_mask=torch.cat(mask_batches),
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, float]:
    config = load_config(args.config)
    prepared = prepare_checkpoint_evaluator(
        config,
        args.checkpoint_dir,
        batch_size=args.batch_size,
        ece_bins=args.ece_bins,
        readout=args.readout,
    )
    examples = load_jsonl(args.data)
    evaluation = evaluate_examples(
        prepared,
        examples,
        batch_size=args.batch_size,
    )
    metrics = evaluation.metrics(ece_bins=args.ece_bins).as_dict()
    encoded = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
