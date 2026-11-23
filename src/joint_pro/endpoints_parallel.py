"""Predictive endpoints: held-out n=120 and Final8 for one objective-switch arm.

Same metrics as the sequential evaluation (``reviewer_evaluate``: NLL,
accuracy, ECE with 10 bins, Brier, mean confidence, confidence gap; Final8
equal-dataset macro and pooled), but the eight particles are spread over the
torchrun ranks with the training allocation (2 per GPU on 4 GPUs) instead of
running all eight sequentially on cuda:0.  Each rank scores its own particles
on the same batches, the per-particle probabilities are gathered in particle
order, and rank 0 forms the normalize-then-mean ensemble.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import evaluate
from .canonicalize_final8 import FINAL8_DATASET_COUNTS, FINAL8_DATASET_NAMES, load_strict_canonical_jsonl
from .checkpoint import load_branch_checkpoint, validate_complete_checkpoint, validate_protocol_config
from .config import load_config
from .data import as_hf_dataset, make_class_label_collator, parse_record
from .distributed import (
    allocate_contiguous_branches,
    barrier,
    cleanup_distributed,
    gather_detached_branch_tensor,
    init_distributed,
)
from .lora import LoRAEnsemble
from .metrics import normalize_then_mean
from .objective import masked_candidate_log_probs
from .reviewer_evaluate import METRIC_KEYS, _concatenate, _metric_row, equal_dataset_macro

SCHEMA_VERSION = "joint_pro.endpoints_parallel.v1"
VALIDATION_COUNT = 120
WIDTH = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--validation", required=True, help="canonical validation.jsonl (n=120)")
    parser.add_argument("--final8-root", required=True, help="dir with <dataset>.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--max-examples-per-split", type=int)
    return parser


def load_local_particles(config, checkpoint_dir: Path, local_ids: Sequence[int], device):
    manifest = validate_complete_checkpoint(checkpoint_dir)
    protocol = manifest["protocol"]
    validate_protocol_config(protocol, config)
    base_model, tokenizer, label_token_ids = evaluate.load_base_model_and_tokenizer(config, device)
    ensemble = LoRAEnsemble(
        base_model, tuple(local_ids), config.training.branch_seeds, config.lora.target_modules,
        rank=config.lora.rank, alpha=config.lora.alpha, dropout=config.lora.dropout,
    ).to(device)
    for branch in ensemble.branches:
        load_branch_checkpoint(
            checkpoint_dir / f"branch_{branch.branch_id}", branch,
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
            expected_temperature=protocol["objective"]["temperature"],
            expected_fingerprint=manifest["protocol_fingerprint"],
            expected_completed_epoch=manifest["completed_epoch"],
            expected_optimizer_step=manifest["optimizer_step"],
            expected_objective_alpha=config.objective.alpha,
        )
    ensemble.eval()
    return manifest, ensemble, tokenizer, label_token_ids


def _pad(tensor: torch.Tensor, value) -> torch.Tensor:
    pad = WIDTH - tensor.shape[-1]
    if pad == 0:
        return tensor
    shape = (*tensor.shape[:-1], pad)
    return torch.cat([tensor, torch.full(shape, value, dtype=tensor.dtype, device=tensor.device)], dim=-1)


@torch.no_grad()
def score_split(ensemble, collator, examples, *, batch_size, context, num_branches):
    """Return (ensemble probs, particle probs, gold, mask) on rank 0, None elsewhere."""
    dataset = as_hf_dataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=lambda records: collator([parse_record(dict(r)) for r in records]))
    parts: dict[str, list[torch.Tensor]] = {"probs": [], "particles": [], "gold": [], "mask": []}
    for batch in loader:
        batch.pop("choice_keys", None)
        batch = {k: v.to(ensemble_device(ensemble)) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = ensemble.local_candidate_logits(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            choice_token_ids=batch["choice_token_ids"])
        local = masked_candidate_log_probs(logits, batch["choice_mask"]).exp()
        particles = gather_detached_branch_tensor(local.contiguous(), context)
        if particles.shape[0] != num_branches:
            raise ValueError(f"gathered {particles.shape[0]} particles, expected {num_branches}")
        if context.is_primary:
            mixture = normalize_then_mean(particles, batch["choice_mask"])
            parts["probs"].append(_pad(mixture, 0.0).cpu())
            parts["particles"].append(_pad(particles, 0.0).permute(1, 0, 2).cpu())
            parts["mask"].append(_pad(batch["choice_mask"], False).cpu())
            parts["gold"].append(batch["gold_indices"].cpu())
    if not context.is_primary:
        return None
    return {key: torch.cat(value) for key, value in parts.items()}


def ensemble_device(ensemble) -> torch.device:
    return next(ensemble.parameters()).device


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    context = init_distributed()
    try:
        output_dir = Path(args.output_dir)
        result_path = output_dir / "result.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite {result_path}")
        limit = args.max_examples_per_split
        config = load_config(args.config)
        num_branches = config.training.num_branches
        local_ids = allocate_contiguous_branches(num_branches, context.world_size)[context.rank]
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        manifest, ensemble, tokenizer, label_ids = load_local_particles(
            config, checkpoint_dir, local_ids, context.device)
        collator = make_class_label_collator(config, tokenizer, label_ids)

        def cut(examples):
            return examples if limit is None else examples[: min(limit, len(examples))]

        splits = [("validation", Path(args.validation), VALIDATION_COUNT)]
        splits += [(name, Path(args.final8_root) / f"{name}.jsonl", FINAL8_DATASET_COUNTS[name])
                   for name in FINAL8_DATASET_NAMES]
        scored = {}
        for name, path, count in splits:
            examples = cut(load_strict_canonical_jsonl(path, count))
            scored[name] = (path, count, score_split(
                ensemble, collator, examples, batch_size=args.batch_size,
                context=context, num_branches=num_branches))
        if not context.is_primary:
            barrier()
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, evaluations = [], []
        for name, (path, count, tensors) in scored.items():
            evaluation = evaluate.ProbabilityEvaluation(tensors["probs"], tensors["gold"], tensors["mask"])
            row = {"dataset": name, "source_path": str(path), "source_sha256": _sha256(path),
                   "full_count": count, **_metric_row(evaluation, ece_bins=args.ece_bins)}
            np.savez_compressed(output_dir / f"per_example_{name}.npz",
                                mixture_probs=tensors["probs"].numpy(),
                                particle_probs=tensors["particles"].float().numpy(),
                                gold_index=tensors["gold"].numpy(),
                                candidate_mask=tensors["mask"].numpy())
            if name == "validation":
                validation_row = row
            else:
                rows.append(row)
                evaluations.append(evaluation)
        protocol = manifest["protocol"]
        result = {
            "schema_version": SCHEMA_VERSION,
            "role": args.role,
            "mode": "smoke" if limit is not None else "full",
            "limit": limit,
            "metric_keys": list(METRIC_KEYS),
            "protocol": {
                "model_id": protocol["model"]["id"],
                "model_revision": protocol["model"]["revision"],
                "prompt_style": protocol["model"]["prompt_style"],
                "objective_beta": protocol["objective"]["beta"],
                "protocol_fingerprint": manifest["protocol_fingerprint"],
                "completed_epoch": manifest["completed_epoch"],
                "optimizer_step": manifest["optimizer_step"],
                "checkpoint_path": str(checkpoint_dir),
                "readout": "label_token_normalize_then_mean",
                "dtype": config.training.dtype,
                "batch_size": args.batch_size,
                "ece_bins": args.ece_bins,
                "world_size": context.world_size,
                "particles_per_rank": len(local_ids),
            },
            "validation": validation_row,
            "final8": {
                "dataset_order": list(FINAL8_DATASET_NAMES),
                "processed_total_count": sum(int(r["count"]) for r in rows),
                "per_dataset": rows,
                "equal_dataset_macro": equal_dataset_macro(rows),
                "pooled": _metric_row(_concatenate(evaluations), ece_bins=args.ece_bins),
            },
        }
        temporary = result_path.with_name(f".result.json.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, result_path)
        print(json.dumps({"role": args.role, "validation_nll": validation_row["nll"],
                          "final8_macro_nll": result["final8"]["equal_dataset_macro"]["nll"]}), flush=True)
        barrier()
        return result
    finally:
        cleanup_distributed()


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
