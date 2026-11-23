"""Torchrun training entry point for branch parallel class label PrO."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .checkpoint import (
    branch_metadata,
    canonical_protocol_payload,
    capture_rng_state,
    load_branch_checkpoint,
    protocol_fingerprint,
    read_rank_state,
    restore_optimizer_scheduler_rng,
    restore_rng_state,
    save_branch_checkpoint,
    save_fork_manifest,
    save_matched_renyi_manifest,
    save_manifest,
    save_rank_state,
    validate_complete_checkpoint,
    validate_objective_switch_protocol,
    validate_matched_renyi_protocol,
)
from .config import (
    Config,
    HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE,
    load_config,
    resolve_temperature,
    validate_config,
    validate_matched_renyi_config,
)
from .data import (
    as_hf_dataset,
    load_jsonl,
    make_class_label_collator,
    parse_record,
    validate_label_tokenization,
)
from .distributed import (
    all_gather_objects,
    all_reduce_sum_detached,
    allocate_contiguous_branches,
    barrier,
    cleanup_distributed,
    derive_process_seed,
    epoch_shuffle_seed,
    gather_detached_branch_tensor,
    init_distributed,
)
from .lora import LoRAEnsemble
from .objective import (
    build_alpha_renyi_objective,
    build_joint_objective,
    finalize_accumulated_gradients,
    gold_label_log_probs,
    lora_prior,
    masked_candidate_log_probs,
    renyi_diagnostics,
    rescue_diagnostics,
    validate_finite_gradients,
)
from .optim import LangevinAdamW
from .optimizer_diagnostics import (
    example_hash,
    is_diagnostic_step,
    particle_optimizer_stats,
    particle_parameter_groups,
    snapshot_parameters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a true joint finite particle class label PrO ensemble"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume")
    resume.add_argument("--objective-switch-resume")
    parser.add_argument("--source-beta", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--matched-reference-config")
    parser.add_argument("--dtype", choices=("bf16", "fp32"))
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--diagnostics-data")
    parser.add_argument("--diagnostics-steps", type=int, default=0)
    parser.add_argument("--diagnostics-max-examples", type=int, default=32)
    parser.add_argument("--diagnostics-jsonl")
    # Read-only extras. Not part of the protocol fingerprint.
    parser.add_argument("--extra-diagnostics", action="store_true")
    parser.add_argument("--dense-diagnostics-window", type=int, default=0)
    parser.add_argument("--dense-diagnostics-every", type=int, default=10)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"), default="disabled")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    return parser


def _resolved_config(args: argparse.Namespace, world_size: int) -> Config:
    config = load_config(args.config).with_beta(args.beta)
    if args.dtype is not None:
        from dataclasses import replace

        config = replace(
            config,
            training=replace(config.training, dtype=args.dtype),
        )
    validate_config(config, world_size=world_size)
    return config


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError("dtype must be bf16 or fp32")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _load_model_and_tokenizer(config: Config, device: torch.device):
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
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if not hasattr(model, "enable_input_require_grads"):
            raise TypeError("model does not support input gradients for checkpointing")
        model.enable_input_require_grads()
    return model, tokenizer, label_token_ids


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_optimizer_steps: int,
):
    from transformers import get_linear_schedule_with_warmup

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )


def optimizer_steps_per_epoch(num_samples: int, batch_size: int) -> int:
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("sample count and batch size must be positive")
    return math.ceil(num_samples / batch_size)


def validate_stop_after_epoch(
    stop_after_epoch: int | None,
    configured_epochs: int,
    *,
    completed_epoch: int = 0,
) -> None:
    if stop_after_epoch is None:
        return
    if not 1 <= stop_after_epoch <= configured_epochs:
        raise ValueError("stop_after_epoch must be within configured epochs")
    if stop_after_epoch <= completed_epoch:
        raise ValueError("stop_after_epoch must follow the resumed checkpoint")


def epoch_indices_to_run(
    completed_epoch: int,
    configured_epochs: int,
    stop_after_epoch: int | None,
) -> range:
    if not 0 <= completed_epoch <= configured_epochs:
        raise ValueError("completed epoch is outside the configured schedule")
    validate_stop_after_epoch(
        stop_after_epoch,
        configured_epochs,
        completed_epoch=completed_epoch,
    )
    final_epoch = stop_after_epoch or configured_epochs
    return range(completed_epoch, final_epoch)


def validate_runtime_args(
    args: argparse.Namespace,
    config: Config,
    *,
    completed_epoch: int = 0,
) -> None:
    validate_stop_after_epoch(
        args.stop_after_epoch,
        config.training.epochs,
        completed_epoch=completed_epoch,
    )
    if args.source_beta is not None and not args.objective_switch_resume:
        raise ValueError("source_beta requires objective_switch_resume")
    if config.objective.name == "alpha_renyi" and not args.matched_reference_config:
        raise ValueError("alpha_renyi requires matched_reference_config")
    if config.objective.name != "alpha_renyi" and args.matched_reference_config:
        raise ValueError("matched_reference_config is only valid for alpha_renyi")
    if args.diagnostics_steps < 0:
        raise ValueError("diagnostics_steps must be nonnegative")
    if args.diagnostics_max_examples <= 0:
        raise ValueError("diagnostics_max_examples must be positive")
    dense_window = getattr(args, "dense_diagnostics_window", 0)
    if dense_window < 0 or getattr(args, "dense_diagnostics_every", 10) <= 0:
        raise ValueError("dense diagnostics window must be >= 0 and cadence > 0")
    if dense_window and not getattr(args, "extra_diagnostics", False):
        raise ValueError("dense diagnostics require --extra-diagnostics")
    if (
        args.diagnostics_data
        and Path(args.diagnostics_data).resolve() == Path(args.data).resolve()
    ):
        raise ValueError("diagnostics data must be held out from training data")
    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if args.wandb_mode == "online" and not project:
        raise ValueError("W&B online mode requires a project")


def _build_epoch_loader(
    dataset,
    *,
    collate_fn,
    micro_batch_size: int,
    shuffle_seed: int,
    epoch_index: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(epoch_shuffle_seed(shuffle_seed, epoch_index))
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _load_local_resume(
    resume_root: str | Path,
    ensemble: LoRAEnsemble,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Config,
    context,
    *,
    local_branch_ids: Sequence[int],
    expected_protocol: dict[str, Any],
    expected_fingerprint: str,
    total_optimizer_steps: int,
    objective_switch: bool = False,
    expected_source_beta: float | None = None,
) -> tuple[int, int, dict[str, Any]]:
    try:
        if objective_switch:
            manifest = validate_complete_checkpoint(resume_root)
            switch = validate_objective_switch_protocol(
                manifest["protocol"],
                expected_protocol,
                expected_source_beta=expected_source_beta,
            )
            if not switch["beta_changed"]:
                raise ValueError("objective switch resume must change beta")
        else:
            manifest = validate_complete_checkpoint(
                resume_root,
                expected_protocol=expected_protocol,
                expected_fingerprint=expected_fingerprint,
            )
            switch = validate_objective_switch_protocol(
                manifest["protocol"],
                expected_protocol,
                expected_source_beta=expected_source_beta,
            )
        completed_epoch = int(manifest["completed_epoch"])
        optimizer_step = int(manifest["optimizer_step"])
        source_protocol = manifest["protocol"]
        source_fingerprint = manifest["protocol_fingerprint"]
        objective = source_protocol["objective"]
        for branch in ensemble.branches:
                load_branch_checkpoint(
                Path(resume_root) / f"branch_{branch.branch_id}",
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
                expected_beta=objective["beta"],
                expected_prior_lambda=config.objective.prior_lambda,
                expected_temperature=objective["temperature"],
                expected_fingerprint=source_fingerprint,
                expected_completed_epoch=completed_epoch,
                expected_optimizer_step=optimizer_step,
                expected_objective_alpha=objective.get("alpha"),
            )
        rank_state = read_rank_state(
            resume_root,
            rank=context.rank,
            expected_completed_epoch=completed_epoch,
            expected_optimizer_step=optimizer_step,
            expected_local_branch_ids=local_branch_ids,
            expected_fingerprint=source_fingerprint,
            expected_total_optimizer_steps=total_optimizer_steps,
        )
        restored_state = {
            "adapter_weights": True,
            **restore_optimizer_scheduler_rng(rank_state, optimizer, scheduler),
        }
        summary = {
            "ok": True,
            "rank": context.rank,
            "completed_epoch": completed_epoch,
            "optimizer_step": optimizer_step,
            "local_branch_ids": list(local_branch_ids),
            "protocol_fingerprint": source_fingerprint,
            "source_protocol_fingerprint": source_fingerprint,
            "target_protocol_fingerprint": expected_fingerprint,
            "restored_state": restored_state,
        }
    except Exception as error:
        summary = {
            "ok": False,
            "rank": context.rank,
            "error": f"{type(error).__name__}: {error}",
        }

    summaries = all_gather_objects(summary, context)
    completed_epoch, optimizer_step = validate_resume_summaries(
        summaries,
        context.world_size,
    )
    restored_state = validate_fork_resume_summaries(summaries, context.world_size)
    lineage = {
        "schema_version": manifest["schema_version"],
        "lineage_type": "prospective_objective_fork",
        "resume_mode": "objective_switch" if objective_switch else "strict",
        "source_checkpoint": str(Path(resume_root).resolve()),
        "source_beta": switch["source_beta"],
        "target_beta": switch["target_beta"],
        "source_protocol": source_protocol,
        "target_protocol": expected_protocol,
        "source_protocol_fingerprint": switch["source_fingerprint"],
        "target_protocol_fingerprint": switch["target_fingerprint"],
        "completed_epoch": completed_epoch,
        "optimizer_step": optimizer_step,
        "total_optimizer_steps": total_optimizer_steps,
        "world_size": context.world_size,
        "ranks": [f"rank_{rank}" for rank in range(context.world_size)],
        "branch_allocation": expected_protocol["training"]["branch_allocation"],
        "restored_state": restored_state,
        "only_beta_differs": switch["only_beta_differs"],
        "beta_changed": switch["beta_changed"],
    }
    return completed_epoch, optimizer_step, lineage


def validate_resume_summaries(
    summaries: Sequence[dict[str, Any]],
    world_size: int,
) -> tuple[int, int]:
    if len(summaries) != world_size:
        raise ValueError("resume metadata world size mismatch")
    errors = [item for item in summaries if not item.get("ok")]
    if errors:
        messages = "; ".join(
            f"rank {item.get('rank')}: {item.get('error')}" for item in errors
        )
        raise ValueError(f"synchronous resume validation failed: {messages}")
    expected_ranks = list(range(world_size))
    if sorted(item["rank"] for item in summaries) != expected_ranks:
        raise ValueError("resume metadata rank set mismatch")
    progress = {
        (
            item["completed_epoch"],
            item["optimizer_step"],
            item["protocol_fingerprint"],
        )
        for item in summaries
    }
    if len(progress) != 1:
        raise ValueError("resume metadata disagrees across ranks")
    completed_epoch, optimizer_step, _ = progress.pop()
    return completed_epoch, optimizer_step


def validate_fork_resume_summaries(
    summaries: Sequence[dict[str, Any]],
    world_size: int,
) -> dict[str, bool]:
    if len(summaries) != world_size:
        raise ValueError("fork resume metadata world size mismatch")
    errors = [item for item in summaries if not item.get("ok")]
    if errors:
        raise ValueError("fork resume contains a local restore error")
    keys = {
        (
            item.get("source_protocol_fingerprint"),
            item.get("target_protocol_fingerprint"),
            json.dumps(item.get("restored_state"), sort_keys=True),
        )
        for item in summaries
    }
    if len(keys) != 1:
        raise ValueError("fork resume metadata disagrees across ranks")
    restored = summaries[0].get("restored_state")
    required = {
        "adapter_weights",
        "optimizer_state",
        "scheduler_state",
        "python_rng",
        "torch_cpu_rng",
        "cuda_rng",
    }
    if not isinstance(restored, dict) or set(restored) != required:
        raise ValueError("fork resume restored state categories mismatch")
    return restored


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _init_wandb(args: argparse.Namespace, protocol: dict[str, Any], context):
    run = None
    try:
        if context.is_primary and args.wandb_mode == "online":
            import wandb

            run = wandb.init(
                project=args.wandb_project or os.environ["WANDB_PROJECT"],
                name=args.wandb_run_name,
                mode="online",
                config=protocol,
            )
        status = {"ok": True, "rank": context.rank}
    except Exception as error:
        status = {
            "ok": False,
            "rank": context.rank,
            "error": f"{type(error).__name__}: {error}",
        }
    statuses = all_gather_objects(status, context)
    failures = [item for item in statuses if not item["ok"]]
    if failures:
        raise RuntimeError(f"W&B initialization failed: {failures}")
    return run


def _wandb_log_preserving_rng(run, payload: dict[str, Any], step: int) -> None:
    if run is None:
        return
    state = capture_rng_state()
    try:
        run.log(payload, step=step)
    finally:
        restore_rng_state(state)


@torch.no_grad()
def _fixed_evaluation_diagnostics(
    ensemble: LoRAEnsemble,
    dataset,
    *,
    collate_fn,
    micro_batch_size: int,
    config: Config,
    context,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    ensemble.eval()
    batches = []
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, context.device)
        logits = ensemble.local_candidate_logits(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            choice_token_ids=batch["choice_token_ids"],
        )
        log_probs = masked_candidate_log_probs(logits, batch["choice_mask"])
        local_gold = gold_label_log_probs(
            log_probs,
            batch["gold_indices"],
            batch["choice_mask"],
        )
        batches.append(gather_detached_branch_tensor(local_gold, context).cpu())
    ensemble.train()
    values = torch.cat(batches, dim=1)
    if config.objective.name == "alpha_renyi":
        assert config.objective.alpha is not None
        return renyi_diagnostics(values, config.objective.alpha).as_dict()
    return rescue_diagnostics(values, config.objective.beta).as_dict()


def _build_training_objective(
    config: Config,
    *,
    local_gold_log_probs: torch.Tensor,
    all_gold_log_probs: torch.Tensor,
    local_branch_indices: Sequence[int],
    local_parameters,
    num_training_samples: int,
):
    common = {
        "local_gold_log_probs": local_gold_log_probs,
        "all_gold_log_probs": all_gold_log_probs,
        "local_branch_indices": local_branch_indices,
        "local_parameters": local_parameters,
        "total_branches": config.training.num_branches,
        "num_training_samples": num_training_samples,
        "prior_lambda": config.objective.prior_lambda,
        "include_prior": False,
    }
    if config.objective.name == "alpha_renyi":
        assert config.objective.alpha is not None
        return build_alpha_renyi_objective(alpha=config.objective.alpha, **common)
    return build_joint_objective(beta=config.objective.beta, **common)


def _training_diagnostics(config: Config, values: torch.Tensor) -> dict[str, Any]:
    if config.objective.name == "alpha_renyi":
        assert config.objective.alpha is not None
        return renyi_diagnostics(values, config.objective.alpha).as_dict()
    return rescue_diagnostics(values, config.objective.beta).as_dict()


def build_training_diagnostic_record(
    config: Config,
    *,
    epoch: int,
    step: int,
    num_examples: int,
    report: dict[str, Any],
    all_gold_log_probs: torch.Tensor,
) -> dict[str, Any]:
    return {
        "scope": "training_minibatch",
        "epoch": epoch,
        "step": step,
        "objective": config.objective.name,
        "alpha": config.objective.alpha,
        "beta": config.objective.beta,
        "num_examples": num_examples,
        "mixture_nll": report["mixture_nll"],
        "branch_ce": report["branch_ce"],
        "data_loss": report["data_loss"],
        "prior": report["prior"],
        "total_loss": report["total_loss"],
        "candidate_probability_max_error": report[
            "candidate_probability_max_error"
        ],
        "finite_gradients": report["finite_gradients"],
        **_training_diagnostics(config, all_gold_log_probs),
    }


def _save_local_branches(
    epoch_root: str | Path,
    ensemble: LoRAEnsemble,
    config: Config,
    *,
    temperature: float,
    fingerprint: str,
    completed_epoch: int,
    optimizer_step: int,
) -> None:
    for branch in ensemble.branches:
        state = branch.adapter_state_dict()
        metadata = branch_metadata(
            model_id=config.model.id,
            revision=config.model.revision,
            label_readout=config.model.label_readout,
            prompt_style=config.model.prompt_style,
            branch_id=branch.branch_id,
            seed=config.training.branch_seeds[branch.branch_id],
            target_modules=config.lora.target_modules,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            objective=config.objective.name,
            beta=config.objective.beta,
            prior_lambda=config.objective.prior_lambda,
            temperature=temperature,
            fingerprint=fingerprint,
            completed_epoch=completed_epoch,
            optimizer_step=optimizer_step,
            state=state,
            objective_alpha=config.objective.alpha,
        )
        save_branch_checkpoint(epoch_root, branch, metadata)


def _prepare_epoch_checkpoint(epoch_root: Path, context) -> None:
    statuses = all_gather_objects(epoch_root.exists(), context)
    if any(statuses):
        raise FileExistsError(
            f"immutable checkpoint already exists: {epoch_root.name}"
        )
    if context.is_primary:
        epoch_root.mkdir(parents=False, exist_ok=False)
    barrier()


def run(args: argparse.Namespace) -> None:
    context = init_distributed()
    wandb_run = None
    try:
        config = _resolved_config(args, context.world_size)
        validate_runtime_args(args, config)
        reference_config = None
        if args.matched_reference_config:
            reference_config = load_config(args.matched_reference_config)
            validate_config(reference_config, world_size=context.world_size)
            validate_matched_renyi_config(reference_config, config)
        examples = load_jsonl(args.data)
        temperature = resolve_temperature(config, len(examples))
        allocations = allocate_contiguous_branches(
            config.training.num_branches,
            context.world_size,
        )
        local_branch_ids = allocations[context.rank]

        base_model, tokenizer, label_token_ids = _load_model_and_tokenizer(
            config,
            context.device,
        )
        ensemble = LoRAEnsemble(
            base_model,
            local_branch_ids,
            config.training.branch_seeds,
            config.lora.target_modules,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
        ).to(context.device)
        trainable_parameters = tuple(ensemble.trainable_parameters())
        optimizer = LangevinAdamW(
            trainable_parameters,
            lr=config.training.learning_rate,
            temperature=temperature,
        )
        steps_per_epoch = optimizer_steps_per_epoch(
            len(examples),
            config.training.batch_size,
        )
        total_optimizer_steps = steps_per_epoch * config.training.epochs
        scheduler = _build_scheduler(
            optimizer,
            warmup_steps=config.training.warmup_steps,
            total_optimizer_steps=total_optimizer_steps,
        )
        protocol = canonical_protocol_payload(
            config,
            num_training_samples=len(examples),
            data_path=args.data,
            diagnostics_data_path=args.diagnostics_data,
            diagnostics_steps=args.diagnostics_steps,
            diagnostics_max_examples=args.diagnostics_max_examples,
            world_size=context.world_size,
            branch_allocation=allocations,
            temperature=temperature,
            total_optimizer_steps=total_optimizer_steps,
        )
        fingerprint = protocol_fingerprint(protocol)
        matched_protocol = None
        if reference_config is not None:
            reference_protocol = canonical_protocol_payload(
                reference_config,
                num_training_samples=len(examples),
                data_path=args.data,
                diagnostics_data_path=args.diagnostics_data,
                diagnostics_steps=args.diagnostics_steps,
                diagnostics_max_examples=args.diagnostics_max_examples,
                world_size=context.world_size,
                branch_allocation=allocations,
                temperature=temperature,
                total_optimizer_steps=total_optimizer_steps,
            )
            matched_protocol = validate_matched_renyi_protocol(
                reference_protocol,
                protocol,
                expected_alpha=config.objective.alpha,
            )
            matched_protocol.update(
                {
                    "schema_version": protocol["schema_version"],
                    "reference_config": str(args.matched_reference_config),
                    "treatment_config": str(args.config),
                    "reference_protocol": reference_protocol,
                    "treatment_protocol": protocol,
                }
            )
        wandb_run = _init_wandb(args, protocol, context)

        collator = make_class_label_collator(config, tokenizer, label_token_ids)
        dataset = as_hf_dataset(examples)

        extra_diagnostics = bool(getattr(args, "extra_diagnostics", False))

        def collate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
            batch = collator([parse_record(dict(record)) for record in records])
            if extra_diagnostics:
                batch["example_hashes"] = [example_hash(record) for record in records]
            return batch

        def collate_fixed(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
            return collator([parse_record(dict(record)) for record in records])

        fixed_dataset = None
        if args.diagnostics_data:
            fixed_examples = load_jsonl(args.diagnostics_data)
            fixed_dataset = as_hf_dataset(
                fixed_examples[: args.diagnostics_max_examples]
            )

        start_epoch = 0
        global_step = 0
        lineage = None
        resume_root = args.resume or args.objective_switch_resume
        if resume_root:
            start_epoch, global_step, lineage = _load_local_resume(
                resume_root,
                ensemble,
                optimizer,
                scheduler,
                config,
                context,
                local_branch_ids=local_branch_ids,
                expected_protocol=protocol,
                expected_fingerprint=fingerprint,
                total_optimizer_steps=total_optimizer_steps,
                objective_switch=bool(args.objective_switch_resume),
                expected_source_beta=args.source_beta,
            )
            validate_runtime_args(args, config, completed_epoch=start_epoch)
        else:
            process_seed = derive_process_seed(
                local_branch_ids,
                config.training.branch_seeds,
            )
            random.seed(process_seed)
            torch.manual_seed(process_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(process_seed)

        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        if matched_protocol is not None:
            if context.is_primary:
                save_matched_renyi_manifest(output_root, matched_protocol)
            barrier()
        if lineage is not None:
            if context.is_primary:
                save_fork_manifest(output_root, lineage)
            barrier()
        checkpoint_root = output_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        diagnostics_path = (
            Path(args.diagnostics_jsonl)
            if args.diagnostics_jsonl
            else output_root / "diagnostics.jsonl"
        )
        accumulation = config.training.gradient_accumulation_steps
        start_step = global_step
        particle_groups = particle_parameter_groups(ensemble) if extra_diagnostics else []

        for epoch_index in epoch_indices_to_run(
            start_epoch,
            config.training.epochs,
            args.stop_after_epoch,
        ):
            loader = _build_epoch_loader(
                dataset,
                collate_fn=collate_records,
                micro_batch_size=config.training.micro_batch_size,
                shuffle_seed=config.training.shuffle_seed,
                epoch_index=epoch_index,
            )
            ensemble.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0
            accumulated_micro_batches = 0
            mixture_nll_sum = torch.zeros((), device=context.device)
            branch_ce_sum = torch.zeros((), device=context.device)
            data_loss_sum = torch.zeros((), device=context.device)
            candidate_normalization_error = 0.0
            step_hashes: list[str] = []
            last_micro_hashes: list[str] = []

            for micro_index, cpu_batch in enumerate(loader):
                if extra_diagnostics:
                    last_micro_hashes = cpu_batch.pop("example_hashes")
                    step_hashes.extend(last_micro_hashes)
                batch = _move_batch(cpu_batch, context.device)
                candidate_logits = ensemble.local_candidate_logits(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    choice_token_ids=batch["choice_token_ids"],
                )
                candidate_log_probs = masked_candidate_log_probs(
                    candidate_logits,
                    batch["choice_mask"],
                )
                normalized = candidate_log_probs.exp().masked_fill(
                    ~batch["choice_mask"].unsqueeze(0),
                    0.0,
                ).sum(dim=-1)
                candidate_normalization_error = max(
                    candidate_normalization_error,
                    float((normalized - 1.0).abs().max()),
                )
                local_gold = gold_label_log_probs(
                    candidate_log_probs,
                    batch["gold_indices"],
                    batch["choice_mask"],
                )
                all_gold = gather_detached_branch_tensor(local_gold, context)
                objective = _build_training_objective(
                    config,
                    local_gold_log_probs=local_gold,
                    all_gold_log_probs=all_gold,
                    local_branch_indices=local_branch_ids,
                    local_parameters=trainable_parameters,
                    num_training_samples=len(examples),
                )
                micro_samples = int(local_gold.shape[1])
                (objective.data_backward_loss * micro_samples).backward()
                accumulated_samples += micro_samples
                accumulated_micro_batches += 1
                mixture_nll_sum += objective.exact_mixture_nll * micro_samples
                branch_ce_sum += objective.branch_ce * micro_samples
                data_loss_sum += objective.data_loss_value * micro_samples

                group_end = accumulated_micro_batches == accumulation
                final_batch = micro_index + 1 == len(loader)
                if group_end or final_batch:
                    prior = lora_prior(
                        trainable_parameters,
                        config.objective.prior_lambda,
                        config.training.num_branches,
                        len(examples),
                    )
                    finalize_accumulated_gradients(
                        trainable_parameters,
                        accumulated_samples,
                        prior,
                    )
                    validate_finite_gradients(trainable_parameters)
                    normalization_errors = all_gather_objects(
                        candidate_normalization_error,
                        context,
                    )
                    global_normalization_error = max(normalization_errors)
                    log_step, log_trigger = is_diagnostic_step(
                        global_step + 1,
                        cadence=args.diagnostics_steps,
                        start_step=start_step,
                        dense_window=getattr(args, "dense_diagnostics_window", 0),
                        dense_every=getattr(args, "dense_diagnostics_every", 10),
                    )
                    extra_now = extra_diagnostics and log_step
                    before = snapshot_parameters(particle_groups) if extra_now else None
                    optimizer.step()
                    particle_rows = None
                    if extra_now:
                        local_rows = particle_optimizer_stats(
                            optimizer, particle_groups, before
                        )
                        del before
                        gathered_rows = all_gather_objects(local_rows, context)
                        particle_rows = sorted(
                            (row for rows in gathered_rows for row in rows),
                            key=lambda row: row["particle"],
                        )
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    mixture_nll = mixture_nll_sum / accumulated_samples
                    branch_ce = branch_ce_sum / accumulated_samples
                    data_loss = data_loss_sum / accumulated_samples
                    global_prior = all_reduce_sum_detached(prior, context)
                    if context.is_primary:
                        report = {
                            "epoch": epoch_index + 1,
                            "step": global_step,
                            "objective": config.objective.name,
                            "alpha": config.objective.alpha,
                            "beta": config.objective.beta,
                            "mixture_nll": float(mixture_nll),
                            "branch_ce": float(branch_ce),
                            "data_loss": float(data_loss),
                            "prior": float(global_prior),
                            "total_loss": float(data_loss + global_prior),
                            "candidate_probability_max_error": global_normalization_error,
                            "finite_gradients": True,
                        }
                        print(json.dumps(report, allow_nan=False), flush=True)
                        _wandb_log_preserving_rng(wandb_run, report, global_step)
                        if log_step and (
                            log_trigger == "cadence" or extra_diagnostics
                        ):
                            diagnostic_report = build_training_diagnostic_record(
                                config,
                                epoch=epoch_index + 1,
                                step=global_step,
                                num_examples=int(all_gold.shape[1]),
                                report=report,
                                all_gold_log_probs=all_gold,
                            )
                            if extra_diagnostics:
                                diagnostic_report.update(
                                    diagnostic_trigger=log_trigger,
                                    example_hashes_last_microbatch=list(
                                        last_micro_hashes
                                    ),
                                    example_hashes_step=list(step_hashes),
                                    particle_optimizer=particle_rows,
                                )
                            _append_jsonl(diagnostics_path, diagnostic_report)
                            _wandb_log_preserving_rng(
                                wandb_run,
                                diagnostic_report,
                                global_step,
                            )
                    accumulated_samples = 0
                    accumulated_micro_batches = 0
                    mixture_nll_sum.zero_()
                    branch_ce_sum.zero_()
                    data_loss_sum.zero_()
                    candidate_normalization_error = 0.0
                    step_hashes = []

            completed_epoch = epoch_index + 1
            epoch_root = checkpoint_root / f"epoch_{completed_epoch:04d}"
            _prepare_epoch_checkpoint(epoch_root, context)
            _save_local_branches(
                epoch_root,
                ensemble,
                config,
                temperature=temperature,
                fingerprint=fingerprint,
                completed_epoch=completed_epoch,
                optimizer_step=global_step,
            )
            save_rank_state(
                epoch_root,
                rank=context.rank,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epoch=completed_epoch,
                optimizer_step=global_step,
                local_branch_ids=local_branch_ids,
                fingerprint=fingerprint,
                total_optimizer_steps=total_optimizer_steps,
            )
            barrier()
            if context.is_primary:
                save_manifest(
                    epoch_root,
                    protocol=protocol,
                    fingerprint=fingerprint,
                    completed_epoch=completed_epoch,
                    optimizer_step=global_step,
                )
            barrier()
            if fixed_dataset is not None:
                fixed_report = {
                    "scope": "fixed_evaluation",
                    "epoch": completed_epoch,
                    "step": global_step,
                    "objective": config.objective.name,
                    "alpha": config.objective.alpha,
                    "beta": config.objective.beta,
                    "num_examples": len(fixed_dataset),
                    **_fixed_evaluation_diagnostics(
                        ensemble,
                        fixed_dataset,
                        collate_fn=collate_fixed,
                        micro_batch_size=config.training.micro_batch_size,
                        config=config,
                        context=context,
                    ),
                }
                if context.is_primary:
                    _append_jsonl(diagnostics_path, fixed_report)
                    _wandb_log_preserving_rng(
                        wandb_run,
                        fixed_report,
                        global_step,
                    )
            if args.stop_after_epoch == completed_epoch:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed()


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
