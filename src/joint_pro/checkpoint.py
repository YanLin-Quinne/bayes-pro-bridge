"""Immutable epoch-boundary checkpoints with protocol fingerprints."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Callable, Mapping, Sequence
import uuid

import torch
from safetensors.torch import load_file, save_file

from .config import Config, SCHEMA_VERSION, relative_display_path
from .lora import LoRABranch


REQUIRED_BRANCH_METADATA = {
    "schema_version",
    "model_id",
    "revision",
    "label_readout",
    "prompt_style",
    "branch_id",
    "seed",
    "target_modules",
    "lora",
    "objective",
    "beta",
    "prior_lambda",
    "temperature",
    "protocol_fingerprint",
    "completed_epoch",
    "optimizer_step",
    "tensor_shapes",
}


def canonical_protocol_payload(
    config: Config,
    *,
    num_training_samples: int,
    data_path: str | Path | None = None,
    diagnostics_data_path: str | Path | None = None,
    diagnostics_steps: int = 0,
    diagnostics_max_examples: int = 0,
    world_size: int,
    branch_allocation: Sequence[Sequence[int]],
    temperature: float,
    total_optimizer_steps: int,
) -> dict[str, Any]:
    if (
        num_training_samples <= 0
        or world_size <= 0
        or total_optimizer_steps <= 0
        or diagnostics_steps < 0
        or diagnostics_max_examples < 0
    ):
        raise ValueError("invalid protocol runtime dimensions")
    allocation = [list(map(int, branch_ids)) for branch_ids in branch_allocation]
    if len(allocation) != world_size:
        raise ValueError("branch allocation does not match world size")
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "id": config.model.id,
            "revision": config.model.revision,
            "labels": list(config.model.labels),
            "label_readout": config.model.label_readout,
            "prompt_style": config.model.prompt_style,
        },
        "lora": {
            "target_modules": list(config.lora.target_modules),
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
        },
        "objective": {
            "name": config.objective.name,
            "beta": config.objective.beta,
            "prior_lambda": config.objective.prior_lambda,
            "temperature_mode": config.objective.temperature_mode,
            "temperature": temperature,
            **(
                {}
                if config.objective.alpha is None
                else {"alpha": config.objective.alpha}
            ),
        },
        "diagnostics": {
            "data_path": (
                None
                if diagnostics_data_path is None
                else str(Path(diagnostics_data_path).resolve())
            ),
            "steps": int(diagnostics_steps),
            "max_examples": int(diagnostics_max_examples),
            "training_scope": "training_minibatch",
            "fixed_scope": "fixed_evaluation",
        },
        "training": {
            "paper_mode": config.training.paper_mode,
            "num_branches": config.training.num_branches,
            "branch_seeds": list(config.training.branch_seeds),
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "micro_batch_size": config.training.micro_batch_size,
            "learning_rate": config.training.learning_rate,
            "warmup_steps": config.training.warmup_steps,
            "scheduler": "linear",
            "dtype": config.training.dtype,
            "max_length": config.training.max_length,
            "shuffle_seed": config.training.shuffle_seed,
            "gradient_checkpointing": config.training.gradient_checkpointing,
            "eval_steps": config.training.eval_steps,
            "save_steps": config.training.save_steps,
            "padding_side": config.training.padding_side,
            "attn_implementation": config.training.attn_implementation,
            "num_training_samples": num_training_samples,
            "data_path": (
                None if data_path is None else str(Path(data_path).resolve())
            ),
            "total_optimizer_steps": total_optimizer_steps,
            "world_size": world_size,
            "branch_allocation": allocation,
        },
    }


def protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_objective_switch_protocol(
    source_protocol: Mapping[str, Any],
    target_protocol: Mapping[str, Any],
    *,
    expected_source_beta: float | None = None,
) -> dict[str, Any]:
    source = deepcopy(dict(source_protocol))
    target = deepcopy(dict(target_protocol))
    for name, protocol in (("source", source), ("target", target)):
        objective = protocol.get("objective")
        if not isinstance(objective, dict) or "beta" not in objective:
            raise ValueError(f"{name} protocol beta is missing")
        beta = float(objective["beta"])
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError(f"{name} protocol beta is invalid")
    source_beta = float(source["objective"]["beta"])
    target_beta = float(target["objective"]["beta"])
    if expected_source_beta is not None and source_beta != float(expected_source_beta):
        raise ValueError("source beta does not match checkpoint")
    source["objective"]["beta"] = target_beta
    if source != target:
        raise ValueError("objective switch changes a field other than objective.beta")
    return {
        "source_beta": source_beta,
        "target_beta": target_beta,
        "source_fingerprint": protocol_fingerprint(source_protocol),
        "target_fingerprint": protocol_fingerprint(target_protocol),
        "beta_changed": source_beta != target_beta,
        "only_beta_differs": True,
    }


def validate_matched_renyi_protocol(
    reference_protocol: Mapping[str, Any],
    treatment_protocol: Mapping[str, Any],
    *,
    expected_alpha: float | None = None,
) -> dict[str, Any]:
    reference = deepcopy(dict(reference_protocol))
    treatment = deepcopy(dict(treatment_protocol))
    for name, protocol in (("reference", reference), ("treatment", treatment)):
        objective = protocol.get("objective")
        if not isinstance(objective, dict) or "name" not in objective:
            raise ValueError(f"{name} protocol objective is invalid")
    objective = treatment["objective"]
    if objective["name"] != "alpha_renyi":
        raise ValueError("treatment objective must be alpha_renyi")
    alpha = objective.get("alpha")
    if alpha is None or not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("treatment protocol alpha is invalid")
    alpha = float(alpha)
    if expected_alpha is not None and alpha != float(expected_alpha):
        raise ValueError("treatment alpha does not match expected alpha")
    reference_name = reference["objective"]["name"]
    reference_alpha = reference["objective"].pop("alpha", None)
    treatment["objective"].pop("alpha", None)
    treatment["objective"]["name"] = reference_name
    if reference != treatment:
        raise ValueError(
            "matched Rényi protocol changes a field other than objective kind and alpha"
        )
    return {
        "status": "PASS",
        "reference_objective": reference_name,
        "reference_alpha": reference_alpha,
        "treatment_objective": "alpha_renyi",
        "treatment_alpha": alpha,
        "reference_fingerprint": protocol_fingerprint(reference_protocol),
        "treatment_fingerprint": protocol_fingerprint(treatment_protocol),
        "only_objective_kind_and_alpha_differ": True,
    }


def validate_protocol_config(payload: Mapping[str, Any], config: Config) -> None:
    expected = {
        "model": {
            "id": config.model.id,
            "revision": config.model.revision,
            "labels": list(config.model.labels),
            "label_readout": config.model.label_readout,
            "prompt_style": config.model.prompt_style,
        },
        "lora": {
            "target_modules": list(config.lora.target_modules),
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
        },
    }
    for section, values in expected.items():
        if payload.get(section) != values:
            raise ValueError(f"checkpoint protocol {section} mismatch")
    objective = payload.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("checkpoint protocol objective is invalid")
    expected_objective = {
        "name": config.objective.name,
        "beta": config.objective.beta,
        "prior_lambda": config.objective.prior_lambda,
        "temperature_mode": config.objective.temperature_mode,
    }
    for key, value in expected_objective.items():
        if objective.get(key) != value:
            raise ValueError(f"checkpoint protocol objective {key} mismatch")
    if objective.get("alpha") != config.objective.alpha:
        raise ValueError("checkpoint protocol objective alpha mismatch")
    if (
        config.objective.temperature_mode == "manual"
        and objective.get("temperature") != config.objective.temperature
    ):
        raise ValueError("checkpoint protocol objective temperature mismatch")
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint protocol training is invalid")
    expected_training = {
        "paper_mode": config.training.paper_mode,
        "num_branches": config.training.num_branches,
        "branch_seeds": list(config.training.branch_seeds),
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "micro_batch_size": config.training.micro_batch_size,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "scheduler": "linear",
        "dtype": config.training.dtype,
        "max_length": config.training.max_length,
        "shuffle_seed": config.training.shuffle_seed,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "eval_steps": config.training.eval_steps,
        "save_steps": config.training.save_steps,
        "padding_side": config.training.padding_side,
        "attn_implementation": config.training.attn_implementation,
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise ValueError(f"checkpoint protocol training {key} mismatch")


def branch_metadata(
    *,
    model_id: str,
    revision: str,
    label_readout: str,
    prompt_style: str,
    branch_id: int,
    seed: int,
    target_modules: Sequence[str],
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    objective: str,
    beta: float,
    prior_lambda: float,
    temperature: float,
    fingerprint: str,
    completed_epoch: int,
    optimizer_step: int,
    state: Mapping[str, torch.Tensor],
    objective_alpha: float | None = None,
) -> dict[str, Any]:
    numeric = (beta, prior_lambda, temperature, lora_alpha, lora_dropout)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("checkpoint metadata contains non finite values")
    if completed_epoch <= 0 or optimizer_step <= 0:
        raise ValueError("checkpoint progress must be positive")
    if objective_alpha is not None and not math.isfinite(float(objective_alpha)):
        raise ValueError("checkpoint objective alpha is non finite")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision,
        "label_readout": label_readout,
        "prompt_style": prompt_style,
        "branch_id": int(branch_id),
        "seed": int(seed),
        "target_modules": list(target_modules),
        "lora": {
            "rank": int(lora_rank),
            "alpha": float(lora_alpha),
            "dropout": float(lora_dropout),
        },
        "objective": objective,
        "beta": float(beta),
        "prior_lambda": float(prior_lambda),
        "temperature": float(temperature),
        "protocol_fingerprint": fingerprint,
        "completed_epoch": int(completed_epoch),
        "optimizer_step": int(optimizer_step),
        "tensor_shapes": {key: list(value.shape) for key, value in state.items()},
    }
    if objective_alpha is not None:
        payload["objective_alpha"] = float(objective_alpha)
    return payload


def _validate_state(state: Mapping[str, torch.Tensor]) -> None:
    if not state:
        raise ValueError("adapter state is empty")
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("adapter state must map strings to tensors")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non finite adapter weights in {key}")


def _atomic_directory(target: Path, writer: Callable[[Path], None]) -> None:
    if target.exists():
        raise FileExistsError(f"immutable checkpoint path already exists: {target.name}")
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        writer(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _atomic_file(target: Path, writer: Callable[[Path], None]) -> None:
    if target.exists():
        raise FileExistsError(f"immutable checkpoint path already exists: {target.name}")
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        writer(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_branch_checkpoint(
    epoch_root: str | Path,
    branch: LoRABranch,
    metadata: Mapping[str, Any],
) -> str:
    state = branch.adapter_state_dict()
    _validate_state(state)
    branch_id = int(metadata["branch_id"])
    if branch_id != branch.branch_id:
        raise ValueError("metadata branch ID does not match adapter")
    branch_name = f"branch_{branch_id}"
    branch_dir = Path(epoch_root) / branch_name

    def write(directory: Path) -> None:
        save_file(state, str(directory / "adapter.safetensors"))
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(dict(metadata), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    _atomic_directory(branch_dir, write)
    return relative_display_path(branch_name)


def read_metadata(branch_dir: str | Path) -> dict[str, Any]:
    path = Path(branch_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    missing = REQUIRED_BRANCH_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"checkpoint metadata is missing fields: {sorted(missing)}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    return metadata


def load_branch_checkpoint(
    branch_dir: str | Path,
    branch: LoRABranch,
    *,
    expected_model_id: str,
    expected_revision: str,
    expected_label_readout: str,
    expected_prompt_style: str,
    expected_seed: int,
    expected_target_modules: Sequence[str],
    expected_lora_rank: int,
    expected_lora_alpha: float,
    expected_lora_dropout: float,
    expected_objective: str,
    expected_beta: float,
    expected_prior_lambda: float,
    expected_temperature: float,
    expected_fingerprint: str,
    expected_completed_epoch: int,
    expected_optimizer_step: int,
    expected_objective_alpha: float | None = None,
) -> dict[str, Any]:
    directory = Path(branch_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"missing checkpoint branch {branch.branch_id}")
    metadata = read_metadata(directory)
    expected = {
        "model_id": expected_model_id,
        "revision": expected_revision,
        "label_readout": expected_label_readout,
        "prompt_style": expected_prompt_style,
        "branch_id": branch.branch_id,
        "seed": int(expected_seed),
        "target_modules": list(expected_target_modules),
        "objective": expected_objective,
        "beta": float(expected_beta),
        "prior_lambda": float(expected_prior_lambda),
        "temperature": float(expected_temperature),
        "protocol_fingerprint": expected_fingerprint,
        "completed_epoch": int(expected_completed_epoch),
        "optimizer_step": int(expected_optimizer_step),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    if expected_objective_alpha is not None and metadata.get("objective_alpha") != float(
        expected_objective_alpha
    ):
        raise ValueError("checkpoint objective_alpha mismatch")
    expected_lora = {
        "rank": int(expected_lora_rank),
        "alpha": float(expected_lora_alpha),
        "dropout": float(expected_lora_dropout),
    }
    if metadata.get("lora") != expected_lora:
        raise ValueError("checkpoint LoRA settings mismatch")
    tensor_path = directory / "adapter.safetensors"
    if not tensor_path.is_file():
        raise FileNotFoundError("missing adapter.safetensors")
    state = load_file(str(tensor_path), device="cpu")
    _validate_state(state)
    expected_state = branch.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("checkpoint tensor key mismatch")
    for key, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected_state[key].shape):
            raise ValueError(f"checkpoint shape mismatch for {key}")
        if metadata["tensor_shapes"].get(key) != list(tensor.shape):
            raise ValueError(f"metadata shape mismatch for {key}")
    branch.load_adapter_state_dict(state)
    restored = branch.adapter_state_dict()
    if not _state_equal(restored, state):
        raise ValueError("adapter state restore mismatch")
    return metadata


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    cpu_state = state.get("cpu")
    cuda_state = state.get("cuda")
    if (
        not isinstance(python_state, tuple)
        or not isinstance(cpu_state, torch.Tensor)
        or not isinstance(cuda_state, list)
    ):
        raise ValueError("checkpoint RNG state is invalid")
    random.setstate(python_state)
    torch.set_rng_state(cpu_state)
    if cuda_state:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def save_rank_state(
    epoch_root: str | Path,
    *,
    rank: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_epoch: int,
    optimizer_step: int,
    local_branch_ids: Sequence[int],
    fingerprint: str,
    total_optimizer_steps: int,
) -> str:
    rank_name = f"rank_{rank}"
    rank_dir = Path(epoch_root) / rank_name

    def write(directory: Path) -> None:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "rank": int(rank),
            "completed_epoch": int(completed_epoch),
            "optimizer_step": int(optimizer_step),
            "local_branch_ids": list(map(int, local_branch_ids)),
            "protocol_fingerprint": fingerprint,
            "total_optimizer_steps": int(total_optimizer_steps),
        }
        payload = {
            **metadata,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(),
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        torch.save(payload, directory / "state.pt")

    _atomic_directory(rank_dir, write)
    return relative_display_path(rank_name)


def read_rank_metadata(epoch_root: str | Path, rank: int) -> dict[str, Any]:
    path = Path(epoch_root) / f"rank_{rank}" / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint rank metadata {rank}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_rank_identity(
    payload: Mapping[str, Any],
    *,
    rank: int,
    expected_completed_epoch: int,
    expected_optimizer_step: int,
    expected_local_branch_ids: Sequence[int],
    expected_fingerprint: str,
    expected_total_optimizer_steps: int,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "rank": int(rank),
        "completed_epoch": int(expected_completed_epoch),
        "optimizer_step": int(expected_optimizer_step),
        "local_branch_ids": list(map(int, expected_local_branch_ids)),
        "protocol_fingerprint": expected_fingerprint,
        "total_optimizer_steps": int(expected_total_optimizer_steps),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint rank state {key} mismatch")


def read_rank_state(
    epoch_root: str | Path,
    *,
    rank: int,
    expected_completed_epoch: int,
    expected_optimizer_step: int,
    expected_local_branch_ids: Sequence[int],
    expected_fingerprint: str,
    expected_total_optimizer_steps: int,
) -> dict[str, Any]:
    path = Path(epoch_root) / f"rank_{rank}" / "state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint rank state {rank}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_rank_identity(
        payload,
        rank=rank,
        expected_completed_epoch=expected_completed_epoch,
        expected_optimizer_step=expected_optimizer_step,
        expected_local_branch_ids=expected_local_branch_ids,
        expected_fingerprint=expected_fingerprint,
        expected_total_optimizer_steps=expected_total_optimizer_steps,
    )
    for key in ("optimizer", "scheduler", "rng"):
        if key not in payload:
            raise ValueError(f"checkpoint rank state is missing {key}")
    if set(payload["rng"]) != {"python", "cpu", "cuda"}:
        raise ValueError("checkpoint RNG state categories are incomplete")
    if payload["scheduler"].get("last_epoch") != expected_optimizer_step:
        raise ValueError("checkpoint scheduler step mismatch")
    return payload


def restore_optimizer_scheduler_rng(
    payload: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, bool]:
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    if not _state_equal(optimizer.state_dict(), payload["optimizer"]):
        raise ValueError("optimizer state restore mismatch")
    if not _state_equal(scheduler.state_dict(), payload["scheduler"]):
        raise ValueError("scheduler state restore mismatch")
    restored_rng = capture_rng_state()
    if not _state_equal(restored_rng, payload["rng"]):
        raise ValueError("RNG state restore mismatch")
    return {
        "optimizer_state": True,
        "scheduler_state": True,
        "python_rng": True,
        "torch_cpu_rng": True,
        "cuda_rng": bool(payload["rng"]["cuda"]),
    }


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def save_fork_manifest(output_root: str | Path, payload: Mapping[str, Any]) -> Path:
    required = {
        "schema_version",
        "lineage_type",
        "resume_mode",
        "source_checkpoint",
        "source_beta",
        "target_beta",
        "source_protocol",
        "target_protocol",
        "source_protocol_fingerprint",
        "target_protocol_fingerprint",
        "completed_epoch",
        "optimizer_step",
        "total_optimizer_steps",
        "world_size",
        "ranks",
        "branch_allocation",
        "restored_state",
        "only_beta_differs",
        "beta_changed",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"fork manifest is missing fields: {sorted(missing)}")
    path = Path(output_root) / "fork_manifest.json"

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    _atomic_file(path, write)
    return path


def save_matched_renyi_manifest(
    output_root: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    required = {
        "schema_version",
        "status",
        "reference_config",
        "treatment_config",
        "reference_protocol",
        "treatment_protocol",
        "reference_fingerprint",
        "treatment_fingerprint",
        "reference_objective",
        "treatment_objective",
        "treatment_alpha",
        "only_objective_kind_and_alpha_differ",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"matched Rényi manifest is missing fields: {sorted(missing)}")
    validation = validate_matched_renyi_protocol(
        payload["reference_protocol"],
        payload["treatment_protocol"],
        expected_alpha=float(payload["treatment_alpha"]),
    )
    for key in (
        "status",
        "reference_fingerprint",
        "treatment_fingerprint",
        "reference_objective",
        "treatment_objective",
        "treatment_alpha",
        "only_objective_kind_and_alpha_differ",
    ):
        if payload.get(key) != validation[key]:
            raise ValueError(f"matched Rényi manifest {key} mismatch")
    path = Path(output_root) / "matched_protocol.json"

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    _atomic_file(path, write)
    return path


def save_manifest(
    epoch_root: str | Path,
    *,
    protocol: Mapping[str, Any],
    fingerprint: str,
    completed_epoch: int,
    optimizer_step: int,
) -> None:
    if protocol_fingerprint(protocol) != fingerprint:
        raise ValueError("manifest fingerprint does not match protocol")
    training = protocol["training"]
    num_branches = int(training["num_branches"])
    world_size = int(training["world_size"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_type": "epoch_boundary",
        "completed_epoch": int(completed_epoch),
        "optimizer_step": int(optimizer_step),
        "world_size": world_size,
        "branch_allocation": training["branch_allocation"],
        "branches": [f"branch_{branch_id}" for branch_id in range(num_branches)],
        "ranks": [f"rank_{rank}" for rank in range(world_size)],
        "total_optimizer_steps": int(training["total_optimizer_steps"]),
        "protocol": dict(protocol),
        "protocol_fingerprint": fingerprint,
    }
    path = Path(epoch_root) / "manifest.json"

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    _atomic_file(path, write)


def validate_complete_checkpoint(
    epoch_root: str | Path,
    *,
    expected_protocol: Mapping[str, Any] | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    root = Path(epoch_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing checkpoint manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint manifest schema")
    if manifest.get("checkpoint_type") != "epoch_boundary":
        raise ValueError("checkpoint is not an epoch boundary")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("checkpoint protocol is invalid")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint protocol schema mismatch")
    fingerprint = protocol_fingerprint(protocol)
    if manifest.get("protocol_fingerprint") != fingerprint:
        raise ValueError("checkpoint protocol fingerprint mismatch")
    if expected_protocol is not None and protocol != dict(expected_protocol):
        raise ValueError("checkpoint protocol/config mismatch")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    completed_epoch = manifest.get("completed_epoch")
    optimizer_step = manifest.get("optimizer_step")
    if not isinstance(completed_epoch, int) or completed_epoch <= 0:
        raise ValueError("checkpoint completed epoch is invalid")
    if not isinstance(optimizer_step, int) or optimizer_step <= 0:
        raise ValueError("checkpoint optimizer step is invalid")
    if root.name != f"epoch_{completed_epoch:04d}":
        raise ValueError("checkpoint directory name does not match completed epoch")
    training = protocol.get("training")
    if not isinstance(training, dict):
        raise ValueError("checkpoint training protocol is invalid")
    world_size = int(training["world_size"])
    num_branches = int(training["num_branches"])
    expected_branches = [f"branch_{branch_id}" for branch_id in range(num_branches)]
    expected_ranks = [f"rank_{rank}" for rank in range(world_size)]
    if manifest.get("branches") != expected_branches:
        raise ValueError("checkpoint manifest branch list mismatch")
    if manifest.get("ranks") != expected_ranks:
        raise ValueError("checkpoint manifest rank list mismatch")
    if manifest.get("world_size") != world_size:
        raise ValueError("checkpoint manifest world size mismatch")
    if manifest.get("branch_allocation") != training.get("branch_allocation"):
        raise ValueError("checkpoint manifest branch allocation mismatch")
    if manifest.get("total_optimizer_steps") != training.get("total_optimizer_steps"):
        raise ValueError("checkpoint total optimizer steps mismatch")
    model = protocol["model"]
    lora = protocol["lora"]
    objective = protocol["objective"]
    branch_seeds = training["branch_seeds"]
    for branch_id, branch_name in enumerate(expected_branches):
        branch_dir = root / relative_display_path(branch_name)
        metadata = read_metadata(branch_dir)
        expected_metadata = {
            "model_id": model["id"],
            "revision": model["revision"],
            "label_readout": model["label_readout"],
            "branch_id": branch_id,
            "seed": branch_seeds[branch_id],
            "target_modules": lora["target_modules"],
            "lora": {
                "rank": lora["rank"],
                "alpha": float(lora["alpha"]),
                "dropout": float(lora["dropout"]),
            },
            "objective": objective["name"],
            "beta": float(objective["beta"]),
            "prior_lambda": float(objective["prior_lambda"]),
            "temperature": float(objective["temperature"]),
            "protocol_fingerprint": fingerprint,
            "completed_epoch": completed_epoch,
            "optimizer_step": optimizer_step,
        }
        if "alpha" in objective:
            expected_metadata["objective_alpha"] = float(objective["alpha"])
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"checkpoint branch {key} mismatch")
        if not (branch_dir / "adapter.safetensors").is_file():
            raise FileNotFoundError(f"missing checkpoint branch {branch_id}")
    for rank, rank_name in enumerate(expected_ranks):
        relative_display_path(rank_name)
        metadata = read_rank_metadata(root, rank)
        _validate_rank_identity(
            metadata,
            rank=rank,
            expected_completed_epoch=completed_epoch,
            expected_optimizer_step=optimizer_step,
            expected_local_branch_ids=training["branch_allocation"][rank],
            expected_fingerprint=fingerprint,
            expected_total_optimizer_steps=training["total_optimizer_steps"],
        )
        if not (root / rank_name / "state.pt").is_file():
            raise FileNotFoundError(f"missing checkpoint rank state {rank}")
    return manifest
