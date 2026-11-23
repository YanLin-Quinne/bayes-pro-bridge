"""Small distributed helpers for deterministic branch parallel training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import Any, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedContext:
    present = {name: name in os.environ for name in ("RANK", "WORLD_SIZE")}
    if any(present.values()) and not all(present.values()):
        raise ValueError("RANK and WORLD_SIZE must be set together")
    if not any(present.values()):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return DistributedContext(0, 1, 0, device, None)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid torchrun rank environment")
    if torch.cuda.is_available():
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        backend = "gloo"
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, world_size, local_rank, device, backend)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def allocate_contiguous_branches(
    num_branches: int,
    world_size: int,
) -> tuple[tuple[int, ...], ...]:
    if num_branches <= 0 or world_size <= 0:
        raise ValueError("branch count and world size must be positive")
    if num_branches % world_size:
        raise ValueError("num_branches must be divisible by world_size")
    per_rank = num_branches // world_size
    return tuple(
        tuple(range(rank * per_rank, (rank + 1) * per_rank))
        for rank in range(world_size)
    )


def validate_branch_seeds(branch_seeds: Sequence[int], num_branches: int) -> None:
    if len(branch_seeds) != num_branches:
        raise ValueError("branch seed count must match branch count")
    if len(set(int(seed) for seed in branch_seeds)) != num_branches:
        raise ValueError("branch seeds must be unique")


def derive_process_seed(
    local_branch_ids: Sequence[int],
    branch_seeds: Sequence[int],
) -> int:
    if not local_branch_ids:
        raise ValueError("at least one local branch is required")
    selected = []
    for branch_id in local_branch_ids:
        if branch_id < 0 or branch_id >= len(branch_seeds):
            raise ValueError("local branch ID is outside branch seeds")
        selected.append((int(branch_id), int(branch_seeds[branch_id])))
    payload = ",".join(f"{branch_id}:{seed}" for branch_id, seed in selected)
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def epoch_shuffle_seed(shuffle_seed: int, epoch_index: int) -> int:
    if epoch_index < 0:
        raise ValueError("epoch index must be nonnegative")
    return int(shuffle_seed) + int(epoch_index)


def rank_ordered_concat(rank_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not rank_tensors:
        raise ValueError("at least one rank tensor is required")
    first_shape = rank_tensors[0].shape[1:]
    for tensor in rank_tensors:
        if tensor.ndim < 1 or tensor.shape[1:] != first_shape:
            raise ValueError("rank tensors must have matching trailing shapes")
        if not torch.isfinite(tensor).all():
            raise ValueError("rank tensor contains non finite values")
    return torch.cat(tuple(rank_tensors), dim=0)


def gather_detached_branch_tensor(
    local_tensor: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    if local_tensor.ndim < 1 or not torch.isfinite(local_tensor).all():
        raise ValueError("local branch tensor is invalid")
    detached = local_tensor.detach()
    if context.world_size == 1:
        return detached
    gathered = [torch.empty_like(detached) for _ in range(context.world_size)]
    dist.all_gather(gathered, detached)
    return rank_ordered_concat(gathered)


def all_reduce_sum_detached(
    value: torch.Tensor,
    context: DistributedContext,
) -> torch.Tensor:
    result = value.detach().clone()
    if context.world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def all_gather_objects(value: Any, context: DistributedContext) -> list[Any]:
    if context.world_size == 1:
        return [value]
    gathered: list[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered
