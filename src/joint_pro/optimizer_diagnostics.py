"""Read-only optimizer diagnostics for the objective-switch experiment.

Nothing here draws random numbers or writes to parameters, gradients or
optimizer state, so enabling it cannot change the optimisation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch


def example_hash(record: Mapping[str, Any]) -> str:
    """Stable 16 hex id of a canonical record (prompt, choices, answer_index).

    The same function applied to a line of the canonical train.jsonl gives the
    same id, so logged ids map back to training examples.
    """
    payload = {
        "prompt": record["prompt"],
        "choices": list(record["choices"]),
        "answer_index": int(record["answer_index"]),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def is_diagnostic_step(
    step: int,
    *,
    cadence: int,
    start_step: int,
    dense_window: int,
    dense_every: int,
) -> tuple[bool, str | None]:
    """Return whether ``step`` gets a diagnostic record and why.

    ``cadence`` is the protocol cadence (``--diagnostics-steps``).  The dense
    window covers steps start_step+1 .. start_step+dense_window, logging the
    first step after the start and every ``dense_every`` steps after it.
    """
    if cadence and step % cadence == 0:
        return True, "cadence"
    if dense_window > 0:
        offset = step - start_step
        if 0 < offset <= dense_window and (offset == 1 or offset % dense_every == 0):
            return True, "dense_after_start"
    return False, None


def particle_parameter_groups(ensemble) -> list[tuple[int, tuple[torch.nn.Parameter, ...]]]:
    groups = []
    for branch in ensemble.branches:
        params = tuple(p for p in branch.parameters() if p.requires_grad)
        if not params:
            raise ValueError(f"particle {branch.branch_id} has no trainable parameters")
        groups.append((int(branch.branch_id), params))
    return groups


@torch.no_grad()
def snapshot_parameters(groups) -> list[list[torch.Tensor]]:
    return [[p.detach().clone() for p in params] for _, params in groups]


def _norm(tensors: Sequence[torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64, device=tensors[0].device)
    for tensor in tensors:
        total += tensor.detach().double().square().sum()
    return math.sqrt(float(total))


@torch.no_grad()
def particle_optimizer_stats(
    optimizer: torch.optim.Optimizer,
    groups,
    before: list[list[torch.Tensor]] | None,
) -> list[dict[str, Any]]:
    """Per-particle norms read after ``optimizer.step()`` and before zero_grad.

    grad_norm: L2 norm of the gradient fed to AdamW (data term plus prior).
    exp_avg_sq_norm / exp_avg_norm: L2 norms of the Adam second and first moments.
    lr: scheduled learning rate used by this step.
    adam_update_norm: lr * || m_hat / (sqrt(v_hat) + eps) ||, the deterministic
      AdamW update (weight decay is zero), excluding Langevin noise.
    effective_step_size: adam_update_norm / grad_norm, the scalar step that a
      plain gradient step would need to move as far.
    mean_coordinate_lr: mean over coordinates of lr / (sqrt(v_hat) + eps) / bias_correction1.
    param_delta_norm: || theta_after - theta_before ||, including Langevin noise.
    """
    group_by_param = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            group_by_param[param] = group
    rows = []
    for index, (branch_id, params) in enumerate(groups):
        grads = [p.grad for p in params if p.grad is not None]
        avg_sq, avg, updates, coord_lr = [], [], [], []
        coordinate_count = 0
        lr = None
        for param in params:
            state = optimizer.state.get(param, {})
            if "exp_avg_sq" not in state:
                continue
            group = group_by_param[param]
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            step = float(state["step"])
            bias1 = 1.0 - beta1 ** step
            bias2_sqrt = math.sqrt(1.0 - beta2 ** step)
            denom = state["exp_avg_sq"].double().sqrt() / bias2_sqrt + group["eps"]
            avg_sq.append(state["exp_avg_sq"])
            avg.append(state["exp_avg"])
            updates.append((lr / bias1) * state["exp_avg"].double() / denom)
            coord_lr.append(((lr / bias1) / denom).sum())
            coordinate_count += param.numel()
        row: dict[str, Any] = {
            "particle": branch_id,
            "grad_norm": _norm(grads) if grads else None,
            "exp_avg_sq_norm": _norm(avg_sq) if avg_sq else None,
            "exp_avg_norm": _norm(avg) if avg else None,
            "lr": lr,
            "adam_update_norm": _norm(updates) if updates else None,
            "mean_coordinate_lr": (
                float(torch.stack(coord_lr).sum()) / coordinate_count if coord_lr else None
            ),
            "param_delta_norm": None,
            "effective_step_size": None,
        }
        if row["grad_norm"] and row["adam_update_norm"] is not None:
            row["effective_step_size"] = row["adam_update_norm"] / row["grad_norm"]
        if before is not None:
            row["param_delta_norm"] = _norm(
                [after - prior for after, prior in zip(params, before[index])]
            )
        rows.append(row)
    return rows
