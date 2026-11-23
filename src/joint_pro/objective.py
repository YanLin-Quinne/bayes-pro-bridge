"""Class label probability normalization and joint finite particle losses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non finite values")


def _validate_gold_log_probs(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must have shape [branch, batch]")
    _require_finite(name, tensor)


def validate_beta(beta: float) -> float:
    value = float(beta)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"beta must be finite and in [0, 1], got {beta!r}")
    return value


def validate_renyi_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha!r}")
    return value


def masked_candidate_log_probs(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize each particle and example over its valid candidate labels."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [branch, batch, choice]")
    if candidate_mask.dtype is not torch.bool or candidate_mask.ndim != 2:
        raise ValueError("candidate_mask must be a bool tensor [batch, choice]")
    if tuple(candidate_mask.shape) != tuple(logits.shape[1:]):
        raise ValueError("candidate_mask shape must match logits batch and choice axes")
    _require_finite("logits", logits)
    counts = candidate_mask.sum(dim=-1)
    if torch.any(counts < 2) or torch.any(counts > 5):
        raise ValueError("every example must have two through five valid candidates")
    expanded_mask = candidate_mask.unsqueeze(0)
    masked = logits.float().masked_fill(~expanded_mask, float("-inf"))
    result = F.log_softmax(masked, dim=-1)
    if not torch.isfinite(result.masked_select(expanded_mask)).all():
        raise ValueError("candidate normalization produced non finite valid values")
    return result


def gold_label_log_probs(
    candidate_log_probs: torch.Tensor,
    gold_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    if candidate_log_probs.ndim != 3:
        raise ValueError("candidate_log_probs must have shape [branch, batch, choice]")
    if gold_indices.dtype != torch.long or gold_indices.ndim != 1:
        raise ValueError("gold_indices must be a one dimensional long tensor")
    if gold_indices.shape[0] != candidate_log_probs.shape[1]:
        raise ValueError("gold index count must match batch size")
    if candidate_mask.shape != candidate_log_probs.shape[1:]:
        raise ValueError("candidate mask shape mismatch")
    if torch.any(gold_indices < 0) or torch.any(
        gold_indices >= candidate_log_probs.shape[2]
    ):
        raise ValueError("gold index is outside the candidate axis")
    rows = torch.arange(gold_indices.numel(), device=gold_indices.device)
    if not candidate_mask[rows, gold_indices].all():
        raise ValueError("gold index points to a masked candidate")
    gathered = candidate_log_probs.gather(
        2,
        gold_indices.view(1, -1, 1).expand(candidate_log_probs.shape[0], -1, 1),
    ).squeeze(-1)
    _require_finite("gold label log probabilities", gathered)
    return gathered


def exact_mixture_nll(all_gold_log_probs: torch.Tensor) -> torch.Tensor:
    """Return the exact finite mixture NLL scalar."""
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    branch_count = all_gold_log_probs.shape[0]
    return -(
        torch.logsumexp(all_gold_log_probs, dim=0) - math.log(branch_count)
    ).mean()


def exact_branch_ce(all_gold_log_probs: torch.Tensor) -> torch.Tensor:
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    return -all_gold_log_probs.mean()


def exact_alpha_renyi_nll(
    all_gold_log_probs: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return the exact finite uniform particle alpha Rényi data loss."""
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    alpha = validate_renyi_alpha(alpha)
    if alpha == 0.0:
        return exact_branch_ce(all_gold_log_probs)
    branch_count = all_gold_log_probs.shape[0]
    return -(
        torch.logsumexp(alpha * all_gold_log_probs, dim=0)
        - math.log(branch_count)
    ).mean() / alpha


def detached_responsibility_surrogate(
    local_gold_log_probs: torch.Tensor,
    all_gold_log_probs: torch.Tensor,
    local_branch_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return a local scalar whose first derivative equals mixture NLL locally."""
    _validate_gold_log_probs("local_gold_log_probs", local_gold_log_probs)
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    if local_gold_log_probs.shape[1] != all_gold_log_probs.shape[1]:
        raise ValueError("local and global tensors must share the batch axis")
    if local_branch_indices is None:
        if local_gold_log_probs.shape[0] != all_gold_log_probs.shape[0]:
            raise ValueError("local branch indices are required for a branch subset")
        local_branch_indices = tuple(range(all_gold_log_probs.shape[0]))
    if len(local_branch_indices) != local_gold_log_probs.shape[0]:
        raise ValueError("local branch index count does not match local tensor")
    indices = torch.as_tensor(
        tuple(local_branch_indices),
        dtype=torch.long,
        device=local_gold_log_probs.device,
    )
    if (
        torch.any(indices < 0)
        or torch.any(indices >= all_gold_log_probs.shape[0])
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("local branch indices are invalid")
    responsibilities = torch.softmax(
        all_gold_log_probs.detach().to(local_gold_log_probs.device),
        dim=0,
    ).index_select(0, indices)
    return -(responsibilities * local_gold_log_probs).sum() / local_gold_log_probs.shape[1]


def detached_alpha_renyi_surrogate(
    local_gold_log_probs: torch.Tensor,
    all_gold_log_probs: torch.Tensor,
    alpha: float,
    local_branch_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the author supplied detached responsibility gradient surrogate."""
    _validate_gold_log_probs("local_gold_log_probs", local_gold_log_probs)
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    alpha = validate_renyi_alpha(alpha)
    if local_gold_log_probs.shape[1] != all_gold_log_probs.shape[1]:
        raise ValueError("local and global tensors must share the batch axis")
    if local_branch_indices is None:
        if local_gold_log_probs.shape[0] != all_gold_log_probs.shape[0]:
            raise ValueError("local branch indices are required for a branch subset")
        local_branch_indices = tuple(range(all_gold_log_probs.shape[0]))
    if len(local_branch_indices) != local_gold_log_probs.shape[0]:
        raise ValueError("local branch index count does not match local tensor")
    indices = torch.as_tensor(
        tuple(local_branch_indices),
        dtype=torch.long,
        device=local_gold_log_probs.device,
    )
    if (
        torch.any(indices < 0)
        or torch.any(indices >= all_gold_log_probs.shape[0])
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("local branch indices are invalid")
    responsibilities = torch.softmax(
        alpha * all_gold_log_probs.detach().to(local_gold_log_probs.device),
        dim=0,
    ).index_select(0, indices)
    return -(responsibilities * local_gold_log_probs).sum() / local_gold_log_probs.shape[1]


def branch_ce_local_contribution(
    local_gold_log_probs: torch.Tensor,
    total_branches: int,
) -> torch.Tensor:
    _validate_gold_log_probs("local_gold_log_probs", local_gold_log_probs)
    if total_branches < local_gold_log_probs.shape[0]:
        raise ValueError("total_branches is smaller than the local branch count")
    return -local_gold_log_probs.sum() / (
        total_branches * local_gold_log_probs.shape[1]
    )


def lora_prior(
    parameters: Iterable[torch.Tensor],
    prior_lambda: float,
    total_branches: int,
    num_training_samples: int,
) -> torch.Tensor:
    if (
        not math.isfinite(prior_lambda)
        or prior_lambda < 0
        or total_branches <= 0
        or num_training_samples <= 0
    ):
        raise ValueError("invalid prior scale")
    params = tuple(parameters)
    if not params:
        raise ValueError("prior requires at least one trainable parameter")
    for parameter in params:
        _require_finite("LoRA parameter", parameter)
    squared_sum = sum(parameter.float().square().sum() for parameter in params)
    return prior_lambda * squared_sum / (
        2.0 * total_branches * num_training_samples
    )


@dataclass(frozen=True)
class JointObjective:
    backward_loss: torch.Tensor
    data_backward_loss: torch.Tensor
    exact_mixture_nll: torch.Tensor
    responsibility_surrogate: torch.Tensor
    branch_ce: torch.Tensor
    data_loss_value: torch.Tensor
    prior: torch.Tensor
    total_loss_value: torch.Tensor


@dataclass(frozen=True)
class RescueDiagnostics:
    raw_responsibility_ess: float
    effective_coefficient_ess: float
    branch_responsibility_mass: tuple[float, ...]
    cumulative_responsibility_mass: tuple[float, ...]
    branch_gold_nll: tuple[float, ...]
    branch_nll_dispersion: float
    minimum_effective_coefficient: float
    predictive_mixture_nll: float

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "raw_responsibility_ess": self.raw_responsibility_ess,
            "effective_coefficient_ess": self.effective_coefficient_ess,
            "branch_responsibility_mass": list(self.branch_responsibility_mass),
            "cumulative_responsibility_mass": list(
                self.cumulative_responsibility_mass
            ),
            "branch_gold_nll": list(self.branch_gold_nll),
            "branch_nll_dispersion": self.branch_nll_dispersion,
            "minimum_effective_coefficient": self.minimum_effective_coefficient,
            "predictive_mixture_nll": self.predictive_mixture_nll,
        }


@torch.no_grad()
def rescue_diagnostics(
    all_gold_log_probs: torch.Tensor,
    beta: float,
) -> RescueDiagnostics:
    """Summarize responsibility collapse and beta rescue coefficients."""
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    beta = validate_beta(beta)
    values = all_gold_log_probs.detach().float()
    branches = values.shape[0]
    responsibilities = torch.softmax(values, dim=0)
    coefficients = (1.0 - beta) * responsibilities + beta / branches
    raw_ess = responsibilities.square().sum(dim=0).reciprocal().mean()
    effective_ess = coefficients.square().sum(dim=0).reciprocal().mean()
    responsibility_mass = responsibilities.mean(dim=1)
    cumulative_mass = responsibilities.sum(dim=1)
    branch_nll = -values.mean(dim=1)
    dispersion = branch_nll.std(unbiased=False)
    minimum = coefficients.min()
    predictive_nll = exact_mixture_nll(values)
    tensors = (
        raw_ess,
        effective_ess,
        responsibility_mass,
        cumulative_mass,
        branch_nll,
        dispersion,
        minimum,
        predictive_nll,
    )
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("rescue diagnostics produced non finite values")
    return RescueDiagnostics(
        raw_responsibility_ess=float(raw_ess),
        effective_coefficient_ess=float(effective_ess),
        branch_responsibility_mass=tuple(float(value) for value in responsibility_mass),
        cumulative_responsibility_mass=tuple(float(value) for value in cumulative_mass),
        branch_gold_nll=tuple(float(value) for value in branch_nll),
        branch_nll_dispersion=float(dispersion),
        minimum_effective_coefficient=float(minimum),
        predictive_mixture_nll=float(predictive_nll),
    )


@torch.no_grad()
def renyi_diagnostics(
    all_gold_log_probs: torch.Tensor,
    alpha: float,
) -> RescueDiagnostics:
    """Summarize exact alpha responsibilities without inventing a floor."""
    _validate_gold_log_probs("all_gold_log_probs", all_gold_log_probs)
    alpha = validate_renyi_alpha(alpha)
    values = all_gold_log_probs.detach().float()
    responsibilities = torch.softmax(alpha * values, dim=0)
    ess = responsibilities.square().sum(dim=0).reciprocal().mean()
    responsibility_mass = responsibilities.mean(dim=1)
    cumulative_mass = responsibilities.sum(dim=1)
    branch_nll = -values.mean(dim=1)
    dispersion = branch_nll.std(unbiased=False)
    minimum = responsibilities.min()
    predictive_nll = exact_mixture_nll(values)
    tensors = (
        ess,
        responsibility_mass,
        cumulative_mass,
        branch_nll,
        dispersion,
        minimum,
        predictive_nll,
    )
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("Rényi diagnostics produced non finite values")
    return RescueDiagnostics(
        raw_responsibility_ess=float(ess),
        effective_coefficient_ess=float(ess),
        branch_responsibility_mass=tuple(float(value) for value in responsibility_mass),
        cumulative_responsibility_mass=tuple(float(value) for value in cumulative_mass),
        branch_gold_nll=tuple(float(value) for value in branch_nll),
        branch_nll_dispersion=float(dispersion),
        minimum_effective_coefficient=float(minimum),
        predictive_mixture_nll=float(predictive_nll),
    )


def build_joint_objective(
    *,
    local_gold_log_probs: torch.Tensor,
    all_gold_log_probs: torch.Tensor,
    local_branch_indices: Sequence[int],
    local_parameters: Iterable[torch.Tensor],
    total_branches: int,
    num_training_samples: int,
    beta: float,
    prior_lambda: float,
    include_prior: bool = True,
) -> JointObjective:
    """Construct the local backward scalar and exact detached report values."""
    beta = validate_beta(beta)
    if all_gold_log_probs.shape[0] != total_branches:
        raise ValueError("gathered branch count does not match total_branches")
    pro_surrogate = detached_responsibility_surrogate(
        local_gold_log_probs,
        all_gold_log_probs,
        local_branch_indices,
    )
    local_ce = branch_ce_local_contribution(local_gold_log_probs, total_branches)
    prior = (
        lora_prior(
            local_parameters,
            prior_lambda,
            total_branches,
            num_training_samples,
        )
        if include_prior
        else local_gold_log_probs.new_zeros(())
    )
    data_backward_loss = (1.0 - beta) * pro_surrogate + beta * local_ce
    backward_loss = data_backward_loss + prior
    with torch.no_grad():
        exact_pro = exact_mixture_nll(all_gold_log_probs.detach())
        branch_ce = exact_branch_ce(all_gold_log_probs.detach())
        data_value = (1.0 - beta) * exact_pro + beta * branch_ce
        total_value = data_value + prior.detach()
    _require_finite("backward loss", backward_loss)
    _require_finite("reported total loss", total_value)
    return JointObjective(
        backward_loss=backward_loss,
        data_backward_loss=data_backward_loss,
        exact_mixture_nll=exact_pro,
        responsibility_surrogate=pro_surrogate,
        branch_ce=branch_ce,
        data_loss_value=data_value,
        prior=prior,
        total_loss_value=total_value,
    )


def build_alpha_renyi_objective(
    *,
    local_gold_log_probs: torch.Tensor,
    all_gold_log_probs: torch.Tensor,
    local_branch_indices: Sequence[int],
    local_parameters: Iterable[torch.Tensor],
    total_branches: int,
    num_training_samples: int,
    alpha: float,
    prior_lambda: float,
    include_prior: bool = True,
) -> JointObjective:
    """Construct the matched alpha Rényi backward scalar and report values."""
    alpha = validate_renyi_alpha(alpha)
    if all_gold_log_probs.shape[0] != total_branches:
        raise ValueError("gathered branch count does not match total_branches")
    surrogate = detached_alpha_renyi_surrogate(
        local_gold_log_probs,
        all_gold_log_probs,
        alpha,
        local_branch_indices,
    )
    prior = (
        lora_prior(
            local_parameters,
            prior_lambda,
            total_branches,
            num_training_samples,
        )
        if include_prior
        else local_gold_log_probs.new_zeros(())
    )
    backward_loss = surrogate + prior
    with torch.no_grad():
        exact_pro = exact_mixture_nll(all_gold_log_probs.detach())
        branch_ce = exact_branch_ce(all_gold_log_probs.detach())
        data_value = exact_alpha_renyi_nll(all_gold_log_probs.detach(), alpha)
        total_value = data_value + prior.detach()
    _require_finite("backward loss", backward_loss)
    _require_finite("reported total loss", total_value)
    return JointObjective(
        backward_loss=backward_loss,
        data_backward_loss=surrogate,
        exact_mixture_nll=exact_pro,
        responsibility_surrogate=surrogate,
        branch_ce=branch_ce,
        data_loss_value=data_value,
        prior=prior,
        total_loss_value=total_value,
    )


def finalize_accumulated_gradients(
    parameters: Iterable[torch.Tensor],
    accumulated_samples: int,
    prior: torch.Tensor,
) -> None:
    """Average sample-summed data gradients, then add the prior once."""
    if accumulated_samples <= 0:
        raise ValueError("accumulated_samples must be positive")
    if prior.ndim != 0 or not torch.isfinite(prior):
        raise ValueError("prior must be a finite scalar")
    params = tuple(parameters)
    if not params:
        raise ValueError("at least one trainable parameter is required")
    for parameter in params:
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_samples)
    prior.backward()


def validate_finite_gradients(parameters: Iterable[torch.Tensor]) -> None:
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("gradient contains non finite values")
    if not found:
        raise ValueError("no gradients were produced")
