"""Metrics for normalized candidate probability ensembles."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def _validate_probability_inputs(
    branch_probabilities: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> None:
    if branch_probabilities.ndim != 3:
        raise ValueError("branch_probabilities must have shape [branch, batch, choice]")
    if candidate_mask.dtype is not torch.bool or candidate_mask.shape != branch_probabilities.shape[1:]:
        raise ValueError("candidate_mask must match batch and choice axes")
    if not torch.isfinite(branch_probabilities).all():
        raise ValueError("branch probabilities contain non finite values")
    if torch.any(branch_probabilities < 0):
        raise ValueError("branch probabilities must be nonnegative")
    counts = candidate_mask.sum(dim=-1)
    if torch.any(counts < 2) or torch.any(counts > 5):
        raise ValueError("every example must have two through five candidates")


def normalize_then_mean(
    branch_probabilities: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize each branch over valid choices, then take the branch mean."""
    _validate_probability_inputs(branch_probabilities, candidate_mask)
    mask = candidate_mask.unsqueeze(0)
    masked = branch_probabilities.masked_fill(~mask, 0.0)
    denominators = masked.sum(dim=-1, keepdim=True)
    if torch.any(denominators <= 0):
        raise ValueError("a branch assigns zero mass to all valid candidates")
    normalized = masked / denominators
    ensemble = normalized.mean(dim=0).masked_fill(~candidate_mask, 0.0)
    if not torch.allclose(
        ensemble.sum(dim=-1),
        torch.ones(
            ensemble.shape[0],
            device=ensemble.device,
            dtype=ensemble.dtype,
        ),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("ensemble probabilities do not sum to one")
    return ensemble


def validate_choice_key_consistency(
    branch_choice_keys: Sequence[Sequence[tuple[str, ...]]],
) -> None:
    if not branch_choice_keys:
        raise ValueError("choice key sets are empty")
    reference = tuple(branch_choice_keys[0])
    for keys in branch_choice_keys[1:]:
        if tuple(keys) != reference:
            raise ValueError("choice keys differ across branches")


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    nll: float
    ece: float
    brier_score: float
    mean_confidence: float
    confidence_accuracy_gap: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "NLL": self.nll,
            "ECE": self.ece,
            "Brier score": self.brier_score,
            "Mean confidence": self.mean_confidence,
            "Confidence minus accuracy gap": self.confidence_accuracy_gap,
        }


def evaluate_probabilities(
    probabilities: torch.Tensor,
    gold_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    ece_bins: int = 10,
) -> EvaluationMetrics:
    if probabilities.ndim != 2 or candidate_mask.shape != probabilities.shape:
        raise ValueError("probabilities and candidate_mask must be [batch, choice]")
    if gold_indices.dtype != torch.long or gold_indices.shape != probabilities.shape[:1]:
        raise ValueError("gold_indices must be a long vector matching the batch")
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    masked = probabilities.masked_fill(~candidate_mask, 0.0)
    if not torch.allclose(
        masked.sum(dim=-1),
        torch.ones(
            masked.shape[0],
            device=masked.device,
            dtype=masked.dtype,
        ),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("probabilities must sum to one over valid candidates")
    rows = torch.arange(masked.shape[0], device=masked.device)
    if torch.any(gold_indices < 0) or torch.any(gold_indices >= masked.shape[1]):
        raise ValueError("gold index is outside the candidate axis")
    if not candidate_mask[rows, gold_indices].all():
        raise ValueError("gold index points to a masked candidate")

    confidence, predictions = masked.max(dim=-1)
    correctness = predictions.eq(gold_indices).to(masked.dtype)
    accuracy = correctness.mean()
    gold_probability = masked[rows, gold_indices]
    nll = -gold_probability.clamp_min(torch.finfo(masked.dtype).tiny).log().mean()

    one_hot = torch.zeros_like(masked)
    one_hot[rows, gold_indices] = 1.0
    brier = ((masked - one_hot).square() * candidate_mask).sum(dim=-1).mean()

    ece = torch.zeros((), dtype=masked.dtype, device=masked.device)
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = (confidence > lower) & (confidence <= upper)
        if bin_index == 0:
            members = members | confidence.eq(0)
        if members.any():
            fraction = members.to(masked.dtype).mean()
            ece = ece + fraction * (
                confidence[members].mean() - correctness[members].mean()
            ).abs()
    mean_confidence = confidence.mean()
    values = (
        accuracy,
        nll,
        ece,
        brier,
        mean_confidence,
        mean_confidence - accuracy,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("metric computation produced non finite values")
    return EvaluationMetrics(*(float(value) for value in values))
