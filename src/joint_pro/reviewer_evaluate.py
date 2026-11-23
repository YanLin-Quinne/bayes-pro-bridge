"""Metric helpers shared by the endpoint evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .evaluate import (
    ProbabilityEvaluation,
)


METRIC_KEYS = (
    "nll",
    "accuracy",
    "ece",
    "brier_score",
    "mean_confidence",
    "confidence_gap",
)


def _metric_row(
    evaluation: ProbabilityEvaluation,
    *,
    ece_bins: int,
) -> dict[str, float | int]:
    # Use one reduction precision for rows and pooled metrics.
    metrics = ProbabilityEvaluation(
        evaluation.probabilities.double(),
        evaluation.gold_indices,
        evaluation.candidate_mask,
    ).metrics(ece_bins=ece_bins)
    row: dict[str, float | int] = {
        "count": evaluation.count,
        "nll": metrics.nll,
        "accuracy": metrics.accuracy,
        "ece": metrics.ece,
        "brier_score": metrics.brier_score,
        "mean_confidence": metrics.mean_confidence,
        "confidence_gap": metrics.confidence_accuracy_gap,
    }
    if any(
        not math.isfinite(float(row[key]))
        for key in METRIC_KEYS
    ):
        raise ValueError("reviewer evaluation produced a non finite metric")
    return row


def _concatenate(evaluations: Sequence[ProbabilityEvaluation]) -> ProbabilityEvaluation:
    if not evaluations:
        raise ValueError("cannot pool an empty evaluation set")
    return ProbabilityEvaluation(
        probabilities=torch.cat([value.probabilities for value in evaluations]),
        gold_indices=torch.cat([value.gold_indices for value in evaluations]),
        candidate_mask=torch.cat([value.candidate_mask for value in evaluations]),
    )


def equal_dataset_macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if len(rows) != 8:
        raise ValueError("Final8 macro requires exactly eight dataset rows")
    return {
        "dataset_count": 8,
        **{
            key: sum(float(row[key]) for row in rows) / 8.0
            for key in METRIC_KEYS
        },
    }
