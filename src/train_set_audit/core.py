import hashlib
import math

import numpy as np


NUM_BRANCHES = 8
EXPECTED_TOTAL_ROWS = 170420
EXPECTED_TRAIN_ROWS = 170300
EXPECTED_VALIDATION_ROWS = 120


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_branch_map(world_size=4, num_branches=NUM_BRANCHES):
    if num_branches != NUM_BRANCHES:
        raise ValueError("evaluation requires eight branches")
    if world_size == 1:
        # One process holds all eight branches on one device.
        return {"0": list(range(NUM_BRANCHES))}
    if world_size != 4:
        raise ValueError("evaluation requires one or four ranks and eight branches")
    return {str(rank): [2 * rank, 2 * rank + 1] for rank in range(world_size)}


def logmeanexp_branch_gold(branch_gold_log_probs):
    values = np.asarray(branch_gold_log_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_BRANCHES:
        raise ValueError(f"expected [rows, 8] branch log probabilities, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite branch gold log probability")
    maximum = values.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1)) - math.log(NUM_BRANCHES)


def split_membership(total_rows, validation_indices):
    validation = np.zeros(total_rows, dtype=np.bool_)
    indices = np.asarray(validation_indices, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("validation indices must be unique")
    if len(indices) and (indices.min() < 0 or indices.max() >= total_rows):
        raise ValueError("validation index out of range")
    validation[indices] = True
    return ~validation, validation


def metrics_for_mask(branch_gold_log_probs, mask):
    values = np.asarray(branch_gold_log_probs, dtype=np.float64)
    selected = values[np.asarray(mask, dtype=np.bool_)]
    if not len(selected):
        return {
            "count": 0,
            "predictive_nll": None,
            "mean_individual_branch_nll": None,
            "branch_mean_nll": [None] * NUM_BRANCHES,
        }
    log_mixture = logmeanexp_branch_gold(selected)
    branch_means = (-selected).mean(axis=0, dtype=np.float64)
    return {
        "count": int(len(selected)),
        "predictive_nll": float((-log_mixture).mean(dtype=np.float64)),
        "mean_individual_branch_nll": float((-selected).mean(dtype=np.float64)),
        "branch_mean_nll": [float(value) for value in branch_means],
    }


def reconstruct_metrics(branch_gold_log_probs, is_optimization, is_validation):
    values = np.asarray(branch_gold_log_probs)
    optimization = np.asarray(is_optimization, dtype=np.bool_)
    validation = np.asarray(is_validation, dtype=np.bool_)
    if values.shape[0] != optimization.shape[0] or values.shape[0] != validation.shape[0]:
        raise ValueError("artifact arrays have inconsistent row counts")
    if np.any(optimization == validation):
        raise ValueError("optimization and validation masks must be complements")
    return {
        "all": metrics_for_mask(values, np.ones(len(values), dtype=np.bool_)),
        "optimization": metrics_for_mask(values, optimization),
        "internal_validation": metrics_for_mask(values, validation),
    }


class RankZeroAccumulator:
    def __init__(self):
        self.row_indices = []
        self.branch_gold_log_probs = []

    def add(self, rank, row_indices, branch_gold_log_probs):
        if rank != 0:
            return
        rows = np.asarray(row_indices, dtype=np.int64)
        values = np.asarray(branch_gold_log_probs, dtype=np.float64)
        if values.shape != (len(rows), NUM_BRANCHES):
            raise ValueError("rank zero batch shape mismatch")
        self.row_indices.append(rows)
        self.branch_gold_log_probs.append(values)

    def finalize(self):
        if not self.row_indices:
            raise ValueError("no rank zero batches accumulated")
        return (
            np.concatenate(self.row_indices),
            np.concatenate(self.branch_gold_log_probs, axis=0),
        )
