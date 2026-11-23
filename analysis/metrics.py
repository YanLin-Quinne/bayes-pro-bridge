"""Per-example metrics for the Final8 class-label evaluations.

Definitions follow the historical evaluator
(src/evaluate_*.py: compute_ece, compute_brier and
the final aggregation block), so recomputing from the per-example predictions
reproduces the evaluator's own fullmetrics.json values:

* confidence = max posterior predictive probability (joint_pro_confidence)
* correct    = first argmax prediction equals the gold label
* NLL        = -log(max(p_gold, 1e-12))
* Brier      = sum over candidates of (p_c - 1[c = gold])^2 (multiclass, not / K)
* ECE        = 10 equal-width confidence bins [k/10, (k+1)/10); the last bin
               also takes confidence == 1.0; bin weight = count / n
* gap        = mean confidence - accuracy

Macro  = equal mean over the 8 datasets of the per-dataset values.
Pooled = the same per-example definitions over all 22,419 examples
         (pooled ECE is one 10-bin ECE over the pooled examples).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASETS = [
    "ARC-Challenge", "ARC-Easy", "boolq", "hellaswag",
    "openbookqa", "piqa", "social_i_qa", "winogrande",
]
EXPECTED_N = {
    "ARC-Challenge": 1172, "ARC-Easy": 2376, "boolq": 3270, "hellaswag": 10042,
    "openbookqa": 500, "piqa": 1838, "social_i_qa": 1954, "winogrande": 1267,
}
TOTAL_N = sum(EXPECTED_N.values())  # 22,419
METRICS = ["accuracy", "nll", "ece", "brier", "mean_confidence", "gap"]
N_BINS = 10


@dataclass
class Preds:
    """Per-example arrays of one dataset (or of a pooled concatenation)."""
    ids: list
    conf: np.ndarray
    correct: np.ndarray
    true_prob: np.ndarray
    brier: np.ndarray

    @property
    def n(self) -> int:
        return len(self.ids)


def _seqsum(values) -> float:
    """Left-to-right float sum, the order the evaluator used (Python sum)."""
    return float(sum(np.asarray(values, dtype=np.float64).tolist()))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    total = len(conf)
    if total == 0:
        return 0.0
    out = 0.0
    for idx in range(n_bins):
        low, high = idx / n_bins, (idx + 1) / n_bins
        mask = (conf >= low) & (conf < high)
        if idx == n_bins - 1:
            mask |= conf == 1.0
        count = int(mask.sum())
        if not count:
            continue
        bin_conf = _seqsum(conf[mask]) / count
        bin_acc = _seqsum(correct[mask]) / count
        out += (count / total) * abs(bin_acc - bin_conf)
    return out


def metrics_of(p: Preds) -> dict:
    n = p.n
    acc = _seqsum(p.correct) / n
    conf = _seqsum(p.conf) / n
    return {
        "n": n,
        "accuracy": acc,
        "nll": _seqsum([-math.log(t) for t in p.true_prob.tolist()]) / n,
        "ece": ece(p.conf, p.correct),
        "brier": _seqsum(p.brier) / n,
        "mean_confidence": conf,
        "gap": conf - acc,
    }


def concat(parts: list) -> Preds:
    return Preds(
        ids=[i for p in parts for i in p.ids],
        conf=np.concatenate([p.conf for p in parts]),
        correct=np.concatenate([p.correct for p in parts]),
        true_prob=np.concatenate([p.true_prob for p in parts]),
        brier=np.concatenate([p.brier for p in parts]),
    )


def macro_of(per_dataset: dict) -> dict:
    out = {"n": len(per_dataset)}
    for m in METRICS:
        out[m] = _seqsum([per_dataset[d][m] for d in DATASETS]) / len(DATASETS)
    return out


def summarize(preds_by_ds: dict) -> dict:
    """{'datasets': {ds: metrics}, 'macro': metrics, 'pooled': metrics}."""
    per = {ds: metrics_of(preds_by_ds[ds]) for ds in DATASETS}
    pooled = metrics_of(concat([preds_by_ds[ds] for ds in DATASETS]))
    return {"datasets": per, "macro": macro_of(per), "pooled": pooled}


# ---------------------------------------------------------------- loading

def find_dataset_dir(eval_dir: Path, dataset: str) -> Path:
    """Accept both <eval_dir>/<ds>/ and <eval_dir>/final/<ds>/ layouts."""
    for cand in (eval_dir / dataset, eval_dir / "final" / dataset):
        if (cand / f"{dataset}_joint_pro_predictions.jsonl").exists():
            return cand
    raise FileNotFoundError(f"no predictions for {dataset} under {eval_dir}")


def _parse_row(row: dict, from_branch_probs: bool):
    gold = str(row["gold"])
    if from_branch_probs:
        branches = row["branch_probs"]
        labels = list(branches[0].keys())
        probs = {c: _seqsum([b[c] for b in branches]) / len(branches) for c in labels}
        best = max(probs.values())
        pred = next(c for c in labels if probs[c] == best)  # first argmax, like torch
        conf = probs[pred]
    else:
        probs = row["ensemble_probs"]
        pred = str(row["joint_pro_pred"])
        conf = float(row["joint_pro_confidence"])
    true_prob = max(float(probs.get(gold, 0.0)), 1e-12)
    brier = sum((float(v) - (1.0 if str(k) == gold else 0.0)) ** 2 for k, v in probs.items())
    return str(row["id"]), conf, int(pred == gold), true_prob, brier


def read_jsonl(path: Path, from_branch_probs: bool = False) -> Preds:
    ids, conf, correct, true_prob, brier = [], [], [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            i, c, k, t, b = _parse_row(json.loads(line), from_branch_probs)
            ids.append(i); conf.append(c); correct.append(k); true_prob.append(t); brier.append(b)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate example ids")
    return Preds(ids, np.array(conf), np.array(correct, dtype=np.float64),
                 np.array(true_prob), np.array(brier))


def _cache_key(path: Path, from_branch_probs: bool) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{from_branch_probs}|v1"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def load_preds(path: Path, cache_dir=None, from_branch_probs: bool = False) -> Preds:
    if cache_dir is None:
        return read_jsonl(path, from_branch_probs)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfile = cache_dir / f"{_cache_key(path, from_branch_probs)}.npz"
    if cfile.exists():
        z = np.load(cfile, allow_pickle=False)
        return Preds(z["ids"].tolist(), z["conf"], z["correct"], z["true_prob"], z["brier"])
    p = read_jsonl(path, from_branch_probs)
    tmp = cache_dir / f"{cfile.stem}.tmp.npz"
    np.savez(tmp, ids=np.array(p.ids), conf=p.conf, correct=p.correct,
             true_prob=p.true_prob, brier=p.brier)
    os.replace(tmp, cfile)
    return p


def load_eval_dir(eval_dir, cache_dir=None, from_branch_probs=False, check_counts=True) -> dict:
    eval_dir = Path(eval_dir)
    out = {}
    for ds in DATASETS:
        dsdir = find_dataset_dir(eval_dir, ds)
        p = load_preds(dsdir / f"{ds}_joint_pro_predictions.jsonl", cache_dir, from_branch_probs)
        if check_counts and p.n != EXPECTED_N[ds]:
            raise ValueError(f"{dsdir}: {p.n} rows, expected {EXPECTED_N[ds]}")
        out[ds] = p
    return out


def evaluator_summary(eval_dir, dataset) -> dict:
    """The evaluator's own fullmetrics.json values, under our metric names."""
    dsdir = find_dataset_dir(Path(eval_dir), dataset)
    m = json.load(open(dsdir / f"{dataset}_joint_pro_fullmetrics.json"))
    m = m.get("metrics", m)
    return {
        "n": m["num_examples"], "accuracy": m["final_prob_accuracy"], "nll": m["final_nll"],
        "ece": m["final_ece"], "brier": m["final_brier"],
        "mean_confidence": m["mean_confidence"],
        "gap": m["signed_calibration_gap_conf_minus_acc"],
    }
