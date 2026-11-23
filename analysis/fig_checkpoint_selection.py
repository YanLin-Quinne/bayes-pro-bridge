#!/usr/bin/env python3
"""Figure 9 (Llama 3.2-3B checkpoint trajectories) and its plotted numbers.

Reads, for beta in 0, 0.1, 0.3, 1 (pure mixture, two bridges, independent training):

  runs/llama3.2-3b_beta<b>_m8/checkpoints/checkpoint-<s>/<ds>/<ds>_joint_pro_fullmetrics.json
      the six saved candidate states on BoolQ, HellaSwag and OpenBookQA
  runs/llama3.2-3b_beta<b>_m8/selected/selected_checkpoint.json
      the checkpoint chosen by the lowest validation NLL, checked against
  runs/llama3.2-3b_beta<b>_m8/train.log
      the validation NLL ("[eval step S] {'eval_loss': ...}") logged at the six candidates

The figure shows accuracy, NLL and Brier score. Outlined markers are the selected checkpoints. The plotted
numbers are written to <out-dir>/data/checkpoint_selection.json.

usage: python analysis/fig_checkpoint_selection.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

import fig_common as FC  # noqa: E402

STEM_PREDICTIVE = "figureC1_checkpoint_predictive_scores_revised"

METHODS = ["beta0_pure_pro", "beta01_convex", "beta03_convex", "ce_only"]
METHOD_BETA = {"beta0_pure_pro": "0", "beta01_convex": "0.1", "beta03_convex": "0.3", "ce_only": "1"}
METHOD_PLAIN = {
    "beta0_pure_pro": r"Pure mixture ($\beta$=0)",
    "beta01_convex": r"Bridge ($\beta$=0.1)",
    "beta03_convex": r"Bridge ($\beta$=0.3)",
    "ce_only": r"Independent training ($\beta$=1)",
}
DATASET_DISPLAY = {"boolq": "BoolQ", "hellaswag": "HellaSwag", "openbookqa": "OpenBookQA"}
CURVE_DATASETS = ["boolq", "hellaswag", "openbookqa"]
CHECKPOINTS = [5000, 10000, 15000, 20000, 25000, 30000]
METRICS = ["accuracy", "nll", "brier"]
# Metric -> key in <ds>_joint_pro_fullmetrics.json["metrics"].
FULLMETRICS_KEY = {
    "accuracy": "final_prob_accuracy",
    "nll": "final_nll",
    "brier": "final_brier",
}
EVAL_LINE = re.compile(r"^\[eval step (\d+)\] \{'eval_loss': ([0-9.eE+-]+)")

COLOUR = {
    "beta0_pure_pro": "#0072B2",
    "beta01_convex": "#E69F00",
    "beta03_convex": "#009E73",
    "ce_only": "#CC79A7",
}
MARKER = {"beta0_pure_pro": "o", "beta01_convex": "s", "beta03_convex": "^", "ce_only": "D"}
LINESTYLE = {"beta0_pure_pro": "-", "beta01_convex": "--", "beta03_convex": "-.", "ce_only": ":"}
METRIC_AXIS_DISPLAY = {
    "accuracy": "Accuracy\n(higher is better)",
    "nll": "NLL\n(lower is better)",
    "brier": "Brier score\n(lower is better)",
}
FIGURE_WIDTH_MM = 183.0
FIGURE_DPI = 600
FIGURE_WIDTH_IN = math.ceil(FIGURE_WIDTH_MM / 25.4 * FIGURE_DPI) / FIGURE_DPI
FIGURE_HEIGHT_IN = 4.55
SELECTION_NOTE = "Outlined markers: selected by separate held-out validation NLL (not these three datasets)"
SELECTION_RULE = ("lowest posterior predictive NLL on the fixed 120 example validation set (seed 42), "
                  "ties to the earlier checkpoint")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


# ------------------------------------------------------------------ inputs

def run_dir(runs: Path, method: str) -> Path:
    return runs / f"llama3.2-3b_beta{METHOD_BETA[method]}_m8"


def load_curves(runs: Path) -> list[dict[str, Any]]:
    """One row per (method, checkpoint, dataset) with the three metrics and the example count."""
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for checkpoint in CHECKPOINTS:
            for dataset in CURVE_DATASETS:
                path = (run_dir(runs, method) / "checkpoints" / f"checkpoint-{checkpoint}" / dataset
                        / f"{dataset}_joint_pro_fullmetrics.json")
                if not path.exists():
                    raise ValueError(f"missing fullmetrics: {path}")
                metrics = json.loads(path.read_text())["metrics"]
                if metrics.get("dataset") != dataset:
                    raise ValueError(f"dataset field mismatch in {path}")
                if int(metrics.get("num_branches", -1)) != 8:
                    raise ValueError(f"expected 8 particles in {path}")
                row: dict[str, Any] = {"method": method, "checkpoint": checkpoint, "dataset": dataset}
                for column, key in FULLMETRICS_KEY.items():
                    value = float(metrics[key])
                    if not math.isfinite(value):
                        raise ValueError(f"{path}: {key} is not finite")
                    row[column] = value
                row["n"] = int(metrics["num_examples"])
                rows.append(row)
    return rows


def parse_validation_nll(train_log: Path) -> dict[int, float]:
    """Validation NLL per evaluation step from a trainer log."""
    if not train_log.exists():
        raise ValueError(f"missing train log: {train_log}")
    values: dict[int, float] = {}
    for line in train_log.read_text(errors="replace").splitlines():
        match = EVAL_LINE.match(line)
        if match is None:
            continue
        step = int(match.group(1))
        if step in values:
            raise ValueError(f"duplicate eval step {step} in {train_log}")
        values[step] = float(match.group(2))
    return values


def select_checkpoint(candidates: dict[int, float]) -> int:
    """Lowest validation NLL, ties resolved in favour of the earlier checkpoint."""
    return min(candidates, key=lambda step: (candidates[step], step))


def load_selection(runs: Path) -> dict[str, Any]:
    """Selected checkpoint per method, verified against the trainer logs."""
    methods: dict[str, Any] = {}
    for method in METHODS:
        run = run_dir(runs, method)
        record = json.loads((run / "selected" / "selected_checkpoint.json").read_text())
        candidates = {int(step): float(value) for step, value in record["candidates"].items()}
        if sorted(candidates) != CHECKPOINTS:
            raise ValueError(f"{method}: candidate grid {sorted(candidates)} is not {CHECKPOINTS}")
        logged = parse_validation_nll(run / "train.log")
        for step, value in candidates.items():
            if logged.get(step) != value:
                raise ValueError(f"{method}: validation NLL at {step} is {value} in selected_checkpoint.json "
                                 f"but {logged.get(step)} in train.log")
        selected = int(record["selected_step"])
        if selected != select_checkpoint(candidates):
            raise ValueError(f"{method}: selected step {selected} is not the validation NLL argmin")
        if float(record["validation_nll"]) != candidates[selected]:
            raise ValueError(f"{method}: validation_nll differs from candidate at {selected}")
        if not (run / "selected" / f"checkpoint-{selected}").is_dir():
            raise ValueError(f"{method}: no evaluation of checkpoint-{selected} in {run / 'selected'}")
        methods[method] = {
            "beta": METHOD_BETA[method],
            "selected_step": selected,
            "validation_nll": candidates[selected],
            "candidates": {str(step): candidates[step] for step in CHECKPOINTS},
            "train_log_eval_points": len(logged),
            "train_log_matches_candidates": True,
        }
    return {"candidate_steps": CHECKPOINTS, "selection_rule": SELECTION_RULE, "methods": methods}


def check_curve_keys(curves: list[dict[str, Any]]) -> None:
    keys = [(c["method"], c["checkpoint"], c["dataset"]) for c in curves]
    expected = {(m, s, d) for m in METHODS for s in CHECKPOINTS for d in CURVE_DATASETS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("curves do not cover each (method, checkpoint, dataset) exactly once")


# ------------------------------------------------------------------ figure

def format_axis_tick(value: float, _position: float | None = None) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return f"{value:.3g}"


def build_figure(data: dict[str, Any], stem: Path, rows: list[tuple[str, str]], write_png: bool) -> None:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in data["curves"]:
        by_key.setdefault((r["method"], r["dataset"]), []).append(r)
    for curve in by_key.values():
        curve.sort(key=lambda r: r["checkpoint"])
    figure, axes = plt.subplots(3, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=FIGURE_DPI, sharex=True)
    selected_steps = {m: int(data["selection"]["methods"][m]["selected_step"]) for m in METHODS}
    for row_index, (metric, _) in enumerate(rows):
        for column_index, dataset in enumerate(CURVE_DATASETS):
            axis = axes[row_index, column_index]
            for method in METHODS:
                curve = by_key[(method, dataset)]
                x = np.array([r["checkpoint"] for r in curve]) / 1000
                y = np.array([r[metric] for r in curve])
                axis.plot(x, y, color=COLOUR[method], marker=MARKER[method], linestyle=LINESTYLE[method],
                          markersize=3.7, linewidth=1.25, markeredgewidth=0.65, label=METHOD_PLAIN[method],
                          zorder=3)
                chosen = [r for r in curve if r["checkpoint"] == selected_steps[method]]
                if len(chosen) != 1:
                    raise ValueError(f"missing selected marker source for {method}, {dataset}, {metric}")
                axis.scatter([selected_steps[method] / 1000], [float(chosen[0][metric])], s=38,
                             marker=MARKER[method], facecolors="none", edgecolors="#202020", linewidths=0.9,
                             zorder=5)
            axis.set_xticks([5, 10, 15, 20, 25, 30])
            axis.margins(x=0.035, y=0.11)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            axis.yaxis.set_major_formatter(FuncFormatter(format_axis_tick))
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
            axis.tick_params(axis="x", labelbottom=row_index == 2, length=2.5)
            axis.tick_params(axis="y", length=2.5)
            if row_index == 0:
                axis.set_title(DATASET_DISPLAY[dataset])
            if column_index == 0:
                axis.set_ylabel(METRIC_AXIS_DISPLAY[metric], labelpad=5)
            axis.text(-0.16, 1.02, chr(ord("a") + row_index * 3 + column_index), transform=axis.transAxes,
                      fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    handles = [
        Line2D([], [], color=COLOUR[method], marker=MARKER[method], linestyle=LINESTYLE[method],
               linewidth=1.25, markersize=4.5, label=METHOD_PLAIN[method])
        for method in METHODS
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.992), ncol=4, fontsize=7.4,
                  columnspacing=1.45, handlelength=2.5, handletextpad=0.55)
    figure.text(0.5, 0.925, SELECTION_NOTE, ha="center", va="center", fontsize=6.8, color="#303030")
    figure.supxlabel("Checkpoint (thousand steps)", x=0.55, y=0.018, fontsize=7.5)
    figure.subplots_adjust(left=0.11, right=0.99, bottom=0.105, top=0.845, wspace=0.25, hspace=0.22)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    if write_png:
        figure.savefig(stem.with_suffix(".png"), facecolor="white", dpi=FIGURE_DPI)
    plt.close(figure)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    curves = load_curves(args.runs)
    check_curve_keys(curves)
    data = FC.write_data(args.out_dir, "checkpoint_selection",
                         {"curves": curves, "selection": load_selection(args.runs)})
    rows = [("accuracy", "Accuracy"), ("nll", "NLL"), ("brier", "Brier score")]
    build_figure(data, FC.figure_stem(args.out_dir, STEM_PREDICTIVE), rows, args.png)


if __name__ == "__main__":
    main()
