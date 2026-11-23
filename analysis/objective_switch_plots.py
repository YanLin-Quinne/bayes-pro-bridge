#!/usr/bin/env python3
"""Objective switch panels: Figure 5c, Figure 5d and Figure 11.

usage: python analysis/objective_switch_plots.py [--runs runs] [--out-dir outputs] [--png]
                                                 [--all-records]

Reads the diagnostics of the three segments written by scripts/objective_switch.sh

    <runs>/objective_switch/leadin/diagnostics.jsonl    shared lead-in, beta = 0
    <runs>/objective_switch/control/diagnostics.jsonl   arm that continues with beta = 0
    <runs>/objective_switch/rescue/diagnostics.jsonl    arm that switches to the bridge, beta = 0.1

and writes four PDF panels to <out-dir>/figures. Figure 5c is the minimum effective responsibility, Figure 5d is
the training-set mixture risk (mixture NLL on minibatches) and Figure 11 shows the effective responsibility ESS
and the population SD of the particle NLL. A summary of the numbers quoted in the text is written to
<out-dir>/data/objective_switch_summary.json.

Nothing is interpolated: the lead-in fixed evaluation at the fork step is drawn as a separate hollow marker,
the arms start at their first logged training record, and the held-out endpoint is the fixed evaluation record
each arm wrote at the end of training. The trainer also logs dense records in the first steps of each arm
(diagnostic_trigger "dense_after_start"). The paper figures keep only the records of the 100-step cadence, which is
the default here. Use --all-records to draw the dense records as well.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import panel_style as ps

ARMS = ("pure", "bridge")
ARM_DIRS = {"leadin": "leadin", "pure": "control", "bridge": "rescue"}
ARM_METHOD = {"pure": "beta0_pure_pro", "bridge": "beta01_convex"}
MOVING_AVERAGE_WINDOW = 10

PANELS = (
    # metric key in the jsonl, file stem, y label, log scale
    ("effective_coefficient_ess", "objective_switch_effective_ess_trajectory",
     "Effective responsibility ESS", False),
    ("minimum_effective_coefficient", "objective_switch_minimum_coefficient_trajectory",
     "Minimum effective responsibility", True),
    ("branch_nll_dispersion", "objective_switch_branch_nll_dispersion_trajectory",
     "Population SD of particle NLL", False),
    ("mixture_nll", "objective_switch_predictive_mixture_nll_trajectory",
     "Training-set $R_{\\mathrm{mix}}$\n(minibatch)", False),
)


class SourceError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise SourceError(message)


def read_jsonl(path):
    path = Path(path)
    _require(path.is_file(), f"missing diagnostics file: {path}")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        for key, value in record.items():
            if isinstance(value, float):
                _require(math.isfinite(value), f"nonfinite {key} in {path}")
    return records


class SwitchData:
    """Training records and fixed evaluations of the lead-in and both arms."""

    def __init__(self, source_dir, cadence_only=True):
        source_dir = Path(source_dir)
        self.records = {
            name: read_jsonl(source_dir / sub / "diagnostics.jsonl")
            for name, sub in ARM_DIRS.items()
        }
        fixed_leadin = [r for r in self.records["leadin"]
                        if r["scope"] == "fixed_evaluation"]
        _require(len(fixed_leadin) == 1, "expected one lead-in fixed evaluation")
        self.leadin_fixed = fixed_leadin[0]
        self.fork_step = int(self.leadin_fixed["step"])
        self.training = {
            arm: [r for r in self.records[arm]
                  if r["scope"] == "training_minibatch"
                  and (not cadence_only or _is_cadence(r))]
            for arm in ARMS
        }
        self.fixed = {}
        for arm in ARMS:
            fixed = [r for r in self.records[arm] if r["scope"] == "fixed_evaluation"]
            _require(len(fixed) == 1, f"expected one fixed evaluation for {arm}")
            self.fixed[arm] = fixed[0]
        steps = [[int(r["step"]) for r in self.training[arm]] for arm in ARMS]
        _require(steps[0] == steps[1], "arms log different training steps")
        self.steps = np.asarray(steps[0])
        _require(len(self.steps) >= MOVING_AVERAGE_WINDOW, "too few records")
        _require(int(self.steps[0]) > self.fork_step,
                 "first arm record must come after the fork")
        self.particles = len(self.training["pure"][0]["branch_gold_nll"])
        self.beta = {arm: float(self.training[arm][0]["beta"]) for arm in ARMS}
        _require(self.beta["pure"] == 0.0 and self.beta["bridge"] > 0.0,
                 "arm betas must be 0 and positive")

    def series(self, arm, metric):
        values = np.asarray([float(r[metric]) for r in self.training[arm]])
        _require(np.isfinite(values).all(), f"nonfinite series {arm} {metric}")
        return values

    def fixed_value(self, arm, metric):
        return float(self.fixed[arm][metric])

    @property
    def arm_labels(self):
        return {
            "pure": ps.METHOD_LABELS["beta0_pure_pro"],
            "bridge": f"Bridge (β={self.beta['bridge']:g})",
        }


def _is_cadence(record):
    return record.get("diagnostic_trigger", "cadence") == "cadence"


def trailing_moving_average(values, window=MOVING_AVERAGE_WINDOW):
    values = np.asarray(values, dtype=float)
    _require(window > 0 and len(values) >= window, "bad moving average window")
    return np.convolve(values, np.ones(window) / window, mode="valid")


FIXED_KEYS = (
    "step", "num_examples", "effective_coefficient_ess", "raw_responsibility_ess",
    "minimum_effective_coefficient", "branch_nll_dispersion",
    "predictive_mixture_nll",
)


def risk_gap_start(data):
    """First step from which the moving average of the training-set risk of
    the bridge arm stays below that of the pure arm (None if it never does)."""
    pure, bridge = (trailing_moving_average(data.series(arm, "mixture_nll"))
                    for arm in ARMS)
    below = bridge < pure
    if not below[-1]:
        return None
    start = len(below)
    while start > 0 and below[start - 1]:
        start -= 1
    return int(data.steps[MOVING_AVERAGE_WINDOW - 1 + start])


def summarise(data):
    """The values the paper quotes, per arm, from the plotted records."""
    def stats(values):
        return {
            "first": float(values[0]), "min": float(values.min()),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
        }

    arms = {}
    for arm in ARMS:
        arms[arm] = {metric: stats(data.series(arm, metric))
                     for metric, *_ in PANELS}
        arms[arm]["beta"] = data.beta[arm]
        arms[arm]["fixed_evaluation"] = {k: data.fixed[arm].get(k) for k in FIXED_KEYS}
    return {
        "fork_step": data.fork_step, "particles": data.particles,
        "records_per_arm": len(data.steps), "first_arm_step": int(data.steps[0]),
        "last_arm_step": int(data.steps[-1]),
        "leadin_fixed_evaluation": {k: data.leadin_fixed.get(k) for k in FIXED_KEYS},
        "risk_moving_average_bridge_below_pure_from_step": risk_gap_start(data),
        "arms": arms,
    }


def _top_margin(metric):
    return 15.5 if metric == "mixture_nll" else 11.5


def plot_panel(data, metric, ylabel, log_scale):
    """One panel. All metrics except the training-set risk end with the held-out
    endpoint, the fixed evaluation record each arm wrote at the end of training."""
    width, height = ps.PANEL_WIDTH_MM, 62.0
    fig = ps.new_canvas(width, height)
    top = _top_margin(metric)
    ax = ps.add_axes_mm(fig, 17.5, 11.5, 5.0, top)
    steps = data.steps
    for arm in ARMS:
        color = ps.METHOD_COLORS[ARM_METHOD[arm]]
        marker = ps.METHOD_MARKERS[ARM_METHOD[arm]]
        values = data.series(arm, metric)
        if log_scale:
            _require(np.all(values > 0), "log scale needs positive values")
        if metric == "mixture_nll":
            ax.plot(steps, values, color=color, marker=marker, ms=1.8,
                    mew=0, lw=0.5, alpha=0.22, zorder=2)
            ax.plot(steps[MOVING_AVERAGE_WINDOW - 1:],
                    trailing_moving_average(values), color=color, lw=1.4,
                    zorder=3)
        else:
            ax.plot(steps, values, color=color, marker=marker, ms=2.2, mew=0,
                    lw=1.05, alpha=0.92, zorder=3)
    if log_scale:
        ax.set_yscale("log")
        _sparse_decade_ticks(ax)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ps.wrap_label(ylabel))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45, zorder=0)

    fork = data.fork_step
    ax.axvline(fork, color=ps.GREY, lw=0.75, ls="--", zorder=1)
    switch_text = (
        f"β switched on, step {fork:,}" if metric == "minimum_effective_coefficient"
        else f"Objective switch, step {fork:,}"
    )
    ax.annotate(
        switch_text, xy=(fork, 1.0), xycoords=("data", "axes fraction"),
        xytext=(3, 2.0), textcoords="offset points", ha="left", va="bottom",
        fontsize=ps.FONT_NOTE, color=ps.GREY, annotation_clip=False,
    )

    if metric == "effective_coefficient_ess":
        _annotate_leadin(ax, data, metric)
    if metric == "minimum_effective_coefficient":
        _draw_floor(ax, data)

    last = int(steps[-1])
    if metric != "mixture_nll":
        _draw_endpoint(ax, data, metric, last)
    else:
        ax.set_xlim(fork - 300, last + 300)
        ticks = [t for t in ax.get_xticks() if steps[0] <= t <= last]
        ax.set_xticks(ticks, [f"{int(t / 1000)}k" for t in ticks])

    _legends(fig, data, metric, height)
    return fig


def _sparse_decade_ticks(ax, max_ticks=5):
    """Decade ticks, thinned to at most max_ticks, counted down from the top."""
    from matplotlib.ticker import FixedLocator, NullLocator

    low, high = ax.get_ylim()
    decades = list(range(math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1))
    step = max(1, math.ceil(len(decades) / max_ticks))
    ticks = [10.0 ** k for k in decades[::-1][::step]]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(low, high)


def _annotate_leadin(ax, data, metric):
    value = float(data.leadin_fixed[metric])
    fork = data.fork_step
    ax.scatter([fork], [value], marker="D", s=22, facecolor="white",
               edgecolor="#333333", linewidth=0.9, zorder=6)
    lower = np.median(data.series("pure", metric))
    upper = np.median(data.series("bridge", metric))
    y_text = 0.5 * (lower + upper)
    ax.annotate(
        f"Lead-in fixed evaluation\n(step {fork:,}, n={int(data.leadin_fixed['num_examples'])})",
        xy=(fork, value), xytext=(fork + 900, y_text),
        textcoords="data", ha="left", va="center",
        fontsize=ps.FONT_NOTE, color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.6,
                    "shrinkA": 1, "shrinkB": 3},
        zorder=6,
    )


def _draw_floor(ax, data):
    floor = data.beta["bridge"] / data.particles
    color = ps.BRIDGE_COLOR
    ax.axhline(floor, color=color, lw=0.9, ls="--", zorder=1)
    ax.annotate(
        f"β/m = {floor:g}", xy=(0.0, floor), xycoords=("axes fraction", "data"),
        xytext=(8, -2.5), textcoords="offset points", ha="left", va="top",
        fontsize=ps.FONT_NOTE, color=color,
    )


def _draw_endpoint(ax, data, metric, last):
    endpoint_x = last + 1500
    half = 450
    ax.axvspan(endpoint_x - half, endpoint_x + half, color="#F1F1F1", zorder=0)
    ax.axvline(endpoint_x - 2 * half, color="#888888", lw=0.6, ls=":")
    for arm in ARMS:
        value = data.fixed_value(arm, metric)
        method = ARM_METHOD[arm]
        ax.scatter([endpoint_x], [value], color=ps.METHOD_COLORS[method],
                   marker=ps.METHOD_MARKERS[method], s=20, edgecolor="white",
                   linewidth=0.45, zorder=5)
    ticks = [t for t in ax.get_xticks() if data.steps[0] <= t <= last]
    labels = [f"{int(t / 1000)}k" for t in ticks]
    ax.set_xticks(ticks + [endpoint_x], labels + ["Held-out\nn=120"])
    ax.set_xlim(data.fork_step - 300, endpoint_x + half)


def _legends(fig, data, metric, height):
    from matplotlib.lines import Line2D

    labels = data.arm_labels
    handles = [
        Line2D([0], [0], color=ps.METHOD_COLORS[ARM_METHOD[a]],
               marker=ps.METHOD_MARKERS[ARM_METHOD[a]], lw=1.3, ms=3.6,
               label=labels[a])
        for a in ARMS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0 - 0.6 / height), frameon=False,
               columnspacing=1.2, handlelength=2.0, handletextpad=0.5)
    if metric == "mixture_nll":
        grey = [
            Line2D([0], [0], color="#999999", marker="o", ms=2.2, mew=0,
                   lw=0.6, alpha=0.6, label="Logged minibatch"),
            Line2D([0], [0], color="#555555", lw=1.4,
                   label=f"Moving average of {MOVING_AVERAGE_WINDOW} records"),
        ]
        fig.legend(handles=grey, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.0 - 4.6 / height), frameon=False,
                   columnspacing=1.2, handlelength=2.0, handletextpad=0.5)


def build_all(source_dir, out_dir, cadence_only=True):
    data = SwitchData(source_dir, cadence_only)
    names = []
    for metric, stem, ylabel, log_scale in PANELS:
        fig = plot_panel(data, metric, ylabel, log_scale)
        names.append(ps.save_panel(fig, Path(out_dir) / stem, stem, ps.PANEL_WIDTH_MM).name)
    return data, names


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="runs",
                        help="folder that holds objective_switch/{leadin,control,rescue}")
    parser.add_argument("--out-dir", default="outputs",
                        help="output root: figures/*.pdf and data/objective_switch_summary.json")
    parser.add_argument("--png", action="store_true", help="also write 600 dpi PNG files")
    parser.add_argument("--all-records", action="store_true",
                        help="draw the dense records logged right after the fork as well "
                             "(the paper figures keep the 100-step cadence only)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ps.WRITE_PNG = args.png
    data, names = build_all(Path(args.runs) / "objective_switch", Path(args.out_dir) / "figures",
                            cadence_only=not args.all_records)
    summary = Path(args.out_dir) / "data" / "objective_switch_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(summarise(data), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fork_step": data.fork_step, "particles": data.particles,
        "records_per_arm": len(data.steps), "panels": names,
    }))


if __name__ == "__main__":
    main()
