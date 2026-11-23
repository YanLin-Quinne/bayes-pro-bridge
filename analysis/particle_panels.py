"""Shared plotters for the four post hoc particle trajectory panels.

Used by starvation.py (Llama 3.2-3B, Figures 5a, 5b and 10) and by
starvation_qwen3.py (Qwen3-4B, Figure 12). The particle count m is read from
the data, never hard-coded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from panel_style import (
    DATASET_NAMES, FONT_CELL, FONT_NOTE, GREY, PANEL_WIDTH_MM,
    METHOD_COLORS, METHOD_HEADERS, METHOD_LABELS, METHOD_MARKERS, METHOD_ORDER,
    add_axes_mm, canvas_mm, nice_ticks_avoiding, new_canvas, particle_ticks,
    rect_mm,
)

MASS_CMAP = "BrBG"
NLL_CMAP = "viridis"
LABEL_COLORS = {
    "A": "#4E79A7", "B": "#F28E2B", "C": "#59A14F", "D": "#B07AA1",
}
UNIFORM_GUESSING_NLL = math.log(4)
LEFT_MM, RIGHT_MM, GAP_MM = 9.5, 1.5, 2.5

# Common vertical layout of all heatmap-style panels and the ESS panel, so the
# rows of a composite figure align at 62 mm canvas height (m = 8).
CANVAS_HEIGHT_MM = 62.0
BOTTOM_MM = 12.5
TOP_MM = 9.0
# The ESS panel stacks two legends above its axes and needs more room.
ESS_TOP_MM = 12.0
ESS_BOTTOM_MM = 9.0
# Right-hand zone of the two heatmap panels that carry a vertical colour bar:
# gap to the bar, bar width, and the room for tick labels plus rotated label.
BAR_GAP_MM, BAR_WIDTH_MM, BAR_ZONE_MM = 1.6, 2.0, 16.0
# Drop of the 'Final' label below the axes: tick length 2 pt, pad 2 pt and one
# 7.5 pt tick label line at 1.2 line spacing.
FINAL_LABEL_DROP_PT = 13.0
# Minimum centre distance of two colour bar tick labels (7.5 pt text is 2.7 mm).
BAR_TICK_GAP_MM = 3.6


@dataclass
class TrajectoryData:
    """Matrices are indexed [particle, checkpoint]."""
    particles: int
    steps: list
    final_step: int
    datasets: list
    ess_macro: dict
    ess_low: dict
    ess_high: dict
    mass: dict
    nll: dict
    dominant: dict = field(default_factory=dict)  # method -> [[(label, fraction)]]
    methods: tuple = METHOD_ORDER


def checkpoint_ticks(data):
    """Tick labels in thousand steps. The final column keeps its tick but gets
    no tick label: add_final_label writes 'Final' one line lower, so it cannot
    collide with '30' even when the columns are only 3.6 mm wide."""
    labels = [
        "" if step == data.final_step else str(step // 1000)
        for step in data.steps
    ]
    return list(range(len(data.steps))), labels


def add_final_label(ax, data):
    """'Final' on the second tick label line below the last column."""
    last = len(data.steps) - 1
    ax.annotate(
        "Final", xy=(last, 0.0), xycoords=("data", "axes fraction"),
        xytext=(0, -FINAL_LABEL_DROP_PT), textcoords="offset points",
        ha="center", va="top", fontsize=mpl.rcParams["xtick.labelsize"],
    )


def checkpoint_axis_label(data):
    return f"Checkpoint in thousand steps (Final = {data.final_step / 1000:.1f})"


def _heat_height_mm(particles):
    """Heatmap height. 40.5 mm for m up to 12, grows for larger m so that
    rows stay at least 3.2 mm tall."""
    return max(CANVAS_HEIGHT_MM - BOTTOM_MM - TOP_MM, 3.2 * particles)


def _canvas_height_mm(particles):
    return BOTTOM_MM + _heat_height_mm(particles) + TOP_MM


def _style_particle_axis(ax, data, first):
    positions, labels = particle_ticks(data.particles)
    ax.set_yticks(positions, labels if first else [""] * len(labels))
    ticks, tick_labels = checkpoint_ticks(data)
    ax.set_xticks(ticks, tick_labels)
    ax.tick_params(axis="both", length=2.0)
    add_final_label(ax, data)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _dataset_list(datasets):
    return ", ".join(DATASET_NAMES.get(name, name) for name in datasets)


def _top_legend(fig, handles, ncol, y_mm_from_top=0.6, **kwargs):
    _, height = canvas_mm(fig)
    return fig.legend(
        handles=handles, loc="upper center", ncol=ncol,
        bbox_to_anchor=(0.5, 1.0 - y_mm_from_top / height), frameon=False,
        **kwargs,
    )


def plot_ess(data):
    particles = data.particles
    height = _canvas_height_mm(particles)
    fig = new_canvas(PANEL_WIDTH_MM, height)
    ax = add_axes_mm(fig, 12.5, ESS_BOTTOM_MM, 2.0, ESS_TOP_MM)
    x = np.asarray(data.steps, dtype=float) / 1000.0
    for method in data.methods:
        color = METHOD_COLORS[method]
        ax.fill_between(
            x, data.ess_low[method], data.ess_high[method], color=color,
            alpha=0.14, linewidth=0,
        )
        ax.plot(
            x, data.ess_macro[method], color=color,
            marker=METHOD_MARKERS[method], lw=1.3, ms=3.6,
        )
    ax.axhline(particles, color=GREY, lw=0.8, ls=":", zorder=0)
    low = min(float(np.min(data.ess_low[m])) for m in data.methods)
    span = particles - low
    ax.set_ylim(low - 0.04 * span, particles + 0.14 * span)
    ax.annotate(
        f"Uniform, ESS = {particles}", xy=(x[0] - 0.4, particles),
        xytext=(0, 2.0), textcoords="offset points", va="bottom", ha="left",
        fontsize=FONT_NOTE, color=GREY,
    )
    final_x = data.final_step / 1000.0
    ax.axvline(final_x, color="#777777", lw=0.7, ls="--", zorder=0)
    ax.annotate(
        "Final", xy=(final_x, 0.03), xycoords=("data", "axes fraction"),
        xytext=(-2.0, 0), textcoords="offset points", ha="right", va="bottom",
        fontsize=FONT_NOTE, color=GREY, rotation=90,
    )
    ax.set_xlim(x[0] - 0.7, final_x + 0.8)
    ax.set_xticks([t for t in (5, 10, 15, 20, 25, 30) if x[0] - 0.7 <= t <= x[-1]])
    ax.set_xlabel("Checkpoint in thousand steps")
    ax.set_ylabel("Mean raw responsibility ESS")

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
               lw=1.3, ms=3.6, label=METHOD_LABELS[m])
        for m in data.methods
    ]
    _top_legend(fig, method_handles, 2, 0.6, columnspacing=1.2,
                handlelength=2.0, handletextpad=0.5)
    band = Patch(
        facecolor="#9A9A9A", alpha=0.35, linewidth=0,
        label=f"Band: range over {_dataset_list(data.datasets)}",
    )
    _top_legend(fig, [band], 1, 5.4, handlelength=1.6, handletextpad=0.5)
    return fig


def _thin(ticks, special, gap, transform=float):
    """Greedy thinning so neighbouring colour bar labels never touch."""
    kept = []
    for tick in ticks:
        if tick in special or all(
                abs(transform(tick) - transform(k)) >= gap for k in kept):
            kept.append(tick)
    return sorted(kept)


def _vertical_colorbar(fig, image, label, ticks, tick_labels, bottom_mm,
                       height_mm):
    """One colour bar to the right of the two heatmaps, as tall as the maps."""
    width, _ = canvas_mm(fig)
    left = width - BAR_ZONE_MM + BAR_GAP_MM
    cax = rect_mm(fig, left, bottom_mm, BAR_WIDTH_MM, height_mm)
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.outline.set_linewidth(0.5)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(tick_labels)
    colorbar.minorticks_off()
    colorbar.set_label(label, labelpad=2.5)
    cax.tick_params(length=2.0, pad=1.5)
    return colorbar


def _two_heatmap_axes(fig, bottom_mm, top_mm, right_mm=RIGHT_MM):
    width_each = (PANEL_WIDTH_MM - LEFT_MM - right_mm - GAP_MM) / 2.0
    axes = []
    for index in range(2):
        left = LEFT_MM + index * (width_each + GAP_MM)
        axes.append(add_axes_mm(
            fig, left, bottom_mm, PANEL_WIDTH_MM - left - width_each, top_mm,
        ))
    return axes


def _shared_xlabel(fig, data, y_mm, right_mm=RIGHT_MM):
    width, height = canvas_mm(fig)
    fig.text(
        (LEFT_MM + (width - LEFT_MM - right_mm) / 2) / width, y_mm / height,
        checkpoint_axis_label(data), ha="center", va="center",
        fontsize=mpl.rcParams["axes.labelsize"],
    )


def plot_mass(data):
    particles = data.particles
    heat = _heat_height_mm(particles)
    fig = new_canvas(PANEL_WIDTH_MM, _canvas_height_mm(particles))
    axes = _two_heatmap_axes(fig, BOTTOM_MM, TOP_MM, BAR_ZONE_MM)
    values = np.concatenate([data.mass[m].ravel() for m in data.methods])
    center = 1.0 / particles
    norm = TwoSlopeNorm(
        vmin=min(float(values.min()), center * 0.99), vcenter=center,
        vmax=max(float(values.max()), center * 1.01),
    )
    image = None
    for column, (ax, method) in enumerate(zip(axes, data.methods)):
        image = ax.imshow(
            data.mass[method], aspect="auto", cmap=MASS_CMAP, norm=norm,
            interpolation="nearest",
        )
        ax.set_title(METHOD_HEADERS[method], pad=2.5)
        _style_particle_axis(ax, data, column == 0)
    axes[0].set_ylabel("Particle")
    _shared_xlabel(fig, data, BOTTOM_MM - 9.7, BAR_ZONE_MM)
    span = float(norm.vmax) - float(norm.vmin)
    gap = BAR_TICK_GAP_MM / heat * max(span, 1e-9)
    candidates = [c / 20 for c in range(0, 21)]
    ticks = nice_ticks_avoiding(
        [c for c in candidates if norm.vmin - 1e-9 <= c <= norm.vmax + 1e-9],
        [center], gap,
    )
    ticks = _thin(ticks, [center], gap)
    labels = [
        f"1/{particles}" if math.isclose(t, center) else f"{t:g}" for t in ticks
    ]
    colorbar = _vertical_colorbar(
        fig, image, "Mean raw responsibility", ticks, labels, BOTTOM_MM, heat,
    )
    colorbar.ax.axhline(center, color="#222222", lw=0.8)
    return fig


def nll_norm_and_ticks(values, bar_mm=None):
    """Colour norm and tick values. With bar_mm (height of the vertical colour
    bar) neighbouring tick labels are at least BAR_TICK_GAP_MM apart."""
    positive = values[values > 0]
    vmin, vmax = float(values.min()), float(values.max())
    use_log = len(positive) > 0 and positive.max() / positive.min() > 30
    if use_log:
        norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
        candidates = [
            m * 10.0 ** k for k in range(-4, 3) for m in (1.0, 2.0, 5.0)
            if norm.vmin <= m * 10.0 ** k <= norm.vmax
        ]
        transform, gap = math.log, 0.45
        if bar_mm:
            gap = BAR_TICK_GAP_MM / bar_mm * math.log(norm.vmax / norm.vmin)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        candidates = list(np.linspace(vmin, vmax, 5))
        transform, gap = float, 0.1 * (vmax - vmin)
        if bar_mm:
            gap = BAR_TICK_GAP_MM / bar_mm * (vmax - vmin)
    special = (
        [UNIFORM_GUESSING_NLL]
        if norm.vmin <= UNIFORM_GUESSING_NLL <= norm.vmax else []
    )
    ticks = nice_ticks_avoiding(candidates, special, gap, transform)
    return norm, _thin(ticks, special, gap, transform)


def plot_nll(data):
    particles = data.particles
    heat = _heat_height_mm(particles)
    fig = new_canvas(PANEL_WIDTH_MM, _canvas_height_mm(particles))
    axes = _two_heatmap_axes(fig, BOTTOM_MM, TOP_MM, BAR_ZONE_MM)
    values = np.concatenate([data.nll[m].ravel() for m in data.methods])
    norm, ticks = nll_norm_and_ticks(values, heat)
    image = None
    for column, (ax, method) in enumerate(zip(axes, data.methods)):
        image = ax.imshow(
            data.nll[method], aspect="auto", cmap=NLL_CMAP, norm=norm,
            interpolation="nearest",
        )
        ax.set_title(METHOD_HEADERS[method], pad=2.5)
        _style_particle_axis(ax, data, column == 0)
    axes[0].set_ylabel("Particle")
    _shared_xlabel(fig, data, BOTTOM_MM - 9.7, BAR_ZONE_MM)
    labels = [
        "log 4" if math.isclose(t, UNIFORM_GUESSING_NLL) else f"{t:g}"
        for t in ticks
    ]
    colorbar = _vertical_colorbar(
        fig, image, "Particle-level gold-label NLL", ticks, labels,
        BOTTOM_MM, heat,
    )
    if norm.vmin <= UNIFORM_GUESSING_NLL <= norm.vmax:
        colorbar.ax.axhline(
            UNIFORM_GUESSING_NLL, color="#111111", lw=1.0, ls=(0, (2.0, 1.2)),
        )
    return fig


def plot_occupancy(data):
    particles = data.particles
    fig = new_canvas(PANEL_WIDTH_MM, _canvas_height_mm(particles))
    axes = _two_heatmap_axes(fig, BOTTOM_MM, TOP_MM)
    colours = {
        label: np.array(mpl.colors.to_rgb(colour))
        for label, colour in LABEL_COLORS.items()
    }
    for column, (ax, method) in enumerate(zip(axes, data.methods)):
        image = np.ones((particles, len(data.steps), 3))
        for row_index in range(particles):
            for col_index in range(len(data.steps)):
                label, fraction = data.dominant[method][row_index][col_index]
                strength = min(1.0, max(0.0, (fraction - 0.25) / 0.75))
                image[row_index, col_index] = (
                    (1.0 - strength) * np.ones(3) + strength * colours[label]
                )
                ax.text(
                    col_index, row_index, label, ha="center", va="center",
                    fontsize=FONT_CELL, fontweight="bold",
                    color="white" if strength > 0.72 else "#111111",
                )
        ax.imshow(image, aspect="auto", interpolation="nearest")
        ax.set_title(METHOD_HEADERS[method], pad=2.5)
        _style_particle_axis(ax, data, column == 0)
    axes[0].set_ylabel("Particle")
    _shared_xlabel(fig, data, 2.8)
    handles = [Patch(facecolor="none", edgecolor="none", label="Dominant label:")]
    handles += [Patch(facecolor=colours[label], label=label) for label in LABEL_COLORS]
    _top_legend(fig, handles, len(handles), 0.6, handlelength=1.0,
                handletextpad=0.4, columnspacing=1.0)
    return fig
