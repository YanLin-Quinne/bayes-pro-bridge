"""Shared style, canvas geometry, export and layout checks for all paper panels.

Every panel is drawn on a canvas whose width equals the width at which LaTeX
includes it (subfigure of 0.485 textwidth, or full textwidth). The inclusion
scale is therefore 1.0 and font sizes in points are the sizes printed on the
page. All panel scripts import the constants below, so the top and bottom rows
of a composite figure cannot drift apart.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# main.tex: \usepackage{fullpage} on a letter page gives \textwidth = 6.5 in.
TEXTWIDTH_MM = 6.5 * 25.4
SUBFIGURE_FRACTION = 0.485
PANEL_WIDTH_MM = round(TEXTWIDTH_MM * SUBFIGURE_FRACTION, 2)
FULL_WIDTH_MM = round(TEXTWIDTH_MM, 2)

# Font sizes in points, identical for every panel at inclusion width.
FONT_LABEL = 8.0
FONT_TICK = 7.5
FONT_LEGEND = 7.5
FONT_NOTE = 7.0
FONT_CELL = 7.0
FONT_PANEL_LETTER = 9.0
MIN_FONT_PT = 6.5

PURE_COLOR = "#7A5195"
BRIDGE_COLOR = "#E39C37"
GREY = "#555555"

METHOD_ORDER = ("beta0_pure_pro", "beta01_convex")
METHOD_COLORS = {"beta0_pure_pro": PURE_COLOR, "beta01_convex": BRIDGE_COLOR}
METHOD_MARKERS = {"beta0_pure_pro": "o", "beta01_convex": "^"}
METHOD_LABELS = {
    "beta0_pure_pro": "Pure mixture (β=0)",
    "beta01_convex": "Bridge (β=0.1)",
}
METHOD_HEADERS = {"beta0_pure_pro": "β=0", "beta01_convex": "β=0.1"}
DATASET_NAMES = {
    "boolq": "BoolQ", "hellaswag": "HellaSwag", "openbookqa": "OpenBookQA",
}

MM = 1.0 / 25.4


class LayoutError(RuntimeError):
    """Raised when a panel has clipped or overlapping text."""


def configure_matplotlib():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.default": "regular",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FONT_LABEL,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.labelpad": 2.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def new_canvas(width_mm, height_mm):
    configure_matplotlib()
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def canvas_mm(fig):
    width, height = fig.get_size_inches() / MM
    return float(width), float(height)


def add_axes_mm(fig, left, bottom, right, top, **kwargs):
    """Axes with margins given in mm from the figure edges."""
    width, height = canvas_mm(fig)
    return fig.add_axes(
        [left / width, bottom / height,
         (width - left - right) / width, (height - bottom - top) / height],
        **kwargs,
    )


def rect_mm(fig, left, bottom, width_mm, height_mm, **kwargs):
    width, height = canvas_mm(fig)
    return fig.add_axes(
        [left / width, bottom / height, width_mm / width, height_mm / height],
        **kwargs,
    )


def wrap_label(text, max_chars=22):
    """Break a long y label into two balanced lines at a space."""
    if len(text) <= max_chars or "\n" in text:
        return text
    spaces = [i for i, c in enumerate(text) if c == " "]
    cut = min(spaces, key=lambda i: abs(i - len(text) / 2))
    return text[:cut] + "\n" + text[cut + 1:]


def particle_tick_step(particles):
    return 1 if particles <= 10 else 2


def particle_ticks(particles):
    step = particle_tick_step(particles)
    positions = list(range(0, particles, step))
    return positions, [f"P{index + 1}" for index in positions]


def nice_ticks_avoiding(candidates, special, min_gap, transform=lambda v: v):
    """Candidates at least min_gap (after transform) away from every special
    tick, together with the special ticks."""
    kept = [
        value for value in candidates
        if all(abs(transform(value) - transform(s)) >= min_gap for s in special)
    ]
    return sorted(set(kept) | set(special))


def positive_log_ticks(low, high):
    start = math.floor(math.log10(low))
    stop = math.ceil(math.log10(high))
    return [10.0 ** k for k in range(start, stop + 1) if low <= 10.0 ** k <= high]


def _hidden_tick_labels(fig):
    """Tick labels whose tick lies outside the view limits are not drawn."""
    hidden = set()
    for axis in fig.axes:
        for ax_axis, limits in ((axis.xaxis, axis.get_xlim()),
                                (axis.yaxis, axis.get_ylim())):
            low, high = sorted(limits)
            span = high - low
            for tick in ax_axis.get_major_ticks():
                loc = tick.get_loc()
                if not (low - 1e-9 * span <= loc <= high + 1e-9 * span):
                    hidden.add(id(tick.label1))
                    hidden.add(id(tick.label2))
    return hidden


def text_artists(fig):
    hidden = _hidden_tick_labels(fig)
    return [
        artist for artist in fig.findobj(Text)
        if artist.get_visible() and artist.get_text().strip()
        and id(artist) not in hidden
    ]


def _describe(artist):
    return artist.get_text().replace("\n", " / ")[:48]


@dataclass
class LayoutReport:
    name: str
    width_mm: float
    height_mm: float
    inclusion_width_mm: float
    scale: float
    min_font_pt: float
    min_effective_pt: float
    label_pt: float
    tick_pt: float
    legend_texts: list = field(default_factory=list)
    clipped: list = field(default_factory=list)
    overlaps: list = field(default_factory=list)
    texts: list = field(default_factory=list)

    def as_dict(self):
        return self.__dict__.copy()


def layout_report(fig, name, inclusion_width_mm, tolerance_px=0.75):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    width_mm, height_mm = canvas_mm(fig)
    scale = inclusion_width_mm / width_mm
    entries = []
    for artist in text_artists(fig):
        box = artist.get_window_extent(renderer=renderer)
        if box.width >= 0.5 and box.height >= 0.5:
            entries.append((artist, box))

    clipped = [
        _describe(artist) for artist, box in entries
        if (box.x0 < figure_box.x0 - tolerance_px
            or box.y0 < figure_box.y0 - tolerance_px
            or box.x1 > figure_box.x1 + tolerance_px
            or box.y1 > figure_box.y1 + tolerance_px)
    ]
    overlaps = []
    for index, (first, box_a) in enumerate(entries):
        for second, box_b in entries[index + 1:]:
            x_overlap = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            y_overlap = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if x_overlap > tolerance_px and y_overlap > tolerance_px:
                overlaps.append([_describe(first), _describe(second)])

    sizes = [artist.get_fontsize() for artist, _ in entries]
    legend_texts = []
    for legend in fig.findobj(mpl.legend.Legend):
        legend_texts.extend(text.get_text() for text in legend.get_texts())
    label_sizes = [
        axis_label.get_fontsize() for axis in fig.axes
        for axis_label in (axis.xaxis.label, axis.yaxis.label)
        if axis_label.get_text()
    ]
    tick_sizes = [
        tick.get_fontsize() for axis in fig.axes
        for tick in axis.get_xticklabels() + axis.get_yticklabels()
        if tick.get_text()
    ]
    return LayoutReport(
        name=name, width_mm=round(width_mm, 2), height_mm=round(height_mm, 2),
        inclusion_width_mm=inclusion_width_mm, scale=round(scale, 4),
        min_font_pt=min(sizes), min_effective_pt=round(min(sizes) * scale, 2),
        label_pt=label_sizes[0] if label_sizes else float("nan"),
        tick_pt=tick_sizes[0] if tick_sizes else float("nan"),
        legend_texts=legend_texts, clipped=clipped, overlaps=overlaps,
        texts=sorted({_describe(artist) for artist, _ in entries}),
    )


# Set to True (the --png flag of the figure scripts) to also write a 600 dpi PNG next to each PDF.
WRITE_PNG = False


def save_panel(fig, stem, name, inclusion_width_mm, qa_dir=None, strict=True,
               dpi=600):
    """Write <stem>.pdf (and <stem>.png when WRITE_PNG) at the canvas size (no tight bbox,
    so the inclusion scale stays exactly inclusion_width_mm / canvas width)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    report = layout_report(fig, name, inclusion_width_mm)
    problems = []
    if report.clipped:
        problems.append(f"clipped text: {report.clipped}")
    if report.overlaps:
        problems.append(f"overlapping text: {report.overlaps}")
    if report.min_effective_pt < MIN_FONT_PT:
        problems.append(f"font below {MIN_FONT_PT} pt: {report.min_effective_pt}")
    fig.savefig(stem.with_suffix(".pdf"))
    if WRITE_PNG:
        fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    plt.close(fig)
    if qa_dir is not None:
        qa_dir = Path(qa_dir)
        qa_dir.mkdir(parents=True, exist_ok=True)
        payload = report.as_dict()
        payload["problems"] = problems
        (qa_dir / f"{stem.name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if problems and strict:
        raise LayoutError(f"{name}: " + "; ".join(problems))
    return report
