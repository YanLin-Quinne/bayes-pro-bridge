#!/usr/bin/env python3
"""Figure 13 (competence) and its plotted numbers.

x: mean correct-label probability of the frozen backbone on the training set (pre-adaptation).
(a) near-starved fraction under pure mixture (beta = 0) on the eight test sets.
(b) macro accuracy of independent training (beta = 1) minus pure mixture.

usage: python analysis/fig_competence.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import json

import matplotlib as mpl

import fig_common as FC
import metrics as M
import panel_style as PS

HEIGHT_MM = 62.0
N_TRAIN = 170300
BACKBONES = (  # (key, run name, display name)
    ("llama3_2_1b", "llama3.2-1b", "Llama 3.2-1B"),
    ("llama3_2_3b", "llama3.2-3b", "Llama 3.2-3B"),
    ("qwen2_1_5b_instruct", "qwen2-1.5b-instruct", "Qwen2-1.5B-Instruct"),
    ("qwen3_4b_base", "qwen3-4b-base", "Qwen3-4B"),
)
STYLE = {  # same colours and markers as the beta sweep figure
    "llama3_2_1b": dict(color="#56B4E9", marker="o", mfc="white"),
    "llama3_2_3b": dict(color="#0072B2", marker="s", mfc="#0072B2"),
    "qwen2_1_5b_instruct": dict(color="#D55E00", marker="^", mfc="#D55E00"),
    "qwen3_4b_base": dict(color="#009E73", marker="D", mfc="#009E73"),
}
# Label offsets in points (layout only), per panel.
OFFSETS = {
    "a": {"llama3_2_1b": (6, -3), "llama3_2_3b": (6, 5), "qwen2_1_5b_instruct": (6, -9), "qwen3_4b_base": (-4, 8)},
    "b": {"llama3_2_1b": (6, -3), "llama3_2_3b": (6, 5), "qwen2_1_5b_instruct": (0, -10), "qwen3_4b_base": (-4, 10)},
}
HA = {"a": {"qwen3_4b_base": "right"}, "b": {"qwen2_1_5b_instruct": "center", "qwen3_4b_base": "right"}}
LOW = {"a": -0.04, "b": -0.15}  # lower y limit as a fraction of the largest value
PANELS = (
    ("a", "pure_pro_starved_frac", "Near-starved fraction, pure mixture"),
    ("b", "bayes_minus_pro_acc", "Accuracy gap,\nindependent training $-$ pure mixture"),
)


def compute(runs):
    points = []
    for key, run_name, name in BACKBONES:
        summary = json.loads((runs / "pre_adaptation" / run_name / "summary.json").read_text())
        pre = summary["statistics"]["pooled"]
        assert pre["count"] == N_TRAIN, (key, pre["count"])
        pure = runs / f"{run_name}_beta0_m8"
        diag = FC.responsibility_stats(pure)
        assert diag["m"] == 8
        pro = M.summarize(M.load_eval_dir(pure))["macro"]
        bayes = M.summarize(M.load_eval_dir(runs / f"{run_name}_beta1_m8"))["macro"]
        assert pro["n"] == 8 and bayes["n"] == 8
        points.append({
            "key": key, "name": name,
            "pre_adapt_mean_correct_prob": pre["mean"],
            "pre_adapt_frac_below_1_over_8": pre["fraction_below_1_over_8"],
            "pure_pro_starved_frac": diag["starved_frac"],
            "pure_pro_macro_acc": pro["accuracy"],
            "pure_bayes_macro_acc": bayes["accuracy"],
            "bayes_minus_pro_acc": bayes["accuracy"] - pro["accuracy"],
        })
    return {"points": points, "n_records": len(points)}


def draw(d, stem):
    pts = d["points"]
    assert d["n_records"] == len(pts) == 4
    fig = PS.new_canvas(PS.FULL_WIDTH_MM, HEIGHT_MM)
    mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    left, right, top, bottom, hgap = 16.0, 3.0, 6.0, 12.0, 19.0
    width = (PS.FULL_WIDTH_MM - left - right - hgap) / 2
    height = HEIGHT_MM - top - bottom
    for i, (letter, field, ylabel) in enumerate(PANELS):
        ax = PS.rect_mm(fig, left + i * (width + hgap), bottom, width, height)
        for p in pts:
            st = STYLE[p["key"]]
            x, y = p["pre_adapt_mean_correct_prob"], p[field]
            ax.plot([x], [y], ls="none", ms=5.5, mec=st["color"], mew=1.2, **st)
            ax.annotate(p["name"], (x, y), xytext=OFFSETS[letter][p["key"]],
                        textcoords="offset points", fontsize=PS.FONT_NOTE,
                        color=st["color"] if p["key"] != "llama3_2_1b" else "#2A7FB0",
                        ha=HA[letter].get(p["key"], "left"), va="center")
        ymax = max(p[field] for p in pts)
        ax.set_xlim(0.38, 0.60)
        ax.set_ylim(LOW[letter] * ymax, 1.12 * ymax)
        ax.set_xlabel("Mean correct-label probability before adaptation")
        ax.set_ylabel(ylabel)
        ax.text(-0.20, 1.02, f"({letter})", transform=ax.transAxes,
                fontsize=PS.FONT_PANEL_LETTER, fontweight="bold", va="bottom", ha="left")
    FC.save_figure(fig, stem, "fig_competence", PS.FULL_WIDTH_MM)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    data = FC.write_data(args.out_dir, "competence", compute(args.runs))
    draw(data, FC.figure_stem(args.out_dir, "fig_competence"))


if __name__ == "__main__":
    main()
