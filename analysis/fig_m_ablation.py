#!/usr/bin/env python3
"""Figure 7 (number of particles) and its plotted numbers.

For Llama 3.2-3B and Qwen3-4B at beta 0 and 0.1 and m = 4, 8, 16 particles: the near-starved fraction
Pr(r < 0.01) and the macro held-out NLL over the eight test sets. The m = 8 runs are the main runs.

usage: python analysis/fig_m_ablation.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.lines import Line2D

import fig_common as FC
import metrics as M
import panel_style as PS

RUN_NAME = {"llama3_2_3b": "llama3.2-3b", "qwen3_4b_base": "qwen3-4b-base"}
BETAS_RUN = ("0", "0.1")
M_VALUES = (4, 8, 16)
N_ITEMS = sum(M.EXPECTED_N[d] for d in M.DATASETS)
BACKBONES = {  # markers as in the beta sweep figure, backbones told apart by marker and line style
    "llama3_2_3b": dict(name="Llama 3.2-3B", marker="s", ls="-"),
    "qwen3_4b_base": dict(name="Qwen3-4B", marker="D", ls=":"),
}
BETAS = {"0": ("beta0_pure_pro", PS.METHOD_LABELS["beta0_pure_pro"]),
         "0.1": ("beta01_convex", PS.METHOD_LABELS["beta01_convex"])}
MS = 3.2
YLAB = {"starved_frac": r"Near-starved $\mathrm{Pr}(\mathit{r}<0.01)$", "nll": "Macro held-out NLL"}


def compute(runs):
    res = {"datasets": list(M.DATASETS), "threshold": FC.STARVED_THRESHOLD, "n_items_total": N_ITEMS, "values": {}}
    for bb, name in RUN_NAME.items():
        for b in BETAS_RUN:
            rows = []
            for m in M_VALUES:
                run = runs / f"{name}_beta{b}_m{m}"
                diag = FC.responsibility_stats(run)
                assert diag["m"] == m, (bb, m, b, diag["m"])
                rows.append({"m": m, "nll": M.summarize(M.load_eval_dir(run))["macro"]["nll"],
                             "starved_frac": diag["starved_frac"], "ess_over_m": diag["ess_over_m"]})
            res["values"].setdefault(bb, {})[b] = rows
    return res


def draw(d, stem):
    data = d["values"]
    mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    fig = PS.new_canvas(PS.FULL_WIDTH_MM, 52)
    gap, wp = 14.0, (PS.FULL_WIDTH_MM - 13 - 2 - 14) / 2
    axes = [PS.rect_mm(fig, 13, 10, wp, 38), PS.rect_mm(fig, 13 + wp + gap + 0, 10, wp, 38)]
    for ax, key, letter in zip(axes, ("starved_frac", "nll"), "ab"):
        for bb, st in BACKBONES.items():
            for b, (mk, _) in BETAS.items():
                rows = data[bb][b]
                ax.plot([r["m"] for r in rows], [r[key] for r in rows], color=PS.METHOD_COLORS[mk],
                        ls=st["ls"], marker=st["marker"], ms=MS, lw=1.2, mfc=PS.METHOD_COLORS[mk],
                        mec=PS.METHOD_COLORS[mk])
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16])
        ax.set_xticklabels(["4", "8", "16"])
        ax.minorticks_off()
        ax.set_xlim(3.4, 18.5)
        ax.set_xlabel(r"Particles $\mathit{m}$")
        ax.set_ylabel(YLAB[key])
        ax.margins(y=0.08)
        ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes,
                fontsize=PS.FONT_PANEL_LETTER, fontweight="bold", va="bottom", ha="left")
    handles = [Line2D([], [], color=PS.METHOD_COLORS[mk], lw=1.2, label=lab) for mk, lab in BETAS.values()]
    handles += [Line2D([], [], color=PS.GREY, ls=st["ls"], marker=st["marker"], ms=MS, lw=1.2, label=st["name"])
                for st in BACKBONES.values()]
    axes[0].legend(handles=handles, loc="upper left", fontsize=PS.FONT_LEGEND, handlelength=2.2,
                   labelspacing=0.25, borderaxespad=0.2)
    FC.save_figure(fig, stem, "fig_m_ablation", PS.FULL_WIDTH_MM)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    data = FC.write_data(args.out_dir, "m_ablation", compute(args.runs))
    draw(data, FC.figure_stem(args.out_dir, "fig_m_ablation"))


if __name__ == "__main__":
    main()
