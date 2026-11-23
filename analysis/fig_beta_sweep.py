#!/usr/bin/env python3
"""Figure 4 (beta sweep) and its plotted numbers.

For four backbones and six values of beta: the macro held-out NLL, accuracy and signed calibration gap over
the eight test sets, and the near-starved fraction Pr(r < 0.01) of the raw responsibilities.

usage: python analysis/fig_beta_sweep.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import fig_common as FC
import metrics as M
import panel_style as PS

HEIGHT_MM = 120.0
BACKBONES = (  # (key, run name, display name)
    ("llama3_2_1b", "llama3.2-1b", "Llama 3.2-1B"),
    ("llama3_2_3b", "llama3.2-3b", "Llama 3.2-3B"),
    ("qwen2_1_5b_instruct", "qwen2-1.5b-instruct", "Qwen2-1.5B-Instruct"),
    ("qwen3_4b_base", "qwen3-4b-base", "Qwen3-4B"),
)
BETAS = ("0", "0.01", "0.1", "0.2", "0.3", "1")
# Okabe-Ito colours. The unstable Llama 3.2-1B is lighter, dashed and hollow.
STYLE = {
    "llama3_2_1b": dict(color="#56B4E9", ls="--", marker="o", mfc="white", lw=1.0),
    "llama3_2_3b": dict(color="#0072B2", ls="-", marker="s", mfc="#0072B2", lw=1.2),
    "qwen2_1_5b_instruct": dict(color="#D55E00", ls="-", marker="^", mfc="#D55E00", lw=1.2),
    "qwen3_4b_base": dict(color="#009E73", ls="-", marker="D", mfc="#009E73", lw=1.2),
}
PANELS = (  # (field, y label, panel letter)
    ("nll", "Held-out NLL ($R_{\\mathrm{mix}}$)", "a"),
    ("accuracy", "Macro accuracy", "b"),
    ("starved_frac", "Near-starved fraction", "c"),
    ("signed_gap", "Signed gap (conf. $-$ acc.)", "d"),
)


def compute(runs):
    out = {"betas": list(BETAS), "backbones": [{"key": k, "name": n} for k, _, n in BACKBONES], "data": {}}
    for key, run_name, _ in BACKBONES:
        per = {}
        for beta in BETAS:
            run = runs / f"{run_name}_beta{beta}_m8"
            macro = M.summarize(M.load_eval_dir(run))["macro"]
            assert macro["n"] == 8, (key, beta, macro["n"])
            diag = FC.responsibility_stats(run)
            assert diag["m"] == 8, (key, beta, diag["m"])
            per[beta] = {"nll": macro["nll"], "accuracy": macro["accuracy"],
                         "mean_confidence": macro["mean_confidence"], "signed_gap": macro["gap"],
                         "starved_frac": diag["starved_frac"], "ess_over_m": diag["ess_over_m"]}
        out["data"][key] = per
    return out


def draw(d, stem):
    betas, backbones = d["betas"], d["backbones"]
    assert len(betas) == 6 and len(backbones) == 4
    assert sum(len(d["data"][b["key"]]) for b in backbones) == 24
    x = list(range(len(betas)))

    fig = PS.new_canvas(PS.FULL_WIDTH_MM, HEIGHT_MM)
    left, right, top, bottom, hgap, vgap = 17.0, 3.0, 10.0, 11.0, 17.0, 11.0
    width = (PS.FULL_WIDTH_MM - left - right - hgap) / 2
    height = (HEIGHT_MM - top - bottom - vgap) / 2
    handles = []
    for i, (field, ylabel, letter) in enumerate(PANELS):
        row, col = divmod(i, 2)
        x0 = left + col * (width + hgap)
        y0 = bottom + (1 - row) * (height + vgap)
        ax = PS.rect_mm(fig, x0, y0, width, height)
        for bb in backbones:
            ys = [d["data"][bb["key"]][b][field] for b in betas]
            st = STYLE[bb["key"]]
            line, = ax.plot(x, ys, label=bb["name"], ms=3.6, mec=st["color"], mew=0.9, **st)
            if i == 0:
                handles.append(line)
        if field == "signed_gap":
            ax.axhline(0.0, color=PS.GREY, lw=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(betas)
        ax.set_xlim(-0.3, len(betas) - 0.7)
        ax.set_ylabel(ylabel)
        if row == 1:
            ax.set_xlabel(r"$\beta$")
        ax.text(-0.19 if col == 0 else -0.22, 1.05, f"({letter})", transform=ax.transAxes,
                fontsize=PS.FONT_PANEL_LETTER, fontweight="bold", va="bottom", ha="left")
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.8)
    FC.save_figure(fig, stem, "figure_beta_sweep", PS.FULL_WIDTH_MM)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    data = FC.write_data(args.out_dir, "beta_sweep", compute(args.runs))
    draw(data, FC.figure_stem(args.out_dir, "figure_beta_sweep"))


if __name__ == "__main__":
    main()
