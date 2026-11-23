#!/usr/bin/env python3
"""Table 7 and Figures 2 and 3 (Case Study 1) from the summaries in results/case_study1.

Reads summary.json, traces.npz and phase_transition.json, which analysis/case_study1_summary.py computes from the
chains of scripts/case_study1.sh and which the repository ships for the chains of the paper. Writes

  <out-dir>/tables/cs1.tex                  Table 7 (Case Study 1), the rows as printed
  <out-dir>/figures/fig_case_study1.pdf     Figure 2 (Case Study 1)
  <out-dir>/figures/fig_cs1_phase.pdf       Figure 3 (phase transition in beta)

usage: python analysis/case_study1.py --results results/case_study1 --out-dir outputs [--png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import case_study1_summary as CS
import fig_common as FC
import panel_style as PS

N_COLORS = {200: "#F0B566", 1000: "#E39C37", 5000: "#9C5B12"}
N_MARKERS = {200: "s", 1000: "o", 5000: "^"}


def sd_tex(x):
    return "{<}10^{-4}" if x < 0.00005 else f"{x:.4f}"


def table_tex(S):
    """Table 7: held-out NLL from the three initialisations, ECE, ESS / m and the time to halve the gap."""
    rows = []
    for b in CS.BETAS:
        cells = []
        for i in CS.INITS:
            st = S["main"][f"{b:g}|{CS.INIT_NAME[i]}"]
            cells.append(f"${st['heldout_mean']:.4f}\\pm{sd_tex(st['heldout_sd'])}$")
        label = {0.0: "\\ac{PrO} posterior, $\\beta=0$",
                 1.0: "Bayesian posterior, $\\beta=1$"}.get(b, f"Bridge, $\\beta={b:g}$")
        half = S["main"][f"{b:g}|slot M=16"]["half_time_median"]
        ece = S["ece_t200"][f"{b:g}|prior"]["mean"]
        ess = S["main"][f"{b:g}|prior"]["ess_over_m"]
        rows.append(f"{label} & " + " & ".join(cells)
                    + f" & ${ece:.3f}$ & ${ess:.2f}$ & ${round(half + 1e-9, 2):.2f}$ \\\\")
    return "\n".join(rows) + "\n"


def figure_case_study1(S, z, out_dir):
    """Figure 2: training-set mixture risk, minimum responsibility, held-out NLL and the decomposition for m = 8."""
    from matplotlib.lines import Line2D

    PS.configure_matplotlib()
    W, H = PS.FULL_WIDTH_MM, 104.0
    fig = PS.new_canvas(W, H)
    left, right, gap_x, bottom, top, gap_y = 15.0, 3.0, 17.0, 10.0, 9.0, 15.0
    pw = (W - left - right - gap_x) / 2
    ph = (H - bottom - top - gap_y) / 2
    axes = {"a": PS.rect_mm(fig, left, bottom + ph + gap_y, pw, ph),
            "b": PS.rect_mm(fig, left + pw + gap_x, bottom + ph + gap_y, pw, ph),
            "c": PS.rect_mm(fig, left, bottom, pw, ph),
            "d": PS.rect_mm(fig, left + pw + gap_x, bottom, pw, ph)}
    col = {0.0: PS.PURE_COLOR, 0.01: PS.BRIDGE_COLOR}
    for b, key in ((0.0, "pure"), (0.01, "bridge")):
        st, mix, minr = z[f"steps_{key}"], z[f"r_mix_{key}"], z[f"min_r_{key}"]
        t = st * float(z[f"eps_{key}"])
        keep = t > 0
        for k in range(mix.shape[1]):
            axes["a"].plot(t[keep], mix[keep, k], color=col[b], lw=0.6, alpha=0.75)
            axes["b"].plot(t[keep], minr[keep, k], color=col[b], lw=0.6, alpha=0.75)
    best = S["task"]["best_risk"]
    ax = axes["a"]
    ax.set_xscale("log")
    ax.axhline(best, color=PS.GREY, ls=":", lw=0.8)
    ax.text(0.22, best + 0.006, "$H(Y\\mid X)$", fontsize=PS.FONT_NOTE, color=PS.GREY, va="bottom")
    ax.set_xlim(0.2, 1000)
    ax.set_ylim(0.33, 0.72)
    ax.set_xlabel("MFLD time $t$")
    ax.set_ylabel("Training-set $R_{\\mathrm{mix}}(Q_m^t)$")
    ax = axes["b"]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(0.01 / 32, color=PS.BRIDGE_COLOR, ls="--", lw=0.8)
    ax.text(0.22, 0.01 / 32 * 8.0, "$\\beta/m$, floor of $\\rho_{ij}$ for $\\beta=0.01$", fontsize=PS.FONT_NOTE,
            color=PS.BRIDGE_COLOR, va="bottom")
    ax.set_xlim(0.2, 1000)
    ax.set_ylim(1e-55, 1.0)
    ax.set_yticks([1e-48, 1e-36, 1e-24, 1e-12, 1.0])
    ax.set_xlabel("MFLD time $t$")
    ax.set_ylabel("Minimum raw\nresponsibility $r_{ij}$")
    # (c) held-out NLL of every chain at t = 200, by beta and initialisation
    ax = axes["c"]
    marks = {CS.INITS[0]: "o", CS.INITS[1]: "s", CS.INITS[2]: "^"}
    jitter = np.random.default_rng(0)
    for xi, b in enumerate(CS.BETAS):
        for j, i in enumerate(CS.INITS):
            v = S["main"][f"{b:g}|{CS.INIT_NAME[i]}"]["heldout_chains"]
            x = xi + (j - 1) * 0.22 + jitter.uniform(-0.06, 0.06, len(v))
            ax.plot(x, v, ls="none", marker=marks[i], ms=3.0, color=col.get(b, PS.GREY), mew=0)
    ax.set_xticks(range(len(CS.BETAS)))
    ax.set_xticklabels([f"{b:g}" for b in CS.BETAS])
    ax.set_xlim(-0.6, len(CS.BETAS) - 0.4)
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("Held-out NLL,\n$t$ from 150 to 200")
    handles = [Line2D([], [], ls="none", marker=marks[i], ms=3.5, color=PS.GREY, mew=0,
                      label=CS.INIT_NAME[i].replace("M=", "$M=$")) for i in CS.INITS]
    ax.legend(handles=handles, loc="lower right", fontsize=PS.FONT_LEGEND, handletextpad=0.3, borderaxespad=0.3,
              labelspacing=0.3)
    # (d) fraction of m = 8 chains in the intended decomposition
    ax = axes["d"]
    t0, f0, t1, f1 = z["t_intended_pure"], z["intended_pure"], z["t_intended_bridge"], z["intended_bridge"]
    ax.plot(t0[t0 > 0], f0[t0 > 0].mean(1), color=PS.PURE_COLOR, lw=1.2)
    ax.plot(t1[t1 > 0], f1[t1 > 0].mean(1), color=PS.BRIDGE_COLOR, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 1000)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("MFLD time $t$")
    ax.set_ylabel("Chains in the intended\ndecomposition ($m=8$)")
    legend = [Line2D([], [], color=PS.PURE_COLOR, lw=1.2, label="PrO posterior ($\\beta=0$)"),
              Line2D([], [], color=PS.BRIDGE_COLOR, lw=1.2, label="Bridge ($\\beta=0.01$)")]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=PS.FONT_LEGEND)
    for letter, ax in axes.items():
        ax.text(-0.21, 1.03, f"({letter})", transform=ax.transAxes, fontsize=PS.FONT_PANEL_LETTER,
                fontweight="bold", va="bottom", ha="left")
    FC.save_figure(fig, FC.figure_stem(out_dir, "fig_case_study1"), "fig_case_study1", W)


def figure_phase(data, out_dir):
    """Figure 3: held-out NLL and particle variance along v against beta, with the Gaussian prediction."""
    from matplotlib.lines import Line2D

    W, ph = PS.FULL_WIDTH_MM, 35.0
    left, right, gap_x, bottom, top = 18.0, 3.0, 17.0, 10.0, 13.0
    fig = PS.new_canvas(W, bottom + ph + top)
    pw = (W - left - right - gap_x) / 2
    axes = [PS.rect_mm(fig, left, bottom, pw, ph), PS.rect_mm(fig, left + pw + gap_x, bottom, pw, ph)]
    handles = []
    for key, d in data.items():
        n = int(key)
        rows = [r for r in d["rows"] if r["beta"] > 0]
        c, mk = N_COLORS[n], N_MARKERS[n]
        b = np.array([r["beta"] for r in rows])
        bayes = next(r["heldout"] for r in rows if r["beta"] == 1)
        axes[0].plot(b, [r["heldout"] - bayes for r in rows], marker=mk, ms=3.2, lw=0.9, color=c, mew=0)
        axes[1].plot(b, [r["var_v"] for r in rows], ls="none", marker=mk, ms=3.2, color=c, mew=0)
        st = CS.stability(n)
        grid = np.linspace(st["beta_crit"] + 1e-4, 1.0, 400)
        axes[1].plot(grid, [CS.gaussian_var_v(st, n, g) for g in grid], color=c, lw=0.9, ls="--")
        for ax in axes:
            ax.axvline(d["beta_crit"], color=c, lw=0.8, ls=":")
        handles.append(Line2D([], [], color=c, marker=mk, ms=3.5, mew=0, lw=0.9,
                              label=f"$n$ = {n:,}, $\\hat{{\\beta}}_{{crit}}$ = {d['beta_crit']:.3f}"))
    axes[0].axhline(0.0, color="black", lw=0.6)
    axes[0].set_ylabel("Held-out NLL minus\nthat at $\\beta=1$")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Variance of $\\mathbf{v}^{\\!\\top}\\!\\mathbf{\\theta}$\nacross particles")
    handles.append(Line2D([], [], color="black", lw=0.9, ls="--",
                          label="Gaussian prediction $\\mathbf{v}^{\\!\\top}\\!\\hat{\\mathbf{\\Sigma}}_{\\beta}\\mathbf{v}$"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=PS.FONT_LEGEND, handletextpad=0.4, columnspacing=1.2)
    for letter, ax in zip("ab", axes):
        ax.set_xlabel("$\\beta$")
        ax.text(-0.21, 1.03, f"({letter})", transform=ax.transAxes, fontsize=PS.FONT_PANEL_LETTER,
                fontweight="bold", va="bottom", ha="left")
    FC.save_figure(fig, FC.figure_stem(out_dir, "fig_cs1_phase"), "fig_cs1_phase", W)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/case_study1"),
                    help="folder of summary.json, traces.npz and phase_transition.json")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"),
                    help="the table goes to <out-dir>/tables and the figures to <out-dir>/figures")
    ap.add_argument("--png", action="store_true", help="also write a 600 dpi PNG next to each PDF")
    a = ap.parse_args(argv)
    PS.WRITE_PNG = a.png
    S = json.loads((a.results / "summary.json").read_text())
    tables = a.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "cs1.tex").write_text(table_tex(S))
    figure_case_study1(S, np.load(a.results / "traces.npz"), a.out_dir)
    figure_phase(json.loads((a.results / "phase_transition.json").read_text())["sweep"], a.out_dir)


if __name__ == "__main__":
    main()
