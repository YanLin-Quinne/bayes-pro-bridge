#!/usr/bin/env python3
"""Figure 1 (mechanism and puzzle) and its plotted numbers.

(a) Llama 3.2-3B HellaSwag test, a fixed-seed subset of 24 examples sorted by gold label. Raw responsibility
    r_ij = w_ij / sum_k w_ik with w_ij = p_j(y_i | x_i), and the effective coefficient
    rho_ij = (1 - beta) r_ij + beta / m.
(b) Qwen3-4B, six betas: training-set R_mix from the training-set audit and the held-out macro NLL
    (which is the held-out R_mix).

usage: python analysis/fig_mechanism_puzzle.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import fig_common as FC
import metrics as M
import panel_style as PS

SEED = 20261027
N_SHOW = 24
M_PARTICLES = 8
N_TRAIN = 170300
BETAS = ["0", "0.01", "0.1", "0.2", "0.3", "1"]
HEIGHT_MM = 78.0
CMAP = "cividis"
VMIN, VMAX = 1e-4, 1.0          # display range of the log colour scale (clipped below)
QWEN3_GREEN = "#009E73"          # Qwen3-4B colour of the beta sweep figure


# ------------------------------------------------------------------ numbers

def read_predictions(runs: Path, run: str):
    path = M.find_dataset_dir(runs / run, "hellaswag") / "hellaswag_joint_pro_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == M.EXPECTED_N["hellaswag"], (run, len(rows))
    return rows


def weights(rows, idx):
    """w[i, j] = p_j(y_i | x_i) for the selected examples."""
    w = np.array([[rows[i]["branch_probs"][j][rows[i]["gold"]] for j in range(M_PARTICLES)]
                  for i in idx], dtype=np.float64)
    assert w.shape == (len(idx), M_PARTICLES)
    return w


def panel_a(runs: Path):
    pure = read_predictions(runs, "llama3.2-3b_beta0_m8")
    bridge = read_predictions(runs, "llama3.2-3b_beta0.1_m8")
    n_test = M.EXPECTED_N["hellaswag"]
    assert all(a["id"] == b["id"] and a["gold"] == b["gold"] for a, b in zip(pure, bridge))
    assert all(len(r["branch_probs"]) == M_PARTICLES for r in pure + bridge)
    rng = np.random.default_rng(SEED)
    subset = rng.choice(n_test, size=N_SHOW, replace=False)
    idx = sorted((int(i) for i in subset), key=lambda i: (pure[i]["gold"], i))  # gold label, then dataset index
    beta = 0.1
    out = {"seed": SEED, "n_examples": N_SHOW, "m": M_PARTICLES, "beta": beta, "floor": beta / M_PARTICLES,
           "example_index": idx, "example_id": [pure[i]["id"] for i in idx],
           "gold": [pure[i]["gold"] for i in idx]}
    for key, rows, b in (("pure", pure, 0.0), ("bridge", bridge, beta)):
        w = weights(rows, idx)
        r = w / w.sum(axis=1, keepdims=True)
        rho = (1.0 - b) * r + b / M_PARTICLES
        assert np.allclose(r.sum(axis=1), 1.0) and np.allclose(rho.sum(axis=1), 1.0)
        assert rho.min() >= b / M_PARTICLES - 1e-12
        out[key] = {"w": w.tolist(), "r": r.tolist(), "rho": rho.tolist(),
                    "r_min": float(r.min()), "rho_min": float(rho.min()),
                    "frac_r_below_0.01": float((r < 0.01).mean()),
                    "frac_r_below_0.0125": float((r < beta / M_PARTICLES).mean()),
                    "n_cells": int(r.size),
                    "n_below_0.01_per_particle": (r < 0.01).sum(axis=0).tolist(),
                    "ess_over_m_mean": float(np.mean(1.0 / (r ** 2).sum(axis=1)) / M_PARTICLES)}
        wa = weights(rows, range(n_test))
        ra = wa / wa.sum(axis=1, keepdims=True)
        out[key]["full_test"] = {"n": n_test, "frac_r_below_0.01": float((ra < 0.01).mean()),
                                 "frac_below_0.01_per_particle": (ra < 0.01).mean(axis=0).tolist(),
                                 "ess_over_m_mean": float(np.mean(1.0 / (ra ** 2).sum(axis=1)) / M_PARTICLES)}
    assert out["pure"]["r"] == out["pure"]["rho"] or np.allclose(out["pure"]["r"], out["pure"]["rho"])
    return out


def panel_b(runs: Path):
    train, held = {}, {}
    for b in BETAS:
        run = runs / f"qwen3-4b-base_beta{b}_m8"
        z = np.load(run / "train_set_audit" / "train_nll_artifact.npz")
        opt = z["is_optimization"]
        assert int(opt.sum()) == N_TRAIN, (b, int(opt.sum()))
        nll = z["predictive_nll"][opt]
        lp = z["branch_gold_log_probs"][opt].astype(np.float64)
        alt = -(np.logaddexp.reduce(lp, axis=1) - np.log(M_PARTICLES))  # -log mean_j p_j
        assert abs(nll.mean() - alt.mean()) < 5e-3, (b, nll.mean(), alt.mean())
        train[b] = float(nll.mean())
        macro = M.summarize(M.load_eval_dir(run))["macro"]
        assert macro["n"] == 8
        held[b] = float(macro["nll"])
    return {"betas": BETAS, "train_r_pro": train, "heldout_r_pro": held,
            "n_train": N_TRAIN, "n_records": 2 * len(BETAS),
            "pure_worst_train": train["0"] == max(train.values()),
            "pure_worst_heldout": held["0"] == max(held.values()),
            "train_ratio_pure_over_bridge01": train["0"] / train["0.1"],
            "heldout_min_beta": min(held, key=held.get)}


# ------------------------------------------------------------------ drawing

def draw_map(fig, x0, y0, w, h, data, gold, color, title, note, first):
    ax = PS.rect_mm(fig, x0, y0, w, h)
    norm = LogNorm(vmin=VMIN, vmax=VMAX, clip=True)
    img = ax.imshow(np.clip(data, VMIN, VMAX), aspect="auto", cmap=CMAP, norm=norm,
                    interpolation="nearest")
    n, m = data.shape
    ax.set_xticks(range(m))
    ax.set_xticklabels([str(j + 1) for j in range(m)])
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.35)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="x", length=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    groups = [(g, [i for i, v in enumerate(gold) if v == g]) for g in sorted(set(gold))]
    for _, rows in groups[:-1]:
        ax.axhline(rows[-1] + 0.5, color="white", lw=1.4)
    if first:
        ax.set_yticks([np.mean(rows) for _, rows in groups])
        ax.set_yticklabels([g for g, _ in groups])
        ax.tick_params(axis="y", length=2.0)
        ax.set_ylabel("Example, grouped by gold label")
    else:
        ax.set_yticks([])
    ax.set_xlabel("Particle $j$", labelpad=1.5)
    ax.text(0.5, -0.17, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=PS.FONT_NOTE, color=PS.GREY)
    ax.plot([0, 1], [1.03, 1.03], transform=ax.transAxes, color=color, lw=2.2,
            solid_capstyle="butt", clip_on=False)
    ax.text(0.5, 1.07, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=PS.FONT_TICK, linespacing=1.15)
    return ax, img


def draw_colorbar(fig, img, x0, y0, h, floor):
    cax = PS.rect_mm(fig, x0, y0, 2.2, h)
    cb = fig.colorbar(img, cax=cax, extend="min")
    ticks = [1e-4, 1e-3, floor, 0.1, 1.0]
    labels = [r"$\leq 10^{-4}$", r"$10^{-3}$", "β/m\n0.0125", "0.1", "1"]
    assert abs(floor - 0.0125) < 1e-12
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=2.0, labelsize=PS.FONT_TICK)
    cb.ax.axhline(floor, color="white", lw=1.0)
    cb.outline.set_visible(False)
    cb.set_label("Responsibility (log scale)", fontsize=PS.FONT_LABEL, labelpad=4.0)
    return cb


def draw_puzzle(fig, x0, y0, w, h, b):
    betas = b["betas"]
    x = list(range(len(betas)))
    train = [b["train_r_pro"][k] for k in betas]
    held = [b["heldout_r_pro"][k] for k in betas]
    ax = PS.rect_mm(fig, x0, y0, w, h)
    ax.plot(x, train, color=QWEN3_GREEN, lw=1.2, ls="-", zorder=2)
    ax.plot(x, held, color=QWEN3_GREEN, lw=1.2, ls=(0, (3.0, 1.6)), zorder=2)
    for xs, ys, mk, fill in ((x, train, "D", True), (x, held, "D", False)):
        for xi, yi in zip(xs, ys):
            col = {0: PS.PURE_COLOR, 2: PS.BRIDGE_COLOR}.get(xi, QWEN3_GREEN)
            hi = xi in (0, 2)
            ax.plot([xi], [yi], marker=mk, ms=5.4 if hi else 3.8, mec=col, mew=1.0,
                    mfc=col if fill else "white", zorder=3, ls="none")
    ax.set_yscale("log")
    ax.set_ylim(2.5e-3, 0.7)
    yt = [0.003, 0.01, 0.03, 0.1, 0.3]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(betas)
    ax.set_xlim(-0.4, len(betas) - 0.6)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$R_{\mathrm{mix}}$ of Qwen3-4B (log scale)")
    handles = [
        Line2D([], [], color=QWEN3_GREEN, lw=1.2, marker="D", ms=3.8, label="Training set"),
        Line2D([], [], color=QWEN3_GREEN, lw=1.2, ls=(0, (3.0, 1.6)), marker="D", ms=3.8,
               mfc="white", label="Held-out"),
        Line2D([], [], ls="none", marker="D", ms=5.4, color=PS.PURE_COLOR, label="Pure mixture"),
        Line2D([], [], ls="none", marker="D", ms=5.4, color=PS.BRIDGE_COLOR, label="β=0.1"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
              handlelength=1.8, columnspacing=0.6, handletextpad=0.4, labelspacing=0.3,
              borderaxespad=0.0)
    return ax


def draw(d, stem):
    a, b = d["a"], d["b"]
    assert a["n_examples"] == 24 and a["m"] == 8 and len(a["gold"]) == 24
    assert len(b["betas"]) == 6 and b["n_records"] == 12 and b["n_train"] == N_TRAIN
    floor = a["floor"]
    gold = a["gold"]
    pure_r = np.array(a["pure"]["r"])
    bridge_r = np.array(a["bridge"]["r"])
    bridge_rho = np.array(a["bridge"]["rho"])
    assert pure_r.shape == bridge_r.shape == bridge_rho.shape == (24, 8)

    fig = PS.new_canvas(PS.FULL_WIDTH_MM, HEIGHT_MM)
    mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    left, bottom, mw, mh, gap = 13.0, 15.0, 20.5, 48.0, 3.0
    specs = [
        (pure_r, PS.PURE_COLOR, "Pure mixture (β=0)\nraw $r_{ij}$",
         f"$r_{{ij}}<0.01$: {a['pure']['frac_r_below_0.01']:.0%}"),
        (bridge_r, PS.BRIDGE_COLOR, "β=0.1\nraw $r_{ij}$",
         f"$r_{{ij}}<0.01$: {a['bridge']['frac_r_below_0.01']:.0%}"),
        (bridge_rho, PS.BRIDGE_COLOR, "β=0.1\neffective $\\rho_{ij}$",
         f"min $\\rho_{{ij}}$: {a['bridge']['rho_min']:.4f}"),
    ]
    img = None
    for k, (data, color, title, note) in enumerate(specs):
        _, img = draw_map(fig, left + k * (mw + gap), bottom, mw, mh, data, gold, color,
                          title, note, first=(k == 0))
    bar_x = left + 3 * mw + 2 * gap + 3.0
    draw_colorbar(fig, img, bar_x, bottom, mh, floor)

    px = 119.0
    draw_puzzle(fig, px, bottom, PS.FULL_WIDTH_MM - px - 2.0, mh, b)
    for letter, xmm in (("a", 0.5), ("b", 106.5)):
        fig.text(xmm / PS.FULL_WIDTH_MM, 1 - 1.0 / HEIGHT_MM, f"({letter})",
                 fontsize=PS.FONT_PANEL_LETTER, fontweight="bold", va="top", ha="left")
    FC.save_figure(fig, stem, "fig_mechanism_puzzle", PS.FULL_WIDTH_MM)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    data = FC.write_data(args.out_dir, "mechanism_puzzle",
                         {"a": panel_a(args.runs), "b": panel_b(args.runs)}, indent=1)
    draw(data, FC.figure_stem(args.out_dir, "fig_mechanism_puzzle"))


if __name__ == "__main__":
    main()
