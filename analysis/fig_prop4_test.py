#!/usr/bin/env python3
"""Figure 6 (gradient against slot margin, Proposition 2) and its plotted numbers.

Rows: Llama 3.2-3B and Qwen3-4B. Columns: pure mixture (beta = 0) and the bridge (beta = 0.1). Each panel is the
density of (example, particle) pairs over the slot margin M and the exact score-gradient norm, with the
bound of part 1 (beta = 0), the upper bound and the 2 beta / m level (beta = 0.1), and the median over
wrong-slot pairs. The numbers come from fig_prop4_test_compute.py.

usage: python analysis/fig_prop4_test.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import math

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

import fig_common as FC
import fig_prop4_test_compute as CP
import panel_style as PS

BACKBONE_COLORS = {"llama3_2_3b": "#0072B2", "qwen3_4b_base": "#009E73"}  # as in the beta sweep figure
BETA_METHOD = {"0": "beta0_pure_pro", "0.1": "beta01_convex"}
HEIGHT_MM = 112.0
LEFT, RIGHT, TOP, BOTTOM = 19.0, 18.0, 18.0, 10.0
HGAP, VGAP = 9.0, 4.0
BAR_W, BAR_GAP = 2.0, 2.5
Y_LOW_BRIDGE = 1e-8  # the bridge particles have no gradient below about 1e-7


def cmap_for(color):
    return LinearSegmentedColormap.from_list("m", [(1, 1, 1), to_rgb(color)])


def darker(color, f=0.55):
    return tuple(c * f for c in to_rgb(color))


def use_arial_mathtext():
    # Keep mathtext glyphs in Arial so the PDF embeds one font family only.
    mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "mathtext.default": "regular"})


def fmt_int(n):
    return f"{n:,}"


def draw_panel(ax, d, run, beta_key, norm, x_max, show_x, show_y):
    method = BETA_METHOD[beta_key]
    color = PS.METHOD_COLORS[method]
    m_edges = np.array(d["grid"]["M_edges"])
    g_edges = 10.0 ** np.array(d["grid"]["log10_grad_edges"])
    counts = (np.array(run["density_wrong_slot"]["counts"])
              + np.array(run["density_correct_slot"]["counts"])).astype(float)
    counts[counts == 0] = np.nan
    mesh = ax.pcolormesh(m_edges, g_edges, counts.T, cmap=cmap_for(color), norm=norm,
                         shading="flat", rasterized=True, zorder=1)
    curve_m = np.array(d["bound_curve"]["M"])
    if float(beta_key) == 0.0:
        ax.plot(curve_m, d["bound_curve"]["values"][str(d["bound_curve"]["K_max"])],
                color="#111111", lw=1.0, ls=(0, (4, 2)), zorder=3)
    else:
        ax.plot(curve_m, d["bound_curve"]["envelope"][beta_key],
                color="#111111", lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), zorder=3)
        ax.axhline(run["floor_2beta_over_m"], color="#111111", lw=1.0, ls=(0, (1, 1.5)), zorder=3)
    med = run["median_wrong_slot"]
    mc = np.array(med["M_center"])
    ax.plot(mc, 10.0 ** np.array(med["log10_median"]), color=darker(color), lw=1.2, zorder=4)
    n_wrong = run["wrong_slot_pairs"]
    if float(beta_key) == 0.0:
        note = (f"{fmt_int(n_wrong)} wrong-slot pairs\n"
                f"{100 * run['fraction_wrong_slot_grad_below_1e-6']:.0f}% below $10^{{-6}}$")
    else:
        floor = run["floor_2beta_over_m"]
        low = run["grad_wrong_slot"]["q00"]
        assert low >= floor, (low, floor)
        note = (f"{fmt_int(n_wrong)} wrong-slot pairs\n"
                f"minimum gradient {low:.3f} = $2\\beta/m$")
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=PS.FONT_NOTE, color=PS.GREY, zorder=5, linespacing=1.15,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))
    ax.set_yscale("log")
    ax.set_xlim(0.0, x_max)
    if float(beta_key) == 0.0:
        ax.set_ylim(g_edges[0], g_edges[-1])
        ax.set_yticks([10.0 ** k for k in range(-16, 1, 4)])
    else:
        ax.set_ylim(Y_LOW_BRIDGE, g_edges[-1])
        ax.set_yticks([10.0 ** k for k in range(-8, 1, 2)])
    ax.minorticks_off()
    if show_x:
        ax.set_xlabel("Slot margin $M$")
    else:
        ax.tick_params(labelbottom=False)
    if show_y:
        ax.set_ylabel(r"$\Vert\partial\ell_{\beta,i}/\partial s_{ij}\Vert_1$")
    return mesh


def draw(d, stem):
    backbones, betas = d["backbones"], d["betas"]
    assert len(backbones) == 2 and len(betas) == 2
    for bb in backbones:
        for b in betas:
            run = d["runs"][bb][b]
            assert run["n_pairs"] == run["n_examples"] * run["m"]
            dens = run["density_wrong_slot"]["pairs"] + run["density_correct_slot"]["pairs"]
            assert dens == run["assumption"]["pairs_tested"], (bb, b)
    vmax = max(np.max(np.array(d["runs"][bb][b]["density_wrong_slot"]["counts"])
                      + np.array(d["runs"][bb][b]["density_correct_slot"]["counts"]))
               for bb in backbones for b in betas)
    norm = LogNorm(vmin=1, vmax=vmax)
    x_max = math.ceil(d["max_M_plotted"])

    fig = PS.new_canvas(PS.FULL_WIDTH_MM, HEIGHT_MM)
    use_arial_mathtext()
    pw = (PS.FULL_WIDTH_MM - LEFT - RIGHT - HGAP) / 2
    ph = (HEIGHT_MM - TOP - BOTTOM - VGAP) / 2
    for r, bb in enumerate(backbones):
        bottom = BOTTOM + (1 - r) * (ph + VGAP)
        for c, b in enumerate(betas):
            ax = PS.rect_mm(fig, LEFT + c * (pw + HGAP), bottom, pw, ph)
            draw_panel(ax, d, d["runs"][bb][b], b, norm, x_max, show_x=(r == 1), show_y=(c == 0))
            if r == 0:
                method = BETA_METHOD[b]
                ax.set_title(PS.METHOD_LABELS[method], color=PS.METHOD_COLORS[method],
                             fontsize=PS.FONT_LABEL, pad=3)
        fig.text(4.0 / PS.FULL_WIDTH_MM, (bottom + ph / 2) / HEIGHT_MM, d["backbone_names"][bb],
                 rotation=90, ha="center", va="center", fontsize=PS.FONT_LABEL,
                 color=BACKBONE_COLORS[bb])

    # One neutral colour bar: all four panels share one log scale of pairs per cell.
    bar_h = 2 * ph + VGAP
    cax = PS.rect_mm(fig, PS.FULL_WIDTH_MM - RIGHT + BAR_GAP, BOTTOM, BAR_W, bar_h)
    grey_map = LinearSegmentedColormap.from_list("g", [(1, 1, 1), (0.25, 0.25, 0.25)])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=grey_map), cax=cax,
                      orientation="vertical")
    cb.outline.set_linewidth(0.5)
    cb.minorticks_off()
    cb.set_label("Pairs per cell (shared)", labelpad=2.0)
    bc = d["bound_curve"]
    handles = [
        Line2D([0], [0], color="#111111", lw=1.0, ls=(0, (4, 2))),
        Line2D([0], [0], color="#111111", lw=1.0, ls=(0, (1, 1.5))),
        Line2D([0], [0], color="#111111", lw=1.0, ls=(0, (5, 1.5, 1, 1.5))),
        (Line2D([0], [0], color=darker(PS.PURE_COLOR), lw=1.2),
         Line2D([0], [0], color=darker(PS.BRIDGE_COLOR), lw=1.2)),
    ]
    labels = [
        rf"Bound $2(K-1)/(m_{{\min}}e^{{M}})$ with $K={bc['K_max']}$, $m_{{\min}}={bc['m_min']}$",
        r"Level $2\beta/m$, the lowest wrong-slot gradient observed",
        r"Bridge upper bound, $(1-\beta)$ times the bound plus $2\beta/m$",
        "Median over wrong-slot pairs",
    ]
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=PS.FONT_LEGEND, handlelength=2.4,
               columnspacing=1.4, handler_map={tuple: HandlerTuple(ndivide=2, pad=0)})
    FC.save_figure(fig, stem, "fig_prop4_test", PS.FULL_WIDTH_MM)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    data = FC.write_data(args.out_dir, "prop4_test", CP.compute(args.runs), indent=1)
    draw(data, FC.figure_stem(args.out_dir, "fig_prop4_test"))


if __name__ == "__main__":
    main()
