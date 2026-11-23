#!/usr/bin/env python3
"""Figure 8 (reliability diagrams) and its plotted numbers.

Pooled over the 22,419 test examples of the eight test sets. The confidence (maximum ensemble probability)
is binned into 10 equal-width bins with the rule of metrics.py, where the last bin includes 1.0.

usage: python analysis/fig_reliability.py --runs runs --out-dir outputs [--png]
"""
from __future__ import annotations

import fig_common as FC
import metrics as M
import panel_style as PS

HEIGHT_MM = 148.0
N_BINS = M.N_BINS
BACKBONES = (("llama3_2_3b", "llama3.2-3b", "Llama 3.2-3B"), ("qwen3_4b_base", "qwen3-4b-base", "Qwen3-4B"))
BETA_STYLE = {  # same purple and orange as Figure 1, independent training in neutral dark grey
    "0": (PS.PURE_COLOR, "o", "Pure mixture (β=0)"),
    "0.1": (PS.BRIDGE_COLOR, "^", "Bridge (β=0.1)"),
    "1": ("#444444", "s", "Independent training (β=1)"),
}
MIN_BIN_SOLID = 30  # bins with fewer examples are drawn hollow (noisy accuracy)
MIN_BIN_DRAW = 5  # bins with fewer examples are not drawn (the histogram strip still shows them)


def bin_stats(conf, correct):
    rows = []
    for k in range(N_BINS):
        lo, hi = k / N_BINS, (k + 1) / N_BINS
        mask = (conf >= lo) & (conf < hi)
        if k == N_BINS - 1:
            mask |= conf == 1.0
        n = int(mask.sum())
        rows.append({"lo": lo, "hi": hi, "n": n,
                     "conf": float(conf[mask].mean()) if n else None,
                     "acc": float(correct[mask].mean()) if n else None})
    assert sum(r["n"] for r in rows) == len(conf)
    return rows


def ece_from_bins(rows, total):
    return sum(r["n"] / total * abs(r["acc"] - r["conf"]) for r in rows if r["n"])


def compute(runs):
    out = {"n_bins": N_BINS, "total_n": M.TOTAL_N, "backbones": [], "data": {}}
    for key, run_name, name in BACKBONES:
        out["backbones"].append({"key": key, "name": name})
        out["data"][key] = {}
        for beta in BETA_STYLE:
            preds = M.load_eval_dir(runs / f"{run_name}_beta{beta}_m8")
            pooled = M.concat([preds[d] for d in M.DATASETS])
            assert pooled.n == M.TOTAL_N == 22419
            rows = bin_stats(pooled.conf, pooled.correct)
            ece = ece_from_bins(rows, pooled.n)
            assert abs(ece - M.ece(pooled.conf, pooled.correct)) < 1e-12, (key, beta)
            below = sum(r["n"] for r in rows if r["n"] and r["acc"] > r["conf"])  # bins above the diagonal
            out["data"][key][beta] = {
                "bins": rows, "ece": ece, "accuracy": float(pooled.correct.mean()),
                "mean_confidence": float(pooled.conf.mean()),
                "frac_in_underconfident_bins": below / pooled.n,
                "n_underconfident_bins": sum(1 for r in rows if r["n"] and r["acc"] > r["conf"]),
                "n_nonempty_bins": sum(1 for r in rows if r["n"]),
                "frac_conf_below_0.5": float((pooled.conf < 0.5).mean()),
                "frac_conf_ge_0.9": float((pooled.conf >= 0.9).mean()),
            }
    return out


def draw_panel(ax_main, ax_hist, rec, beta, first_panel):
    color, marker, _ = BETA_STYLE[beta]
    bins = rec["bins"]
    ax_main.plot([0, 1], [0, 1], color=PS.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    full = [b for b in bins if b["n"] >= MIN_BIN_DRAW]
    solid_bins = [b for b in full if b["n"] >= MIN_BIN_SOLID]
    ax_main.plot([b["conf"] for b in solid_bins], [b["acc"] for b in solid_bins],
                 color=color, lw=1.1, zorder=2, clip_on=False)
    for b in full:
        solid = b["n"] >= MIN_BIN_SOLID
        ax_main.plot(b["conf"], b["acc"], marker=marker, ms=4.2, mec=color, mew=0.9,
                     mfc=color if solid else "white", ls="none", zorder=3, clip_on=False)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_aspect("equal")
    ax_main.set_xticks([0, 0.5, 1])
    ax_main.set_yticks([0, 0.5, 1])
    ax_main.tick_params(labelbottom=False)
    gap = rec["mean_confidence"] - rec["accuracy"]
    ax_main.text(0.97, 0.05, f"ECE {rec['ece']:.3f}\nconf. $-$ acc. {'$-$' if gap < 0 else '+'}{abs(gap):.3f}",
                 transform=ax_main.transAxes, ha="right", va="bottom", fontsize=PS.FONT_NOTE)
    if first_panel:
        ax_main.text(0.03, 0.97, "above diagonal:\nunderconfident", transform=ax_main.transAxes,
                     ha="left", va="top", fontsize=PS.FONT_NOTE, color=PS.GREY)
    for b in bins:
        ax_hist.bar((b["lo"] + b["hi"]) / 2, b["n"] / 1000.0, width=(b["hi"] - b["lo"]) * 0.92,
                    color=color, alpha=0.35, lw=0)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xticks([0, 0.5, 1])


def draw(d, stem):
    betas = list(BETA_STYLE)
    assert d["total_n"] == 22419 and len(d["backbones"]) == 2
    for bb in d["backbones"]:
        for beta in betas:
            rec = d["data"][bb["key"]][beta]
            assert sum(b["n"] for b in rec["bins"]) == d["total_n"] and len(rec["bins"]) == d["n_bins"]
    hist_top = max(b["n"] for bb in d["backbones"] for beta in betas
                   for b in d["data"][bb["key"]][beta]["bins"]) / 1000.0
    hist_ymax = 5.0 * (int(hist_top / 5.0) + 1)

    fig = PS.new_canvas(PS.FULL_WIDTH_MM, HEIGHT_MM)
    left, right, bottom, hgap = 21.0, 3.0, 10.0, 6.0
    side = (PS.FULL_WIDTH_MM - left - right - 2 * hgap) / 3
    hist_h, gap_main_hist, vgap = 12.0, 5.0, 11.0
    row_h = side + gap_main_hist + hist_h
    for r, bb in enumerate(d["backbones"]):
        y_hist = bottom + (1 - r) * (row_h + vgap)
        y_main = y_hist + hist_h + gap_main_hist
        for c, beta in enumerate(betas):
            x0 = left + c * (side + hgap)
            ax_main = PS.rect_mm(fig, x0, y_main, side, side)
            ax_hist = PS.rect_mm(fig, x0, y_hist, side, hist_h)
            draw_panel(ax_main, ax_hist, d["data"][bb["key"]][beta], beta, r == 0 and c == 0)
            ax_hist.set_ylim(0, hist_ymax)
            ax_hist.set_yticks([0, hist_ymax / 2, hist_ymax] if hist_ymax % 10 == 0 else [0, hist_ymax])
            if c == 0:
                ax_main.set_ylabel("Accuracy")
                ax_hist.set_ylabel("Count (k)")
                ax_main.text(-0.40, 0.5, bb["name"], transform=ax_main.transAxes, rotation=90,
                             ha="center", va="center", fontsize=PS.FONT_LABEL, fontweight="bold")
            else:
                ax_main.tick_params(labelleft=False)
                ax_hist.tick_params(labelleft=False)
            if r == 1:
                ax_hist.set_xlabel("Confidence (max ensemble probability)" if c == 1 else "")
            else:
                ax_hist.set_xlabel("")
            if r == 0:
                ax_main.set_title(BETA_STYLE[beta][2], fontsize=PS.FONT_LABEL,
                                  color=BETA_STYLE[beta][0], pad=4)
    FC.save_figure(fig, stem, "fig_reliability", PS.FULL_WIDTH_MM)


def main() -> None:
    args = FC.parse_args(__doc__.splitlines()[0])
    data = FC.write_data(args.out_dir, "reliability", compute(args.runs), indent=1)
    draw(data, FC.figure_stem(args.out_dir, "fig_reliability"))


if __name__ == "__main__":
    main()
