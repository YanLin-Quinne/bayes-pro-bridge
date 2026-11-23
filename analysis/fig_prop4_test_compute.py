"""Numbers of Figure 6 (gradient against slot margin), an empirical look at Proposition 2.

For every held-out example i and particle j of the eight test sets of Llama 3.2-3B and Qwen3-4B under
beta = 0 and beta = 0.1 this module computes, in closed form from the particle probabilities,

  slot          c_j(x_i) = argmax_c p_ij(c)
  m_i           number of particles whose slot is the gold label y_i
  slot margin   M_ij = s_j(c_j) - max_{c != c_j} s_j(c)
                     = log p_ij(c_j) - log max_{c != c_j} p_ij(c)
  r_ij          w_ij / sum_k w_ik with w_ij = p_ij(y_i)
  rho_ij        (1 - beta) r_ij + beta / m
  gradient      ||grad_{s_ij} l_{beta,i}||_1 = rho_ij * ||p_ij - e_y||_1
                     = rho_ij * 2 * sum_{c != y} p_ij(c)

and compares the gradient with the bound 2(K-1)/(m e^M) of part 1 and the limit 2 beta / m of part 2.
Off the slot-margin family, M is read as the smallest slot margin among particle j itself and the
particles whose slot is the gold label (the reading named min_relevant below).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import metrics as MT

BACKBONES = (("llama3_2_3b", "llama3.2-3b", "Llama 3.2-3B"), ("qwen3_4b_base", "qwen3-4b-base", "Qwen3-4B"))
BETAS = (("0", 0.0), ("0.1", 0.1))
M_EXPECTED = 8
# float32 resolution floor: branch_probs are stored from float32 softmax outputs.
PROB_FLOOR = float(np.finfo(np.float32).smallest_subnormal)
# Display grid of the density panels.
M_EDGES = np.linspace(0.0, 30.0, 61)
LOG10_G_EDGES = np.linspace(-16.0, 0.5, 67)
MEDIAN_MIN_COUNT = 100
MEDIAN_EDGES = np.arange(0.0, 30.5, 1.0)
BOUND_CURVE_M = np.linspace(0.0, 30.0, 301)
PRIMARY_MARGIN = "min_relevant"


def load_run(run_dir: Path):
    """Per-pair arrays for one run, pooled over the eight test sets."""
    probs_rows, gold_rows, ds_rows, k_rows = [], [], [], []
    per_ds = {}
    for ds in MT.DATASETS:
        f = MT.find_dataset_dir(run_dir, ds) / f"{ds}_joint_pro_predictions.jsonl"
        n = 0
        ks = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                branches = row["branch_probs"]
                assert len(branches) == M_EXPECTED, (f, len(branches))
                labels = list(branches[0].keys())
                gold = str(row["gold"])
                assert gold in labels, (f, row["id"], gold, labels)
                assert all(list(b.keys()) == labels for b in branches), f
                probs_rows.append(np.array([[b[c] for c in labels] for b in branches], float))
                gold_rows.append(labels.index(gold))
                ds_rows.append(ds)
                k_rows.append(len(labels))
                ks[len(labels)] = ks.get(len(labels), 0) + 1
                n += 1
        assert n == MT.EXPECTED_N[ds], (f, n, MT.EXPECTED_N[ds])
        per_ds[ds] = {"n_examples": n, "K_histogram": {str(k): v for k, v in sorted(ks.items())}}
    return probs_rows, np.array(gold_rows), np.array(ds_rows), np.array(k_rows), per_ds


def pair_quantities(probs_rows, gold, k_arr, beta):
    """Flattened (example, particle) arrays. Examples have different K, so loop by K."""
    m = M_EXPECTED
    n = len(probs_rows)
    out = {key: np.empty((n, m)) for key in
           ("grad", "slot_margin", "r", "w", "rho", "wrong_margin", "one_minus_w")}
    correct_slot = np.empty((n, m), bool)
    floored = 0
    for K in sorted(set(k_arr.tolist())):
        idx = np.flatnonzero(k_arr == K)
        P = np.stack([probs_rows[i] for i in idx])          # (n_K, m, K)
        P = P / P.sum(axis=2, keepdims=True)
        floored += int((P < PROB_FLOOR).sum())
        Pc = np.maximum(P, PROB_FLOOR)
        y = gold[idx]
        onehot = np.zeros_like(P, bool)
        onehot[np.arange(len(idx)), :, y] = True
        slot = P.argmax(axis=2)
        logp = np.log(Pc)
        top = np.take_along_axis(logp, slot[..., None], axis=2)[..., 0]
        masked = np.where(np.eye(K, dtype=bool)[slot], -np.inf, logp)
        second = masked.max(axis=2)
        w = P[np.arange(len(idx)), :, y]                     # (n_K, m)
        one_minus_w = np.where(onehot, 0.0, P).sum(axis=2)   # exact 1 - w_ij
        total = w.sum(axis=1, keepdims=True)
        r = w / total
        rho = (1.0 - beta) * r + beta / m
        out["grad"][idx] = rho * 2.0 * one_minus_w
        out["slot_margin"][idx] = top - second
        out["r"][idx] = r
        out["w"][idx] = w
        out["rho"][idx] = rho
        out["one_minus_w"][idx] = one_minus_w
        logw = np.log(np.maximum(w, PROB_FLOOR))
        out["wrong_margin"][idx] = top - logw                # s_j(c_j) - s_j(y_i)
        correct_slot[idx] = slot == y[:, None]
    return out, correct_slot, floored


def quantiles(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    qs = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs} | {"n": int(x.size)}


def binned_median(m_vals, g_vals):
    centers, med, lo, hi, counts = [], [], [], [], []
    for a, b in zip(MEDIAN_EDGES[:-1], MEDIAN_EDGES[1:]):
        sel = (m_vals >= a) & (m_vals < b)
        if sel.sum() < MEDIAN_MIN_COUNT:
            continue
        lg = np.log10(np.maximum(g_vals[sel], 1e-300))
        centers.append(float((a + b) / 2))
        med.append(float(np.median(lg)))
        lo.append(float(np.quantile(lg, 0.1)))
        hi.append(float(np.quantile(lg, 0.9)))
        counts.append(int(sel.sum()))
    return {"M_center": centers, "log10_median": med, "log10_q10": lo,
            "log10_q90": hi, "count": counts}


def margins(q, correct_slot):
    """Three per-pair readings of the family margin M off the slot-margin family.

    min_relevant: smallest slot margin among particle j itself and the particles
      whose slot is the gold label (the only particles the proof's bounds use).
    min_all: smallest slot margin over all m particles of the example.
    median_all: median particle slot margin of the example (sensitivity only).
    All three equal M on the slot-margin family."""
    sm = q["slot_margin"]
    m = sm.shape[1]
    s_min = np.where(correct_slot, sm, np.inf).min(axis=1)        # +inf if m_i = 0
    relevant = np.minimum(sm, s_min[:, None])
    return {
        "min_relevant": relevant,
        "min_all": np.repeat(sm.min(axis=1)[:, None], m, axis=1),
        "median_all": np.repeat(np.median(sm, axis=1)[:, None], m, axis=1),
    }


def check(values, bound, sel, upper=True):
    v, b = values[sel], bound[sel]
    bad = v > b * (1.0 + 1e-9) if upper else v < b * (1.0 - 1e-9)
    ratio = v / b
    return {"pairs_tested": int(sel.sum()), "violations": int(bad.sum()),
            "fraction": float(bad.mean()),
            ("max_ratio" if upper else "min_ratio"): float(ratio.max() if upper else ratio.min())}


def density(M, grad, sel):
    lg = np.log10(np.maximum(grad[sel], 1e-300))
    Mx = M[sel]
    H, _, _ = np.histogram2d(np.minimum(Mx, M_EDGES[-1] - 1e-9),
                             np.clip(lg, LOG10_G_EDGES[0], LOG10_G_EDGES[-1] - 1e-9),
                             bins=[M_EDGES, LOG10_G_EDGES])
    assert int(H.sum()) == int(sel.sum())
    return {"counts": H.astype(int).tolist(), "pairs": int(sel.sum()),
            "pairs_clipped_to_bottom_row": int((lg < LOG10_G_EDGES[0]).sum()),
            "pairs_with_M_beyond_axis": int((Mx >= M_EDGES[-1]).sum())}


def analyse(run_dir, beta):
    probs_rows, gold, ds_arr, k_arr, per_ds = load_run(run_dir)
    q, correct_slot, floored = pair_quantities(probs_rows, gold, k_arr, beta)
    n, m = q["grad"].shape
    wrong = ~correct_slot
    m_i = correct_slot.sum(axis=1)
    K_pair = np.repeat(k_arr[:, None], m, axis=1)
    mi_pair = np.repeat(m_i[:, None], m, axis=1)
    ds_pair = np.repeat(ds_arr[:, None], m, axis=1)
    holds = mi_pair >= 1
    m_min_ds = {ds: int(m_i[(ds_arr == ds) & (m_i >= 1)].min()) for ds in MT.DATASETS}
    mmin_pair = np.vectorize(m_min_ds.get)(ds_pair)

    grad = q["grad"]
    grad_pure = q["r"] * 2.0 * q["one_minus_w"]   # gradient under the pure mixture risk at this configuration
    floor = 2.0 * beta / m
    marg = margins(q, correct_slot)

    tests = {}
    for name, Mm in marg.items():
        Mh = np.where(holds, Mm, 0.0)
        b_example = 2.0 * (K_pair - 1) / (np.maximum(mi_pair, 1) * np.exp(Mh))
        b_stated = 2.0 * (K_pair - 1) / (mmin_pair * np.exp(Mh))
        if name == PRIMARY_MARGIN:
            ratio = np.where(holds, grad_pure / b_example, -np.inf)
            i, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
            tight = {"ratio": float(ratio[i, j]), "M": float(Mm[i, j]), "m_i": int(m_i[i]),
                     "K": int(k_arr[i]), "correct_slot": bool(correct_slot[i, j]),
                     "dataset": str(ds_arr[i]),
                     "pairs_with_ratio_above_0.5": int((ratio > 0.5).sum())}
        t = {"pure_gradient_vs_per_example_bound": check(grad_pure, b_example, holds),
             "pure_gradient_vs_stated_bound": check(grad_pure, b_stated, holds)}
        if beta > 0:
            envelope = (1.0 - beta) * b_example + (beta / m) * 2.0 * q["one_minus_w"]
            t["bridge_gradient_vs_envelope"] = check(grad, envelope, holds)
            t["bridge_gradient_vs_pure_bound"] = check(grad, b_example, holds)
        tests[name] = t

    res = {
        "beta": beta, "m": m, "n_examples": int(n), "n_pairs": int(n * m),
        "datasets": per_ds, "m_min_by_dataset": m_min_ds,
        "probabilities_below_float32_floor": floored,
        "assumption": {
            "examples_with_m_i_ge_1": int((m_i >= 1).sum()),
            "fraction_examples_with_m_i_ge_1": float((m_i >= 1).mean()),
            "pairs_tested": int(holds.sum()),
            "m_min_over_examples_with_m_i_ge_1": int(m_i[m_i >= 1].min()),
            "m_i_histogram": {str(k): int((m_i == k).sum()) for k in range(m + 1)},
        },
        "margin_quantiles": {name: quantiles(Mm[holds]) for name, Mm in marg.items()},
        "pair_slot_margin": quantiles(q["slot_margin"].ravel()),
        "tests": tests,
        "tightest_pair_primary": tight,
        "assumption_fails": {
            "pairs": int((~holds).sum()),
            "pure_gradient": quantiles(grad_pure[~holds]),
            "pure_gradient_exceeds_bound_with_m_i_set_to_1_min_all": int(
                (grad_pure > 2.0 * (K_pair - 1) / np.exp(marg["min_all"]))[~holds].sum()),
        },
        "wrong_slot_pairs": int(wrong.sum()),
        "correct_slot_pairs": int(correct_slot.sum()),
        "grad_wrong_slot": quantiles(grad[wrong]),
        "grad_correct_slot": quantiles(grad[correct_slot]),
        "pure_grad_wrong_slot": quantiles(grad_pure[wrong]),
        "r_wrong_slot": quantiles(q["r"][wrong]),
        "fraction_wrong_slot_r_below_0.01": float((q["r"][wrong] < 0.01).mean()),
        "fraction_wrong_slot_grad_below_1e-6": float((grad[wrong] < 1e-6).mean()),
        "floor_2beta_over_m": floor,
    }
    if beta > 0:
        gw = grad[wrong]
        res["floor_check_wrong_slot"] = {
            "pairs": int(wrong.sum()),
            "min_grad_over_floor": float((gw / floor).min()),
            "fraction_ge_floor": float((gw >= floor).mean()),
            "max_w_wrong_slot": float(q["w"][wrong].max()),
            "identity_lower_bound_holds_all_pairs": bool(
                np.all(grad >= (beta / m) * 2.0 * q["one_minus_w"] * (1 - 1e-9))),
        }
    M_plot = marg[PRIMARY_MARGIN]
    res["primary_margin"] = PRIMARY_MARGIN
    res["density_wrong_slot"] = density(M_plot, grad, holds & wrong)
    res["density_correct_slot"] = density(M_plot, grad, holds & correct_slot)
    res["median_wrong_slot"] = binned_median(M_plot[holds & wrong], grad[holds & wrong])
    res["median_correct_slot"] = binned_median(M_plot[holds & correct_slot], grad[holds & correct_slot])
    return res


def compute(runs: Path) -> dict:
    out = {
        "grid": {"M_edges": M_EDGES.tolist(), "log10_grad_edges": LOG10_G_EDGES.tolist()},
        "bound_curve": {"M": BOUND_CURVE_M.tolist()},
        "backbones": [b[0] for b in BACKBONES],
        "backbone_names": {b[0]: b[2] for b in BACKBONES},
        "betas": [b[0] for b in BETAS],
        "runs": {},
    }
    for bb, run_name, _ in BACKBONES:
        out["runs"][bb] = {}
        for key, beta in BETAS:
            run = f"{run_name}_beta{key}_m{M_EXPECTED}"
            out["runs"][bb][key] = analyse(runs / run, beta)
            out["runs"][bb][key]["run_dir"] = f"{run}/final"
    ks = sorted({int(k) for r in out["runs"].values() for x in r.values()
                 for d in x["datasets"].values() for k in d["K_histogram"]})
    mmin = min(x["assumption"]["m_min_over_examples_with_m_i_ge_1"]
               for r in out["runs"].values() for x in r.values())
    out["bound_curve"]["K_values"] = ks
    out["bound_curve"]["m_min"] = mmin
    out["bound_curve"]["values"] = {
        str(K): (2.0 * (K - 1) / (mmin * np.exp(BOUND_CURVE_M))).tolist() for K in ks}
    kmax = max(ks)
    pure_max = 2.0 * (kmax - 1) / (mmin * np.exp(BOUND_CURVE_M))
    out["bound_curve"]["K_max"] = kmax
    out["bound_curve"]["envelope"] = {}
    for key, beta in BETAS:
        out["bound_curve"]["envelope"][key] = ((1.0 - beta) * pure_max + 2.0 * beta / M_EXPECTED).tolist()
    out["max_M_plotted"] = max(
        x["margin_quantiles"][PRIMARY_MARGIN]["q100"] for r in out["runs"].values() for x in r.values())
    out["summary"] = {
        f"{bb}|{b}": {
            "pairs_tested": x["tests"][PRIMARY_MARGIN]["pure_gradient_vs_per_example_bound"]["pairs_tested"],
            "violations": x["tests"][PRIMARY_MARGIN]["pure_gradient_vs_per_example_bound"]["violations"],
            "max_ratio": x["tests"][PRIMARY_MARGIN]["pure_gradient_vs_per_example_bound"]["max_ratio"],
            "fraction_examples_assumption_holds": x["assumption"]["fraction_examples_with_m_i_ge_1"],
            "median_grad_wrong_slot": x["grad_wrong_slot"]["q50"],
            "min_grad_wrong_slot": x["grad_wrong_slot"]["q00"],
        } for bb, r in out["runs"].items() for b, x in r.items()}
    total = sum(x["n_examples"] for r in out["runs"].values() for x in r.values())
    assert total == 4 * sum(MT.EXPECTED_N.values()), total
    return out
