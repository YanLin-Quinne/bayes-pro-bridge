#!/usr/bin/env python3
"""Case Study 1: reduce the chains of scripts/case_study1.sh to the small files behind Table 7 and Figures 2 and 3.

Reads <runs>/case_study1/<group>/*.json and *.npz, the output of src/case_study1_mfld.py, and writes to <out-dir>

  summary.json           every number of Case Study 1 quoted in the paper, per setting
  traces.npz             the curves of Figure 2 (panels a, b and d)
  phase_transition.json  the estimated threshold beta_crit and the beta sweep of Figure 3

The repository ships these three files, computed from the chains of the paper, in results/case_study1, so
analysis/case_study1.py redraws the table and the figures without rerunning the chains.

usage: python analysis/case_study1_summary.py --runs runs --out-dir outputs/data/case_study1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import case_study1_mfld as M  # noqa: E402

BETAS = (0.0, 0.01, 0.1, 0.2, 0.3, 1.0)
INITS = (("prior", 0.0), ("slot", 8.0), ("slot", 16.0))
INIT_NAME = {("prior", 0.0): "prior", ("slot", 8.0): "slot M=8", ("slot", 16.0): "slot M=16"}
PHASE_NS = (200, 1000, 5000)
# The n = 5,000 sweep is completed at small beta by two chains of the sample size comparison (group n5000).
PHASE_EXTRA = {5000: ("n5000", (0.01, 0.1))}


class Chains:
    """Access to the run files of one <runs>/case_study1 folder."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, group, **want):
        out = []
        for f in sorted(glob.glob(str(self.root / group / "*.json"))):
            r = json.load(open(f))
            c = r["config"]
            if c["init"] == "prior":
                c["margin"] = 0.0
            if all(c[k] == v for k, v in want.items()):
                out.append(r)
        return out

    def one(self, group, **want):
        rs = self.load(group, **want)
        if len(rs) != 1:
            sys.exit(f"error: expected one run in {self.root / group} with {want}, found {len(rs)}")
        return rs[0]

    def particles(self, pattern):
        return sorted(glob.glob(str(self.root / pattern)))


def traces(r):
    st = np.array([x["step"] for x in r["train"]])
    mix = np.array([x["r_mix"] for x in r["train"]])
    minr = np.array([x["min_r"] for x in r["train"]])
    ns = np.array([x["near_starved"] for x in r["train"]])
    tst = np.array([x["step"] for x in r["test"]])
    te = np.array([x["nll"] for x in r["test"]])
    return st, mix, minr, ns, tst, te


def at_time(r, t):
    """Held-out NLL of every chain at MFLD time t."""
    _, _, _, _, tst, te = traces(r)
    k = int(np.argmin(np.abs(tst * r["config"]["eps"] - t)))
    return te[k]


def stats(r):
    """Diagnostics of one setting, averaged over the last quarter of each chain."""
    c = r["config"]
    st, mix, minr, ns, tst, te = traces(r)
    T = len(st)
    q4 = slice(3 * T // 4, None)
    t4 = te[3 * len(te) // 4:].mean(0)
    gap = mix - c["best_risk"]
    half = [st[np.argmax(gap[:, k] <= gap[0, k] / 2)] if (gap[:, k] <= gap[0, k] / 2).any() else np.nan
            for k in range(mix.shape[1])]
    rho_min = (1 - c["beta"]) * minr[q4] + c["beta"] / c["m"]
    ess = np.array([x["ess"] for x in r["train"]])
    return {"heldout_mean": float(t4.mean()), "heldout_sd": float(t4.std(ddof=1)), "heldout_min": float(t4.min()),
            "ess_over_m": float(ess[q4].mean() / c["m"]), "raw_min_r_median": float(np.median(minr[q4])),
            "heldout_max": float(t4.max()), "heldout_chains": np.sort(t4).round(4).tolist(),
            "train_rmix": float(mix[q4].mean()), "half_steps_median": float(np.nanmedian(half)),
            "half_time_median": float(np.nanmedian(half) * c["eps"]), "rho_min_median": float(np.median(rho_min)),
            "near_starved": float(ns[q4].mean()), "chains": int(mix.shape[1]), "steps": int(c["steps"]),
            "eps": c["eps"], "T": c["steps"] * c["eps"]}


def kind(t, thr=3.0, tol=30.0):
    """A: slope vector of norm at least thr within tol degrees of +x0, B: the same for -x1, o: flat, x: other."""
    ang = np.degrees(np.arctan2(t[2], t[1]))
    if np.hypot(t[1], t[2]) < thr:
        return "o"
    if abs(ang) < tol:
        return "A"
    if abs(ang + 90) < tol:
        return "B"
    return "x"


def intended(theta, thr=3.0, tol=30.0):
    """Every particle steep along +x0 or along -x1, half of them each (the decomposition that recovers P0)."""
    k = [kind(t, thr, tol) for t in theta]
    m = len(k)
    return k.count("A") == m // 2 and k.count("B") == m // 2


def decomposition_fraction(ch, pattern):
    """MFLD times and, per kept snapshot and chain, whether the m = 8 particles are in the intended decomposition."""
    fr = None
    for f in ch.particles(pattern):
        z = np.load(f)
        th, st = z["theta"], z["steps"]
        ok = np.array([[intended(th[s, c]) for c in range(th.shape[1])] for s in range(len(st))])
        fr = ok if fr is None else np.concatenate([fr, ok], axis=1)
    return st * 1e-4, fr  # the m = 8 runs use eps = 1e-4


def ece_of(pbar, y, bins=10):
    conf = np.maximum(pbar, 1 - pbar)
    correct = ((pbar > 0.5) == (y > 0)).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[idx == k].mean() - correct[idx == k].mean()) * (idx == k).mean()
                     for k in range(bins) if (idx == k).any()))


def ece_main(ch):
    """ECE (10 equal-width bins) of the held-out mixture predictive at t = 200, m = 32, per setting."""
    task = M.mclatchie_task(1000)
    out = {}
    for b in BETAS:
        for i in INITS:
            pat = f"main/*_b{b:g}_m32_{'prior8' if i[0] == 'prior' else 'slot' + str(int(i[1]))}_*.npz"
            (f,) = ch.particles(pat)
            th = np.load(f)["theta"][-1]
            pbar = M.expit(th @ task.phi_test.T).mean(axis=1)
            e = [ece_of(pbar[c], task.sgn_test) for c in range(th.shape[0])]
            out[f"{b:g}|{INIT_NAME[i]}"] = {"mean": float(np.mean(e)), "sd": float(np.std(e, ddof=1))}
    return out


def rule_robustness(ch):
    """Chains in the intended decomposition at t = 200 and t = 1,000 for other thresholds of the rule."""
    res = {}
    zs = [np.load(f) for f in ch.particles("long/*_b0_m8_*.npz")]
    for thr in (1, 3, 5):
        for tol in (15, 30, 45):
            counts = []
            for T in (200, 1000):
                n = 0
                for z in zs:
                    k = int(np.argmin(np.abs(z["steps"] * 1e-4 - T)))
                    n += sum(intended(z["theta"][k, c], thr, tol) for c in range(z["theta"].shape[1]))
                counts.append(n)
            res[f"thr{thr}_tol{tol}"] = counts
    return res


def mechanism(ch):
    """Final particles of the m = 8 chains at beta = 0 (t = 200): what the misoriented particles receive."""
    task = M.mclatchie_task(1000)
    rows = {"intended": [], "misoriented": []}
    for f in ch.particles("main/*_b0_m8_*.npz"):
        th = np.load(f)["theta"][-1]
        lw, _s = M.log_w(th, task.phi, task.sgn)
        r = M.responsibilities(lw)
        for c in range(th.shape[0]):
            for j in range(th.shape[1]):
                k = kind(th[c, j])
                key = "intended" if k in "AB" else "misoriented"
                wrong = np.sign(_s[c, j]) != task.sgn
                rows[key].append((r[c, j].mean() * th.shape[1], (r[c, j] < 0.01).mean(), -lw[c, j].mean(),
                                  wrong.mean(), (r[c, j][wrong] < 0.01).mean(), r[c, j][wrong].mean() * th.shape[1]))
    out = {}
    for key, v in rows.items():
        v = np.array(v)
        out[key] = {"particles": len(v), "share_of_uniform": float(v[:, 0].mean()),
                    "frac_r_below_0.01": float(v[:, 1].mean()), "particle_nll": float(v[:, 2].mean()),
                    "misclassified": float(v[:, 3].mean()), "r_below_0.01_on_misclassified": float(v[:, 4].mean()),
                    "share_on_misclassified": float(v[:, 5].mean())}
    return out


def summary(ch):
    """The summary of every setting and the curves of Figure 2."""
    name = lambda b, i: f"{b:g}|{INIT_NAME[i]}"  # noqa: E731
    S = {"task": {"n": 1000, "d": 3, "prior_sd": 10.0, "best_risk": float(M.mclatchie_task(1000).best_risk)}}
    S["main"] = {name(b, i): stats(ch.one("main", m=32, beta=b, init=i[0], margin=i[1])) for b in BETAS for i in INITS}
    S["long_m32"] = {
        name(b, i): {f"T{t}": np.sort(at_time(ch.one("long", m=32, beta=b, init=i[0], margin=i[1]), t)).round(4).tolist()
                     for t in (1, 10, 100, 200, 500, 1000)}
        for b in (0.0, 0.01) for i in INITS}
    S["m8"] = {name(b, i): stats(ch.one("main", m=8, beta=b, init=i[0], margin=i[1])) for b in BETAS for i in INITS}
    S["m128_long"] = {name(b, i): stats(r) for b in BETAS for i in INITS
                      for r in ch.load("m128_long", m=128, beta=b, init=i[0], margin=i[1])}
    S["n200"] = {name(b, i): stats(ch.one("n200", beta=b, init=i[0], margin=i[1])) for b in BETAS for i in INITS}
    S["n5000_T20"] = {name(b, i): stats(ch.one("n5000", beta=b, init=i[0], margin=i[1]))
                      for b in (0.0, 0.01, 0.1) for i in INITS}
    S["n1000_T20"] = {
        name(b, i): {"heldout_chains": np.sort(at_time(ch.one("main", m=32, beta=b, init=i[0], margin=i[1]), 20)).round(4).tolist()}
        for b in (0.0, 0.01, 0.1) for i in INITS}
    S["half_step"] = {name(b, i): stats(ch.one("half_step", beta=b, init=i[0], margin=i[1]))
                      for b in (0.0, 0.01, 0.1) for i in INITS}
    # The main and long chains of one setting share the seed, so they are the same chains up to t = 200.
    # The main chains give the early part at a finer spacing.
    ta, fa = decomposition_fraction(ch, "main/*_b0_m8_*.npz")
    tb, fb = decomposition_fraction(ch, "long/*_b0_m8_*.npz")
    t0, f0 = np.r_[ta, tb[tb > ta.max()]], np.r_[fa, fb[tb > ta.max()]]
    t1, f1 = decomposition_fraction(ch, "main/*_b0.01_m8_*.npz")
    S["m8_intended_fraction"] = {
        "beta0": {f"T{t}": float(f0[np.argmin(np.abs(t0 - t))].mean()) for t in (1, 10, 100, 200, 500, 1000)},
        "beta0.01": {f"T{t}": float(f1[np.argmin(np.abs(t1 - t))].mean()) for t in (1, 10, 100, 200)},
        "chains": int(f0.shape[1])}
    S["m8_mechanism_t200"] = mechanism(ch)
    S["ece_t200"] = ece_main(ch)
    S["decomposition_rule_robustness"] = rule_robustness(ch)
    S["half_step_check_T20"] = {name(b, i): {
        "eps1e-4": float(at_time(ch.one("main", m=32, beta=b, init=i[0], margin=i[1]), 20).mean()),
        "eps5e-5": float(at_time(ch.one("half_step", beta=b, init=i[0], margin=i[1]), 20).mean())}
        for b in (0.0, 0.01, 0.1) for i in INITS}
    curves = {"t_intended_pure": t0, "intended_pure": f0, "t_intended_bridge": t1, "intended_bridge": f1}
    for key, b in (("pure", 0.0), ("bridge", 0.01)):  # Figure 2 (a) and (b): long chains from the slot partition
        r = ch.one("long", m=32, beta=b, init="slot", margin=16.0)
        st, mix, minr, _, _, _ = traces(r)
        curves.update({f"steps_{key}": st, f"r_mix_{key}": mix, f"min_r_{key}": minr, f"eps_{key}": r["config"]["eps"]})
    return S, curves


# ---------------------------------------------------------------- phase transition in beta

def stability(n):
    """H, G and the threshold beta_crit = max(0, 1 - 1 / gamma) at the mode of the Bayesian posterior.

    H is the average negative Hessian of the log likelihood and G the average outer product of its gradient
    over the n training examples. gamma is the largest value of v'Gv / v'Hv, attained by the unit vector v.
    """
    task = M.mclatchie_task(n)
    th = np.zeros(task.phi.shape[1])
    # Some BLAS builds raise spurious floating-point flags in matmul. The flags are ignored and the result is checked.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for _ in range(100):  # Newton on n L_n + lambda0 |theta|^2 / 2, the centre of the Bayesian posterior
            s = task.phi @ th
            p = M.expit(s)
            g = (task.sgn * M.expit(-task.sgn * s)) @ task.phi - M.LAMBDA0 * th
            h = (task.phi * (p * (1 - p))[:, None]).T @ task.phi + M.LAMBDA0 * np.eye(len(th))
            th = th + np.linalg.solve(h, g)
        p = M.expit(task.phi @ th)
        y = (task.sgn + 1) / 2
        H = (task.phi * (p * (1 - p))[:, None]).T @ task.phi / n
        score = task.phi * (y - p)[:, None]
        G = score.T @ score / n
    if not (np.isfinite(th).all() and np.isfinite(H).all() and np.isfinite(G).all()):
        raise FloatingPointError(f"n = {n}: the posterior mode, H or G is not finite")
    kappa, vecs = eigh(H, G)  # ascending values of v'Hv / v'Gv, so kappa[0] = 1 / gamma
    v = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
    v = v if v[np.argmax(np.abs(v))] > 0 else -v
    return {"theta": th, "H": H, "G": G, "v": v, "gamma": float(1 / kappa[0]),
            "beta_crit": float(max(0.0, 1 - kappa[0]))}


def gaussian_var_v(st, n, beta):
    """v' [n (H - (1 - beta) G) + lambda0 I]^{-1} v, or nan when the matrix is not positive definite."""
    A = n * (st["H"] - (1 - beta) * st["G"]) + M.LAMBDA0 * np.eye(len(st["v"]))
    if np.linalg.eigvalsh(A).min() <= 0:
        return float("nan")
    return float(st["v"] @ np.linalg.solve(A, st["v"]))


def summarise_run(path, st, n):
    """One chain group of the sweep, averaged over the second half of each chain."""
    d = json.loads(Path(path).read_text())
    z = np.load(str(path)[:-5] + ".npz")
    theta, steps = z["theta"], z["steps"]
    late = steps >= steps[-1] / 2
    proj = theta[late] @ st["v"]  # [snapshots, chains, m]
    var_v = proj.var(axis=-1, ddof=1)
    cov = np.einsum("scmi,scmj->scij", theta[late] - theta[late].mean(2, keepdims=True),
                    theta[late] - theta[late].mean(2, keepdims=True)) / (theta.shape[2] - 1)
    test = [r for r in d["test"] if r["step"] >= steps[-1] / 2]
    nll = np.array([r["nll"] for r in test])  # [times, chains]
    train = [r for r in d["train"] if r["step"] >= steps[-1] / 2]
    ess = np.array([r["ess"] for r in train]) / d["config"]["m"]
    beta = d["config"]["beta"]
    g = gaussian_var_v(st, n, beta)
    return {"n": n, "beta": beta, "heldout": float(nll.mean()), "heldout_chain_sd": float(nll.mean(0).std(ddof=1)),
            "ess_over_m": float(ess.mean()), "var_v": float(var_v.mean()),
            "var_v_chain_sd": float(var_v.mean(0).std(ddof=1)), "trace_cov": float(np.trace(cov.mean((0, 1)))),
            "gaussian_var_v": None if np.isnan(g) else g,
            "T": d["config"]["steps"] * d["config"]["eps"], "chains": d["config"]["chains"]}


def phase_transition(root: Path):
    """beta_crit for the three sample sizes and for 200,000 examples, and the summary of the beta sweep."""
    out = {"stability": {}, "sweep": {}}
    for n in PHASE_NS + (200000,):
        st = stability(n)
        out["stability"][str(n)] = {"beta_crit": st["beta_crit"], "gamma": st["gamma"], "v": st["v"].tolist(),
                                    "theta_hat": st["theta"].tolist()}
    for n in PHASE_NS:
        st = stability(n)
        files = sorted(glob.glob(str(root / f"phase_n{n}" / "*.json")))
        if n in PHASE_EXTRA:
            group, betas = PHASE_EXTRA[n]
            for b in betas:
                (f,) = glob.glob(str(root / group / f"*_b{b:g}_m32_prior8_*.json"))
                files.append(f)
        rows = sorted((summarise_run(f, st, n) for f in files), key=lambda r: r["beta"])
        out["sweep"][str(n)] = {"beta_crit": st["beta_crit"], "rows": rows}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="run directory written by the launchers")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/data/case_study1"),
                    help="folder for summary.json, traces.npz and phase_transition.json")
    a = ap.parse_args(argv)
    root = a.runs / "case_study1"
    if not root.is_dir():
        sys.exit(f"error: {root} not found, run scripts/case_study1.sh first")
    S, curves = summary(Chains(root))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
    np.savez_compressed(a.out_dir / "traces.npz", **curves)
    (a.out_dir / "phase_transition.json").write_text(json.dumps(phase_transition(root), indent=1) + "\n")
    print(f"wrote summary.json, traces.npz and phase_transition.json to {a.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
