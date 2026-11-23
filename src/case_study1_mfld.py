#!/usr/bin/env python3
"""Case Study 1: Euler-Maruyama mean-field Langevin dynamics for the bridge on a misspecified logistic regression.

Particle update, with F = sum_j log pi(theta_j) - m n R_beta(Q_m):
    theta_{1:m} <- theta_{1:m} + eps grad F + sqrt(2 eps) Z
    grad_j F = -lambda0 theta_j + m sum_i rho_ij grad log p_j(y_i | x_i),  rho_ij = (1 - beta) r_ij + beta / m
All chains of one setting run as one array [chains, m, d]. numpy only, deterministic seeds.

usage: python src/case_study1_mfld.py --beta 0.1 --m 32 --init prior --steps 200000 --eps 1e-4 --out DIR
Writes DIR/<tag>.json (training and held-out traces of every chain) and DIR/<tag>.npz (kept particles).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

LAMBDA0 = 0.01  # prior N(0, I / lambda0)
DATA_SEED = 20261101


@dataclass
class Task:
    phi: np.ndarray  # [n, d] features with intercept
    sgn: np.ndarray  # [n] +1 for y = 1, -1 for y = 0
    phi_test: np.ndarray
    sgn_test: np.ndarray
    best_risk: float  # lowest attainable R_mix (entropy of the labels given x)
    name: str


def expit(z):
    return np.exp(-np.logaddexp(0.0, -z))


def log_w(theta, phi, sgn):
    """log p_j(y_i | x_i) for all chains, particles and examples: [C, m, n], with the score s = theta phi."""
    s = theta @ phi.T
    return -np.logaddexp(0.0, -sgn * s), s


def responsibilities(lw):
    lw = lw - lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def grad_f(theta, task, beta):
    m = theta.shape[1]
    lw, s = log_w(theta, task.phi, task.sgn)
    rho = (1.0 - beta) * responsibilities(lw) + beta / m
    # d log w / d s = sgn * sigmoid(-sgn * s), computed without loss of accuracy in the tails
    dlw = task.sgn * expit(-task.sgn * s)
    return -LAMBDA0 * theta + m * ((rho * dlw) @ task.phi)


def f_value(theta, task, beta):
    """F per chain, used only by the gradient check."""
    m = theta.shape[1]
    n = task.phi.shape[0]
    lw, _ = log_w(theta, task.phi, task.sgn)
    mix = -(np.logaddexp.reduce(lw, axis=1) - np.log(m)).mean(axis=-1)
    ind = -lw.mean(axis=(1, 2))
    r = (1.0 - beta) * mix + beta * ind
    return -0.5 * LAMBDA0 * (theta ** 2).sum(axis=(1, 2)) - m * n * r


def risks(theta, phi, sgn):
    """R_mix and R_ind per chain, with the responsibility diagnostics."""
    m = theta.shape[1]
    lw, _ = log_w(theta, phi, sgn)
    mix = -(np.logaddexp.reduce(lw, axis=1) - np.log(m)).mean(axis=-1)
    ind = -lw.mean(axis=(1, 2))
    r = responsibilities(lw)
    ess = (1.0 / (r ** 2).sum(axis=1)).mean(axis=-1)
    return {"r_mix": mix, "r_ind": ind, "min_r": r.min(axis=(1, 2)), "ess": ess,
            "near_starved": (r < 0.01).mean(axis=(1, 2))}


def mclatchie_task(n, n_test=10000, seed=DATA_SEED):
    """Section 5.2 of McLatchie et al. (2025): two deterministic quadrants and two Bernoulli(0.5) quadrants."""
    rng = np.random.default_rng(seed)

    def draw(k):
        x = rng.uniform(-2.0, 2.0, size=(k, 2))
        y = rng.integers(0, 2, size=k).astype(float)
        y[(x[:, 0] < 0) & (x[:, 1] > 0)] = 0.0
        y[(x[:, 0] > 0) & (x[:, 1] < 0)] = 1.0
        return np.column_stack([np.ones(k), x]), 2.0 * y - 1.0

    phi, sgn = draw(n)
    phi_t, sgn_t = draw(n_test)
    return Task(phi, sgn, phi_t, sgn_t, 0.5 * np.log(2.0), "mclatchie")


def init_particles(kind, chains, m, d, rng, margin=8.0):
    if kind == "prior":
        return rng.normal(size=(chains, m, d)) / np.sqrt(LAMBDA0)
    if kind == "slot":
        theta = rng.normal(size=(chains, m, d)) * 0.01
        theta[:, : m // 2, 0] += margin
        theta[:, m // 2:, 0] -= margin
        return theta
    raise ValueError(kind)


def run(task, beta, m, init, margin, eps, steps, chains, seed, log_every=200, test_every=2000, keep_every=2000):
    rng = np.random.default_rng([seed, m, int(round(beta * 1000)), int(margin), 0 if init == "prior" else 1])
    d = task.phi.shape[1]
    theta = init_particles(init, chains, m, d, rng, margin)
    rows, test_rows, kept, kept_steps = [], [], [], []
    root = np.sqrt(2.0 * eps)
    t0 = time.time()
    for t in range(steps + 1):
        if t % log_every == 0:
            q = risks(theta, task.phi, task.sgn)
            if not np.isfinite(q["r_mix"]).all() or np.abs(theta).max() > 1e6:
                raise FloatingPointError(f"diverged at step {t}, reduce eps")
            rows.append({"step": t, **{k: v.tolist() for k, v in q.items()}})
        if t % test_every == 0:
            q = risks(theta, task.phi_test, task.sgn_test)
            test_rows.append({"step": t, "nll": q["r_mix"].tolist()})
        if t % keep_every == 0:
            kept.append(theta.copy())
            kept_steps.append(t)
        if t < steps:
            theta = theta + eps * grad_f(theta, task, beta) + root * rng.normal(size=theta.shape)
    return {"train": rows, "test": test_rows, "seconds": time.time() - t0}, np.array(kept), np.array(kept_steps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--init", required=True, choices=["prior", "slot"])
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--test-every", type=int, default=2000)
    ap.add_argument("--keep-every", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    task = mclatchie_task(a.n)
    res, kept, kept_steps = run(task, a.beta, a.m, a.init, a.margin, a.eps, a.steps, a.chains, a.seed,
                                a.log_every, a.test_every, a.keep_every)
    res["config"] = {**vars(a), "best_risk": task.best_risk}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{task.name}_n{task.phi.shape[0]}_b{a.beta:g}_m{a.m}_{a.init}{a.margin:g}_eps{a.eps:g}_s{a.seed}"
    (out / f"{tag}.json").write_text(json.dumps(res))
    np.savez_compressed(out / f"{tag}.npz", theta=kept, steps=kept_steps)
    print(tag, f"{res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
