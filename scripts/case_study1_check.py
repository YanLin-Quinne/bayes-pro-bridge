#!/usr/bin/env python3
"""Checks of src/case_study1_mfld.py: gradient against finite differences, and beta = 1 against a Laplace approximation.

usage: python scripts/case_study1_check.py     (one core, about eight minutes)
The paper reports a relative error below 1e-8, means within 0.006 and standard deviations within 1%.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import case_study1_mfld as M  # noqa: E402


def gradient_check():
    task = M.mclatchie_task(200, 10)
    rng = np.random.default_rng(0)
    worst = 0.0
    for beta in (0.0, 0.01, 0.1, 1.0):
        th = rng.normal(size=(2, 4, 3)) * 2.0
        g = M.grad_f(th, task, beta)
        for _ in range(30):
            c, j, k = rng.integers(2), rng.integers(4), rng.integers(3)
            e = np.zeros_like(th)
            e[c, j, k] = 1e-5
            fd = (M.f_value(th + e, task, beta)[c] - M.f_value(th - e, task, beta)[c]) / 2e-5
            worst = max(worst, abs(fd - g[c, j, k]) / max(1.0, abs(fd)))
    print(f"gradient check: worst relative error {worst:.2e}")
    assert worst < 1e-6


def laplace_bayes():
    """At beta = 1 the particles are independent draws from the Bayesian posterior."""
    task = M.mclatchie_task(1000, 10)
    th = np.zeros(3)
    for _ in range(50):
        s = task.phi @ th
        p = M.expit(s)
        g = (task.sgn * M.expit(-task.sgn * s)) @ task.phi - M.LAMBDA0 * th
        H = (task.phi * (p * (1 - p))[:, None]).T @ task.phi + M.LAMBDA0 * np.eye(3)
        th = th + np.linalg.solve(H, g)
    sd = np.sqrt(np.diag(np.linalg.inv(H)))
    res, kept, steps = M.run(task, 1.0, 32, "prior", 8.0, 1e-4, 60000, 8, 3, log_every=10**6, test_every=10**9,
                             keep_every=100)
    pool = kept[steps > 20000].reshape(-1, 3)
    print("beta = 1 against Laplace: mean", np.round(pool.mean(0), 3), "laplace", np.round(th, 3))
    print("                          sd  ", np.round(pool.std(0), 4), "laplace", np.round(sd, 4))


if __name__ == "__main__":
    gradient_check()
    laplace_bayes()
