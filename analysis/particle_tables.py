#!/usr/bin/env python3
r"""Tables 6 (cost against the number of particles) and 11 (individual particles) from the run outputs.

Table 11. For every main run (four backbones, six values of beta, m = 8) the held-out accuracy and NLL of each
particle on its own, from the per-example particle probabilities saved by the evaluation, with the definitions
of metrics.py (first argmax, NLL = -log(max(p_gold, 1e-12)), equal mean over the eight test sets). The mixture
is recomputed from the same probabilities and checked against the values of the evaluator.

Table 6. Training GPU hours and peak memory per GPU (resources.json of the training job) and evaluation time per
question (runtime_seconds of the evaluator, one GPU, batch size 8), for m in 4, 8 and 16 at beta = 0.1.

usage: python analysis/particle_tables.py --runs runs --out-dir outputs
Writes <out-dir>/tables/member-metrics.tex and cost-particles.tex, holding the rows between \midrule and
\bottomrule as the paper prints them, and the unrounded numbers to <out-dir>/data/particle_tables.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import make_tables as T
import metrics as M

COST_BETA = "0.1"
FLOOR = 1e-12
TOL = 1e-6  # recomputed mixture NLL against the value of the evaluator


def dataset_members(ds_dir: Path, ds: str) -> dict:
    """Accuracy and NLL of each particle on one test set, the mixture recomputed from the same probabilities,
    and the uniform-guess reference."""
    hit, nll, mix_hit, mix_nll, k = [], [], [], [], []
    with open(ds_dir / f"{ds}_joint_pro_predictions.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            labels = list(row["branch_probs"][0])
            p = np.array([[b[c] for c in labels] for b in row["branch_probs"]], dtype=np.float64)  # [m, K]
            g = labels.index(str(row["gold"]))
            hit.append(p.argmax(1) == g)  # first argmax, as in metrics.py
            nll.append(-np.log(np.maximum(p[:, g], FLOOR)))
            mix = p.mean(0)
            mix_hit.append(mix.argmax() == g)
            mix_nll.append(-math.log(max(mix[g], FLOOR)))
            k.append(len(labels))
    if len(k) != M.EXPECTED_N[ds]:
        raise ValueError(f"{ds_dir}: {len(k)} examples, expected {M.EXPECTED_N[ds]}")
    return {"n": len(k), "member_acc": np.mean(hit, axis=0).tolist(), "member_nll": np.mean(nll, axis=0).tolist(),
            "mix_acc_recomputed": float(np.mean(mix_hit)), "mix_nll_recomputed": float(np.mean(mix_nll)),
            "unif_acc": float(np.mean(1.0 / np.array(k))), "unif_nll": float(np.mean(np.log(k)))}


def run_members(eval_dir: Path) -> dict:
    """Particle and mixture metrics of one evaluation folder. The mixture values are those of the evaluator, the
    source of Table 8. The recomputation from the saved particle probabilities must agree in NLL, and in accuracy
    up to one float32 near-tie per test set."""
    per = {}
    for ds in M.DATASETS:
        d = M.find_dataset_dir(eval_dir, ds)
        per[ds] = dataset_members(d, ds)
        ev = T.read_json(d / f"{ds}_joint_pro_fullmetrics.json")["metrics"]
        flips = abs(per[ds]["mix_acc_recomputed"] - ev["final_prob_accuracy"]) * per[ds]["n"]
        if abs(per[ds]["mix_nll_recomputed"] - ev["final_nll"]) > TOL or flips > 1 + TOL:
            raise ValueError(f"{d}: mixture recomputed from the particle probabilities differs from the evaluator")
        per[ds].update(mix_acc=float(ev["final_prob_accuracy"]), mix_nll=float(ev["final_nll"]),
                       tie_flips=round(flips), runtime_seconds=float(ev["runtime_seconds"]))
    macro = lambda k: float(np.mean([per[ds][k] for ds in M.DATASETS]))  # noqa: E731
    acc = np.mean([per[ds]["member_acc"] for ds in M.DATASETS], axis=0)  # equal mean over test sets, per particle
    nll = np.mean([per[ds]["member_nll"] for ds in M.DATASETS], axis=0)
    seconds = sum(per[ds]["runtime_seconds"] for ds in M.DATASETS)
    return {"member_acc": acc.tolist(), "member_nll": nll.tolist(), "m": len(acc),
            "mix_acc": macro("mix_acc"), "mix_nll": macro("mix_nll"),
            "unif_acc": macro("unif_acc"), "unif_nll": macro("unif_nll"),
            "tie_flips": sum(per[ds]["tie_flips"] for ds in M.DATASETS),
            "eval_seconds": seconds, "eval_ms_per_question": 1000 * seconds / M.TOTAL_N}


def train_resources(path: Path) -> dict:
    r = T.read_json(path)
    return {"gpu_hours": r["gpu_hours"], "ngpus": r["ngpus"], "peak_gib": r["peak_gpu_memory_gib"],
            "wall_seconds": r["wall_seconds"]}


def compute_members(man: dict, runs: Path, log) -> dict:
    members = {}
    for bb in man["backbones"]:
        for b in man["betas"]:
            log(f"particles {bb} beta={b}")
            members.setdefault(bb, {})[b] = run_members(T.at(runs, man, man["final"], bb, b))
    return members


def compute_cost(man: dict, runs: Path, members: dict, log) -> dict:
    """Cost against m at beta = 0.1: training resources of each run and evaluation time per question."""
    cost = {}
    for bb in man["m_ablation"]["backbones"]:
        for m in man["m_ablation"]["particles"]:
            log(f"cost {bb} m={m}")
            ev = members[bb][COST_BETA] if m == 8 else run_members(T.at(runs, man, man["final"], bb, COST_BETA, m))
            if ev["m"] != m:
                raise ValueError(f"{bb} m={m}: the evaluation has {ev['m']} particles")
            cost.setdefault(bb, {})[str(m)] = {
                **train_resources(T.at(runs, man, man["resources"], bb, COST_BETA, m)),
                "eval_ms_per_question": ev["eval_ms_per_question"]}
    return cost


def compute(man: dict, runs: Path, log) -> tuple[dict, dict]:
    """Table 11 needs the evaluations of the main runs. Table 6 also needs the particle count runs and the
    resources.json that training writes. A table whose inputs are missing is left out and the reason recorded."""
    res, missing = {}, {}
    try:
        res["members"] = compute_members(man, runs, log)
        res["cost_m"] = compute_cost(man, runs, res["members"], log)
    except FileNotFoundError as exc:
        missing["members" if "members" not in res else "cost_m"] = str(exc)
    res["cost_beta"] = COST_BETA
    return res, missing


def table_member_metrics(res: dict) -> str:
    """Table 11 (individual particles): mixture, mean particle and worst particle, accuracy and NLL."""
    rows = []
    for k, (bb, label) in enumerate(T.BACKBONE_LABEL.items()):
        if k:
            rows.append("\\midrule")
        for i, (b, r) in enumerate(res["members"][bb].items()):
            a, n = r["member_acc"], r["member_nll"]
            cells = [T.f4(r["mix_acc"]), T.f4(float(np.mean(a))), T.f4(min(a)),
                     T.f4(r["mix_nll"]), T.f4(float(np.mean(n))), T.f4(max(n))]
            rows.append(f"{label if i == 0 else ''} & {T.method_label(b)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def table_cost_particles(res: dict) -> str:
    """Table 6 (cost against the number of particles): training GPU hours, peak memory and evaluation time."""
    rows = []
    for k, (bb, runs) in enumerate(res["cost_m"].items()):
        if k:
            rows.append("\\midrule")
        for i, (m, c) in enumerate(runs.items()):
            cells = [m, str(c["ngpus"]), f"{c['gpu_hours']:.1f}", f"{c['gpu_hours'] / int(m):.2f}",
                     f"{c['peak_gib']:.2f}", f"{c['eval_ms_per_question']:.1f}",
                     f"{c['eval_ms_per_question'] / int(m):.1f}"]
            rows.append(f"{T.BACKBONE_LABEL[bb] if i == 0 else ''} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Tables 6 and 11 of the paper from the run outputs.")
    ap.add_argument("--runs", default="runs", help="folder holding the runs (default: runs)")
    ap.add_argument("--out-dir", default="outputs", help="output root: tables/<label>.tex and data/particle_tables.json")
    ap.add_argument("--manifest", default=str(T.HERE / "manifest.json"), help="run layout description")
    args = ap.parse_args(argv)
    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    try:
        res, missing = compute(man, Path(args.runs), log=lambda s: print(s, file=sys.stderr))
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    if "members" in missing:
        sys.exit(f"error: Tables 6 and 11 need the evaluations of the main runs: {missing['members']}")
    out, data = Path(args.out_dir) / "tables", Path(args.out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    (data / "particle_tables.json").write_text(json.dumps(res, indent=1))
    (out / "member-metrics.tex").write_text(table_member_metrics(res))
    if "cost_m" in missing:
        print(f"skipped tab:cost-particles: {missing['cost_m']}", file=sys.stderr)
        print(f"wrote Table 11 to {out}", file=sys.stderr)
        return
    (out / "cost-particles.tex").write_text(table_cost_particles(res))
    print(f"wrote Tables 6 and 11 to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
