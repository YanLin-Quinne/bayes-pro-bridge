#!/usr/bin/env python3
r"""Regenerate the paper tables that are built from evaluation outputs.

usage: python analysis/make_tables.py --runs RUNS --out-dir outputs [--cache DIR]

The run layout is the one written by the launchers and is described by manifest.json. The script writes
one file per table to <out-dir>/tables, named after the table label (tab:main-results becomes
main-results.tex), holding the rows between \midrule and \bottomrule exactly as the paper prints them, and
<out-dir>/data/tables.json with the unrounded numbers. This script covers Tables 1, 3 to 5, 8 to 10 and 12 to 14.
Table 2 comes from analysis/starvation.py, Tables 6 and 11 from analysis/particle_tables.py and Table 7 from
analysis/case_study1.py.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

import metrics as M

HERE = Path(__file__).resolve().parent
BACKBONE_LABEL = {
    "llama3_2_1b": "Llama 3.2-1B", "llama3_2_3b": "Llama 3.2-3B",
    "qwen2_1_5b_instruct": "Qwen2-1.5B-Instruct", "qwen3_4b_base": "Qwen3-4B",
}
DIRECTION = {"accuracy": "max", "nll": "min", "ece": "min", "brier": "min", "gap": "abs0"}
STARVED_THRESHOLD = 0.01
TRAIN_SET_SIZE = 170300


# ======================================================================= inputs

def run_name(man: dict, backbone: str, beta: str, m: int = 8) -> str:
    return man["run"].format(name=man["backbones"][backbone], beta=beta, m=m)


def at(runs: Path, man: dict, template: str, backbone: str, beta: str, m: int = 8) -> Path:
    """Path of a manifest template for one run inside the runs folder."""
    return runs / template.format(run=run_name(man, backbone, beta, m))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing input {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def field(record: dict, keys: list, path: Path):
    try:
        for k in keys:
            record = record[k]
    except KeyError as exc:
        raise ValueError(f"{path}: missing field {'.'.join(keys)}") from exc
    return record


def single_match(pattern: Path) -> Path:
    hits = sorted(glob.glob(str(pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern} must match exactly one folder, found {len(hits)}")
    return Path(hits[0])


def evaluate(path: Path, cache) -> dict:
    """Macro, pooled and per dataset metrics of one evaluation folder."""
    return M.summarize(M.load_eval_dir(path, cache))


# ======================================================================= compute

def parse_val_curve(train_log: Path) -> dict:
    """{step: eval_loss} from '[eval step N] {... 'eval_loss': x ...}' lines of the trainer log."""
    if not train_log.exists():
        raise FileNotFoundError(f"missing input {train_log}")
    pat = re.compile(r"\[eval step (\d+)\].*?'eval_loss': ([0-9.eE+-]+)")
    out = {}
    with open(train_log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def select_checkpoint(curve: dict, candidates: list, source: Path) -> tuple[int, float]:
    """Lowest validation NLL among the candidate steps, ties to the earlier step."""
    missing = [s for s in candidates if s not in curve]
    if missing:
        raise ValueError(f"{source}: validation NLL missing at steps {missing}")
    best = min(candidates, key=lambda s: (curve[s], s))
    return best, curve[best]


def compute_final(man: dict, runs: Path, cache, log) -> dict:
    out = {}
    for bb in man["backbones"]:
        for beta in man["betas"]:
            log(f"final {bb} beta={beta}")
            out.setdefault(bb, {})[beta] = evaluate(at(runs, man, man["final"], bb, beta), cache)
    return out


def compute_selected(man: dict, runs: Path, cache, log) -> dict:
    """Validation selected checkpoints of one backbone."""
    cfg = man["selected"]
    bb = cfg["backbone"]
    methods = {}
    for beta in cfg["betas"]:
        train_log = at(runs, man, man["train_log"], bb, beta)
        step, val = select_checkpoint(parse_val_curve(train_log), cfg["candidate_steps"], train_log)
        folder = single_match(at(runs, man, cfg["evaluation"], bb, beta))
        if folder.name != f"checkpoint-{step}":
            raise ValueError(f"{folder}: the training log selects step {step}")
        log(f"selected {bb} beta={beta} step={step}")
        methods[beta] = {"step": step, "val_nll": val, **M.summarize(M.load_eval_dir(folder, cache))}
    return {"backbone": bb, "methods": methods}


def compute_sequence(man: dict, runs: Path, cache, log) -> dict:
    out = {}
    for noise, readouts in man["sequence"].items():
        for readout, rel in readouts.items():
            log(f"sequence {noise} {readout}")
            out.setdefault(noise, {})[readout] = evaluate(runs / rel, cache)
    return out


def train_nll_stats(npz: Path) -> dict:
    if not npz.exists():
        raise FileNotFoundError(f"missing input {npz}")
    z = np.load(npz, allow_pickle=False)
    mask = z["is_optimization"]
    return {"optimization": {
        "n": int(mask.sum()),
        "mixture_nll": float(np.mean(z["predictive_nll"][mask])),
        "particle_nll": float(np.mean(-z["branch_gold_log_probs"][mask].astype(np.float64))),
    }}


def compute_train_nll(man: dict, runs: Path, log) -> dict:
    cfg = man["train_nll"]
    methods = {}
    for beta in man["betas"]:
        log(f"training set audit beta={beta}")
        methods[beta] = train_nll_stats(at(runs, man, cfg["npz"], cfg["backbone"], beta))
    return {"backbone": cfg["backbone"], "methods": methods}


def read_resources(path: Path) -> dict:
    """Wall seconds, GPU hours and the largest per GPU peak memory (GiB, two decimals) of a training job."""
    r = read_json(path)
    per_gpu = field(r, ["per_gpu_peak_memory_gib"], path)
    wall = int(field(r, ["wall_seconds"], path))
    if not per_gpu or wall <= 0:
        raise ValueError(f"{path}: no GPU memory samples or non positive wall time")
    return {"wall_seconds": wall, "peak_mem_gib": round(max(float(v) for v in per_gpu.values()), 2),
            "gpu_hours": field(r, ["gpu_hours"], path)}


def compute_timing(man: dict, runs: Path) -> dict:
    return {bb: {beta: read_resources(at(runs, man, man["resources"], bb, beta)) for beta in man["betas"]}
            for bb in man["backbones"]}


def compute_pre_adaptation(man: dict, runs: Path) -> dict:
    """Pooled correct label probability of the frozen backbone on the training set."""
    out = {}
    for bb, name in man["backbones"].items():
        path = runs / man["pre_adaptation"].format(name=name)
        pooled = field(read_json(path), ["statistics", "pooled"], path)
        if pooled["count"] != TRAIN_SET_SIZE:
            raise ValueError(f"{path}: {pooled['count']} examples, expected {TRAIN_SET_SIZE}")
        out[bb] = {k: float(pooled[k]) for k in ("mean", "median", "q10", "fraction_below_1_over_8")}
    return out


def compute_zero_shot(man: dict, runs: Path) -> dict:
    """Argmax accuracy and NLL of the frozen backbone on the test sets, with the equal dataset macro."""
    out = {}
    for bb, name in man["backbones"].items():
        per = {}
        for ds in M.DATASETS:
            path = runs / man["zero_shot"].format(name=name, dataset=ds)
            pooled = field(read_json(path), ["statistics", "pooled"], path)
            if pooled["count"] != M.EXPECTED_N[ds]:
                raise ValueError(f"{path}: {pooled['count']} examples, expected {M.EXPECTED_N[ds]}")
            per[ds] = {"accuracy": float(pooled["accuracy_argmax"]), "nll": float(pooled["mean_gold_nll"])}
        out[bb] = {"datasets": per,
                   "macro": {k: sum(per[d][k] for d in M.DATASETS) / len(M.DATASETS)
                             for k in ("accuracy", "nll")}}
    return out


def example_responsibility(row: dict, particles: int, path: Path) -> tuple[float, int]:
    """ESS and number of particles below the threshold for one example."""
    branches = row["branch_probs"]
    if len(branches) != particles:
        raise ValueError(f"{path}: {len(branches)} particles, expected {particles}")
    gold = [float(b[str(row["gold"])]) for b in branches]
    total = sum(gold)
    if total <= 0.0:
        raise ValueError(f"{path}: gold probability sum must be positive")
    resp = [v / total for v in gold]
    return 1.0 / sum(v * v for v in resp), sum(v < STARVED_THRESHOLD for v in resp)


def responsibility_counts(path: Path, particles: int) -> tuple[int, float, int]:
    """Number of examples, sum of their ESS and number of near starved (example, particle) pairs."""
    ess_sum, below, count = 0.0, 0, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                ess, low = example_responsibility(json.loads(line), particles, path)
                ess_sum += ess
                below += low
                count += 1
    return count, ess_sum, below


def particle_diagnostics(eval_dir: Path, particles: int) -> dict:
    """Raw responsibility ESS / m and Pr(r < 0.01) per dataset and their equal dataset macro.

    The responsibility of particle j on an example is its probability of the gold label divided by the
    sum over particles. ESS is 1 / sum of squared responsibilities, averaged over the examples.
    """
    per = {}
    for ds in M.DATASETS:
        path = M.find_dataset_dir(eval_dir, ds) / f"{ds}_joint_pro_predictions.jsonl"
        count, ess_sum, below = responsibility_counts(path, particles)
        if count != M.EXPECTED_N[ds]:
            raise ValueError(f"{path}: {count} rows, expected {M.EXPECTED_N[ds]}")
        per[ds] = {"n": count, "m": particles, "ess_over_m": ess_sum / count / particles,
                   "starved_frac": below / (count * particles)}
    macro = {k: sum(per[d][k] for d in M.DATASETS) / len(M.DATASETS) for k in ("ess_over_m", "starved_frac")}
    return {"m": particles, "threshold": STARVED_THRESHOLD, "datasets": per, "macro": macro}


def compute_particle_count(man: dict, runs: Path, final: dict, cache, log) -> dict:
    """Metrics and particle diagnostics for the particle count ablation."""
    cfg = man["m_ablation"]
    out = {}
    for bb, m, beta in itertools.product(cfg["backbones"], cfg["particles"], cfg["betas"]):
        log(f"particle count {bb} m={m} beta={beta}")
        folder = at(runs, man, man["final"], bb, beta, m)
        s = dict(final[bb][beta]) if m == 8 else evaluate(folder, cache)
        s["particles"] = particle_diagnostics(folder, m)
        out.setdefault(bb, {}).setdefault(str(m), {})[beta] = s
    return out


def compute(man: dict, runs: Path, cache, log=print) -> tuple[dict, dict]:
    """The numbers behind the tables, part by part. A part whose run outputs are missing is left out and its
    reason recorded, so that the tables of the other parts can still be written."""
    parts = {
        "final8": lambda res: compute_final(man, runs, cache, log),
        "selected": lambda res: compute_selected(man, runs, cache, log),
        "sequence": lambda res: compute_sequence(man, runs, cache, log),
        "train_nll": lambda res: compute_train_nll(man, runs, log),
        "timing": lambda res: compute_timing(man, runs),
        "pre_adaptation": lambda res: compute_pre_adaptation(man, runs),
        "zero_shot": lambda res: compute_zero_shot(man, runs),
        "m_ablation": lambda res: compute_particle_count(man, runs, needs(res, "final8"), cache, log),
    }
    res, missing = {}, {}
    for name, fn in parts.items():
        try:
            res[name] = fn(res)
        except FileNotFoundError as exc:
            missing[name] = str(exc)
    return res, missing


def needs(res: dict, part: str):
    """The result of another part, or FileNotFoundError when that part is missing."""
    if part not in res:
        raise FileNotFoundError(f"needs the runs of {part}")
    return res[part]


# ======================================================================= format

def f4(x: float) -> str:
    s = f"{x:.4f}"
    return "0.0000" if s == "-0.0000" else s


def betas_of(d: dict) -> list:
    return sorted(d, key=float)


def best_set(values: dict, direction: str) -> set:
    """Keys attaining the best unrounded value, all tied keys included."""
    if direction == "max":
        target = max(values.values())
        return {k for k, v in values.items() if v == target}
    if direction == "min":
        target = min(values.values())
        return {k for k, v in values.items() if v == target}
    target = min(abs(v) for v in values.values())
    return {k for k, v in values.items() if abs(v) == target}


def cell(x: float, bold: bool) -> str:
    """Four decimals, bold when best, negative numbers in math mode."""
    s = f4(x)
    if s.startswith("-"):
        return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"
    return f"\\textbf{{{s}}}" if bold else s


def math_cell(x: float, bold: bool) -> str:
    s = f4(x)
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def method_label(beta: str) -> str:
    return {"0": "Pure mixture, $\\beta=0$", "1": "Independent training, $\\beta=1$"}.get(beta, f"Bridge, $\\beta={beta}$")


def hms(sec: int) -> str:
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def table_pro_optimisation_puzzle(res: dict) -> str:
    """Table 1 (optimisation puzzle): training set R_mix and R_ind of Qwen3-4B."""
    meth = res["train_nll"]["methods"]
    betas = betas_of(meth)
    cols = ["mixture_nll", "particle_nll"]
    bold = {k: best_set({b: meth[b]["optimization"][k] for b in betas}, "min") for k in cols}
    out = []
    for b in betas:
        c = [math_cell(meth[b]["optimization"][k], b in bold[k]) for k in cols]
        out.append(f"{method_label(b)} & {c[0]} & {c[1]} \\\\")
    return "\n".join(out) + "\n"


def table_cross_backbone_summary(res: dict) -> str:
    """Table 3 (cross-backbone summary): zero shot, pure mixture and the best bridge setting for each metric."""
    out = []
    for metric, title, pick in (("accuracy", "Accuracy $\\uparrow$", max),
                                ("nll", "\\ac{NLL} $\\downarrow$", min)):
        if out:
            out.append("\\midrule")
        out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for bb, label in BACKBONE_LABEL.items():
            v = {b: res["final8"][bb][b]["macro"][metric] for b in res["final8"][bb]}
            bridge = {b: x for b, x in v.items() if 0 < float(b) < 1}
            best = pick(bridge, key=lambda b: (bridge[b], -float(b) if pick is max else float(b)))
            out.append(f"{label} & {f4(res['zero_shot'][bb]['macro'][metric])} & {f4(v['0'])} & "
                       f"{f4(bridge[best])} ($\\beta={best}$) \\\\")
    return "\n".join(out) + "\n"


def table_renyi_native_comparison(res: dict) -> str:
    """Table 4 (alpha-Renyi comparison): sequence space rows against the bridge with beta = 0.1."""
    cols = ["accuracy", "nll", "ece", "brier"]
    rows = [("Sequence \\(\\alpha=0.8\\), without noise", res["sequence"]["zero"]["qseq"]["macro"]),
            ("Sequence \\(\\alpha=0.8\\), with noise", res["sequence"]["auto"]["qseq"]["macro"]),
            ("Bridge, \\(\\beta=0.1\\)", res["final8"]["llama3_2_3b"]["0.1"]["macro"])]
    bold = {m: best_set({i: r[1][m] for i, r in enumerate(rows)}, DIRECTION[m]) for m in cols}
    return "\n".join(f"{label}\n& " + " & ".join(cell(mac[m], i in bold[m]) for m in cols) + "\n\\\\"
                     for i, (label, mac) in enumerate(rows)) + "\n"


def span(values: list) -> str:
    """Smallest and largest value to three decimals, or the single value when both print the same."""
    low, high = f"{min(values):.3f}", f"{max(values):.3f}"
    return low if low == high else f"{low}--{high}"


def table_computational_resources(res: dict) -> str:
    """Table 5 (cost of joint training): wall time and peak memory of independent training (beta = 1), and
    the range over the other values of beta of the time and memory relative to it."""
    out = []
    for bb, label in BACKBONE_LABEL.items():
        runs = res["timing"][bb]
        ref = runs["1"]
        others = [runs[b] for b in betas_of(runs) if b != "1"]
        rel_t = span([r["wall_seconds"] / ref["wall_seconds"] for r in others])
        rel_m = span([r["peak_mem_gib"] / ref["peak_mem_gib"] for r in others])
        out.append(f"{label} & {hms(ref['wall_seconds'])} & {ref['peak_mem_gib']:.2f} & {rel_t} & {rel_m} \\\\")
    return "\n".join(out) + "\n"


def table_main_results(res: dict) -> str:
    """Table 8 (held-out results): macro test results of the four backbones."""
    cols = ["accuracy", "nll", "ece", "brier", "gap"]
    out = []
    for bb, label in BACKBONE_LABEL.items():
        runs = res["final8"][bb]
        betas = betas_of(runs)
        bold = {m: best_set({b: runs[b]["macro"][m] for b in betas}, DIRECTION[m]) for m in cols}
        if out:
            out.append("\\midrule")
        for i, b in enumerate(betas):
            cells = [(math_cell if m == "gap" else cell)(runs[b]["macro"][m], b in bold[m]) for m in cols]
            out.append(f"{label if i == 0 else ''} & {method_label(b)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


def table_per_benchmark(res: dict, metric: str) -> str:
    """Tables 9 and 10 (accuracy and NLL on each test set), with the macro average in the last column."""
    keys = M.DATASETS + ["macro"]
    out = []
    for bb, label in BACKBONE_LABEL.items():
        runs = res["final8"][bb]
        betas = betas_of(runs)

        def val(b, k):
            return runs[b]["macro"][metric] if k == "macro" else runs[b]["datasets"][k][metric]

        if out:
            out.append("\\addlinespace[3pt]")
        out.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
        win = {k: best_set({b: val(b, k) for b in betas}, DIRECTION[metric]) for k in keys}
        for b in betas:
            out.append(f"{method_label(b)} & " + " & ".join(cell(val(b, k), b in win[k]) for k in keys) + " \\\\")
    return "\n".join(out) + "\n"


def table_llama3b_revised_selected_aggregate(res: dict) -> str:
    """Table 12 (validation-selected checkpoints) of Llama 3.2-3B, macro averages."""
    meth = res["selected"]["methods"]
    betas = betas_of(meth)
    cols = ["accuracy", "nll", "ece", "brier"]
    bold = {m: best_set({b: meth[b]["macro"][m] for b in betas}, DIRECTION[m]) for m in cols}
    out = []
    for b in betas:
        r = meth[b]
        cells = [f"{r['step'] // 1000}k"] + [cell(r["macro"][m], b in bold[m]) for m in cols]
        out.append(f"{method_label(b)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


def table_m_ablation(res: dict) -> str:
    """Table 13 (number of particles): macro accuracy, NLL and signed gap, ESS / m and Pr(r < 0.01)."""
    cols = ["accuracy", "nll", "gap"]
    out = []
    for k, (bb, runs) in enumerate(res["m_ablation"].items()):
        if k:
            out.append("\\midrule")
        ms = sorted(runs, key=int)
        for j, b in enumerate(runs[ms[0]]):
            bold = {c: best_set({m: runs[m][b]["macro"][c] for m in ms}, DIRECTION[c]) for c in cols}
            if j:
                out.append("\\addlinespace[2pt]")
            for i, m in enumerate(ms):
                s = runs[m][b]
                cells = [(math_cell if c == "gap" else cell)(s["macro"][c], m in bold[c]) for c in cols]
                p = s["particles"]["macro"]
                cells += [f4(p["ess_over_m"]), f4(p["starved_frac"])]
                head = BACKBONE_LABEL[bb] if i == 0 and j == 0 else ""
                out.append(f"{head} & {method_label(b) if i == 0 else ''} & {m} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out) + "\n"


def table_pre_adaptation(res: dict) -> str:
    """Table 14 (competence of the frozen backbones), three decimals."""
    out = []
    for bb, label in BACKBONE_LABEL.items():
        p = res["pre_adaptation"][bb]
        vals = [res["zero_shot"][bb]["macro"]["accuracy"], p["mean"], p["median"], p["q10"],
                p["fraction_below_1_over_8"]]
        out.append(f"{label} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\")
    return "\n".join(out) + "\n"


TABLES = {  # label without the tab: prefix -> (table, parts of compute() that it needs)
    "pro_optimisation_puzzle": (table_pro_optimisation_puzzle, ["train_nll"]),
    "cross-backbone-summary": (table_cross_backbone_summary, ["final8", "zero_shot"]),
    "renyi-native-comparison": (table_renyi_native_comparison, ["sequence", "final8"]),
    "computational_resources": (table_computational_resources, ["timing"]),
    "main-results": (table_main_results, ["final8"]),
    "per-benchmark-accuracy": (lambda res: table_per_benchmark(res, "accuracy"), ["final8"]),
    "per-benchmark-nll": (lambda res: table_per_benchmark(res, "nll"), ["final8"]),
    "llama3b_revised_selected_aggregate": (table_llama3b_revised_selected_aggregate, ["selected"]),
    "m_ablation": (table_m_ablation, ["m_ablation"]),
    "pre_adaptation": (table_pre_adaptation, ["pre_adaptation", "zero_shot"]),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Regenerate the table bodies of the paper from the run outputs.")
    ap.add_argument("--runs", default="runs", help="folder holding the runs (default: runs)")
    ap.add_argument("--out-dir", default="outputs", help="output root: tables/<label>.tex and data/tables.json")
    ap.add_argument("--cache", help="folder for the parsed per example arrays, reused between runs")
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"), help="run layout description")
    args = ap.parse_args(argv)

    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    cache = Path(args.cache) if args.cache else None
    try:
        res, missing = compute(man, Path(args.runs), cache, log=lambda s: print(s, file=sys.stderr))
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    out, data = Path(args.out_dir) / "tables", Path(args.out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    (data / "tables.json").write_text(json.dumps(res, indent=1, default=float))
    written = 0
    for name, (fn, needs) in TABLES.items():
        gone = [p for p in needs if p in missing]
        if gone:
            print(f"skipped tab:{name} (part {gone[0]}): {missing[gone[0]]}", file=sys.stderr)
            continue
        (out / f"{name}.tex").write_text(fn(res))
        written += 1
    print(f"wrote {written} of {len(TABLES)} tables to {out} and the unrounded numbers to {data / 'tables.json'}",
          file=sys.stderr)
    if not written:
        sys.exit("error: no table could be computed, see the missing inputs above")


if __name__ == "__main__":
    main()
