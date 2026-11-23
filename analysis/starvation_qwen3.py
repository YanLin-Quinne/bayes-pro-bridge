#!/usr/bin/env python3
"""Particle starvation dynamics of Qwen3-4B (Figure 12).

Post hoc statistics on the BoolQ, HellaSwag and OpenBookQA test sets at the
saved checkpoints 5,000 to 30,000 and at the final step 31,932, computed from
the per-example particle probabilities saved by the evaluation scripts. Writes
the four panels of Figure 12 and their aggregated source data under
<out-dir>/data/starvation_qwen3.

usage: python analysis/starvation_qwen3.py --runs runs --out-dir outputs [--png] [--reuse-data]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import panel_style
import particle_panels
from starvation import (
    DATASET, FINAL_STEP, MACRO, POOLED, TRAJECTORY_CHECKPOINTS as CHECKPOINTS,
    TRAJECTORY_DATASETS as DATASETS, ValidationError, analyze_file, mean,
    prediction_path, read_tsv, require_files, write_tsv,
)

LABELS = "ABCD"
# (method id, label, beta, beta as written in the run name)
METHODS = (
    ("beta0_pure_pro", r"Pure mixture ($\beta=0$)", 0.0, "0"),
    ("beta01_convex", r"Bridge ($\beta=0.1$)", 0.1, "0.1"),
)
COMMON_FIELDS = [
    "method", "method_label", "beta", "checkpoint", "aggregation", "dataset",
    "sample_count",
]
TSV_FIELDS = {
    "responsibility_ess_trajectory": COMMON_FIELDS + ["raw_responsibility_ess"],
    "cumulative_responsibility_mass_trajectory": COMMON_FIELDS + [
        "branch", "cumulative_responsibility_mass"],
    "branch_nll_trajectory": COMMON_FIELDS + ["branch", "per_branch_nll"],
    "label_occupancy_trajectory": COMMON_FIELDS + ["branch", "label", "occupancy"],
}
PANEL_BUILDERS = {
    "figure_qwen3_responsibility_ess_trajectory": "plot_ess",
    "figure_qwen3_cumulative_responsibility_mass_trajectory": "plot_mass",
    "figure_qwen3_branch_nll_trajectory": "plot_nll",
    "figure_qwen3_label_occupancy_trajectory": "plot_occupancy",
}


def list_sources(runs):
    """Every predictions file as (method, beta, checkpoint, dataset, path)."""
    sources = [
        (method, beta, checkpoint, dataset,
         prediction_path(runs, "qwen3-4b-base", beta_name, checkpoint, dataset))
        for method, _, beta, beta_name in METHODS
        for checkpoint in CHECKPOINTS for dataset in DATASETS
    ]
    require_files([source[-1] for source in sources])
    return sources


def summary_units(results):
    """Equal dataset macro and sample pooled aggregation of the three datasets."""
    particles = range(len(results[0]["nll"]))
    total = sum(result["count"] for result in results)
    macro = {
        "aggregation": MACRO,
        "ess": mean([r["raw_ess"] for r in results]),
        "mass": [mean([r["raw_mass"][p] for r in results]) for p in particles],
        "nll": [mean([r["nll"][p] for r in results]) for p in particles],
    }
    pooled = {
        "aggregation": POOLED,
        "ess": sum(r["raw_ess_sum"] for r in results) / total,
        "mass": [sum(r["raw_mass_sums"][p] for r in results) / total for p in particles],
        "nll": [sum(r["nll_sums"][p] for r in results) / total for p in particles],
    }
    return macro, pooled, total


def aggregate(runs, data_dir):
    """Analyze every source and write the four aggregated TSV files."""
    sources = list_sources(runs)
    results = {}
    for index, (method, beta, checkpoint, dataset, path) in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {method} {checkpoint} {dataset}", flush=True)
        results[(method, checkpoint, dataset)] = analyze_file(path, beta, DATASETS[dataset])
    tables = {name: [] for name in TSV_FIELDS}
    for method, label, beta, _ in METHODS:
        for checkpoint in CHECKPOINTS:
            per_dataset = [results[(method, checkpoint, dataset)] for dataset in DATASETS]
            meta = {"method": method, "method_label": label, "beta": beta,
                    "checkpoint": checkpoint}
            for dataset, result in zip(DATASETS, per_dataset):
                base = dict(meta, aggregation=DATASET, dataset=dataset,
                            sample_count=result["count"])
                add_rows(tables, base, result["raw_ess"], result["raw_mass"], result["nll"])
                if dataset == "hellaswag":
                    for particle, counts in enumerate(result["occupancy"], 1):
                        for answer in LABELS:
                            tables["label_occupancy_trajectory"].append(dict(
                                base, branch=particle, label=answer,
                                occupancy=counts.get(answer, 0.0)))
            macro, pooled, total = summary_units(per_dataset)
            for unit in (macro, pooled):
                base = dict(meta, aggregation=unit["aggregation"], dataset="all",
                            sample_count=total)
                add_rows(tables, base, unit["ess"], unit["mass"], unit["nll"])
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, fields in TSV_FIELDS.items():
        write_tsv(data_dir / f"{name}.tsv", fields, tables[name])


def add_rows(tables, base, ess, mass, nll):
    tables["responsibility_ess_trajectory"].append(dict(base, raw_responsibility_ess=ess))
    for particle in range(len(mass)):
        tables["cumulative_responsibility_mass_trajectory"].append(dict(
            base, branch=particle + 1, cumulative_responsibility_mass=mass[particle]))
        tables["branch_nll_trajectory"].append(dict(
            base, branch=particle + 1, per_branch_nll=nll[particle]))


def load_trajectory_data(data_dir):
    """Read the aggregated TSV files into plot data."""
    data_dir = Path(data_dir)
    ess = read_tsv(data_dir / "responsibility_ess_trajectory.tsv")
    mass = read_tsv(data_dir / "cumulative_responsibility_mass_trajectory.tsv")
    nll = read_tsv(data_dir / "branch_nll_trajectory.tsv")
    occupancy = read_tsv(data_dir / "label_occupancy_trajectory.tsv")
    steps = list(CHECKPOINTS)
    particles = max(int(row["branch"]) for row in nll)

    def matrix(rows, method, field):
        values = {(int(row["branch"]), int(row["checkpoint"])): float(row[field])
                  for row in rows if row["method"] == method and row["aggregation"] == MACRO}
        return np.array([[values[(particle, step)] for step in steps]
                         for particle in range(1, particles + 1)])

    ess_macro, ess_low, ess_high, mass_by, nll_by, dominant = {}, {}, {}, {}, {}, {}
    for method in panel_style.METHOD_ORDER:
        method_rows = [row for row in ess if row["method"] == method]
        macro = {int(row["checkpoint"]): float(row["raw_responsibility_ess"])
                 for row in method_rows if row["aggregation"] == MACRO}
        per_dataset = [row for row in method_rows if row["aggregation"] == DATASET]
        ess_macro[method] = np.array([macro[step] for step in steps])
        for store, pick in ((ess_low, min), (ess_high, max)):
            store[method] = np.array([
                pick(float(row["raw_responsibility_ess"]) for row in per_dataset
                     if int(row["checkpoint"]) == step) for step in steps])
        mass_by[method] = matrix(mass, method, "cumulative_responsibility_mass")
        nll_by[method] = matrix(nll, method, "per_branch_nll")
        lookup = {(int(row["branch"]), int(row["checkpoint"]), row["label"]): float(row["occupancy"])
                  for row in occupancy if row["method"] == method}
        dominant[method] = []
        for particle in range(1, particles + 1):
            line = []
            for step in steps:
                label = max(LABELS, key=lambda x: lookup[(particle, step, x)])
                line.append((label, lookup[(particle, step, label)]))
            dominant[method].append(line)
    return particle_panels.TrajectoryData(
        particles=particles, steps=steps, final_step=FINAL_STEP,
        datasets=list(DATASETS), ess_macro=ess_macro, ess_low=ess_low,
        ess_high=ess_high, mass=mass_by, nll=nll_by, dominant=dominant)


def plot_trajectories(data_dir, figure_dir):
    data = load_trajectory_data(data_dir)
    for stem, builder in PANEL_BUILDERS.items():
        figure = getattr(particle_panels, builder)(data)
        panel_style.save_panel(figure, Path(figure_dir) / stem, stem,
                               panel_style.PANEL_WIDTH_MM)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="runs", help="root of the training runs")
    parser.add_argument("--out-dir", default="outputs", help="output root")
    parser.add_argument("--png", action="store_true", help="also write 600 dpi PNG files")
    parser.add_argument("--reuse-data", action="store_true",
                        help="skip the analysis and plot from the existing TSV files")
    args = parser.parse_args(argv)
    panel_style.WRITE_PNG = args.png
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data" / "starvation_qwen3"
    try:
        if not args.reuse_data:
            aggregate(args.runs, data_dir)
        plot_trajectories(data_dir, out_dir / "figures")
    except (ValidationError, OSError) as error:
        sys.exit(f"error: {error}")
    except KeyError as error:
        sys.exit(f"error: the source data lack the entry {error}")


if __name__ == "__main__":
    main()
