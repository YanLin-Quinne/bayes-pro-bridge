#!/usr/bin/env python3
"""Particle starvation diagnostics of Llama 3.2-3B and the Qwen3-4B final endpoint.

Post hoc statistics on fixed evaluation examples, computed from the per-example
particle probabilities that the evaluation scripts save. Writes

* Figures 5a, 5b and 10 (starvation dynamics of Llama 3.2-3B, four panels),
* Table 2 (post hoc particle diagnostics), the rows as the paper prints them,
* the aggregated source data of these under <out-dir>/data/starvation.

usage: python analysis/starvation.py --runs runs --out-dir outputs [--png] [--reuse-data]
"""

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import panel_style
import particle_panels

FINAL_STEP = 31932
SAVED_CHECKPOINTS = (5000, 10000, 15000, 20000, 25000, 30000)
TRAJECTORY_CHECKPOINTS = SAVED_CHECKPOINTS + (FINAL_STEP,)
TRAJECTORY_DATASETS = {"boolq": 3270, "hellaswag": 10042, "openbookqa": 500}
EIGHT_DATASETS = {
    "ARC-Challenge": 1172, "ARC-Easy": 2376, "boolq": 3270, "hellaswag": 10042,
    "openbookqa": 500, "piqa": 1838, "social_i_qa": 1954, "winogrande": 1267,
}
# (analysis id, architecture, evidence scope, run name prefix, checkpoints, datasets).
# The Qwen3-4B final endpoint covers all eight test sets and feeds Table 2 only.
ANALYSES = (
    ("llama3b_trajectory", "Llama 3.2 3B", "checkpoint trajectory",
     "llama3.2-3b", TRAJECTORY_CHECKPOINTS, TRAJECTORY_DATASETS),
    ("qwen4b_final_endpoint", "Qwen3 4B", "final endpoint",
     "qwen3-4b-base", ("final",), EIGHT_DATASETS),
)
LLAMA_ID, QWEN_ID = ANALYSES[0][0], ANALYSES[1][0]
# (method id, label, beta, beta as written in the run name)
METHODS = (
    ("beta0_pure_pro", "Pure mixture", 0.0, "0"),
    ("beta01_convex", "beta 0.1", 0.1, "0.1"),
)
PROBABILITY_TOLERANCE = 1e-5
STARVED_THRESHOLD = 1e-2
DATASET, MACRO, POOLED = "dataset", "equal_dataset_macro", "pooled_examples"

KEY_FIELDS = [
    "analysis_id", "architecture", "evidence_scope", "method", "method_label",
    "beta", "checkpoint",
]
COMMON_FIELDS = KEY_FIELDS + ["aggregation", "dataset", "sample_count"]
TSV_FIELDS = {
    "ess": COMMON_FIELDS + ["raw_responsibility_ess", "effective_coefficient_ess"],
    "cumulative_mass": COMMON_FIELDS + [
        "branch", "cumulative_raw_responsibility_mass",
        "cumulative_effective_responsibility_mass"],
    "branch_nll": COMMON_FIELDS + ["branch", "per_branch_nll"],
    "label_occupancy": COMMON_FIELDS + ["branch", "label", "occupancy", "is_dominant"],
    "mechanism_summary": COMMON_FIELDS + [
        "raw_responsibility_ess", "probability_responsibility_below_0_01",
        "soft_particle_answer_mi_nats", "mean_dominant_hard_occupancy",
        "max_dominant_hard_occupancy", "branch_nll_range"],
    "summary": KEY_FIELDS + ["aggregation", "metric", "value"],
}


class ValidationError(ValueError):
    """Raised when an input file or a source data file is not as expected."""


def mean(values):
    return sum(values) / len(values)


def weighted_mean(values, weights):
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


# One predictions file

def _clean_particle(probabilities, gold, where):
    if not isinstance(probabilities, dict) or gold not in probabilities:
        raise ValidationError(f"{where}: malformed particle probabilities")
    clean = {}
    for label, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{where}: nonnumeric probability")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValidationError(f"{where}: probability outside [0, 1]")
        clean[str(label)] = value
    if abs(sum(clean.values()) - 1.0) > PROBABILITY_TOLERANCE:
        raise ValidationError(f"{where}: probabilities do not sum to one")
    return clean


class Accumulator:
    """Running sums over the examples of one predictions file."""

    def __init__(self, beta):
        self.beta = beta
        self.count = 0
        self.particles = None
        self.raw_ess_sum = 0.0
        self.effective_ess_sum = 0.0
        self.labels = set()
        self.starved = 0

    def _allocate(self, particles):
        self.particles = particles
        self.mass_sums = [0.0] * particles
        self.raw_mass_sums = [0.0] * particles
        self.nll_sums = [0.0] * particles
        self.occupancy = [Counter() for _ in range(particles)]
        self.soft_sums = [Counter() for _ in range(particles)]

    def add(self, record, where):
        gold = record.get("gold")
        particle_probs = record.get("branch_probs")
        if not isinstance(particle_probs, list) or not particle_probs:
            raise ValidationError(f"{where}: particle probabilities are missing")
        clean_particles = [_clean_particle(p, gold, where) for p in particle_probs]
        labels = tuple(sorted(clean_particles[0]))
        if any(tuple(sorted(clean)) != labels for clean in clean_particles):
            raise ValidationError(f"{where}: answer labels differ between particles")
        if self.particles is None:
            self._allocate(len(clean_particles))
        if len(clean_particles) != self.particles:
            raise ValidationError(f"{where}: particle count changed")
        particles = self.particles
        gold_probabilities = [clean[gold] for clean in clean_particles]
        predictions = [max(sorted(clean), key=clean.get) for clean in clean_particles]

        denominator = sum(gold_probabilities)
        if denominator <= 0.0:
            raise ValidationError(f"{where}: gold probability sum must be positive")
        responsibility = [value / denominator for value in gold_probabilities]
        coefficients = [
            (1.0 - self.beta) * value + self.beta / particles
            for value in responsibility
        ]
        if (abs(sum(responsibility) - 1.0) > 1e-12
                or abs(sum(coefficients) - 1.0) > 1e-12):
            raise ValidationError(f"{where}: responsibilities do not normalize")

        self.labels.update(labels)
        self.count += 1
        self.raw_ess_sum += 1.0 / sum(value * value for value in responsibility)
        self.effective_ess_sum += 1.0 / sum(value * value for value in coefficients)
        self.starved += sum(value < STARVED_THRESHOLD for value in responsibility)
        for particle in range(particles):
            self.raw_mass_sums[particle] += responsibility[particle]
            self.mass_sums[particle] += coefficients[particle]
            self.nll_sums[particle] += -math.log(max(gold_probabilities[particle], 1e-12))
            self.occupancy[particle][predictions[particle]] += 1
            for label, value in clean_particles[particle].items():
                self.soft_sums[particle][label] += value

    def finish(self, expected_rows, where):
        if self.count != expected_rows:
            raise ValidationError(
                f"{where}: expected {expected_rows} rows but found {self.count}")
        labels = tuple(sorted(self.labels))
        soft_occupancy = [
            {label: sums[label] / self.count for label in labels}
            for sums in self.soft_sums
        ]
        return {
            "count": self.count,
            "particles": self.particles,
            "labels": list(labels),
            "raw_ess": self.raw_ess_sum / self.count,
            "effective_ess": self.effective_ess_sum / self.count,
            "mass": [value / self.count for value in self.mass_sums],
            "raw_mass": [value / self.count for value in self.raw_mass_sums],
            "nll": [value / self.count for value in self.nll_sums],
            "occupancy": [
                {label: count / self.count for label, count in counts.items()}
                for counts in self.occupancy
            ],
            "soft_mi": conditional_soft_mi(soft_occupancy),
            "starved": self.starved / (self.count * self.particles),
            "raw_ess_sum": self.raw_ess_sum,
            "effective_ess_sum": self.effective_ess_sum,
            "mass_sums": self.mass_sums,
            "raw_mass_sums": self.raw_mass_sums,
            "nll_sums": self.nll_sums,
        }


def analyze_file(path, beta, expected_rows):
    """Per-dataset statistics of one predictions file (JSON lines)."""
    accumulator = Accumulator(beta)
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"{path}:{number}: malformed JSON") from error
            accumulator.add(record, f"{path}:{number}")
    return accumulator.finish(expected_rows, str(path))


def conditional_soft_mi(soft_occupancy):
    """Mutual information in nats between particle index and answer, given the dataset.

    Computed from the soft answer occupancy of each particle."""
    labels = tuple(sorted(soft_occupancy[0]))
    mean_probabilities = {
        label: mean([probabilities[label] for probabilities in soft_occupancy])
        for label in labels
    }
    value = 0.0
    for probabilities in soft_occupancy:
        for label in labels:
            probability = probabilities[label]
            if probability > 0.0:
                value += probability * math.log(
                    probability / mean_probabilities[label]
                ) / len(soft_occupancy)
    if value < -PROBABILITY_TOLERANCE:
        raise ValidationError("soft particle answer MI is negative")
    return max(0.0, value)


# Locating the predictions

def prediction_path(runs, prefix, beta_name, checkpoint, dataset):
    """Predictions of a saved checkpoint, or of the final state for any other value."""
    run = Path(runs) / f"{prefix}_beta{beta_name}_m8"
    saved = checkpoint in SAVED_CHECKPOINTS
    folder = run / (f"checkpoints/checkpoint-{checkpoint}" if saved else "final")
    return folder / dataset / f"{dataset}_joint_pro_predictions.jsonl"


def require_files(sources):
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValidationError(
            f"{len(missing)} prediction files are missing, first: {missing[0]}")


def list_sources(runs):
    """Every predictions file of the two analyses."""
    sources = []
    for analysis_id, architecture, scope, prefix, checkpoints, datasets in ANALYSES:
        for method, label, beta, beta_name in METHODS:
            for checkpoint, (dataset, rows) in itertools.product(checkpoints, datasets.items()):
                sources.append({
                    "analysis_id": analysis_id, "architecture": architecture,
                    "evidence_scope": scope, "method": method, "method_label": label,
                    "beta": beta, "checkpoint": checkpoint, "dataset": dataset,
                    "expected_rows": rows,
                    "path": prediction_path(runs, prefix, beta_name, checkpoint, dataset),
                })
    require_files([source["path"] for source in sources])
    return sources


def analyze_sources(runs):
    """Run the per-file analysis for every source."""
    sources = list_sources(runs)
    items = []
    for index, source in enumerate(sources):
        print(f"[{index + 1}/{len(sources)}] {source['analysis_id']} "
              f"{source['method']} {source['checkpoint']} {source['dataset']}", flush=True)
        result = analyze_file(source["path"], source["beta"], source["expected_rows"])
        items.append({"source": source, "result": result})
    return items


# Aggregation

def dataset_unit(item):
    """Statistics of one dataset in the form shared with the summary units."""
    result = item["result"]
    dominant = [max(counts.values()) for counts in result["occupancy"]]
    return {
        "aggregation": DATASET, "dataset": item["source"]["dataset"],
        "sample_count": result["count"],
        "raw_ess": result["raw_ess"], "effective_ess": result["effective_ess"],
        "mass": result["mass"], "raw_mass": result["raw_mass"], "nll": result["nll"],
        "starved": result["starved"], "soft_mi": result["soft_mi"],
        "dominant": dominant, "mean_dominant": mean(dominant),
        "max_dominant": max(dominant),
        "nll_range": max(result["nll"]) - min(result["nll"]),
        "labels": result["labels"], "hard_occupancy": result["occupancy"],
        "raw_ess_sum": result["raw_ess_sum"],
        "effective_ess_sum": result["effective_ess_sum"],
        "mass_sums": result["mass_sums"], "raw_mass_sums": result["raw_mass_sums"],
        "nll_sums": result["nll_sums"],
    }


def summary_units(units):
    """Equal dataset macro and sample pooled aggregation of one group."""
    particles = range(len(units[0]["nll"]))
    counts = [unit["sample_count"] for unit in units]
    total = sum(counts)

    def over_units(key):
        return [unit[key] for unit in units]

    def pooled_sum(key, particle=None):
        values = [u[key] if particle is None else u[key][particle] for u in units]
        return sum(values) / total

    macro = {
        "aggregation": MACRO, "dataset": "all", "sample_count": total,
        "raw_ess": mean(over_units("raw_ess")),
        "effective_ess": mean(over_units("effective_ess")),
        "mass": [mean([u["mass"][p] for u in units]) for p in particles],
        "raw_mass": [mean([u["raw_mass"][p] for u in units]) for p in particles],
        "nll": [mean([u["nll"][p] for u in units]) for p in particles],
        "starved": mean(over_units("starved")),
        "soft_mi": mean(over_units("soft_mi")),
        "mean_dominant": mean(over_units("mean_dominant")),
        "max_dominant": mean(over_units("max_dominant")),
        "nll_range": mean(over_units("nll_range")),
    }
    pooled = {
        "aggregation": POOLED, "dataset": "all", "sample_count": total,
        "raw_ess": pooled_sum("raw_ess_sum"),
        "effective_ess": pooled_sum("effective_ess_sum"),
        "mass": [pooled_sum("mass_sums", p) for p in particles],
        "raw_mass": [pooled_sum("raw_mass_sums", p) for p in particles],
        "nll": [pooled_sum("nll_sums", p) for p in particles],
        "starved": weighted_mean(over_units("starved"), counts),
        "soft_mi": weighted_mean(over_units("soft_mi"), counts),
        "mean_dominant": weighted_mean(over_units("mean_dominant"), counts),
        "max_dominant": weighted_mean(over_units("max_dominant"), counts),
        "nll_range": weighted_mean(over_units("nll_range"), counts),
    }
    for unit in (macro, pooled):
        unit["labels"], unit["hard_occupancy"] = [], []
    return macro, pooled


def emit_particle_rows(tables, base, unit):
    """Rows that have one entry per particle (and per answer label)."""
    if abs(sum(unit["raw_mass"]) - 1.0) > 1e-12 or abs(sum(unit["mass"]) - 1.0) > 1e-12:
        raise ValidationError("mean responsibility masses do not sum to one")
    for particle in range(len(unit["nll"])):
        tables["cumulative_mass"].append(dict(
            base, branch=particle + 1,
            cumulative_raw_responsibility_mass=unit["raw_mass"][particle],
            cumulative_effective_responsibility_mass=unit["mass"][particle]))
        tables["branch_nll"].append(dict(
            base, branch=particle + 1, per_branch_nll=unit["nll"][particle]))
    for particle, counts in enumerate(unit["hard_occupancy"], 1):
        dominant = max(unit["labels"], key=lambda label: (counts.get(label, 0.0), label))
        for label in unit["labels"]:
            tables["label_occupancy"].append(dict(
                base, branch=particle, label=label,
                occupancy=counts.get(label, 0.0), is_dominant=int(label == dominant)))


def emit_scalar_rows(tables, base, unit):
    """Rows that have one entry per dataset or summary."""
    tables["ess"].append(dict(
        base, raw_responsibility_ess=unit["raw_ess"],
        effective_coefficient_ess=unit["effective_ess"]))
    tables["mechanism_summary"].append(dict(
        base, raw_responsibility_ess=unit["raw_ess"],
        probability_responsibility_below_0_01=unit["starved"],
        soft_particle_answer_mi_nats=unit["soft_mi"],
        mean_dominant_hard_occupancy=unit["mean_dominant"],
        max_dominant_hard_occupancy=unit["max_dominant"],
        branch_nll_range=unit["nll_range"]))


def summary_metrics(units, summary):
    """The scalar summaries of a macro or pooled unit written to summary.tsv."""
    values = {
        "raw_responsibility_ess": summary["raw_ess"],
        "effective_coefficient_ess": summary["effective_ess"],
        "mass_min": min(summary["mass"]), "mass_max": max(summary["mass"]),
        "mass_range": max(summary["mass"]) - min(summary["mass"]),
        "branch_nll_min": min(summary["nll"]), "branch_nll_max": max(summary["nll"]),
    }
    dominant = [value for unit in units if unit["dataset"].lower() == "hellaswag"
                for value in unit["dominant"]]
    if dominant:
        values["hellaswag_mean_dominant_occupancy"] = mean(dominant)
        values["hellaswag_max_dominant_occupancy"] = max(dominant)
    return values


def aggregate(runs, data_dir):
    """Analyze every source and write the aggregated TSV files."""
    items = analyze_sources(runs)
    groups = {}
    for item in items:
        source = item["source"]
        key = tuple(source[field] if field != "checkpoint" else str(source[field])
                    for field in KEY_FIELDS)
        groups.setdefault(key, []).append(item)
    if len({item["result"]["particles"] for item in items}) != 1:
        raise ValidationError("all sources must share one particle count")
    tables = {name: [] for name in TSV_FIELDS}
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item["source"]["dataset"])
        meta = dict(zip(KEY_FIELDS, key))
        units = [dataset_unit(item) for item in group]
        macro, pooled = summary_units(units)
        for unit in units + [macro, pooled]:
            base = dict(meta, aggregation=unit["aggregation"], dataset=unit["dataset"],
                        sample_count=unit["sample_count"])
            emit_scalar_rows(tables, base, unit)
            emit_particle_rows(tables, base, unit)
        for summary in (macro, pooled):
            for metric, value in summary_metrics(units, summary).items():
                tables["summary"].append(dict(
                    meta, aggregation=summary["aggregation"], metric=metric, value=value))
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, fields in TSV_FIELDS.items():
        write_tsv(data_dir / f"{name}.tsv", fields, tables[name])


def write_tsv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# Figures

def load_trajectory_data(data_dir):
    """Read the aggregated TSV files of the Llama trajectory into plot data."""
    data_dir = Path(data_dir)

    def llama_rows(name):
        return [row for row in read_tsv(data_dir / f"{name}.tsv")
                if row["analysis_id"] == LLAMA_ID]

    ess, mass, nll, occupancy = (
        llama_rows(name) for name in ("ess", "cumulative_mass", "branch_nll", "label_occupancy"))
    steps = list(TRAJECTORY_CHECKPOINTS)
    particles = max(int(row["branch"]) for row in nll)

    def select(rows, method, aggregation, dataset=None):
        return [row for row in rows if row["method"] == method
                and row["aggregation"] == aggregation
                and (dataset is None or row["dataset"] == dataset)]

    def matrix(rows, method, field):
        values = {(int(row["branch"]), int(row["checkpoint"])): float(row[field])
                  for row in select(rows, method, MACRO)}
        return np.array([[values[(particle, step)] for step in steps]
                         for particle in range(1, particles + 1)])

    ess_macro, ess_low, ess_high, mass_by, nll_by, dominant = {}, {}, {}, {}, {}, {}
    for method in panel_style.METHOD_ORDER:
        macro = {int(row["checkpoint"]): float(row["raw_responsibility_ess"])
                 for row in select(ess, method, MACRO)}
        per_dataset = select(ess, method, DATASET)
        ess_macro[method] = np.array([macro[step] for step in steps])
        for store, pick in ((ess_low, min), (ess_high, max)):
            store[method] = np.array([
                pick(float(row["raw_responsibility_ess"]) for row in per_dataset
                     if int(row["checkpoint"]) == step) for step in steps])
        mass_by[method] = matrix(mass, method, "cumulative_raw_responsibility_mass")
        if float(np.abs(mass_by[method].sum(axis=0) - 1.0).max()) > 1e-8:
            raise ValidationError("mean responsibility mass does not normalize")
        nll_by[method] = matrix(nll, method, "per_branch_nll")
        top = {(int(row["branch"]), int(row["checkpoint"])): (row["label"], float(row["occupancy"]))
               for row in select(occupancy, method, DATASET, "hellaswag")
               if int(row["is_dominant"])}
        dominant[method] = [[top[(particle, step)] for step in steps]
                            for particle in range(1, particles + 1)]
    return particle_panels.TrajectoryData(
        particles=particles, steps=steps, final_step=FINAL_STEP,
        datasets=list(TRAJECTORY_DATASETS), ess_macro=ess_macro, ess_low=ess_low,
        ess_high=ess_high, mass=mass_by, nll=nll_by, dominant=dominant)


PANEL_BUILDERS = {
    "figure_llama3b_responsibility_ess_trajectory": "plot_ess",
    "figure_llama3b_cumulative_responsibility_mass_trajectory": "plot_mass",
    "figure_llama3b_branch_nll_trajectory": "plot_nll",
    "figure_llama3b_label_occupancy_trajectory": "plot_occupancy",
}


def plot_trajectory_panels(data_dir, figure_dir):
    data = load_trajectory_data(data_dir)
    for stem, builder in PANEL_BUILDERS.items():
        figure = getattr(particle_panels, builder)(data)
        panel_style.save_panel(figure, Path(figure_dir) / stem, stem,
                               panel_style.PANEL_WIDTH_MM)


# Tables

# (analysis, method, checkpoint, backbone as printed, method label, checkpoint as printed)
SLOT_TABLE_ROWS = (
    (LLAMA_ID, "beta0_pure_pro", "5000", "Llama 3.2-3B", r"Pure mixture, $\beta=0$", "5k"),
    (LLAMA_ID, "beta01_convex", "5000", "", r"Bridge, $\beta=0.1$", "5k"),
    (LLAMA_ID, "beta0_pure_pro", str(FINAL_STEP), "", r"Pure mixture, $\beta=0$", "Final"),
    (LLAMA_ID, "beta01_convex", str(FINAL_STEP), "", r"Bridge, $\beta=0.1$", "Final"),
    (QWEN_ID, "beta0_pure_pro", "final", "Qwen3-4B", r"Pure mixture, $\beta=0$", "Final"),
    (QWEN_ID, "beta01_convex", "final", "", r"Bridge, $\beta=0.1$", "Final"),
)
SLOT_COLUMNS = (
    "raw_responsibility_ess", "probability_responsibility_below_0_01", "soft_particle_answer_mi_nats",
    "mean_dominant_hard_occupancy", "max_dominant_hard_occupancy", "branch_nll_range",
)


def slot_cell(field, value):
    """Three decimals. A mutual information that would print as 0.000 is written as below 0.001."""
    if field == "soft_particle_answer_mi_nats" and value < 0.0005:
        return "${<}0.001$"
    return format(value, ".3f")


def write_slot_table(data_dir, path):
    """Table 2: macro diagnostics at the first checkpoint and at the end."""
    lookup = {(row["analysis_id"], row["method"], row["checkpoint"]): row
              for row in read_tsv(Path(data_dir) / "mechanism_summary.tsv")
              if row["aggregation"] == MACRO}
    rows = []
    for analysis, method, checkpoint, model, label, point in SLOT_TABLE_ROWS:
        source = lookup[(analysis, method, checkpoint)]
        cells = " & ".join(slot_cell(field, float(source[field])) for field in SLOT_COLUMNS)
        rows.append(f"{model} & {label} & {point} & {cells} \\\\")
    lines = [*rows[:4], r"\midrule", *rows[4:], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# Command line

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
    data_dir = out_dir / "data" / "starvation"
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.reuse_data:
            aggregate(args.runs, data_dir)
        plot_trajectory_panels(data_dir, out_dir / "figures")
        write_slot_table(data_dir, table_dir / "particle_slot_diagnostics.tex")
    except (ValidationError, OSError) as error:
        sys.exit(f"error: {error}")
    except KeyError as error:
        sys.exit(f"error: the source data lack the entry {error}")


if __name__ == "__main__":
    main()
