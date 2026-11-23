#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core import logmeanexp_branch_gold, reconstruct_metrics, sha256_file


ARTIFACT_KEYS = {
    "original_row_index",
    "is_optimization",
    "is_internal_validation",
    "branch_gold_log_probs",
    "log_mixture_probability",
    "predictive_nll",
}


def close(a, b, tolerance=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tolerance


def audit_one(summary_path, artifact_path, verify_sources=False):
    summary_path = Path(summary_path)
    artifact_path = Path(artifact_path)
    summary = json.loads(summary_path.read_text())
    if sha256_file(artifact_path) != summary["artifact"]["sha256"]:
        raise ValueError("artifact SHA256 mismatch")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        if set(artifact.files) != ARTIFACT_KEYS:
            raise ValueError(f"unexpected artifact keys: {set(artifact.files)}")
        arrays = {key: artifact[key] for key in artifact.files}
    rows = arrays["original_row_index"]
    branch = arrays["branch_gold_log_probs"]
    optimization = arrays["is_optimization"]
    validation = arrays["is_internal_validation"]
    expected_rows = np.arange(len(rows), dtype=np.int64)
    if summary.get("selected_rows") is not None:
        # Verification subset: rows are the recorded original indices, in increasing order.
        expected_rows = np.asarray(summary["selected_rows"], dtype=np.int64)
    if rows.ndim != 1 or not np.array_equal(rows, expected_rows):
        raise ValueError("row indices are not exact original order")
    if branch.shape != (len(rows), 8):
        raise ValueError("branch log probability shape mismatch")
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("nonfinite artifact value")
    if np.any(optimization == validation):
        raise ValueError("split masks do not partition evaluated rows")
    recomputed_log_mixture = logmeanexp_branch_gold(branch)
    max_log_error = float(np.max(np.abs(recomputed_log_mixture - arrays["log_mixture_probability"])))
    max_nll_error = float(np.max(np.abs(-recomputed_log_mixture - arrays["predictive_nll"])))
    if max_log_error > 1e-10 or max_nll_error > 1e-10:
        raise ValueError(f"predictive formula mismatch: {max_log_error} {max_nll_error}")
    metrics = reconstruct_metrics(branch, optimization, validation)
    for split_name, split_metrics in metrics.items():
        recorded = summary["metrics"][split_name]
        if split_metrics["count"] != recorded["count"]:
            raise ValueError(f"count mismatch for {split_name}")
        for key in ("predictive_nll", "mean_individual_branch_nll"):
            if not close(split_metrics[key], recorded[key]):
                raise ValueError(f"summary mismatch for {split_name}.{key}")
        for actual, expected in zip(split_metrics["branch_mean_nll"], recorded["branch_mean_nll"]):
            if not close(actual, expected):
                raise ValueError(f"branch summary mismatch for {split_name}")
    checked_hashes = []
    if verify_sources:
        provenance = summary["provenance"]
        sources = [
            provenance["evaluation_script"], provenance["authoritative_trainer"],
            provenance["training_json"], provenance["run_config"], provenance["trainer_state"],
            *provenance["adapters"],
        ]
        for source in sources:
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError(f"source SHA256 mismatch: {source['path']}")
            checked_hashes.append(source["path"])
    return {
        "status": "PASS",
        "method": summary["method"],
        "summary_path": str(summary_path),
        "artifact_path": str(artifact_path),
        "evaluated_rows": len(rows),
        "max_log_mixture_absolute_error": max_log_error,
        "max_predictive_nll_absolute_error": max_nll_error,
        "metrics": metrics,
        "verified_source_hashes": checked_hashes,
    }


def parse_method(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be LABEL=OUTPUT_DIR")
    label, directory = value.split("=", 1)
    return label, Path(directory)


def write_aggregate(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "qwen_full_train_nll.tsv"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["method", "split", "count", "predictive_nll", "mean_individual_branch_nll", *[f"branch_{i}_nll" for i in range(8)]])
        for result in results:
            for split_name in ("all", "optimization", "internal_validation"):
                metrics = result["metrics"][split_name]
                writer.writerow([
                    result["method"], split_name, metrics["count"], metrics["predictive_nll"],
                    metrics["mean_individual_branch_nll"], *metrics["branch_mean_nll"],
                ])
    audit = {"status": "PASS", "methods": results}
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    all_values = [(result["method"], result["metrics"]["all"]["predictive_nll"]) for result in results]
    observed = min(all_values, key=lambda item: item[1])
    lines = [
        "# Qwen3 4B full training predictive NLL audit",
        "",
        "All four artifacts independently reproduce the reported summary formula and split counts.",
        "Predictive NLL is negative log of the arithmetic mean of eight branch gold probabilities.",
        "Mean individual branch NLL is a diagnostic and is not the predictive metric.",
        "",
        f"The lowest observed full JSON predictive NLL in these completed artifacts is {observed[0]} at {observed[1]:.10f}.",
        "This is an empirical comparison of final checkpoints. It is not an optimization guarantee for Pure PrO or any beta value.",
        "",
    ]
    markdown_path = output_dir / "interpretation.md"
    markdown_path.write_text("\n".join(lines))
    return tsv_path, audit_path, markdown_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary")
    parser.add_argument("--artifact")
    parser.add_argument("--single-output")
    parser.add_argument("--method", action="append", type=parse_method, default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    if args.summary or args.artifact or args.single_output:
        if not (args.summary and args.artifact and args.single_output):
            raise SystemExit("single audit requires summary, artifact, and single output")
        result = audit_one(args.summary, args.artifact, args.verify_sources)
        Path(args.single_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("SINGLE_ARTIFACT_AUDIT_PASS")
        return
    if len(args.method) != 4 or not args.output_dir:
        raise SystemExit("aggregate audit requires four methods and output dir")
    results = []
    for label, directory in args.method:
        result = audit_one(directory / "summary.json", directory / "train_nll_artifact.npz", args.verify_sources)
        if result["method"] != label:
            raise ValueError(f"method label mismatch: {label} != {result['method']}")
        results.append(result)
    write_aggregate(results, Path(args.output_dir))
    print("FOUR_METHOD_AGGREGATE_AUDIT_PASS")


if __name__ == "__main__":
    main()
