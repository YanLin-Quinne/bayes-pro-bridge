"""Audit prospective control and objective switch fork manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint import protocol_fingerprint, validate_objective_switch_protocol
from .config import SCHEMA_VERSION


REQUIRED_FIELDS = {
    "schema_version",
    "lineage_type",
    "resume_mode",
    "source_checkpoint",
    "source_beta",
    "target_beta",
    "source_protocol",
    "target_protocol",
    "source_protocol_fingerprint",
    "target_protocol_fingerprint",
    "completed_epoch",
    "optimizer_step",
    "total_optimizer_steps",
    "world_size",
    "ranks",
    "branch_allocation",
    "restored_state",
    "only_beta_differs",
    "beta_changed",
}
RESTORED_CATEGORIES = {
    "adapter_weights",
    "optimizer_state",
    "scheduler_state",
    "python_rng",
    "torch_cpu_rng",
    "cuda_rng",
}


def _manifest_path(value: str | Path) -> Path:
    path = Path(value)
    return path / "fork_manifest.json" if path.is_dir() else path


def load_fork_manifest(value: str | Path) -> dict[str, Any]:
    path = _manifest_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"missing fork manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("fork manifest must be a JSON object")
    missing = REQUIRED_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"fork manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("fork manifest schema mismatch")
    if manifest["lineage_type"] != "prospective_objective_fork":
        raise ValueError("fork manifest lineage type mismatch")
    source = manifest["source_protocol"]
    target = manifest["target_protocol"]
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("fork manifest protocols are invalid")
    switch = validate_objective_switch_protocol(
        source,
        target,
        expected_source_beta=manifest["source_beta"],
    )
    expected = {
        "target_beta": switch["target_beta"],
        "source_protocol_fingerprint": protocol_fingerprint(source),
        "target_protocol_fingerprint": protocol_fingerprint(target),
        "only_beta_differs": True,
        "beta_changed": switch["beta_changed"],
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"fork manifest {key} mismatch")
    training = source.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("fork source training protocol is invalid")
    identity = {
        "total_optimizer_steps": training.get("total_optimizer_steps"),
        "world_size": training.get("world_size"),
        "branch_allocation": training.get("branch_allocation"),
        "ranks": [f"rank_{rank}" for rank in range(int(training["world_size"]))],
    }
    for key, expected_value in identity.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"fork manifest {key} mismatch")
    restored = manifest["restored_state"]
    if not isinstance(restored, dict) or set(restored) != RESTORED_CATEGORIES:
        raise ValueError("fork manifest restored state categories mismatch")
    if not all(isinstance(value, bool) for value in restored.values()):
        raise ValueError("fork manifest restored state values are invalid")
    if not all(restored.values()):
        raise ValueError("fork manifest has an unrestored state category")
    return manifest


def _audit_diagnostics(
    manifest_value: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    path = _manifest_path(manifest_value).parent / "diagnostics.jsonl"
    if not path.is_file():
        raise FileNotFoundError("missing fork diagnostics JSONL")
    branches = int(manifest["source_protocol"]["training"]["num_branches"])
    target_beta = float(manifest["target_beta"])
    required = {
        "raw_responsibility_ess",
        "effective_coefficient_ess",
        "branch_responsibility_mass",
        "branch_gold_nll",
        "branch_nll_dispersion",
        "minimum_effective_coefficient",
    }
    counts = {"training_minibatch": 0, "fixed_evaluation": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid fork diagnostics JSON at line {line_number}"
                ) from error
            scope = record.get("scope")
            if scope not in counts:
                raise ValueError("fork diagnostics scope is invalid")
            if record.get("beta") != target_beta:
                raise ValueError("fork diagnostics beta mismatch")
            missing = required - record.keys()
            if missing:
                raise ValueError(
                    f"fork diagnostics are missing fields: {sorted(missing)}"
                )
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        raise ValueError(f"fork diagnostics {key} is not finite")
            for key in ("branch_responsibility_mass", "branch_gold_nll"):
                values = record[key]
                if not isinstance(values, list) or len(values) != branches:
                    raise ValueError(f"fork diagnostics {key} branch count mismatch")
                if not all(math.isfinite(float(value)) for value in values):
                    raise ValueError(f"fork diagnostics {key} is not finite")
            if scope == "training_minibatch":
                losses = {"mixture_nll", "branch_ce", "data_loss", "prior", "total_loss"}
                if losses - record.keys():
                    raise ValueError("training diagnostics are missing finite loss values")
            counts[scope] += 1
    if not all(counts.values()):
        raise ValueError("fork diagnostics must contain both diagnostic scopes")
    return counts


def compare_fork_manifests(
    control_value: str | Path,
    rescue_value: str | Path,
) -> dict[str, Any]:
    control = load_fork_manifest(control_value)
    rescue = load_fork_manifest(rescue_value)
    expected_modes = {"control": "strict", "rescue": "objective_switch"}
    for name, manifest in (("control", control), ("rescue", rescue)):
        if manifest["resume_mode"] != expected_modes[name]:
            raise ValueError(f"{name} resume mode mismatch")
    shared_fields = (
        "source_checkpoint",
        "source_beta",
        "source_protocol",
        "source_protocol_fingerprint",
        "completed_epoch",
        "optimizer_step",
        "total_optimizer_steps",
        "world_size",
        "ranks",
        "branch_allocation",
        "restored_state",
    )
    for key in shared_fields:
        if control[key] != rescue[key]:
            raise ValueError(f"fork manifests disagree on {key}")
    if control["target_beta"] != 0.0 or rescue["target_beta"] != 0.1:
        raise ValueError("fork target betas must be 0 and 0.1")
    if control["source_beta"] != 0.0:
        raise ValueError("fork source beta must be 0")
    if control["beta_changed"] or not rescue["beta_changed"]:
        raise ValueError("fork beta change flags are invalid")
    normalized = validate_objective_switch_protocol(
        control["target_protocol"],
        rescue["target_protocol"],
        expected_source_beta=0.0,
    )
    if not normalized["beta_changed"]:
        raise ValueError("fork target protocols do not differ in beta")
    control_diagnostics = _audit_diagnostics(control_value, control)
    rescue_diagnostics = _audit_diagnostics(rescue_value, rescue)
    return {
        "status": "PASS",
        "source_checkpoint": control["source_checkpoint"],
        "control_beta": 0.0,
        "rescue_beta": 0.1,
        "completed_epoch": control["completed_epoch"],
        "optimizer_step": control["optimizer_step"],
        "total_optimizer_steps": control["total_optimizer_steps"],
        "control_diagnostics": control_diagnostics,
        "rescue_diagnostics": rescue_diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit prospective objective forks")
    parser.add_argument("--control", required=True)
    parser.add_argument("--rescue", required=True)
    parser.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = compare_fork_manifests(args.control, args.rescue)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
