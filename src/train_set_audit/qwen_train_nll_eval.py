#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import (
    EXPECTED_TOTAL_ROWS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VALIDATION_ROWS,
    NUM_BRANCHES,
    RankZeroAccumulator,
    logmeanexp_branch_gold,
    rank_branch_map,
    reconstruct_metrics,
    sha256_file,
    split_membership,
)


TRAINER_SHA = "867a4ae38e5c9c2ac4ee581aeda0f330f7729d8f2bbe6946e0871ea61324b3c1"
DATA_SHA = "4651a3e707fbe28be03f486f96fcf57f8c38b2173f3e38812003108970793ac8"
MODEL_NAME = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
# The run_config of the scored run must carry the method's beta.
METHOD_BETAS = {"beta0": 0.0, "beta0.01": 0.01, "beta0.1": 0.1, "beta0.2": 0.2, "beta0.3": 0.3, "beta1": 1.0}
FINAL_GLOBAL_STEP = 31932
OUTPUT_FILES = ("train_nll_artifact.npz", "summary.json", "structural_audit.json")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
BRANCH_SEEDS = list(range(42, 50))


class IndexedCollator:
    def __init__(self, trainer_module, tokenizer, cutoff_len):
        self.base = trainer_module.ClassLabelCollator(tokenizer, cutoff_len=cutoff_len)

    def __call__(self, examples):
        batch = self.base(examples)
        batch["original_row_index"] = torch.tensor(
            [int(example["__original_row_index"]) for example in examples], dtype=torch.long
        )
        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHOD_BETAS))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cutoff", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rows", default="", help="comma list of original row indices (verification subset)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--expected-global-step", type=int, default=FINAL_GLOBAL_STEP)
    return parser.parse_args()


def parse_rows(text):
    if not text:
        return None
    rows = [int(value) for value in text.split(",") if value.strip()]
    if not rows or any(b <= a for a, b in zip(rows, rows[1:])) or rows[0] < 0:
        raise ValueError("--rows must be strictly increasing nonnegative indices")
    return rows


def load_trainer(path):
    trainer_path = Path(path).resolve()
    sys.path.insert(0, str(trainer_path.parent))
    spec = importlib.util.spec_from_file_location("authoritative_qwen_trainer", trainer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_run_and_hashes(args):
    run_root = Path(args.run_root).resolve()
    if sha256_file(args.trainer) != TRAINER_SHA:
        raise ValueError("authoritative trainer SHA256 mismatch")
    if sha256_file(args.data) != DATA_SHA:
        raise ValueError("training JSON SHA256 mismatch")
    state_path = run_root / "trainer_state.json"
    config_path = run_root / "run_config.json"
    state = json.loads(state_path.read_text())
    config = json.loads(config_path.read_text())
    if state.get("global_step") != args.expected_global_step or state.get("finished") is not True:
        raise ValueError(f"final trainer state is not finished at global step {args.expected_global_step}")
    expected_config = {
        "base_model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "num_branches": NUM_BRANCHES,
        "cutoff_len": 256,
        "micro_batch_size": 8,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": TARGET_MODULES,
        "attn_implementation": "sdpa",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"run_config mismatch for {key}: {config.get(key)!r} != {expected!r}")
    if config.get("objective_mode") != "pro_ce_convex" or float(config.get("beta_ce", -1)) != METHOD_BETAS[args.method]:
        raise ValueError(f"run_config objective/beta mismatch for {args.method}: "
                         f"{config.get('objective_mode')!r} beta_ce={config.get('beta_ce')!r}")
    if config.get("data_path") and sha256_file(config["data_path"]) != DATA_SHA:
        raise ValueError("training JSON of the run differs from the audited JSON")
    adapters = []
    for branch_id in range(NUM_BRANCHES):
        path = run_root / f"branch_{branch_id}" / "adapter_model.safetensors"
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty adapter: {path}")
        adapters.append({"branch_id": branch_id, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "run_root": str(run_root),
        "run_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "trainer_state": {"path": str(state_path), "sha256": sha256_file(state_path), "global_step": state["global_step"]},
        "adapters": adapters,
    }


def load_indexed_data(data_path, limit, rows=None):
    raw = load_dataset("json", data_files=data_path, keep_in_memory=True)["train"]
    if len(raw) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"training row count mismatch: {len(raw)}")
    indexed = raw.add_column("__original_row_index", list(range(len(raw))))
    split = indexed.train_test_split(test_size=EXPECTED_VALIDATION_ROWS, shuffle=True, seed=42, keep_in_memory=True)
    validation_indices = split["test"]["__original_row_index"]
    optimization_mask, validation_mask = split_membership(len(indexed), validation_indices)
    if int(optimization_mask.sum()) != EXPECTED_TRAIN_ROWS or int(validation_mask.sum()) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("recreated split counts are wrong")
    if rows is not None:
        if limit != 0 or rows[-1] >= len(indexed):
            raise ValueError("--rows excludes --limit and must index existing rows")
        return indexed.select(rows, keep_in_memory=True), optimization_mask, validation_mask
    selected = indexed if limit == 0 else indexed.select(range(min(limit, len(indexed))), keep_in_memory=True)
    return selected, optimization_mask, validation_mask


def validate_local_choice_distributions(local_log_probs, choice_mask, local_branches=NUM_BRANCHES):
    if local_log_probs.ndim != 3 or local_log_probs.shape[0] != local_branches:
        raise ValueError(f"the process must produce {local_branches} branches, got {tuple(local_log_probs.shape)}")
    expanded_mask = choice_mask.unsqueeze(0).expand_as(local_log_probs)
    valid = local_log_probs.masked_select(expanded_mask)
    if not torch.isfinite(valid).all():
        raise ValueError("nonfinite valid choice log probability")
    probabilities = local_log_probs.exp().masked_fill(~expanded_mask, 0.0)
    if torch.any(probabilities < -1e-7) or torch.any(probabilities > 1.0 + 1e-7):
        raise ValueError("choice probability outside [0, 1]")
    totals = probabilities.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=2e-6, rtol=2e-6):
        raise ValueError("choice probabilities do not normalize")


def run(args):
    # One process holds all eight branches (no torch.distributed); the trainer's
    # _gather_branch_logps(x, world_size=1) returns x, so the scoring path is unchanged.
    if args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("H100 BF16 CUDA support is required")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    rank = 0
    world_size = 1
    branch_map = rank_branch_map(world_size)
    rows = parse_rows(args.rows)
    started = time.time()
    try:
        output_dir = Path(args.output_dir).resolve()
        if args.batch_size != 8 or args.cutoff != 256:
            raise ValueError("batch size and cutoff must remain 8 and 256")
        if args.limit < 0:
            raise ValueError("limit must be nonnegative")
        provenance = None
        if rank == 0:
            # The launcher keeps its log in the output directory, so refuse only existing results.
            existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
            if existing:
                raise ValueError(f"output directory already holds results {existing}: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            provenance = validate_run_and_hashes(args)
        trainer = load_trainer(args.trainer)
        dataset, optimization_mask, validation_mask = load_indexed_data(args.data, args.limit, rows)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
        )
        tokenizer_contract = trainer._configure_qwen3_tokenizer(tokenizer)
        collator = IndexedCollator(trainer, tokenizer, args.cutoff)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
            drop_last=False,
            num_workers=0,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=False,
            attn_implementation="sdpa",
        ).to(device)
        base.config.use_cache = False
        backbone_contract = trainer._validate_qwen3_backbone(base, TARGET_MODULES)
        local_branch_ids = branch_map[str(rank)]
        model = trainer.BranchParallelLoRAEnsemble(
            base_model=base,
            local_branch_ids=local_branch_ids,
            branch_seeds=BRANCH_SEEDS,
            target_modules=TARGET_MODULES,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
        ).to(device)
        model.load_local_branches(args.run_root)
        model.eval()
        torch.set_grad_enabled(False)
        accumulator = RankZeroAccumulator()
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                row_indices = batch.pop("original_row_index")
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                local_choice = model.local_choice_log_probs(
                    batch["input_ids"], batch["attention_mask"], batch["choice_token_ids"], batch["choice_mask"]
                )
                validate_local_choice_distributions(local_choice, batch["choice_mask"], len(local_branch_ids))
                all_choice = trainer._gather_branch_logps(local_choice, world_size)
                if all_choice.shape[0] != NUM_BRANCHES:
                    raise ValueError("all_gather did not produce eight branches")
                gold = batch["gold_choice_indices"]
                branch_gold = all_choice.gather(
                    2, gold.view(1, -1, 1).expand(NUM_BRANCHES, -1, 1)
                ).squeeze(-1)
                if not torch.isfinite(branch_gold).all():
                    raise ValueError("nonfinite branch gold log probability")
                accumulator.add(rank, row_indices.cpu().numpy(), branch_gold.transpose(0, 1).double().cpu().numpy())
                if rank == 0 and (batch_number == 1 or batch_number % 1000 == 0):
                    print(f"PROGRESS batches={batch_number} rows={min(batch_number * args.batch_size, len(dataset))} "
                          f"elapsed_s={time.time() - started:.0f}", flush=True)

        if rank == 0:
            row_indices, branch_gold = accumulator.finalize()
            expected_evaluated = len(dataset)
            expected_rows = np.arange(expected_evaluated) if rows is None else np.asarray(rows, dtype=np.int64)
            if len(row_indices) != expected_evaluated or not np.array_equal(row_indices, expected_rows):
                raise ValueError("row indices are incomplete, duplicated, or out of order")
            is_optimization = optimization_mask[row_indices]
            is_validation = validation_mask[row_indices]
            log_mixture = logmeanexp_branch_gold(branch_gold)
            predictive_nll = -log_mixture
            if not np.isfinite(log_mixture).all() or not np.isfinite(predictive_nll).all():
                raise ValueError("nonfinite formal predictive NLL")
            metrics = reconstruct_metrics(branch_gold, is_optimization, is_validation)
            if args.limit == 0 and rows is None:
                counts = [metrics["all"]["count"], metrics["optimization"]["count"], metrics["internal_validation"]["count"]]
                if counts != [EXPECTED_TOTAL_ROWS, EXPECTED_TRAIN_ROWS, EXPECTED_VALIDATION_ROWS]:
                    raise ValueError(f"full evaluation counts mismatch: {counts}")
            artifact_path = output_dir / "train_nll_artifact.npz"
            np.savez_compressed(
                artifact_path,
                original_row_index=row_indices.astype(np.int64),
                is_optimization=is_optimization.astype(np.bool_),
                is_internal_validation=is_validation.astype(np.bool_),
                branch_gold_log_probs=branch_gold.astype(np.float32),
                log_mixture_probability=log_mixture.astype(np.float64),
                predictive_nll=predictive_nll.astype(np.float64),
            )
            summary = {
                "status": "PASS",
                "method": args.method,
                "metric_definition": "mean over rows of negative natural log of arithmetic mean of eight branch gold choice probabilities",
                "diagnostic_definition": "mean over rows and branches of negative branch gold log probability",
                "formal_metric_probability_clamp": None,
                "evaluated_rows": int(expected_evaluated),
                "full_dataset_rows": EXPECTED_TOTAL_ROWS,
                "limit": int(args.limit),
                "selected_rows": rows,
                "metrics": metrics,
                "artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path), "bytes": artifact_path.stat().st_size},
                "provenance": {
                    **provenance,
                    "evaluation_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
                    "authoritative_trainer": {"path": str(Path(args.trainer).resolve()), "sha256": TRAINER_SHA},
                    "training_json": {"path": str(Path(args.data).resolve()), "sha256": DATA_SHA},
                    "model": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "rank_branch_map": branch_map,
                    "dtype": "bfloat16",
                    "cutoff": args.cutoff,
                    "batch_size": args.batch_size,
                    "shuffle": False,
                    "drop_last": False,
                    "loader_sharded": False,
                    "split_seed": 42,
                    "split_test_size": EXPECTED_VALIDATION_ROWS,
                    "tokenizer_contract": tokenizer_contract,
                    "backbone_contract": backbone_contract,
                    "device": str(device),
                    "world_size": world_size,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "expected_global_step": args.expected_global_step,
                    "host": socket.gethostname(),
                    "elapsed_seconds": time.time() - started,
                },
            }
            summary_path = output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(f"TRAIN_NLL_EVALUATION_PASS method={args.method} rows={expected_evaluated}", flush=True)
    finally:
        print(f"ELAPSED_SECONDS={time.time() - started:.0f}", flush=True)


if __name__ == "__main__":
    run(parse_args())
