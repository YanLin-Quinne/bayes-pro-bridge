#!/usr/bin/env python3
"""Qwen2 Instruct posterior-predictive full-metrics eval for true-joint branches.

This evaluates branch_i/lora_params.bin checkpoints saved by the true joint
PRO/MFLD script. For class-label PrO it computes per-branch choice
distributions and averages those probabilities:

    p_bar(c | x) = (1 / N) * sum_j p_j(c | x)

For sequence-level PrO, use --posterior_predictive_mode mean_then_normalize to
first average answer-string probabilities across branches and then normalize
over choices. The reported confidence, NLL, ECE, and auxiliary accuracy are all
based on the selected posterior-predictive probability over answer choices.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ALL_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
    "mmlu",
]

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
CANONICAL_LABELS = ["A", "B", "C", "D", "E"]
QWEN2_BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
QWEN2_MODEL_REVISION = "ba1cf1846d7df0a0591d6c00649f57e798519da8"
QWEN2_MODEL_TYPE = "qwen2"
QWEN2_ARCHITECTURE = "Qwen2ForCausalLM"
QWEN2_NUM_HIDDEN_LAYERS = 28
QWEN2_PAD_TOKEN_ID = 151643
QWEN2_EOS_TOKEN_ID = 151645
QWEN2_LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
QWEN2_PROMPT_STYLE = "qwen2_chatml_single_user_generation_prompt"


def configure_qwen2_instruct_tokenizer(tokenizer) -> Dict[str, object]:
    label_token_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"]
        for label in CANONICAL_LABELS
    }
    expected = {label: [token_id] for label, token_id in QWEN2_LABEL_TOKEN_IDS.items()}
    if tokenizer.pad_token_id != QWEN2_PAD_TOKEN_ID:
        raise ValueError(
            f"Unexpected Qwen2 pad token id: {tokenizer.pad_token_id} != {QWEN2_PAD_TOKEN_ID}"
        )
    if tokenizer.eos_token_id != QWEN2_EOS_TOKEN_ID:
        raise ValueError(
            f"Unexpected Qwen2 EOS token id: {tokenizer.eos_token_id} != {QWEN2_EOS_TOKEN_ID}"
        )
    if label_token_ids != expected:
        raise ValueError(f"Unexpected Qwen2 label tokens: {label_token_ids} != {expected}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Qwen2 Instruct tokenizer is missing its official chat template")
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with A only."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(probe, str) or not probe.endswith("<|im_start|>assistant\n"):
        raise ValueError(f"Unexpected Qwen2 ChatML generation boundary: {probe[-80:]!r}")
    tokenizer.padding_side = "left"
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "label_token_ids": dict(QWEN2_LABEL_TOKEN_IDS),
        "prompt_style": QWEN2_PROMPT_STYLE,
        "chat_template_probe_suffix": probe[-80:],
    }


def resolve_qwen2_target_modules(
    modules: Dict[str, nn.Module],
    target_modules: List[str],
) -> tuple[List[tuple[str, nn.Linear]], Dict[str, int]]:
    target_set = set(target_modules)
    if target_set != set(DEFAULT_TARGET_MODULES):
        raise ValueError(f"Unexpected Qwen2 target modules: {target_modules}")
    resolved = []
    counts = {target: 0 for target in target_modules}
    for name, module in modules.items():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in target_set:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Qwen2 target is not linear: {name} ({type(module).__name__})")
        resolved.append((name, module))
        counts[leaf_name] += 1
    expected_counts = {target: QWEN2_NUM_HIDDEN_LAYERS for target in target_modules}
    if counts != expected_counts or len(resolved) != 140:
        raise ValueError(
            f"Qwen2 LoRA target mismatch: counts={counts}, total={len(resolved)}"
        )
    return resolved, counts


def validate_qwen2_model(base_model: nn.Module) -> None:
    config = base_model.config
    if getattr(config, "model_type", None) != QWEN2_MODEL_TYPE:
        raise ValueError(f"Expected model_type={QWEN2_MODEL_TYPE}, got {config.model_type!r}")
    if getattr(config, "architectures", None) != [QWEN2_ARCHITECTURE]:
        raise ValueError(f"Unexpected Qwen2 architectures: {config.architectures!r}")
    if getattr(config, "num_hidden_layers", None) != QWEN2_NUM_HIDDEN_LAYERS:
        raise ValueError(f"Unexpected Qwen2 layer count: {config.num_hidden_layers!r}")


class LoRAParameterGroup(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_for_lora = x.to(self.lora_A.dtype)
        delta = x_for_lora @ self.lora_A.T @ self.lora_B.T
        return (delta * self.scaling).to(original_dtype)


class LoRABranch(nn.Module):
    def __init__(self, rank: int, alpha: int):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_params = nn.ModuleDict()
        self.name_mapping: Dict[str, str] = {}

    def add_lora_layer(self, name: str, in_features: int, out_features: int) -> None:
        safe_name = name.replace(".", "_")
        self.lora_params[safe_name] = LoRAParameterGroup(
            in_features=in_features,
            out_features=out_features,
            rank=self.rank,
            alpha=self.alpha,
        )
        self.name_mapping[safe_name] = name


class JointPROEnsemble:
    def __init__(
        self,
        base_model: nn.Module,
        ensemble_dir: str,
        num_branches: int,
        target_modules: List[str],
    ):
        self.base_model = base_model
        self.ensemble_dir = ensemble_dir
        self.num_branches = num_branches
        self.device = next(base_model.parameters()).device
        self.dtype = next(base_model.parameters()).dtype
        self.branches: List[LoRABranch] = []

        for param in self.base_model.parameters():
            param.requires_grad = False

        modules = dict(self.base_model.named_modules())
        validate_qwen2_model(self.base_model)
        self.target_linear_modules, self.target_module_counts = resolve_qwen2_target_modules(
            modules, target_modules
        )
        for branch_id in range(num_branches):
            branch_dir = os.path.join(ensemble_dir, f"branch_{branch_id}")
            branch = self._load_branch(branch_id, branch_dir, modules, target_modules)
            branch.lora_params = branch.lora_params.to(device=self.device, dtype=self.dtype)
            self.branches.append(branch)
            print(f"[load] branch_{branch_id} rank={branch.rank} alpha={branch.alpha}", flush=True)

    def _load_branch(
        self,
        branch_id: int,
        branch_dir: str,
        modules: Dict[str, nn.Module],
        target_modules: List[str],
    ) -> LoRABranch:
        params_path = os.path.join(branch_dir, "lora_params.bin")
        if not os.path.exists(params_path):
            raise FileNotFoundError(params_path)

        state = torch.load(params_path, map_location="cpu", weights_only=False)

        config_bin_path = os.path.join(branch_dir, "config.bin")
        config_json_path = os.path.join(branch_dir, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        elif os.path.exists(config_bin_path):
            config = torch.load(config_bin_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(config_json_path)

        expected_config = {
            "base_model": QWEN2_BASE_MODEL,
            "model_revision": QWEN2_MODEL_REVISION,
            "model_type": QWEN2_MODEL_TYPE,
            "architecture": QWEN2_ARCHITECTURE,
            "num_hidden_layers": QWEN2_NUM_HIDDEN_LAYERS,
            "branch_id": branch_id,
            "seed": 42 + branch_id,
            "target_modules": target_modules,
            "target_module_counts": self.target_module_counts,
            "num_lora_layers": 140,
            "state_format": "custom_lora_hook",
        }
        for key, expected_value in expected_config.items():
            if config.get(key) != expected_value:
                raise ValueError(
                    f"Branch metadata mismatch {branch_dir}/{key}: "
                    f"{config.get(key)!r} != {expected_value!r}"
                )

        run_config = {}
        run_config_path = os.path.join(os.path.dirname(branch_dir), "run_config.json")
        if os.path.exists(run_config_path):
            with open(run_config_path, "r", encoding="utf-8") as handle:
                run_config = json.load(handle)

        rank = config.get("rank", config.get("lora_r"))
        alpha = config.get("alpha", config.get("lora_alpha"))
        if rank is None:
            rank = run_config.get("lora_r")
        if alpha is None:
            alpha = run_config.get("lora_alpha")
        if rank is None:
            for key, value in state.items():
                stripped_key = key[len("lora_params."):] if key.startswith("lora_params.") else key
                is_lora_a = stripped_key == "lora_A" or stripped_key.endswith(".lora_A")
                if is_lora_a and isinstance(value, torch.Tensor) and value.ndim >= 1:
                    rank = value.shape[0]
                    break
        if rank is None:
            raise ValueError(
                f"Missing LoRA rank for {branch_dir}; check branch config.json/config.bin, "
                "run_config.json, or lora_params.bin lora_A tensors"
            )
        if alpha is None:
            raise ValueError(
                f"Missing LoRA alpha for {branch_dir}; check branch config.json/config.bin or run_config.json"
            )

        branch = LoRABranch(rank=int(rank), alpha=int(alpha))

        for name, module in self.target_linear_modules:
            branch.add_lora_layer(name, module.in_features, module.out_features)

        stripped = {
            key[len("lora_params."):] if key.startswith("lora_params.") else key: value
            for key, value in state.items()
        }
        branch.lora_params.load_state_dict(stripped)
        for key, value in branch.lora_params.state_dict().items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite adapter tensor: {branch_dir}/{key}")
        return branch

    @torch.no_grad()
    def branch_logits(
        self,
        branch: LoRABranch,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hooks = []

        def make_hook(lora_group: LoRAParameterGroup):
            def hook_fn(module, inputs, output):
                return output + lora_group(inputs[0])

            return hook_fn

        modules = dict(self.base_model.named_modules())
        for safe_name, lora_group in branch.lora_params.items():
            original = branch.name_mapping[safe_name]
            hooks.append(modules[original].register_forward_hook(make_hook(lora_group)))

        try:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits
        finally:
            for hook in hooks:
                hook.remove()


def generate_prompt(tokenizer, instruction: str, input_text: Optional[str] = None) -> str:
    if input_text:
        content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        content = instruction
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def candidate_labels(dataset: str, instruction: str) -> List[str]:
    match = re.search(r"Answer format:\s*([^\n]+)", instruction or "", flags=re.IGNORECASE)
    if match:
        labels = re.findall(
            r"true|false|solution\d+|answer\d+|ending\d+|option\d+",
            match.group(1).lower(),
        )
        if labels:
            return labels

    defaults = {
        "boolq": ["true", "false"],
        "piqa": ["solution1", "solution2"],
        "social_i_qa": ["answer1", "answer2", "answer3"],
        "hellaswag": ["ending1", "ending2", "ending3", "ending4"],
        "winogrande": ["option1", "option2"],
        "ARC-Challenge": ["answer1", "answer2", "answer3", "answer4", "answer5"],
        "ARC-Easy": ["answer1", "answer2", "answer3", "answer4", "answer5"],
        "openbookqa": ["answer1", "answer2", "answer3", "answer4"],
        "mmlu": ["answer1", "answer2", "answer3", "answer4"],
    }
    return defaults[dataset]


def candidate_response(label: str) -> str:
    return f"the correct answer is {label}"


def canonicalise_labels(source_labels: List[str]) -> Dict[str, str]:
    if len(source_labels) > len(CANONICAL_LABELS):
        raise ValueError(f"too many labels for canonical mapping: {source_labels}")
    return {
        source_label: CANONICAL_LABELS[idx]
        for idx, source_label in enumerate(source_labels)
    }


def label_token_instruction(instruction: str, source_labels: List[str]) -> str:
    label_map = canonicalise_labels(source_labels)
    mapping = ", ".join(
        f"{letter} = {source_label}"
        for source_label, letter in label_map.items()
    )
    return (
        f"{instruction}\n\n"
        f"Use the following option-letter mapping: {mapping}.\n"
        "Answer with the single option letter only."
    )


def compute_ece(confidences: List[float], correctness: List[float], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    ece = 0.0
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        in_bin = [
            pos for pos, conf in enumerate(confidences)
            if (low <= conf < high) or (idx == n_bins - 1 and conf == 1.0)
        ]
        if not in_bin:
            continue
        bin_conf = sum(confidences[pos] for pos in in_bin) / len(in_bin)
        bin_acc = sum(correctness[pos] for pos in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(bin_acc - bin_conf)
    return ece


def compute_brier(ensemble_probs: Dict[str, float], gold: str) -> float:
    if not ensemble_probs:
        return 0.0
    return sum(
        (prob - (1.0 if label == gold else 0.0)) ** 2
        for label, prob in ensemble_probs.items()
    )


def compute_bins(
    confidences: List[float],
    correctness: List[float],
    n_bins: int = 10,
) -> List[Dict]:
    bins = []
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        positions = [
            pos for pos, conf in enumerate(confidences)
            if (low <= conf < high) or (idx == n_bins - 1 and conf == 1.0)
        ]
        if not positions:
            bins.append({"bin": idx, "low": low, "high": high, "count": 0})
            continue
        bins.append({
            "bin": idx,
            "low": low,
            "high": high,
            "count": len(positions),
            "mean_confidence": sum(confidences[pos] for pos in positions) / len(positions),
            "empirical_accuracy": sum(correctness[pos] for pos in positions) / len(positions),
        })
    return bins


def load_data(dataset: str, limit: int = -1) -> List[Dict]:
    path = os.path.join("dataset", dataset, "test.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit > 0:
        return data[:limit]
    return data


def encode_choice_batch(
    tokenizer,
    dataset: str,
    batch: List[Dict],
    max_length: int,
    scoring_mode: str,
):
    encoded = []
    groups: Dict[int, List[int]] = {idx: [] for idx in range(len(batch))}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    stats = {
        "boundary_checked_count": 0,
        "boundary_mismatch_count": 0,
        "label_token_target_count": 0,
    }

    for example_idx, item in enumerate(batch):
        instruction = item.get("instruction", "")
        source_labels = candidate_labels(dataset, instruction)
        label_map = canonicalise_labels(source_labels)
        if scoring_mode == "label_token":
            prompt = generate_prompt(
                tokenizer,
                label_token_instruction(instruction, source_labels),
                item.get("input") or None,
            )
        else:
            prompt = generate_prompt(tokenizer, instruction, item.get("input") or None)
        prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for source_label in source_labels:
            label = label_map[source_label] if scoring_mode == "label_token" else source_label
            response = label if scoring_mode == "label_token" else candidate_response(source_label)
            if scoring_mode == "label_token":
                target_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if not target_ids:
                    continue
                if len(target_ids) != 1:
                    stats["label_token_target_count"] += len(target_ids)
                    raise ValueError(
                        "label_token scoring expects a single target token; "
                        f"dataset={dataset}, label={label!r}, target_len={len(target_ids)}"
                    )
                stats["label_token_target_count"] += 1
                joint_ids = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
                stats["boundary_checked_count"] += 1
                if (
                    len(joint_ids) < len(prefix_ids)
                    or joint_ids[:len(prefix_ids)] != prefix_ids
                    or joint_ids[len(prefix_ids):] != target_ids
                ):
                    stats["boundary_mismatch_count"] += 1
                    raise ValueError(
                        "Tokenizer boundary mismatch under matched label_token prompt; "
                        f"dataset={dataset}, label={label!r}. "
                        "Eval prompt template no longer matches the training template."
                    )
                full_ids = list(prefix_ids) + target_ids
                target_len = len(target_ids)
            else:
                full_ids = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
                if len(full_ids) < len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
                    raise ValueError(
                        "Tokenizer boundary mismatch between prompt and response; "
                        f"dataset={dataset}, scoring_mode={scoring_mode}, label={label!r}"
                    )
                stats["boundary_checked_count"] += 1
                target_len = len(full_ids) - len(prefix_ids)
                if target_len <= 0:
                    continue
            if scoring_mode == "label_token" and target_len != 1:
                raise ValueError(
                    "label_token scoring expects a single target token; "
                    f"dataset={dataset}, label={label!r}, target_len={target_len}"
                )
            if target_len > max_length:
                continue
            prefix_len = len(prefix_ids)
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                full_ids = full_ids[overflow:]
                prefix_len = max(0, prefix_len - overflow)
                if prefix_len == 0:
                    continue
            row_idx = len(encoded)
            encoded.append({
                "example_idx": example_idx,
                "label": label,
                "source_label": source_label,
                "ids": full_ids,
                "prefix_len": prefix_len,
                "target_len": target_len,
            })
            groups[example_idx].append(row_idx)

    if not encoded:
        return None, None, None, None, stats

    max_len = max(len(row["ids"]) for row in encoded)
    input_ids = []
    attention_mask = []
    for row in encoded:
        pad_len = max_len - len(row["ids"])
        input_ids.append(row["ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(row["ids"]) + [0] * pad_len)

    return (
        encoded,
        groups,
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        stats,
    )


@torch.no_grad()
def score_batch(
    model: JointPROEnsemble,
    tokenizer,
    dataset: str,
    batch: List[Dict],
    max_length: int,
    scoring_mode: str,
    posterior_predictive_mode: str,
    choice_score_normalization: str,
) -> List[Dict]:
    def empty_result() -> Dict:
        return {
            "pred": "",
            "confidence": 0.0,
            "true_probability": 0.0,
            "scores": {},
            "branch_probs": [],
            "ensemble_probs": {},
            "gold": "",
            "source_gold": "",
            "source_label_map": {},
            "branch_mean_entropy": 0.0,
            "sequence_mixture_scores": {},
            "posterior_predictive_mode": posterior_predictive_mode,
            "choice_score_normalization": choice_score_normalization,
            "local_scores_num_branches": 0,
            "num_choices": 0,
        }

    encoded, groups, input_ids, attention_mask, batch_stats = encode_choice_batch(
        tokenizer, dataset, batch, max_length, scoring_mode
    )
    if encoded is None:
        return [empty_result() for _ in batch], batch_stats

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    branch_scores = []

    for branch in model.branches:
        logits = model.branch_logits(branch, input_ids, attention_mask)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        choice_scores = []
        for row_idx, row in enumerate(encoded):
            ids = row["ids"]
            prefix_len = row["prefix_len"]
            target_len = row["target_len"]
            positions = torch.arange(
                prefix_len,
                prefix_len + target_len,
                device=log_probs.device,
            )
            token_ids = torch.tensor(
                ids[prefix_len:prefix_len + target_len],
                device=log_probs.device,
                dtype=torch.long,
            )
            token_scores = log_probs[row_idx, positions - 1, token_ids]
            total = token_scores.sum().item()
            if choice_score_normalization == "length":
                total = total / max(1, target_len)
            choice_scores.append(total)
        branch_scores.append(choice_scores)

    branch_scores_tensor = torch.tensor(branch_scores, dtype=torch.float32)
    results = []

    for example_idx, item in enumerate(batch):
        row_indices = groups[example_idx]
        if not row_indices:
            results.append(empty_result())
            continue
        labels = [encoded[row_idx]["label"] for row_idx in row_indices]
        source_labels = [encoded[row_idx]["source_label"] for row_idx in row_indices]
        local_scores = branch_scores_tensor[:, row_indices]
        if local_scores.shape[0] != model.num_branches or not torch.isfinite(local_scores).all():
            raise ValueError(
                f"Invalid local branch scores: shape={tuple(local_scores.shape)}"
            )
        source_to_label = {
            source_label: label
            for source_label, label in zip(source_labels, labels)
        }
        source_label_map = {
            label: source_label
            for source_label, label in zip(source_labels, labels)
        }
        label = source_to_label.get(item.get("answer"), item.get("answer"))
        if posterior_predictive_mode == "normalize_then_mean":
            branch_probs = torch.softmax(local_scores, dim=-1)
            ensemble_probs = branch_probs.mean(dim=0)
            sequence_mixture_scores = torch.logsumexp(local_scores, dim=0) - math.log(
                max(1, local_scores.shape[0])
            )
        elif posterior_predictive_mode == "mean_then_normalize":
            branch_probs = torch.softmax(local_scores, dim=-1)
            sequence_mixture_scores = torch.logsumexp(local_scores, dim=0) - math.log(
                max(1, local_scores.shape[0])
            )
            ensemble_probs = torch.softmax(sequence_mixture_scores, dim=-1)
        else:
            raise ValueError(f"Unsupported posterior_predictive_mode={posterior_predictive_mode}")

        if not torch.allclose(
            branch_probs.sum(dim=-1),
            torch.ones(branch_probs.shape[0]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("Branch probabilities do not sum to one")
        if not torch.isclose(ensemble_probs.sum(), torch.tensor(1.0), atol=1e-5, rtol=0.0):
            raise ValueError("Ensemble probabilities do not sum to one")

        best = int(torch.argmax(ensemble_probs).item())
        score_map = {
            choice: float(prob.item())
            for choice, prob in zip(labels, ensemble_probs)
        }
        sequence_score_map = {
            choice: float(score.item())
            for choice, score in zip(labels, sequence_mixture_scores)
        }
        branch_prob_maps = [
            {
                choice: float(prob.item())
                for choice, prob in zip(labels, branch_probs[branch_idx])
            }
            for branch_idx in range(branch_probs.shape[0])
        ]
        results.append({
            "pred": labels[best],
            "confidence": float(ensemble_probs[best].item()),
            "true_probability": float(max(score_map.get(label, 0.0), 1e-12)),
            "scores": score_map,
            "branch_probs": branch_prob_maps,
            "ensemble_probs": score_map,
            "sequence_mixture_scores": sequence_score_map,
            "gold": label,
            "source_gold": item.get("answer"),
            "source_label_map": source_label_map,
            "posterior_predictive_mode": posterior_predictive_mode,
            "choice_score_normalization": choice_score_normalization,
            "branch_mean_entropy": float(
                (-(branch_probs * branch_probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
            ),
            "local_scores_num_branches": int(local_scores.shape[0]),
            "num_choices": len(labels),
        })
    return results, batch_stats


def create_batches(data: List[Dict], batch_size: int) -> List[List[Dict]]:
    return [data[idx: idx + batch_size] for idx in range(0, len(data), batch_size)]


def evaluate(args) -> Dict:
    if args.base_model != QWEN2_BASE_MODEL:
        raise ValueError(f"This evaluator is pinned to {QWEN2_BASE_MODEL}")
    if args.model_revision != QWEN2_MODEL_REVISION:
        raise ValueError(f"This evaluator is pinned to revision {QWEN2_MODEL_REVISION}")
    if args.num_branches != 8:
        raise ValueError("Qwen2 Instruct evaluation requires exactly 8 branches")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    target_modules = [
        part.strip().strip("'\"")
        for part in args.target_modules.strip("[]").split(",")
        if part.strip()
    ] or DEFAULT_TARGET_MODULES

    print(f"[eval] dataset={args.dataset} device={device} ensemble={args.ensemble_dir}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.model_dtype],
        device_map={"": device},
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    )
    base_model.eval()
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    tokenizer_contract = configure_qwen2_instruct_tokenizer(tokenizer)

    ensemble = JointPROEnsemble(
        base_model=base_model,
        ensemble_dir=args.ensemble_dir,
        num_branches=args.num_branches,
        target_modules=target_modules,
    )

    data = load_data(args.dataset, args.limit)
    batches = create_batches(data, args.batch_size)

    confidences: List[float] = []
    correctness: List[float] = []
    nlls: List[float] = []
    briers: List[float] = []
    details = []
    per_k: Dict[str, Dict[str, float]] = {}
    boundary_checked_count = 0
    boundary_mismatch_count = 0
    label_token_target_count = 0
    start = time.time()

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"{args.dataset} joint-pro")):
        scored, batch_stats = score_batch(
            ensemble,
            tokenizer,
            args.dataset,
            batch,
            args.max_length,
            args.scoring_mode,
            args.posterior_predictive_mode,
            args.choice_score_normalization,
        )
        boundary_checked_count += int(batch_stats.get("boundary_checked_count", 0))
        boundary_mismatch_count += int(batch_stats.get("boundary_mismatch_count", 0))
        label_token_target_count += int(batch_stats.get("label_token_target_count", 0))
        for item, output in zip(batch, scored):
            if output["num_choices"] <= 0:
                raise ValueError(f"No valid answer choices for dataset={args.dataset}")
            example_id = item.get("id") or f"{args.dataset}-{len(details)}"
            gold = output.get("gold", item.get("answer"))
            correct = output["pred"] == gold
            true_probability = max(output["true_probability"], 1e-12)
            nll = -math.log(true_probability)
            brier = compute_brier(output["ensemble_probs"], gold)
            confidences.append(output["confidence"])
            correctness.append(1.0 if correct else 0.0)
            nlls.append(nll)
            briers.append(brier)

            k_key = str(output["num_choices"])
            bucket = per_k.setdefault(
                k_key,
                {"n": 0.0, "correct": 0.0, "conf": 0.0, "nll": 0.0, "brier": 0.0},
            )
            bucket["n"] += 1.0
            bucket["correct"] += 1.0 if correct else 0.0
            bucket["conf"] += output["confidence"]
            bucket["nll"] += nll
            bucket["brier"] += brier

            detail = dict(item)
            detail.update({
                "dataset": args.dataset,
                "id": example_id,
                "gold": gold,
                "source_gold": output.get("source_gold", item.get("answer")),
                "branch_probs": output["branch_probs"],
                "ensemble_probs": output["ensemble_probs"],
                "sequence_mixture_scores": output["sequence_mixture_scores"],
                "source_label_map": output["source_label_map"],
                "posterior_predictive_mode": output["posterior_predictive_mode"],
                "choice_score_normalization": output["choice_score_normalization"],
                "prob_pred": output["pred"],
                "prob_confidence": output["confidence"],
                "prob_correct": correct,
                "joint_pro_pred": output["pred"],
                "joint_pro_confidence": output["confidence"],
                "joint_pro_true_probability": true_probability,
                "joint_pro_scores": output["scores"],
                "joint_pro_correct": correct,
                "joint_pro_nll": nll,
                "prob_brier": brier,
                "joint_pro_brier": brier,
                "num_choices": output["num_choices"],
                "branch_mean_entropy": output["branch_mean_entropy"],
                "local_scores_num_branches": output["local_scores_num_branches"],
            })
            details.append(detail)

        if (batch_idx + 1) % args.log_every == 0 or batch_idx + 1 == len(batches):
            print(
                f"test:{batch_idx + 1}/{len(batches)} | "
                f"nll {sum(nlls) / len(nlls):.6f} | "
                f"ece {compute_ece(confidences, correctness):.6f} | "
                f"prob_accuracy {sum(correctness) / len(correctness):.6f} | "
                f"brier {sum(briers) / len(briers):.6f}",
                flush=True,
            )

    k_summary = {}
    for key, bucket in per_k.items():
        denom = max(1.0, bucket["n"])
        k_summary[key] = {
            "n": int(bucket["n"]),
            "prob_accuracy": bucket["correct"] / denom,
            "aux_accuracy": bucket["correct"] / denom,
            "mean_confidence": bucket["conf"] / denom,
            "nll": bucket["nll"] / denom,
            "brier": bucket["brier"] / denom,
        }

    prob_accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    metrics = {
        "dataset": args.dataset,
        "num_examples": len(data),
        "num_branches": args.num_branches,
        "local_scores_num_branches": args.num_branches,
        "invalid_sequence_count": 0,
        "mode": "joint_pro_posterior_predictive_choice_probs",
        "scoring_mode": args.scoring_mode,
        "model_dtype": {"bf16": "bfloat16", "fp16": "float16"}[args.model_dtype],
        "posterior_predictive_mode": args.posterior_predictive_mode,
        "choice_score_normalization": args.choice_score_normalization,
        "tokenizer_boundary_checked_count": boundary_checked_count,
        "tokenizer_boundary_mismatch_count": boundary_mismatch_count,
        "tokenizer_boundary_mismatch_rate": (
            boundary_mismatch_count / boundary_checked_count
            if boundary_checked_count else 0.0
        ),
        "label_token_target_count": label_token_target_count,
        "final_nll": sum(nlls) / len(nlls) if nlls else 0.0,
        "final_ece": compute_ece(confidences, correctness, args.n_bins),
        "final_brier": sum(briers) / len(briers) if briers else 0.0,
        "final_prob_accuracy": prob_accuracy,
        "final_aux_accuracy": prob_accuracy,
        "mean_confidence": mean_confidence,
        "signed_calibration_gap_conf_minus_acc": mean_confidence - prob_accuracy,
        "signed_accuracy_confidence_gap": prob_accuracy - mean_confidence,
        "per_choice_count": k_summary,
        "reliability_bins": compute_bins(confidences, correctness, args.n_bins),
        "runtime_seconds": time.time() - start,
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "model_type": QWEN2_MODEL_TYPE,
        "architecture": QWEN2_ARCHITECTURE,
        "num_hidden_layers": QWEN2_NUM_HIDDEN_LAYERS,
        "prompt_style": QWEN2_PROMPT_STYLE,
        "target_module_counts": ensemble.target_module_counts,
        **tokenizer_contract,
        "ensemble_dir": args.ensemble_dir,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_joint_pro_fullmetrics.json")
    details_path = os.path.join(args.output_dir, f"{args.dataset}_joint_pro_predictions.jsonl")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics}, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with open(details_path, "w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    print(f"final_nll {metrics['final_nll']}")
    print(f"final_ece {metrics['final_ece']}")
    print(f"final_brier {metrics['final_brier']}")
    print(f"final_aux_accuracy {metrics['final_aux_accuracy']}")
    print(f"mean_confidence {metrics['mean_confidence']}")
    print(f"signed_calibration_gap_conf_minus_acc {metrics['signed_calibration_gap_conf_minus_acc']}")
    print(f"scoring_mode {metrics['scoring_mode']}")
    print(f"posterior_predictive_mode {metrics['posterior_predictive_mode']}")
    print(f"choice_score_normalization {metrics['choice_score_normalization']}")
    print(f"summary_path {summary_path}")
    print(f"details_path {details_path}")
    print("test finished")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--model_revision", default=QWEN2_MODEL_REVISION)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--ensemble_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--num_branches", type=int, default=8)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,up_proj,down_proj")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--model_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--scoring_mode",
        choices=["response_string", "length_normalized_response", "label_token"],
        default="length_normalized_response",
        help=(
            "Use response_string for answer-string scoring. "
            "length_normalized_response is a deprecated alias kept for old scripts."
        ),
    )
    parser.add_argument(
        "--posterior_predictive_mode",
        choices=["normalize_then_mean", "mean_then_normalize"],
        default="normalize_then_mean",
        help="q_CL uses normalize_then_mean; q_SEQ uses mean_then_normalize.",
    )
    parser.add_argument(
        "--choice_score_normalization",
        choices=["length", "none"],
        default="length",
        help="Use none for sequence-level answer-string likelihood; length preserves legacy eval.",
    )
    args = parser.parse_args()
    if args.scoring_mode == "length_normalized_response":
        print(
            "WARNING: --scoring_mode length_normalized_response is deprecated; "
            "use --scoring_mode response_string and control length handling with "
            "--choice_score_normalization.",
            file=sys.stderr,
            flush=True,
        )
        if args.choice_score_normalization == "none":
            print(
                "WARNING: legacy scoring_mode name says length-normalized, but "
                "--choice_score_normalization none requests raw string log-probability.",
                file=sys.stderr,
                flush=True,
            )
    return args


if __name__ == "__main__":
    evaluate(parse_args())
