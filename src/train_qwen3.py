# Branch-parallel class-label PRO/MFLD LoRA ensemble training for Qwen3-4B-Base.
#
# This isolated fork preserves the validated convex objective and adds strict
# Qwen3 architecture, tokenizer, LoRA-target, and checkpoint contracts. Existing
# Llama training scripts and outputs are intentionally left untouched.
#
# Objective:
#   data_loss = - mean_i log( (1/N) * sum_j p_j(y_i | x_i) )
#   p_j(y_i | x_i) is the normalized probability of the gold A/B/C/D/E label.
#   prior     = (0.5 * prior_lambda / (N * M)) * sum_j ||theta_j||^2
#   T         = 1 / (N * M) for NoisyAdamW when temperature_mode="auto"
#
# Convex mode ("pro_ce_convex"), with 0 <= beta <= 1:
#   L_beta = (1 - beta) * L_PrO + beta * L_branchCE + R_prior
#   beta = 0 -> pure PrO data objective; beta = 1 -> CE-only data objective.
#   The prior R_prior stays OUTSIDE the (1 - beta)/beta mixture and is unchanged.
#   Both L_PrO and L_branchCE are per-example / per-branch means, exactly as the
#   existing modes normalize them.

import json
import math
import os
import random
import re
import time
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import fire
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from NoisyAdamW_our import NoisyAdamW

try:
    import wandb
except Exception:
    wandb = None

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:
    safe_load_file = None
    safe_save_file = None


DEFAULT_BRANCH_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
CANONICAL_LABELS = ["A", "B", "C", "D", "E"]
QWEN3_BASE_MODEL = "Qwen/Qwen3-4B-Base"
QWEN3_MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
QWEN3_MODEL_TYPE = "qwen3"
QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"
QWEN3_NUM_HIDDEN_LAYERS = 36
QWEN3_PAD_TOKEN_ID = 151643
QWEN3_LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
OBJECTIVE_MODES = {"pro_only", "pro_ce_anchor", "ce_only", "pro_ce_convex"}
OBJECTIVE_MODE_ALIASES = {
    "pro": "pro_only",
    "pro_only": "pro_only",
    "pro_ce_anchor": "pro_ce_anchor",
    # pro_ce_convex: L_beta = (1 - beta) * L_PrO + beta * L_branchCE (+ prior).
    "pro_ce_convex": "pro_ce_convex",
    "convex": "pro_ce_convex",
    "ce": "ce_only",
    "ce_only": "ce_only",
}


def _parse_list(value, default: List, item_type=str) -> List:
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        return [item_type(v) for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return list(default)
    return [item_type(part.strip().strip("'\"")) for part in text.split(",") if part.strip()]


def _resolve_wandb_tags(raw_tags, base_model, objective_mode, beta_ce) -> List[str]:
    """Resolve the W&B run tags (pure, no side effects, no secrets touched).

    ``raw_tags`` is the raw value of the ``WANDB_TAGS`` environment variable
    (a comma-separated string; ``None``/empty when unset). It is parsed by
    splitting on commas, trimming surrounding whitespace and dropping empty
    entries. If the parsed list is non-empty those tags are used verbatim.

    Otherwise a dynamic fallback is derived from ``base_model``,
    ``objective_mode`` and ``beta_ce`` so the tags always describe the actual
    run (no hard-coded / stale model name).
    """
    if raw_tags is not None:
        parsed = [part.strip() for part in str(raw_tags).split(",")]
        parsed = [part for part in parsed if part]
        if parsed:
            return parsed

    model_slug = os.path.basename(str(base_model).rstrip("/")) or "model"
    return [
        "comet",
        "prob-mfld",
        "branch-parallel",
        "class-label-pro",
        model_slug,
        str(objective_mode),
        f"beta-{beta_ce}",
    ]


def _configure_qwen3_tokenizer(tokenizer) -> Dict[str, object]:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise ValueError("Qwen3 tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    label_token_ids = {}
    for label in CANONICAL_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Canonical label {label!r} is not one token: {token_ids}")
        label_token_ids[label] = int(token_ids[0])

    if int(tokenizer.pad_token_id) != QWEN3_PAD_TOKEN_ID:
        raise ValueError(
            f"Unexpected Qwen3 pad token id: {tokenizer.pad_token_id} != {QWEN3_PAD_TOKEN_ID}"
        )
    if label_token_ids != QWEN3_LABEL_TOKEN_IDS:
        raise ValueError(
            f"Unexpected Qwen3 label-token ids: {label_token_ids} != {QWEN3_LABEL_TOKEN_IDS}"
        )
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "padding_side": tokenizer.padding_side,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "label_token_ids": label_token_ids,
    }


def _resolve_target_linear_specs(
    base_model: nn.Module,
    target_modules: List[str],
) -> Tuple[List[Tuple[str, int, int]], Dict[str, int]]:
    if not target_modules:
        raise ValueError("target_modules must not be empty")
    if len(set(target_modules)) != len(target_modules):
        raise ValueError(f"target_modules contains duplicates: {target_modules}")

    target_set = set(target_modules)
    counts = {target: 0 for target in target_modules}
    specs = []
    non_linear = []
    for name, module in base_model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in target_set:
            continue
        if not isinstance(module, nn.Linear):
            non_linear.append((name, type(module).__name__))
            continue
        counts[leaf_name] += 1
        specs.append((name, int(module.in_features), int(module.out_features)))

    if non_linear:
        raise TypeError(f"Matched non-linear target modules: {non_linear[:5]}")
    missing = [target for target, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Missing target modules in backbone: {missing}; counts={counts}")
    return specs, counts


def _validate_qwen3_backbone(
    base_model: nn.Module,
    target_modules: List[str],
) -> Dict[str, object]:
    config = base_model.config
    model_type = str(getattr(config, "model_type", ""))
    architectures = list(getattr(config, "architectures", None) or [])
    num_hidden_layers = int(getattr(config, "num_hidden_layers", -1))
    if model_type != QWEN3_MODEL_TYPE:
        raise ValueError(f"Expected model_type={QWEN3_MODEL_TYPE}, got {model_type!r}")
    if QWEN3_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Expected architecture {QWEN3_ARCHITECTURE}, got {architectures}"
        )
    if num_hidden_layers != QWEN3_NUM_HIDDEN_LAYERS:
        raise ValueError(
            f"Expected {QWEN3_NUM_HIDDEN_LAYERS} hidden layers, got {num_hidden_layers}"
        )

    specs, counts = _resolve_target_linear_specs(base_model, target_modules)
    expected_counts = {
        target: QWEN3_NUM_HIDDEN_LAYERS for target in target_modules
    }
    if counts != expected_counts:
        raise ValueError(f"Unexpected Qwen3 target-module counts: {counts} != {expected_counts}")
    if len(specs) != QWEN3_NUM_HIDDEN_LAYERS * len(target_modules):
        raise ValueError(f"Unexpected number of Qwen3 target linear modules: {len(specs)}")
    return {
        "model_type": model_type,
        "architectures": architectures,
        "num_hidden_layers": num_hidden_layers,
        "target_module_counts": counts,
        "num_target_linear_modules": len(specs),
    }


def _init_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=60),
    )
    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _rank_print(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs, flush=True)


class LoRAParameterGroup(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        init_seed: Optional[int] = None,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        if init_seed is not None:
            torch.manual_seed(init_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(init_seed)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_for_lora = x.to(self.lora_A.dtype)
        result = self.dropout(x_for_lora @ self.lora_A.T @ self.lora_B.T)
        return (result * self.scaling).to(original_dtype)


class LoRABranch(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: List[str],
        rank: int,
        alpha: int,
        dropout: float,
        seed: int,
        branch_id: int,
        target_linear_specs: Optional[List[Tuple[str, int, int]]] = None,
        target_module_counts: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.branch_id = branch_id
        self.seed = seed
        self.lora_params = nn.ModuleDict()

        if target_linear_specs is None:
            target_linear_specs, resolved_counts = _resolve_target_linear_specs(
                base_model, target_modules
            )
        else:
            resolved_counts = dict(target_module_counts or {})
        self.target_module_counts = resolved_counts

        for layer_idx, (name, in_features, out_features) in enumerate(target_linear_specs):
            layer_seed = seed * 100_000 + layer_idx
            self.lora_params[name.replace(".", "_")] = LoRAParameterGroup(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                init_seed=layer_seed,
            )

        self.name_mapping = {
            name.replace(".", "_"): name for name, _, _ in target_linear_specs
        }

    def forward(self, **kwargs):
        hooks = []
        modules = dict(self.base_model.named_modules())

        def make_hook(lora_group: LoRAParameterGroup):
            def hook_fn(module, inputs, output):
                return output + lora_group(inputs[0])

            return hook_fn

        for safe_name, lora_group in self.lora_params.items():
            hook = modules[self.name_mapping[safe_name]].register_forward_hook(
                make_hook(lora_group)
            )
            hooks.append(hook)

        try:
            return self.base_model(**kwargs)
        finally:
            for hook in hooks:
                hook.remove()


class BranchParallelLoRAEnsemble(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        local_branch_ids: List[int],
        branch_seeds: List[int],
        target_modules: List[str],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.local_branch_ids = list(local_branch_ids)
        self.branches = nn.ModuleList()

        target_linear_specs, target_module_counts = _resolve_target_linear_specs(
            base_model, target_modules
        )
        self.target_module_counts = target_module_counts
        self.num_target_linear_modules = len(target_linear_specs)

        for param in self.base_model.parameters():
            param.requires_grad = False

        for branch_id in self.local_branch_ids:
            self.branches.append(
                LoRABranch(
                    base_model=base_model,
                    target_modules=target_modules,
                    rank=lora_r,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    seed=branch_seeds[branch_id],
                    branch_id=branch_id,
                    target_linear_specs=target_linear_specs,
                    target_module_counts=target_module_counts,
                )
            )

    def local_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        token_logps = []
        for branch in self.branches:
            outputs = branch(input_ids=input_ids, attention_mask=attention_mask)
            token_logps.append(_true_token_logp(outputs.logits, labels))
        return torch.stack(token_logps, dim=0)

    def local_choice_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        choice_token_ids: torch.Tensor,
        choice_mask: torch.Tensor,
    ) -> torch.Tensor:
        choice_log_probs = []
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_token_indices = (
            attention_mask.size(1)
            - 1
            - attention_mask.long().flip(dims=[1]).argmax(dim=1)
        )
        safe_choice_ids = choice_token_ids.clamp_min(0)

        for branch in self.branches:
            outputs = branch(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[batch_indices, last_token_indices, :]
            logits = next_token_logits.gather(1, safe_choice_ids)
            logits = logits.masked_fill(~choice_mask, torch.finfo(logits.dtype).min)
            choice_log_probs.append(F.log_softmax(logits.float(), dim=-1))
        return torch.stack(choice_log_probs, dim=0)

    def local_prior_sum(self) -> torch.Tensor:
        prior = None
        for param in self.parameters():
            if not param.requires_grad:
                continue
            term = param.float().square().sum()
            prior = term if prior is None else prior + term
        if prior is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return prior

    def save_local_branches(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for branch_id, branch in zip(self.local_branch_ids, self.branches):
            branch_dir = os.path.join(output_dir, f"branch_{branch_id}")
            os.makedirs(branch_dir, exist_ok=True)
            state = {
                key: value.detach().cpu()
                for key, value in branch.state_dict().items()
                if "lora_params" in key
            }
            torch.save(state, os.path.join(branch_dir, "lora_params.bin"))
            if safe_save_file is not None:
                safe_save_file(state, os.path.join(branch_dir, "adapter_model.safetensors"))
            first_lora_group = next(iter(branch.lora_params.values()), None)
            rank = int(first_lora_group.rank) if first_lora_group is not None else None
            alpha = int(first_lora_group.alpha) if first_lora_group is not None else None
            metadata = {
                "branch_id": branch_id,
                "seed": branch.seed,
                "target_modules": branch.target_modules,
                "target_module_counts": branch.target_module_counts,
                "num_lora_layers": len(branch.lora_params),
                "base_model": QWEN3_BASE_MODEL,
                "model_revision": QWEN3_MODEL_REVISION,
                "model_type": QWEN3_MODEL_TYPE,
                "architecture": QWEN3_ARCHITECTURE,
                "num_hidden_layers": QWEN3_NUM_HIDDEN_LAYERS,
                "rank": rank,
                "alpha": alpha,
                "lora_r": rank,
                "lora_alpha": alpha,
                "state_format": "custom_lora_hook",
            }
            with open(os.path.join(branch_dir, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            torch.save(metadata, os.path.join(branch_dir, "config.bin"))

    def load_local_branches(self, input_dir: str) -> None:
        for branch_id, branch in zip(self.local_branch_ids, self.branches):
            branch_dir = os.path.join(input_dir, f"branch_{branch_id}")
            safe_path = os.path.join(branch_dir, "adapter_model.safetensors")
            bin_path = os.path.join(branch_dir, "lora_params.bin")
            if os.path.exists(safe_path):
                if safe_load_file is None:
                    raise RuntimeError("safetensors is unavailable, cannot load adapter_model.safetensors")
                state = safe_load_file(safe_path, device="cpu")
            elif os.path.exists(bin_path):
                state = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Missing warm-start branch parameters: {branch_dir}")
            missing, unexpected = branch.load_state_dict(state, strict=False)
            bad_missing = [key for key in missing if "lora_params" in key]
            if bad_missing or unexpected:
                raise RuntimeError(
                    f"Bad warm-start state for branch {branch_id}: "
                    f"missing={bad_missing[:5]}, unexpected={unexpected[:5]}"
                )


def _true_token_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    safe_labels = shift_labels.masked_fill(shift_labels.eq(-100), 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    return log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def _gather_branch_logps(local_logps: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_logps.detach()
    gathered = [torch.empty_like(local_logps) for _ in range(world_size)]
    dist.all_gather(gathered, local_logps.detach())
    return torch.cat(gathered, dim=0)


def _hist_string(preds: torch.Tensor, choice_dim: int) -> str:
    labels = CANONICAL_LABELS[:choice_dim]
    counts = torch.bincount(preds.detach().cpu(), minlength=choice_dim)
    return ",".join(
        f"{label}={int(counts[idx])}"
        for idx, label in enumerate(labels)
    )


def _choice_prediction_diagnostics(
    choice_log_probs: torch.Tensor,
    choice_mask: torch.Tensor,
) -> Dict[str, object]:
    choice_dim = min(choice_log_probs.size(-1), len(CANONICAL_LABELS))
    mask = choice_mask.unsqueeze(0)
    probs = choice_log_probs.exp().masked_fill(~mask, 0.0)
    safe_log_probs = choice_log_probs.masked_fill(~mask, 0.0)

    branch_entropy = -(probs * safe_log_probs).sum(dim=-1)
    branch_entropy_per_branch = branch_entropy.mean(dim=1)

    ensemble_probs = probs.mean(dim=0).masked_fill(~choice_mask, 0.0)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ensemble_entropy = -(
        ensemble_probs * ensemble_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)

    masked_log_probs = choice_log_probs.masked_fill(
        ~mask, torch.finfo(choice_log_probs.dtype).min
    )
    branch_preds = masked_log_probs.argmax(dim=-1)
    ensemble_preds = ensemble_probs.argmax(dim=-1)

    branch_hist = "; ".join(
        f"b{branch_idx}:{_hist_string(branch_preds[branch_idx], choice_dim)}"
        for branch_idx in range(branch_preds.size(0))
    )
    ensemble_hist = _hist_string(ensemble_preds, choice_dim)

    pairwise_disagreement = torch.tensor(0.0, device=choice_log_probs.device)
    if branch_preds.size(0) > 1:
        disagreements = []
        for left in range(branch_preds.size(0)):
            for right in range(left + 1, branch_preds.size(0)):
                disagreements.append(
                    branch_preds[left].ne(branch_preds[right]).float().mean()
                )
        pairwise_disagreement = torch.stack(disagreements).mean()

    return {
        "branch_entropy_mean": float(branch_entropy.mean().detach().cpu()),
        "branch_entropy_min": float(branch_entropy_per_branch.min().detach().cpu()),
        "ensemble_entropy": float(ensemble_entropy.mean().detach().cpu()),
        "branch_pred_hist": branch_hist,
        "ensemble_pred_hist": ensemble_hist,
        "pairwise_disagreement": float(pairwise_disagreement.detach().cpu()),
    }


def _merge_diagnostic_buffer(diagnostic_buffer: List[Dict[str, object]]) -> Dict[str, object]:
    if not diagnostic_buffer:
        return {}
    merged: Dict[str, object] = {}
    for key in (
        "branch_entropy_mean",
        "branch_entropy_min",
        "ensemble_entropy",
        "pairwise_disagreement",
    ):
        values = [float(item[key]) for item in diagnostic_buffer if key in item]
        if values:
            merged[key] = sum(values) / len(values)
    for key in ("branch_pred_hist", "ensemble_pred_hist"):
        if key in diagnostic_buffer[0]:
            merged[key] = diagnostic_buffer[0][key]
    return merged


def _branch_parallel_pro_loss(
    model: BranchParallelLoRAEnsemble,
    batch: Dict[str, torch.Tensor],
    num_branches: int,
    num_training_samples: int,
    prior_lambda: float,
    world_size: int,
    objective_mode: str = "pro_only",
    beta_ce: float = 0.0,
    compute_diagnostics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, object]]:
    objective_mode = OBJECTIVE_MODE_ALIASES.get(objective_mode, objective_mode)
    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(f"Unsupported objective_mode={objective_mode!r}")
    if objective_mode == "pro_ce_convex" and not (0.0 <= float(beta_ce) <= 1.0):
        raise ValueError(
            f"pro_ce_convex requires 0 <= beta_ce <= 1, got beta_ce={beta_ce!r}"
        )

    gold = batch["gold_choice_indices"]
    valid_count = torch.tensor(gold.numel(), device=gold.device).clamp_min(1)
    local_choice_log_probs = model.local_choice_log_probs(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        choice_token_ids=batch["choice_token_ids"],
        choice_mask=batch["choice_mask"],
    )
    local_logps = local_choice_log_probs.gather(
        dim=2,
        index=gold.view(1, -1, 1).expand(local_choice_log_probs.size(0), -1, 1),
    ).squeeze(-1)
    branch_ce = -local_logps.sum() / (num_branches * valid_count)

    with torch.no_grad():
        all_logps = _gather_branch_logps(local_logps, world_size)
        global_logsumexp = torch.logsumexp(all_logps, dim=0)
        log_avg_prob = global_logsumexp - math.log(num_branches)
        pro_loss_value = -log_avg_prob.mean()
        local_weights = torch.exp(local_logps.detach() - global_logsumexp.unsqueeze(0))

    pro_data_loss = -(
        local_weights * local_logps
    ).sum() / valid_count
    if objective_mode == "pro_only":
        data_loss = pro_data_loss
    elif objective_mode == "pro_ce_anchor":
        data_loss = pro_data_loss + beta_ce * branch_ce
    elif objective_mode == "pro_ce_convex":
        # L_beta = (1 - beta) * L_PrO + beta * L_branchCE ; prior added below,
        # OUTSIDE the mixture. Both terms are per-example / per-branch means.
        data_loss = (1.0 - beta_ce) * pro_data_loss + beta_ce * branch_ce
    else:
        data_loss = branch_ce

    local_prior = model.local_prior_sum()
    scaled_local_prior = (0.5 * prior_lambda / (num_branches * num_training_samples)) * local_prior
    total_loss = data_loss + scaled_local_prior

    # Fail-fast on a non-finite loss for the convex fork; does not alter the
    # finite-path value/gradient, and leaves the existing modes untouched.
    if objective_mode == "pro_ce_convex" and not torch.isfinite(total_loss):
        raise RuntimeError(
            f"Non-finite pro_ce_convex loss: beta_ce={beta_ce}, "
            f"data_loss={float(data_loss.detach())}, "
            f"prior={float(scaled_local_prior.detach())}"
        )

    diagnostics = {}
    if compute_diagnostics:
        with torch.no_grad():
            all_choice_log_probs = _gather_branch_logps(local_choice_log_probs, world_size)
            diagnostics = _choice_prediction_diagnostics(
                all_choice_log_probs, batch["choice_mask"]
            )

    with torch.no_grad():
        branch_ce_sum = -local_logps.detach().sum().float()
        branch_ce_count = torch.tensor(
            local_logps.numel(), device=gold.device, dtype=torch.float32
        )
        global_branch_ce = _all_reduce_sum(branch_ce_sum.clone()) / _all_reduce_sum(
            branch_ce_count.clone()
        ).clamp_min(1.0)

        if objective_mode == "pro_only":
            data_loss_value = pro_loss_value
        elif objective_mode == "pro_ce_anchor":
            data_loss_value = pro_loss_value + beta_ce * global_branch_ce
        elif objective_mode == "pro_ce_convex":
            data_loss_value = (
                (1.0 - beta_ce) * pro_loss_value + beta_ce * global_branch_ce
            )
        else:
            data_loss_value = global_branch_ce

        global_prior = _all_reduce_sum(local_prior.detach().clone())
        scaled_global_prior = (
            0.5 * prior_lambda / (num_branches * num_training_samples)
        ) * global_prior
        total_loss_value = data_loss_value + scaled_global_prior
        prior_data_ratio = scaled_global_prior / data_loss_value.detach().abs().clamp_min(1e-12)
        metrics = {
            "pro_loss": float(pro_loss_value.detach().cpu()),
            "branch_ce": float(global_branch_ce.detach().cpu()),
            "data_loss": float(data_loss_value.detach().cpu()),
            "prior": float(scaled_global_prior.detach().cpu()),
            "total_loss": float(total_loss_value.detach().cpu()),
            "prior_data_ratio": float(prior_data_ratio.detach().cpu()),
            "example_count": float(valid_count.detach().cpu()),
        }
    return total_loss, metrics, diagnostics


@torch.no_grad()
def _evaluate(
    model: BranchParallelLoRAEnsemble,
    val_loader: DataLoader,
    device: torch.device,
    num_branches: int,
    world_size: int,
    max_eval_batches: int,
    diagnostic_batches: int = 0,
) -> Dict[str, object]:
    model.eval()
    nll_sum = torch.tensor(0.0, device=device)
    correct_sum = torch.tensor(0.0, device=device)
    confidence_sum = torch.tensor(0.0, device=device)
    example_count = torch.tensor(0.0, device=device)
    diagnostic_buffer = []

    for batch_idx, batch in enumerate(val_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        batch = _move_batch_to_device(batch, device)
        gold = batch["gold_choice_indices"]
        local_choice_log_probs = model.local_choice_log_probs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            choice_token_ids=batch["choice_token_ids"],
            choice_mask=batch["choice_mask"],
        )
        local_logps = local_choice_log_probs.gather(
            dim=2,
            index=gold.view(1, -1, 1).expand(local_choice_log_probs.size(0), -1, 1),
        )
        local_logps = local_logps.squeeze(-1)
        all_logps = _gather_branch_logps(local_logps, world_size)
        log_avg_prob = torch.logsumexp(all_logps, dim=0) - math.log(num_branches)
        nll_sum += -log_avg_prob.sum()

        all_choice_log_probs = _gather_branch_logps(local_choice_log_probs, world_size)
        ensemble_probs = all_choice_log_probs.exp().mean(dim=0)
        ensemble_probs = ensemble_probs.masked_fill(~batch["choice_mask"], 0.0)
        if diagnostic_batches > 0 and batch_idx < diagnostic_batches:
            diagnostic_buffer.append(
                _choice_prediction_diagnostics(all_choice_log_probs, batch["choice_mask"])
            )

        confidence, pred = ensemble_probs.max(dim=1)
        correct_sum += pred.eq(gold).float().sum()
        confidence_sum += confidence.sum()
        example_count += gold.numel()

    model.train()
    total_nll = nll_sum
    total_correct = correct_sum
    total_confidence = confidence_sum
    total_examples = example_count.clamp_min(1.0)
    metrics: Dict[str, object] = {
        "eval_loss": float((total_nll / total_examples).detach().cpu()),
        "eval_prob_accuracy": float((total_correct / total_examples).detach().cpu()),
        "eval_mean_confidence": float((total_confidence / total_examples).detach().cpu()),
        "eval_examples": float(total_examples.detach().cpu()),
    }
    diagnostics = _merge_diagnostic_buffer(diagnostic_buffer)
    metrics.update({f"diag_{key}": value for key, value in diagnostics.items()})
    return metrics


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _save_checkpoint(
    model: BranchParallelLoRAEnsemble,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: str,
    step: int,
    rank: int,
    world_size: int,
    metadata: Dict,
) -> None:
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    model.save_local_branches(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
        },
        os.path.join(checkpoint_dir, f"rank_{rank}_optim.pt"),
    )
    _barrier()
    if rank == 0:
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as handle:
            json.dump({**metadata, "global_step": step, "world_size": world_size}, handle, indent=2)
        _verify_saved_branches(checkpoint_dir, metadata["num_branches"])
    _barrier()


def _verify_saved_branches(output_dir: str, num_branches: int, min_bytes: int = 10_000_000) -> None:
    for branch_id in range(num_branches):
        branch_dir = os.path.join(output_dir, f"branch_{branch_id}")
        safe_path = os.path.join(branch_dir, "adapter_model.safetensors")
        bin_path = os.path.join(branch_dir, "lora_params.bin")

        if os.path.exists(safe_path):
            if os.path.getsize(safe_path) <= min_bytes:
                raise RuntimeError(f"Saved adapter is too small: {safe_path}")
            if safe_load_file is None:
                raise RuntimeError("safetensors is unavailable, cannot verify adapter_model.safetensors")
            state = safe_load_file(safe_path, device="cpu")
        else:
            if not os.path.exists(bin_path):
                raise RuntimeError(f"Missing saved branch parameters: {branch_dir}")
            if os.path.getsize(bin_path) <= min_bytes:
                raise RuntimeError(f"Saved branch is too small: {bin_path}")
            state = torch.load(bin_path, map_location="cpu")

        has_lora_a = any("lora_A" in key and torch.is_tensor(value) for key, value in state.items())
        has_lora_b = any("lora_B" in key and torch.is_tensor(value) for key, value in state.items())
        if not (has_lora_a and has_lora_b):
            raise RuntimeError(f"Saved branch lacks readable LoRA A/B tensors: {branch_dir}")
    print(f"[save-check] Verified {num_branches} branches under {output_dir}", flush=True)


def _format_alpaca(data_point: Dict) -> str:
    if data_point.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{data_point['instruction']}\n\n"
            f"### Input:\n{data_point['input']}\n\n"
            f"### Response:\n{data_point['output']}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{data_point['instruction']}\n\n"
        f"### Response:\n{data_point['output']}"
    )


def _candidate_labels(instruction: str) -> List[str]:
    match = re.search(r"Answer format:\s*([^\n]+)", instruction or "", flags=re.IGNORECASE)
    if match:
        labels = re.findall(
            r"true|false|solution\d+|answer\d+|ending\d+|option\d+",
            match.group(1).lower(),
        )
        if labels:
            return labels
    raise ValueError(f"Cannot infer answer labels from instruction: {instruction[:200]}")


def _canonicalise_labels(source_labels: List[str]) -> Dict[str, str]:
    if len(source_labels) > len(CANONICAL_LABELS):
        raise ValueError(f"Too many labels for canonical mapping: {source_labels}")
    return {
        source_label: CANONICAL_LABELS[idx]
        for idx, source_label in enumerate(source_labels)
    }


def _label_token_instruction(instruction: str, source_labels: List[str]) -> str:
    label_map = _canonicalise_labels(source_labels)
    mapping = ", ".join(
        f"{letter} = {source_label}"
        for source_label, letter in label_map.items()
    )
    return (
        f"{instruction}\n\n"
        f"Use the following option-letter mapping: {mapping}.\n"
        "Answer with the single option letter only."
    )


def _format_label_prompt(data_point: Dict) -> str:
    source_labels = _candidate_labels(data_point.get("instruction", ""))
    instruction = _label_token_instruction(data_point.get("instruction", ""), source_labels)
    if data_point.get("input"):
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{data_point['input']}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


class ClassLabelCollator:
    def __init__(self, tokenizer, cutoff_len: int):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must be configured with a pad token before collation")
        self.pad_id = int(tokenizer.pad_token_id)
        self.label_token_ids = dict(QWEN3_LABEL_TOKEN_IDS)

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        rows = []
        max_choices = 0
        for example in examples:
            source_labels = _candidate_labels(example.get("instruction", ""))
            label_map = _canonicalise_labels(source_labels)
            gold_source = str(example.get("answer", "")).lower()
            if gold_source not in label_map:
                raise ValueError(f"Gold answer {gold_source!r} not in labels {source_labels}")

            prompt = _format_label_prompt(example)
            input_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            if len(input_ids) > self.cutoff_len:
                input_ids = input_ids[-self.cutoff_len:]
            choice_labels = [label_map[source_label] for source_label in source_labels]
            choice_token_ids = [self.label_token_ids[label] for label in choice_labels]
            gold_choice_index = choice_labels.index(label_map[gold_source])
            rows.append({
                "input_ids": input_ids,
                "choice_token_ids": choice_token_ids,
                "gold_choice_index": gold_choice_index,
            })
            max_choices = max(max_choices, len(choice_token_ids))

        max_len = max(len(row["input_ids"]) for row in rows)
        if max_len % 8:
            max_len = ((max_len // 8) + 1) * 8

        input_ids = []
        attention_mask = []
        choice_token_ids = []
        choice_mask = []
        gold_choice_indices = []
        for row in rows:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append([self.pad_id] * pad_len + row["input_ids"])
            attention_mask.append([0] * pad_len + [1] * len(row["input_ids"]))
            choice_pad = max_choices - len(row["choice_token_ids"])
            choice_token_ids.append(row["choice_token_ids"] + [0] * choice_pad)
            choice_mask.append([True] * len(row["choice_token_ids"]) + [False] * choice_pad)
            gold_choice_indices.append(row["gold_choice_index"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "choice_token_ids": torch.tensor(choice_token_ids, dtype=torch.long),
            "choice_mask": torch.tensor(choice_mask, dtype=torch.bool),
            "gold_choice_indices": torch.tensor(gold_choice_indices, dtype=torch.long),
        }


def _build_datasets(
    tokenizer,
    data_path: str,
    cutoff_len: int,
    val_set_size: int,
    train_on_inputs: bool,
    max_train_samples: int,
    seed: int,
):
    if data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    if val_set_size > 0:
        split = dataset["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_raw = split["train"].shuffle(seed=seed)
        if max_train_samples > 0:
            train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
        train_data = train_raw
        val_data = split["test"].shuffle(seed=seed)
    else:
        train_raw = dataset["train"].shuffle(seed=seed)
        if max_train_samples > 0:
            train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
        train_data = train_raw
        val_data = None
    return train_data, val_data


def _make_loader(
    dataset,
    tokenizer,
    micro_batch_size: int,
    cutoff_len: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    collator = ClassLabelCollator(tokenizer, cutoff_len=cutoff_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator if shuffle else None,
        pin_memory=True,
        drop_last=False,
    )


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train(
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "./lora-ensemble-pro-branch-parallel",
    batch_size: int = 8,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 200,
    eval_step: int = 250,
    save_step: int = 250,
    logging_steps: int = 10,
    warmup_steps: int = 100,
    max_steps: int = 1000,
    max_train_samples: int = -1,
    max_eval_batches: int = -1,
    num_branches: int = 8,
    branch_seeds=None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules=None,
    train_on_inputs: bool = True,
    warm_start_dir: str = "",
    weight_decay: float = 0.01,
    prior_lambda: Optional[float] = None,
    temperature_mode: str = "auto",
    manual_temperature: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    seed: int = 42,
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    run_notes: str = "",
    experiment_family: str = "",
    endpoint_reuse: str = "",
    objective_mode: str = "pro_only",
    beta_ce: float = 0.0,
    diagnostic_batches: int = 1,
    model_dtype: str = "fp16",
    model_revision: str = QWEN3_MODEL_REVISION,
    attn_implementation: str = "sdpa",
):
    assert base_model, "Please specify base_model"
    assert os.path.exists(data_path), f"Data path not found: {data_path}"
    raw_objective_mode = str(objective_mode).strip()
    objective_mode = OBJECTIVE_MODE_ALIASES.get(raw_objective_mode, raw_objective_mode)
    if objective_mode not in OBJECTIVE_MODES:
        valid_modes = sorted(set(OBJECTIVE_MODES) | set(OBJECTIVE_MODE_ALIASES))
        raise ValueError(f"objective_mode must be one of {valid_modes}")
    beta_ce = float(beta_ce)
    if objective_mode == "pro_ce_convex" and not (0.0 <= beta_ce <= 1.0):
        raise ValueError(
            f"pro_ce_convex requires 0 <= beta_ce <= 1, got beta_ce={beta_ce}"
        )
    if base_model != QWEN3_BASE_MODEL:
        raise ValueError(f"This trainer is pinned to {QWEN3_BASE_MODEL}, got {base_model!r}")
    if model_revision != QWEN3_MODEL_REVISION:
        raise ValueError(
            f"This trainer is pinned to revision {QWEN3_MODEL_REVISION}, got {model_revision!r}"
        )
    if attn_implementation != "sdpa":
        raise ValueError("Qwen3 smoke contract requires attn_implementation='sdpa'")
    if prior_lambda is None:
        prior_lambda = weight_decay
    prior_lambda = float(prior_lambda)
    diagnostic_batches = max(0, int(diagnostic_batches))
    if model_dtype == "fp16":
        torch_model_dtype = torch.float16
    elif model_dtype == "bf16":
        torch_model_dtype = torch.bfloat16
    else:
        raise ValueError("model_dtype must be 'fp16' or 'bf16'")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rank, world_size, local_rank, device = _init_distributed()

    try:
        if num_branches % world_size != 0:
            raise ValueError(
                f"num_branches={num_branches} must be divisible by world_size={world_size}"
            )

        branch_seeds = _parse_list(branch_seeds, DEFAULT_BRANCH_SEEDS, int)[:num_branches]
        target_modules = _parse_list(target_modules, DEFAULT_TARGET_MODULES, str)
        if len(branch_seeds) != num_branches:
            raise ValueError("branch_seeds length must match num_branches")

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.makedirs(output_dir, exist_ok=True)
        branches_per_rank = num_branches // world_size
        start_branch = rank * branches_per_rank
        local_branch_ids = list(range(start_branch, start_branch + branches_per_rank))

        _rank_print(rank, "=" * 80)
        _rank_print(rank, "Branch-parallel class-label PRO/MFLD training")
        _rank_print(rank, f"world_size={world_size}, branches_per_rank={branches_per_rank}")
        _rank_print(rank, f"output_dir={output_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            revision=model_revision,
            trust_remote_code=False,
        )
        tokenizer_contract = _configure_qwen3_tokenizer(tokenizer)

        # Rank 0 builds the cache first to avoid all ranks racing on the same cache files.
        if rank == 0:
            train_data, val_data = _build_datasets(
                tokenizer, data_path, cutoff_len, val_set_size,
                train_on_inputs, max_train_samples, seed
            )
        _barrier()
        if rank != 0:
            train_data, val_data = _build_datasets(
                tokenizer, data_path, cutoff_len, val_set_size,
                train_on_inputs, max_train_samples, seed
            )
        _barrier()

        num_training_samples = len(train_data)
        if max_steps <= 0:
            updates_per_epoch = math.ceil(
                math.ceil(num_training_samples / micro_batch_size)
                / max(1, batch_size // micro_batch_size)
            )
            total_steps = updates_per_epoch * num_epochs
        else:
            total_steps = max_steps

        if temperature_mode == "auto":
            temperature = 1.0 / (num_branches * num_training_samples)
        elif temperature_mode == "zero":
            temperature = 0.0
        else:
            temperature = manual_temperature

        train_loader = _make_loader(train_data, tokenizer, micro_batch_size, cutoff_len, True, seed)
        val_loader = (
            _make_loader(val_data, tokenizer, micro_batch_size, cutoff_len, False, seed)
            if val_data is not None
            else None
        )
        grad_acc = max(1, batch_size // micro_batch_size)

        _rank_print(rank, f"train_samples={num_training_samples}, val_samples={len(val_data) if val_data else 0}")
        _rank_print(rank, f"batch_size={batch_size}, micro_batch_size={micro_batch_size}, grad_acc={grad_acc}")
        _rank_print(rank, f"max_steps={total_steps}, lr={learning_rate}, temperature={temperature:.6e}")
        _rank_print(rank, f"objective_mode={objective_mode}, beta_ce={beta_ce:.6g}, prior_lambda={prior_lambda:.6g}, diagnostic_batches={diagnostic_batches}")
        _rank_print(rank, f"local branches per rank are contiguous: rank0 -> [0, 1], rank1 -> [2, 3], etc.")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=model_revision,
            torch_dtype=torch_model_dtype,
            device_map=None,
            trust_remote_code=False,
            attn_implementation=attn_implementation,
        ).to(device)
        base.config.use_cache = False
        backbone_contract = _validate_qwen3_backbone(base, target_modules)

        model = BranchParallelLoRAEnsemble(
            base_model=base,
            local_branch_ids=local_branch_ids,
            branch_seeds=branch_seeds,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(device)
        if warm_start_dir:
            model.load_local_branches(warm_start_dir)
            _barrier()
            _rank_print(rank, f"warm_start_dir={warm_start_dir}")
        model.train()

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = NoisyAdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.0,
            temperature=temperature,
            no_deprecation_warning=True,
        )
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        metadata = {
            "base_model": base_model,
            "model_revision": model_revision,
            "attn_implementation": attn_implementation,
            **backbone_contract,
            **tokenizer_contract,
            "data_path": data_path,
            "num_branches": num_branches,
            "branch_seeds": branch_seeds,
            "world_size": world_size,
            "branches_per_rank": branches_per_rank,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "warm_start_dir": warm_start_dir,
            "loss": "class-label probability loss controlled by objective_mode + scaled Gaussian prior",
            "objective_mode": objective_mode,
            "beta_ce": beta_ce,
            "diagnostic_batches": diagnostic_batches,
            "scoring_mode": "label_token",
            "prior_lambda": prior_lambda,
            "legacy_weight_decay_arg": weight_decay,
            "temperature": temperature,
            "num_training_samples": num_training_samples,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": grad_acc,
            "cutoff_len": cutoff_len,
            "notes": run_notes,
            "family": experiment_family,
            "endpoint_reuse": endpoint_reuse,
            "job_kind": "smoke" if max_steps > 0 else "train",
            "num_epochs": num_epochs,
            "expected_steps": total_steps,
            "save_step": save_step,
            "eval_step": eval_step,
            "max_steps": max_steps,
        }
        if rank == 0:
            with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

        run = None
        if rank == 0 and wandb_project and wandb is not None:
            run = wandb.init(
                entity=wandb_entity or None,
                project=wandb_project,
                name=wandb_run_name or None,
                group=os.environ.get("WANDB_RUN_GROUP") or None,
                job_type=metadata["job_kind"],
                config=metadata,
                tags=_resolve_wandb_tags(
                    os.environ.get("WANDB_TAGS"),
                    base_model,
                    objective_mode,
                    beta_ce,
                ),
            )
            if run is None or getattr(run, "mode", "online") != "online":
                raise RuntimeError("W&B online initialization guard failed")
            print(
                f"WANDB_ONLINE_INITIALIZED run_id={getattr(run, 'id', 'unknown')}",
                flush=True,
            )

        train_iter = iter(train_loader)
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, total_steps + 1):
            metric_buffer = []
            diagnostic_buffer = []
            should_log_step = step == 1 or step % logging_steps == 0
            for micro_idx in range(grad_acc):
                batch, train_iter = _next_batch(train_iter, train_loader)
                batch = _move_batch_to_device(batch, device)
                compute_diagnostics = (
                    diagnostic_batches > 0
                    and should_log_step
                    and micro_idx < diagnostic_batches
                )
                loss, metrics, diagnostics = _branch_parallel_pro_loss(
                    model=model,
                    batch=batch,
                    num_branches=num_branches,
                    num_training_samples=num_training_samples,
                    prior_lambda=prior_lambda,
                    world_size=world_size,
                    objective_mode=objective_mode,
                    beta_ce=beta_ce,
                    compute_diagnostics=compute_diagnostics,
                )
                (loss / grad_acc).backward()
                metric_buffer.append(metrics)
                if diagnostics:
                    diagnostic_buffer.append(diagnostics)

            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and (step == 1 or step % logging_steps == 0):
                averaged = {
                    key: sum(m[key] for m in metric_buffer) / len(metric_buffer)
                    for key in metric_buffer[0]
                }
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                diagnostics = _merge_diagnostic_buffer(diagnostic_buffer)
                diag_text = ""
                if diagnostics:
                    diag_text = (
                        f" branch_ent={diagnostics['branch_entropy_mean']:.4f}/"
                        f"{diagnostics['branch_entropy_min']:.4f} "
                        f"ens_ent={diagnostics['ensemble_entropy']:.4f} "
                        f"disagree={diagnostics['pairwise_disagreement']:.4f} "
                        f"branch_hist={diagnostics['branch_pred_hist']} "
                        f"ens_hist={diagnostics['ensemble_pred_hist']}"
                    )
                print(
                    f"[step {step}/{total_steps}] "
                    f"loss={averaged['total_loss']:.4f} "
                    f"pro={averaged['pro_loss']:.4f} "
                    f"ce={averaged['branch_ce']:.4f} "
                    f"data={averaged['data_loss']:.4f} "
                    f"prior={averaged['prior']:.6f} "
                    f"prior/data={averaged['prior_data_ratio']:.6f} "
                    f"lr={lr:.3e} "
                    f"elapsed={elapsed/60:.1f}m"
                    f"{diag_text}",
                    flush=True,
                )
                if run is not None:
                    log_payload = {
                        "train/loss": averaged["total_loss"],
                        "train/pro_loss": averaged["pro_loss"],
                        "train/branch_ce": averaged["branch_ce"],
                        "train/data_loss": averaged["data_loss"],
                        "train/prior": averaged["prior"],
                        "train/prior_data_ratio": averaged["prior_data_ratio"],
                        "train/lr": lr,
                        "train/step": step,
                    }
                    log_payload.update({
                        f"train/diag_{key}": value
                        for key, value in diagnostics.items()
                    })
                    run.log(log_payload, step=step)

            if val_loader is not None and eval_step > 0 and step % eval_step == 0:
                eval_metrics = _evaluate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    num_branches=num_branches,
                    world_size=world_size,
                    max_eval_batches=max_eval_batches,
                    diagnostic_batches=diagnostic_batches,
                )
                if rank == 0:
                    print(f"[eval step {step}] {eval_metrics}", flush=True)
                    if run is not None:
                        run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

            if save_step > 0 and step % save_step == 0:
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=output_dir,
                    step=step,
                    rank=rank,
                    world_size=world_size,
                    metadata=metadata,
                )

        model.save_local_branches(output_dir)
        _barrier()
        if rank == 0:
            _verify_saved_branches(output_dir, num_branches)
            with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as handle:
                json.dump({**metadata, "global_step": total_steps, "finished": True}, handle, indent=2)
            print(f"Training complete. Models saved to {output_dir}", flush=True)
            if run is not None:
                run.finish()
        _barrier()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(train)
