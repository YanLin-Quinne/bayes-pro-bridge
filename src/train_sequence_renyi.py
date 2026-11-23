# Branch-parallel true PRO/MFLD LoRA ensemble training.
#
# This script is intentionally separate from finetune_joint_ensemble_PRO_our.py.
# It keeps the proven single-GPU path untouched, while allowing 8 LoRA branches
# to be split over 4 GPUs as 2 branches per rank.
#
# Objective:
#   data_loss = - mean_i (1/alpha) log( (1/N) * sum_j p_j(answer_i | prompt_i)^alpha )
#   where p_j(answer_i | prompt_i) is the full answer-string likelihood.
#   alpha=1 gives sequence-level PrO; alpha=0 gives the Bayes/CE endpoint.
#   prior     = (0.5 * lambda / (N * M)) * sum_j ||theta_j||^2
#   T         = 1 / (N * M) for NoisyAdamW when temperature_mode="auto"

import json
import math
import os
import random
import shutil
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
BRANCH_ARTIFACT_FILES = ("adapter_model.safetensors", "lora_params.bin", "config.json", "config.bin")


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
    ):
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.branch_id = branch_id
        self.seed = seed
        self.lora_params = nn.ModuleDict()

        layer_idx = 0
        for name, module in base_model.named_modules():
            if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
                layer_seed = seed * 100_000 + layer_idx
                self.lora_params[name.replace(".", "_")] = LoRAParameterGroup(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    init_seed=layer_seed,
                )
                layer_idx += 1

        self.name_mapping = {
            name.replace(".", "_"): name
            for name, module in base_model.named_modules()
            if any(target in name for target in target_modules) and isinstance(module, nn.Linear)
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
            first_lora = next(iter(branch.lora_params.values()))
            metadata = {
                "branch_id": branch_id,
                "seed": branch.seed,
                "rank": int(first_lora.rank),
                "alpha": int(first_lora.alpha),
                "target_modules": branch.target_modules,
                "num_lora_layers": len(branch.lora_params),
                "state_format": "custom_lora_hook",
            }
            with open(os.path.join(branch_dir, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            torch.save(metadata, os.path.join(branch_dir, "config.bin"))


def _true_token_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    safe_labels = shift_labels.masked_fill(shift_labels.eq(-100), 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    return log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def _sequence_logps(token_logps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_mask = valid.unsqueeze(0)
    return token_logps.float().masked_fill(~valid_mask, 0.0).sum(dim=-1)


def _gather_branch_logps(local_logps: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_logps.detach()
    gathered = [torch.empty_like(local_logps) for _ in range(world_size)]
    dist.all_gather(gathered, local_logps.detach())
    return torch.cat(gathered, dim=0)


def _branch_parallel_pro_loss(
    model: BranchParallelLoRAEnsemble,
    batch: Dict[str, torch.Tensor],
    num_branches: int,
    num_training_samples: int,
    weight_decay: float,
    sequence_alpha: float,
    world_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    labels = batch["labels"]
    valid = labels[..., 1:].ne(-100)
    sample_valid = valid.any(dim=-1)
    valid_count = sample_valid.sum().clamp_min(1)

    local_token_logps = model.local_token_logps(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=labels,
    )
    local_seq_logps = _sequence_logps(local_token_logps, valid)

    with torch.no_grad():
        all_seq_logps = _gather_branch_logps(local_seq_logps, world_size)
        branch_ce_per_sample = -all_seq_logps.mean(dim=0)
        branch_ce_value = branch_ce_per_sample[sample_valid].mean()

        if sequence_alpha <= 0:
            global_logsumexp = None
            data_loss_value = branch_ce_value
            local_weights = torch.full_like(local_seq_logps, 1.0 / num_branches)
            all_weights = torch.full_like(all_seq_logps, 1.0 / num_branches)
        else:
            alpha_seq_logps = sequence_alpha * all_seq_logps.float()
            global_logsumexp = torch.logsumexp(alpha_seq_logps, dim=0)
            log_avg_power = global_logsumexp - math.log(num_branches)
            data_loss_value = -(log_avg_power[sample_valid] / sequence_alpha).mean()
            local_weights = torch.exp(
                sequence_alpha * local_seq_logps.detach().float()
                - global_logsumexp.unsqueeze(0)
            )
            all_weights = torch.exp(alpha_seq_logps - global_logsumexp.unsqueeze(0))

        valid_weights = all_weights[:, sample_valid]
        if valid_weights.numel() > 0:
            w_sq_sum = valid_weights.square().sum(dim=0)
            finite_w_sq = torch.isfinite(w_sq_sum) & (w_sq_sum > 0)
            ess = torch.full_like(w_sq_sum, float("nan"))
            ess[finite_w_sq] = 1.0 / w_sq_sum[finite_w_sq]
            max_weight = valid_weights.max(dim=0).values
            valid_logps = all_seq_logps[:, sample_valid].float()
            logp_spread = valid_logps.max(dim=0).values - valid_logps.min(dim=0).values
            logp_std = valid_logps.std(dim=0, unbiased=False)
        else:
            ess = torch.ones(1, device=all_seq_logps.device) * num_branches
            max_weight = torch.zeros(1, device=all_seq_logps.device)
            logp_spread = torch.zeros(1, device=all_seq_logps.device)
            logp_std = torch.zeros(1, device=all_seq_logps.device)

    if local_weights.requires_grad:
        raise RuntimeError("sequence PrO detached weights must not require gradients")

    data_loss = -(
        local_weights * local_seq_logps.float() * sample_valid.unsqueeze(0)
    ).sum() / valid_count

    uniform_branch_ce_sanity_abs_diff = torch.zeros(
        (), device=local_seq_logps.device, dtype=torch.float32
    )
    if sequence_alpha <= 0:
        with torch.no_grad():
            global_surrogate_value = _all_reduce_sum(data_loss.detach().float().clone())
            uniform_branch_ce_sanity_abs_diff = (
                global_surrogate_value - data_loss_value.detach().float()
            ).abs()
            rel = uniform_branch_ce_sanity_abs_diff / data_loss_value.detach().abs().clamp_min(1e-8)
            if (
                torch.isfinite(uniform_branch_ce_sanity_abs_diff)
                and torch.isfinite(rel)
                and uniform_branch_ce_sanity_abs_diff > 1e-5
                and rel > 1e-5
            ):
                raise RuntimeError(
                    "sequence_alpha<=0 CE sanity check failed: "
                    f"global_surrogate={float(global_surrogate_value.detach().cpu()):.8f}, "
                    f"branch_ce={float(data_loss_value.detach().cpu()):.8f}, "
                    f"abs_diff={float(uniform_branch_ce_sanity_abs_diff.detach().cpu()):.8e}"
                )

    local_prior = model.local_prior_sum()
    scaled_local_prior = (0.5 * weight_decay / (num_branches * num_training_samples)) * local_prior
    total_loss = data_loss + scaled_local_prior

    with torch.no_grad():
        global_prior = _all_reduce_sum(local_prior.detach().clone())
        scaled_global_prior = (
            0.5 * weight_decay / (num_branches * num_training_samples)
        ) * global_prior
        target_token_count = valid.sum().clamp_min(1)
        invalid_sequence_count = (~sample_valid).sum()
        jensen_gap = branch_ce_value - data_loss_value
        if torch.isfinite(jensen_gap) and jensen_gap < -1e-6:
            raise RuntimeError(
                "sequence Jensen gap should be non-negative: "
                f"branch_ce={float(branch_ce_value.detach().cpu()):.8f}, "
                f"data_loss={float(data_loss_value.detach().cpu()):.8f}, "
                f"gap={float(jensen_gap.detach().cpu()):.8e}"
            )
        metrics = {
            "data_loss": float(data_loss_value.detach().cpu()),
            "branch_ce": float(branch_ce_value.detach().cpu()),
            "jensen_gap": float(jensen_gap.detach().cpu()),
            "uniform_branch_ce_sanity_abs_diff": float(
                uniform_branch_ce_sanity_abs_diff.detach().cpu()
            ),
            "prior": float(scaled_global_prior.detach().cpu()),
            "total_loss": float((data_loss_value + scaled_global_prior).detach().cpu()),
            "sequence_count": float(valid_count.detach().cpu()),
            "target_token_count": float(target_token_count.detach().cpu()),
            "invalid_sequence_count": float(invalid_sequence_count.detach().cpu()),
            "ess_mean": float(ess.mean().detach().cpu()),
            "ess_median": float(ess.median().detach().cpu()),
            "ess_p10": float(torch.quantile(ess, 0.10).detach().cpu()),
            "ess_min": float(ess.min().detach().cpu()),
            "ess_normalized_mean": float((ess / num_branches).mean().detach().cpu()),
            "frac_ess_le_1_5": float((ess <= 1.5).float().mean().detach().cpu()),
            "max_weight_mean": float(max_weight.mean().detach().cpu()),
            "frac_max_weight_ge_0_8": float((max_weight >= 0.8).float().mean().detach().cpu()),
            "frac_max_weight_ge_0_95": float((max_weight >= 0.95).float().mean().detach().cpu()),
            "logp_spread_mean": float(logp_spread.mean().detach().cpu()),
            "logp_spread_max": float(logp_spread.max().detach().cpu()),
            "logp_std_mean": float(logp_std.mean().detach().cpu()),
        }
    return total_loss, metrics

@torch.no_grad()
def _evaluate(
    model: BranchParallelLoRAEnsemble,
    val_loader: DataLoader,
    device: torch.device,
    num_branches: int,
    sequence_alpha: float,
    world_size: int,
    max_eval_batches: int,
) -> Dict[str, float]:
    model.eval()
    nll_sum = torch.tensor(0.0, device=device)
    sequence_count = torch.tensor(0.0, device=device)
    target_token_count = torch.tensor(0.0, device=device)

    for batch_idx, batch in enumerate(val_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        batch = _move_batch_to_device(batch, device)
        labels = batch["labels"]
        valid = labels[..., 1:].ne(-100)
        sample_valid = valid.any(dim=-1)
        local_token_logps = model.local_token_logps(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        local_seq_logps = _sequence_logps(local_token_logps, valid)
        all_seq_logps = _gather_branch_logps(local_seq_logps, world_size)
        if sequence_alpha <= 0:
            eval_loss = -all_seq_logps.mean(dim=0)
        else:
            log_avg_power = (
                torch.logsumexp(sequence_alpha * all_seq_logps.float(), dim=0)
                - math.log(num_branches)
            )
            eval_loss = -log_avg_power / sequence_alpha
        nll_sum += eval_loss[sample_valid].sum()
        sequence_count += sample_valid.sum()
        target_token_count += valid.sum()

    model.train()
    count = sequence_count.clamp_min(1)
    return {
        "eval_loss": float((nll_sum / count).detach().cpu()),
        "eval_sequences": float(sequence_count.detach().cpu()),
        "eval_target_tokens": float(target_token_count.detach().cpu()),
    }

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



def _write_best_checkpoint_restore_report(
    output_dir: str,
    source: Optional[str],
    best_metric: Optional[float],
    copied_files: List[Dict],
    backed_up_files: List[Dict],
    restored: bool,
    reason: str,
) -> str:
    restore_report_path = os.path.join(output_dir, "best_checkpoint_restore.json")
    with open(restore_report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": source,
                "metric": "eval_loss",
                "best_metric": best_metric,
                "copied_files": copied_files,
                "backed_up_files": backed_up_files,
                "restored": restored,
                "reason": reason,
            },
            handle,
            indent=2,
        )
    return restore_report_path


def _restore_best_checkpoint_branches(
    best_checkpoint: str,
    output_dir: str,
    num_branches: int,
    best_metric: Optional[float],
) -> str:
    best_checkpoint = os.path.abspath(best_checkpoint)
    output_dir = os.path.abspath(output_dir)
    _verify_saved_branches(best_checkpoint, num_branches)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    copied_files = []
    backed_up_files = []

    for branch_id in range(num_branches):
        source_branch = os.path.join(best_checkpoint, f"branch_{branch_id}")
        target_branch = os.path.join(output_dir, f"branch_{branch_id}")
        os.makedirs(target_branch, exist_ok=True)

        for filename in BRANCH_ARTIFACT_FILES:
            source_path = os.path.join(source_branch, filename)
            target_path = os.path.join(target_branch, filename)
            if not os.path.exists(source_path):
                raise RuntimeError(f"Missing best checkpoint artifact: {source_path}")
            if os.path.exists(target_path):
                backup_path = f"{target_path}.final_step_bak_{timestamp}"
                shutil.copy2(target_path, backup_path)
                backed_up_files.append({"path": target_path, "backup_path": backup_path})
            shutil.copy2(source_path, target_path)
            copied_files.append({"source": source_path, "destination": target_path})

    _verify_saved_branches(output_dir, num_branches)
    restore_report_path = _write_best_checkpoint_restore_report(
        output_dir=output_dir,
        source=best_checkpoint,
        best_metric=best_metric,
        copied_files=copied_files,
        backed_up_files=backed_up_files,
        restored=True,
        reason="restored best eval_loss checkpoint",
    )
    print(f"[best-checkpoint] Restored best checkpoint branches from {best_checkpoint}", flush=True)
    return restore_report_path

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


def _build_datasets(
    tokenizer,
    data_path: str,
    cutoff_len: int,
    val_set_size: int,
    train_on_inputs: bool,
    max_train_samples: int,
    seed: int,
):
    def tokenize(prompt: str, add_eos: bool = True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize(data_point):
        full_prompt = _format_alpaca(data_point)
        tokenized_full = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = _format_alpaca({**data_point, "output": ""})
            tokenized_user = tokenize(user_prompt, add_eos=False)
            user_len = len(tokenized_user["input_ids"])
            tokenized_full["labels"] = [-100] * user_len + tokenized_full["labels"][user_len:]
        return tokenized_full

    if data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    raw_columns = list(dataset["train"].column_names)
    if val_set_size > 0:
        split = dataset["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_raw = split["train"].shuffle(seed=seed)
        if max_train_samples > 0:
            train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
        val_raw = split["test"].shuffle(seed=seed)
        train_data = train_raw.map(generate_and_tokenize, remove_columns=raw_columns)
        val_data = val_raw.map(generate_and_tokenize, remove_columns=raw_columns)
    else:
        train_raw = dataset["train"].shuffle(seed=seed)
        if max_train_samples > 0:
            train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
        train_data = train_raw.map(generate_and_tokenize, remove_columns=raw_columns)
        val_data = None
    return train_data, val_data


def _make_loader(dataset, tokenizer, micro_batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )
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
    train_on_inputs: bool = False,
    weight_decay: float = 0.01,
    sequence_alpha: float = 1.0,
    temperature_mode: str = "auto",
    manual_temperature: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    seed: int = 42,
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    run_notes: str = "",
    restore_best_checkpoint: bool = False,
    model_dtype: str = "fp16",
):
    assert base_model, "Please specify base_model"
    assert os.path.exists(data_path), f"Data path not found: {data_path}"

    dtype_key = str(model_dtype).lower()
    if dtype_key in ("bf16", "bfloat16"):
        torch_model_dtype = torch.bfloat16
    elif dtype_key in ("fp16", "float16", "half"):
        torch_model_dtype = torch.float16
    elif dtype_key in ("fp32", "float32"):
        torch_model_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported model_dtype={model_dtype!r}; use fp16, bf16, or fp32")

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
        _rank_print(rank, "Branch-parallel sequence-level PRO/MFLD training")
        _rank_print(rank, f"world_size={world_size}, branches_per_rank={branches_per_rank}")
        _rank_print(rank, f"output_dir={output_dir}")

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

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

        train_loader = _make_loader(train_data, tokenizer, micro_batch_size, True, seed)
        val_loader = (
            _make_loader(val_data, tokenizer, micro_batch_size, False, seed)
            if val_data is not None
            else None
        )
        grad_acc = max(1, batch_size // micro_batch_size)

        _rank_print(rank, f"train_samples={num_training_samples}, val_samples={len(val_data) if val_data else 0}")
        _rank_print(rank, f"batch_size={batch_size}, micro_batch_size={micro_batch_size}, grad_acc={grad_acc}")
        _rank_print(rank, f"max_steps={total_steps}, lr={learning_rate}, temperature={temperature:.6e}")
        _rank_print(rank, f"model_dtype={dtype_key}, train_on_inputs={train_on_inputs}, sequence_alpha={sequence_alpha}")
        _rank_print(rank, f"local branches per rank are contiguous: rank0 -> [0, 1], rank1 -> [2, 3], etc.")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_model_dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(device)
        base.config.use_cache = False

        model = BranchParallelLoRAEnsemble(
            base_model=base,
            local_branch_ids=local_branch_ids,
            branch_seeds=branch_seeds,
            target_modules=target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(device)
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
            "data_path": data_path,
            "num_branches": num_branches,
            "branch_seeds": branch_seeds,
            "world_size": world_size,
            "branches_per_rank": branches_per_rank,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "loss": "-(1/alpha) log mean_j p_j(answer_string|prompt)^alpha + scaled Gaussian prior",
            "sequence_alpha": sequence_alpha,
            "model_dtype": dtype_key,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "num_training_samples": num_training_samples,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": grad_acc,
            "cutoff_len": cutoff_len,
            "notes": run_notes,
            "restore_best_checkpoint": restore_best_checkpoint,
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
                config=metadata,
                tags=[
                    part.strip()
                    for part in os.environ.get(
                        "WANDB_TAGS",
                        "comet,h100,4xh100,llama3.2-1b,joint-pro,prob-mfld",
                    ).split(",")
                    if part.strip()
                ],
            )

        train_iter = iter(train_loader)
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        best_metric = None
        best_model_checkpoint = None

        for step in range(1, total_steps + 1):
            saved_checkpoint_this_step = False
            metric_buffer = []
            for _ in range(grad_acc):
                batch, train_iter = _next_batch(train_iter, train_loader)
                batch = _move_batch_to_device(batch, device)
                loss, metrics = _branch_parallel_pro_loss(
                    model=model,
                    batch=batch,
                    num_branches=num_branches,
                    num_training_samples=num_training_samples,
                    weight_decay=weight_decay,
                    sequence_alpha=sequence_alpha,
                    world_size=world_size,
                )
                (loss / grad_acc).backward()
                metric_buffer.append(metrics)

            pre_clip_grad_norm = None
            clip_ratio = 1.0
            clip_applied = 0.0
            if gradient_clip_norm > 0:
                pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, gradient_clip_norm
                )
                pre_clip_value = float(pre_clip_grad_norm.detach().cpu())
                if math.isfinite(pre_clip_value) and pre_clip_value > 0:
                    clip_ratio = min(1.0, gradient_clip_norm / (pre_clip_value + 1e-12))
                    clip_applied = 1.0 if pre_clip_value > gradient_clip_norm else 0.0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and (step == 1 or step % logging_steps == 0):
                averaged = {
                    key: sum(m[key] for m in metric_buffer) / len(metric_buffer)
                    for key in metric_buffer[0]
                }
                averaged["pre_clip_grad_norm"] = (
                    float(pre_clip_grad_norm.detach().cpu())
                    if pre_clip_grad_norm is not None
                    else 0.0
                )
                averaged["clip_ratio"] = clip_ratio
                averaged["clip_applied"] = clip_applied
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[step {step}/{total_steps}] "
                    f"loss={averaged['total_loss']:.4f} "
                    f"data={averaged['data_loss']:.4f} "
                    f"ce={averaged['branch_ce']:.4f} "
                    f"gap={averaged['jensen_gap']:.4f} "
                    f"cechk={averaged['uniform_branch_ce_sanity_abs_diff']:.2e} "
                    f"prior={averaged['prior']:.6f} "
                    f"ess={averaged['ess_mean']:.2f}/{averaged['ess_min']:.2f} "
                    f"maxw={averaged['max_weight_mean']:.3f} "
                    f"spread={averaged['logp_spread_mean']:.2f} "
                    f"clip={averaged['clip_applied']:.0f}:{averaged['clip_ratio']:.3f} "
                    f"lr={lr:.3e} "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )
                if run is not None:
                    run.log(
                        {
                            "train/loss": averaged["total_loss"],
                            "train/data_loss": averaged["data_loss"],
                            "train/branch_ce": averaged["branch_ce"],
                            "train/jensen_gap": averaged["jensen_gap"],
                            "train/uniform_branch_ce_sanity_abs_diff": averaged[
                                "uniform_branch_ce_sanity_abs_diff"
                            ],
                            "train/prior": averaged["prior"],
                            "train/ess_mean": averaged["ess_mean"],
                            "train/ess_median": averaged["ess_median"],
                            "train/ess_p10": averaged["ess_p10"],
                            "train/ess_min": averaged["ess_min"],
                            "train/ess_normalized_mean": averaged["ess_normalized_mean"],
                            "train/frac_ess_le_1_5": averaged["frac_ess_le_1_5"],
                            "train/max_weight_mean": averaged["max_weight_mean"],
                            "train/frac_max_weight_ge_0_8": averaged["frac_max_weight_ge_0_8"],
                            "train/frac_max_weight_ge_0_95": averaged["frac_max_weight_ge_0_95"],
                            "train/logp_spread_mean": averaged["logp_spread_mean"],
                            "train/logp_spread_max": averaged["logp_spread_max"],
                            "train/logp_std_mean": averaged["logp_std_mean"],
                            "train/invalid_sequence_count": averaged["invalid_sequence_count"],
                            "train/pre_clip_grad_norm": averaged["pre_clip_grad_norm"],
                            "train/clip_ratio": averaged["clip_ratio"],
                            "train/clip_applied": averaged["clip_applied"],
                            "train/lr": lr,
                            "train/step": step,
                        },
                        step=step,
                    )

            if val_loader is not None and eval_step > 0 and step % eval_step == 0:
                eval_metrics = _evaluate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    num_branches=num_branches,
                    sequence_alpha=sequence_alpha,
                    world_size=world_size,
                    max_eval_batches=max_eval_batches,
                )
                eval_loss = eval_metrics.get("eval_loss")
                improved = (
                    eval_loss is not None
                    and math.isfinite(float(eval_loss))
                    and (best_metric is None or float(eval_loss) < best_metric)
                )
                if improved:
                    best_metric = float(eval_loss)
                    best_model_checkpoint = os.path.abspath(os.path.join(output_dir, f"checkpoint-{step}"))
                    metadata["best_metric"] = best_metric
                    metadata["best_model_checkpoint"] = best_model_checkpoint
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
                    saved_checkpoint_this_step = True
                if rank == 0:
                    print(f"[eval step {step}] {eval_metrics}", flush=True)
                    if improved:
                        print(f"[best-checkpoint] step={step} eval_loss={best_metric:.6f}", flush=True)
                    if run is not None:
                        run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

            if save_step > 0 and step % save_step == 0 and not saved_checkpoint_this_step:
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
            if restore_best_checkpoint and best_model_checkpoint and os.path.isdir(best_model_checkpoint):
                _restore_best_checkpoint_branches(best_model_checkpoint, output_dir, num_branches, best_metric)
                _verify_saved_branches(output_dir, num_branches)
            else:
                if not restore_best_checkpoint:
                    restore_reason = "restore_best_checkpoint disabled"
                elif not best_model_checkpoint:
                    restore_reason = "no best_model_checkpoint recorded"
                else:
                    restore_reason = "best_model_checkpoint path missing"
                _write_best_checkpoint_restore_report(
                    output_dir=output_dir,
                    source=best_model_checkpoint,
                    best_metric=best_metric,
                    copied_files=[],
                    backed_up_files=[],
                    restored=False,
                    reason=restore_reason,
                )
            final_state = {
                **metadata,
                "global_step": total_steps,
                "finished": True,
                "best_model_checkpoint": best_model_checkpoint,
                "best_metric": best_metric,
                "restore_best_checkpoint": restore_best_checkpoint,
            }
            with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as handle:
                json.dump(final_state, handle, indent=2)
            print(f"Training complete. Models saved to {output_dir}", flush=True)
            if run is not None:
                run.finish()
        _barrier()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    fire.Fire(train)
