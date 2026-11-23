"""Independent low rank adapters applied to a shared frozen causal LM."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .data import last_valid_token_indices
from .objective import masked_candidate_log_probs


def candidate_logits_from_model(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Read candidate-label logits at each prompt's final valid token."""
    if choice_token_ids.ndim != 2 or choice_token_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("choice_token_ids must have shape [batch, choice]")
    batch_rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    last_indices = last_valid_token_indices(attention_mask)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[batch_rows, last_indices].float()
    candidate_logits = logits.gather(1, choice_token_ids)
    if not torch.isfinite(candidate_logits).all():
        raise ValueError("candidate logits contain non finite values")
    return candidate_logits


def candidate_probabilities_from_model(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_token_ids: torch.Tensor,
    choice_mask: torch.Tensor,
) -> torch.Tensor:
    """Return class-label probabilities normalized over valid candidates."""
    candidate_logits = candidate_logits_from_model(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        choice_token_ids=choice_token_ids,
    )
    return masked_candidate_log_probs(
        candidate_logits.unsqueeze(0),
        choice_mask,
    ).exp().squeeze(0)


class LoRAAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        seed: int,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("LoRA dimensions and rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        lora_a = torch.empty(rank, in_features)
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5), generator=generator)
        self.lora_A = nn.Parameter(lora_a)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_dtype = inputs.dtype
        adapted_inputs = self.dropout(inputs.to(self.lora_A.dtype))
        delta = (adapted_inputs @ self.lora_A.T) @ self.lora_B.T
        return (delta * self.scaling).to(original_dtype)


def resolve_target_linear_specs(
    base_model: nn.Module,
    target_modules: Sequence[str],
) -> tuple[tuple[tuple[str, int, int], ...], dict[str, int]]:
    if not target_modules or len(set(target_modules)) != len(target_modules):
        raise ValueError("target_modules must be nonempty and unique")
    target_set = set(target_modules)
    counts = {name: 0 for name in target_modules}
    specs: list[tuple[str, int, int]] = []
    non_linear: list[str] = []
    for full_name, module in base_model.named_modules():
        leaf_name = full_name.rsplit(".", 1)[-1]
        if leaf_name not in target_set:
            continue
        if not isinstance(module, nn.Linear):
            non_linear.append(full_name)
            continue
        counts[leaf_name] += 1
        specs.append((full_name, int(module.in_features), int(module.out_features)))
    if non_linear:
        raise TypeError(f"target names matched non linear modules: {non_linear[:5]}")
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"missing exact target modules: {missing}")
    return tuple(specs), counts


class LoRABranch(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        target_specs: Sequence[tuple[str, int, int]],
        *,
        branch_id: int,
        seed: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_base_model", base_model)
        self.branch_id = int(branch_id)
        self.seed = int(seed)
        self.target_names = tuple(name for name, _, _ in target_specs)
        self.adapters = nn.ModuleList(
            [
                LoRAAdapter(
                    in_features,
                    out_features,
                    rank,
                    alpha,
                    dropout,
                    seed=self.seed * 100_000 + index,
                )
                for index, (_, in_features, out_features) in enumerate(target_specs)
            ]
        )

    @property
    def base_model(self) -> nn.Module:
        return object.__getattribute__(self, "_base_model")

    def forward(self, **model_inputs):
        modules = dict(self.base_model.named_modules())
        hooks = []

        def hook_for(adapter: LoRAAdapter):
            def apply_delta(_module, inputs, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("target linear module did not return a tensor")
                return output + adapter(inputs[0])

            return apply_delta

        try:
            for name, adapter in zip(self.target_names, self.adapters):
                hooks.append(modules[name].register_forward_hook(hook_for(adapter)))
            return self.base_model(**model_inputs)
        finally:
            for hook in hooks:
                hook.remove()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
        }

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)


class LoRAEnsemble(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        local_branch_ids: Sequence[int],
        branch_seeds: Sequence[int],
        target_modules: Sequence[str],
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not local_branch_ids:
            raise ValueError("at least one local branch is required")
        if any(branch_id < 0 or branch_id >= len(branch_seeds) for branch_id in local_branch_ids):
            raise ValueError("local branch ID is outside branch_seeds")
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.target_specs, self.target_module_counts = resolve_target_linear_specs(
            self.base_model,
            target_modules,
        )
        self.local_branch_ids = tuple(int(value) for value in local_branch_ids)
        self.branches = nn.ModuleList(
            [
                LoRABranch(
                    self.base_model,
                    self.target_specs,
                    branch_id=branch_id,
                    seed=int(branch_seeds[branch_id]),
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                for branch_id in self.local_branch_ids
            ]
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for branch in self.branches:
            yield from branch.parameters()

    def local_candidate_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        choice_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        branch_logits = []
        for branch in self.branches:
            branch_logits.append(
                candidate_logits_from_model(
                    branch,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    choice_token_ids=choice_token_ids,
                )
            )
        stacked = torch.stack(branch_logits, dim=0)
        if not torch.isfinite(stacked).all():
            raise ValueError("candidate logits contain non finite values")
        return stacked
