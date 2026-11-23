"""TOML configuration loading and release contract validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PINNED_REVISIONS = {
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "meta-llama/Llama-3.2-3B": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "Qwen/Qwen3-4B-Base": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "Qwen/Qwen2-1.5B-Instruct": "ba1cf1846d7df0a0591d6c00649f57e798519da8",
}
CANONICAL_LABELS = ("A", "B", "C", "D", "E")
LABEL_READOUT = "bare_canonical_token"
PLAIN_PROMPT_STYLE = "plain_mcq"
QWEN2_CHATML_PROMPT_STYLE = "qwen2_chatml_single_user_generation_prompt"
HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE = "historical_alpaca_class_label"
PROMPT_STYLES = {
    PLAIN_PROMPT_STYLE,
    QWEN2_CHATML_PROMPT_STYLE,
    HISTORICAL_ALPACA_CLASS_LABEL_PROMPT_STYLE,
}
SCHEMA_VERSION = 4


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str
    label_readout: str
    labels: tuple[str, ...] = CANONICAL_LABELS
    prompt_style: str = PLAIN_PROMPT_STYLE


@dataclass(frozen=True)
class LoRAConfig:
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    beta: float
    prior_lambda: float
    temperature_mode: str
    temperature: float | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    paper_mode: bool
    num_branches: int
    branch_seeds: tuple[int, ...]
    epochs: int
    batch_size: int
    micro_batch_size: int
    learning_rate: float
    warmup_steps: int
    dtype: str
    max_length: int
    shuffle_seed: int
    gradient_checkpointing: bool
    eval_steps: int | None = None
    save_steps: int | None = None
    padding_side: str = "left"
    attn_implementation: str | None = None

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    lora: LoRAConfig
    objective: ObjectiveConfig
    training: TrainingConfig

    def with_beta(self, beta: float | None) -> "Config":
        if beta is None:
            return self
        return replace(self, objective=replace(self.objective, beta=float(beta)))


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid [{name}] section")
    return value


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    model = _section(raw, "model")
    lora = _section(raw, "lora")
    objective = _section(raw, "objective")
    training = _section(raw, "training")
    config = Config(
        model=ModelConfig(
            id=str(model["id"]),
            revision=str(model["revision"]),
            label_readout=str(model["label_readout"]),
            labels=tuple(str(item) for item in model.get("labels", CANONICAL_LABELS)),
            prompt_style=str(model.get("prompt_style", PLAIN_PROMPT_STYLE)),
        ),
        lora=LoRAConfig(
            target_modules=tuple(str(item) for item in lora["target_modules"]),
            rank=int(lora["rank"]),
            alpha=float(lora["alpha"]),
            dropout=float(lora["dropout"]),
        ),
        objective=ObjectiveConfig(
            name=str(objective["name"]),
            beta=float(objective["beta"]),
            prior_lambda=float(objective["prior_lambda"]),
            temperature_mode=str(objective["temperature_mode"]),
            temperature=(
                None
                if objective.get("temperature") is None
                else float(objective["temperature"])
            ),
            alpha=(
                None
                if objective.get("alpha") is None
                else float(objective["alpha"])
            ),
        ),
        training=TrainingConfig(
            paper_mode=bool(training["paper_mode"]),
            num_branches=int(training["num_branches"]),
            branch_seeds=tuple(int(item) for item in training["branch_seeds"]),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            micro_batch_size=int(training["micro_batch_size"]),
            learning_rate=float(training["learning_rate"]),
            warmup_steps=int(training["warmup_steps"]),
            dtype=str(training["dtype"]),
            max_length=int(training["max_length"]),
            shuffle_seed=int(training["shuffle_seed"]),
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
            eval_steps=(
                None if training.get("eval_steps") is None else int(training["eval_steps"])
            ),
            save_steps=(
                None if training.get("save_steps") is None else int(training["save_steps"])
            ),
            padding_side=str(training.get("padding_side", "left")),
            attn_implementation=(
                None
                if training.get("attn_implementation") is None
                else str(training["attn_implementation"])
            ),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: Config, world_size: int | None = None) -> None:
    expected_revision = PINNED_REVISIONS.get(config.model.id)
    if expected_revision is None:
        raise ValueError(f"Unsupported model ID: {config.model.id}")
    if config.model.revision != expected_revision:
        raise ValueError(
            f"Bad revision for {config.model.id}: expected {expected_revision}"
        )
    if config.model.labels != CANONICAL_LABELS:
        raise ValueError(f"Labels must be exactly {CANONICAL_LABELS}")
    if config.model.label_readout != LABEL_READOUT:
        raise ValueError(f"label_readout must be {LABEL_READOUT}")
    if config.model.prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unsupported prompt_style: {config.model.prompt_style}")

    targets = config.lora.target_modules
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("LoRA target modules must be nonempty and unique")
    if config.lora.rank <= 0 or config.lora.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= config.lora.dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")

    beta = config.objective.beta
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be finite and in [0, 1], got {beta!r}")
    if config.objective.name not in {"pro_ce_convex", "alpha_renyi"}:
        raise ValueError("objective name must be pro_ce_convex or alpha_renyi")
    if config.objective.name == "pro_ce_convex":
        if config.objective.alpha is not None:
            raise ValueError("pro_ce_convex objective must not define alpha")
    else:
        alpha = config.objective.alpha
        if alpha is None or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha_renyi alpha must be finite and in [0, 1]")
        if beta != 0.0:
            raise ValueError("alpha_renyi retains matched beta 0")
    if (
        not math.isfinite(config.objective.prior_lambda)
        or config.objective.prior_lambda < 0
    ):
        raise ValueError("prior_lambda must be finite and nonnegative")
    if config.objective.temperature_mode not in {"auto", "zero", "manual"}:
        raise ValueError("temperature_mode must be auto, zero, or manual")
    if config.objective.temperature_mode == "manual":
        temperature = config.objective.temperature
        if temperature is None or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("manual temperature must be finite and nonnegative")

    training = config.training
    if training.num_branches <= 0:
        raise ValueError("num_branches must be positive")
    if len(training.branch_seeds) != training.num_branches:
        raise ValueError("branch seed count must match num_branches")
    if len(set(training.branch_seeds)) != len(training.branch_seeds):
        raise ValueError("branch seeds must be unique")
    if training.epochs <= 0:
        raise ValueError("epochs must be positive")
    if training.batch_size <= 0 or training.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if training.batch_size % training.micro_batch_size:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    if not math.isfinite(training.learning_rate) or training.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if training.warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if training.dtype not in {"bf16", "fp32"}:
        raise ValueError("dtype must be bf16 or fp32")
    if training.max_length <= 0:
        raise ValueError("max_length must be positive")
    if training.eval_steps is not None and training.eval_steps <= 0:
        raise ValueError("eval_steps must be positive when configured")
    if training.save_steps is not None and training.save_steps <= 0:
        raise ValueError("save_steps must be positive when configured")
    if training.padding_side != "left":
        raise ValueError("padding_side must be left")
    if training.attn_implementation not in {None, "eager", "sdpa", "flash_attention_2"}:
        raise ValueError("unsupported attn_implementation")

    if world_size is not None:
        if world_size <= 0 or training.num_branches % world_size:
            raise ValueError("num_branches must be divisible by world_size")
        if training.paper_mode and world_size != 4:
            raise ValueError("paper mode requires exactly 4 ranks")
    if training.paper_mode and training.num_branches != 8:
        raise ValueError("paper mode requires exactly 8 branches")


def validate_matched_renyi_config(reference: Config, treatment: Config) -> None:
    validate_config(reference)
    validate_config(treatment)
    if reference.objective.name != "pro_ce_convex":
        raise ValueError("matched reference objective must be pro_ce_convex")
    if treatment.objective.name != "alpha_renyi":
        raise ValueError("matched treatment objective must be alpha_renyi")
    reference_payload = deepcopy(asdict(reference))
    treatment_payload = deepcopy(asdict(treatment))
    reference_payload["objective"].pop("alpha", None)
    treatment_payload["objective"].pop("alpha", None)
    treatment_payload["objective"]["name"] = reference_payload["objective"]["name"]
    if reference_payload != treatment_payload:
        raise ValueError(
            "matched Rényi config changes a field other than objective kind and alpha"
        )


def resolve_temperature(config: Config, num_training_samples: int) -> float:
    if num_training_samples <= 0:
        raise ValueError("num_training_samples must be positive")
    mode = config.objective.temperature_mode
    if mode == "auto":
        return 1.0 / (config.training.num_branches * num_training_samples)
    if mode == "zero":
        return 0.0
    assert config.objective.temperature is not None
    return config.objective.temperature


def relative_display_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest display paths must be relative and contained")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("manifest display path must identify an artifact")
    return normalized
