"""AdamW followed by an explicit Langevin noise update."""

from __future__ import annotations

import math
from typing import Iterable

import torch


class LangevinAdamW(torch.optim.AdamW):
    """Clean AdamW plus Gaussian noise implementation.

    Optimizer weight decay is fixed to zero because the Gaussian prior is
    represented explicitly in the objective.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        temperature: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        self.temperature = float(temperature)
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group.get("weight_decay", 0.0) != 0.0:
                raise ValueError("optimizer weight decay must remain zero")
            for parameter in group["params"]:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError("optimizer gradient contains non finite values")
        loss = super().step(closure)
        if self.temperature:
            for group in self.param_groups:
                scale = math.sqrt(2.0 * float(group["lr"]) * self.temperature)
                for parameter in group["params"]:
                    parameter.add_(torch.randn_like(parameter), alpha=scale)
        for group in self.param_groups:
            for parameter in group["params"]:
                if not torch.isfinite(parameter).all():
                    raise ValueError("optimizer produced non finite parameters")
        return loss
