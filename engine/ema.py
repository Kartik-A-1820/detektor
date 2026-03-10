from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn


class ModelEMA:
    """Lightweight exponential moving average wrapper for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the current model parameters."""
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return EMA state dict for checkpointing."""
        return self.ema.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Restore EMA weights from checkpoint state."""
        self.ema.load_state_dict(state_dict, strict=True)

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """Temporarily apply EMA weights to a model."""
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.ema.state_dict(), strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)
