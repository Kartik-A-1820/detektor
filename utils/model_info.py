from __future__ import annotations

from typing import Any, Dict

from torch import nn


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Return lightweight model parameter and size statistics."""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    size_mb = total_params * 4.0 / (1024.0 * 1024.0)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "estimated_size_mb": float(size_mb),
    }
