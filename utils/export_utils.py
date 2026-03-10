from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch import Tensor

from models.chimera import ChimeraODIS


EXPORT_INPUT_NAME = "images"
EXPORT_OUTPUT_NAMES = ["cls_flat", "box_flat", "obj_flat", "mask_coeff_flat", "proto"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML export configuration."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_name: str) -> torch.device:
    """Resolve export device with a clear CUDA error when unavailable."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in the config, but CUDA is not available on this machine")
    return torch.device("cuda" if device_name == "cuda" else "cpu")


def build_model_from_config(cfg: Dict[str, Any], device: torch.device) -> ChimeraODIS:
    """Instantiate ChimeraODIS from config for export."""
    model = ChimeraODIS(
        num_classes=cfg["data"]["num_classes"],
        proto_k=cfg["model"]["proto_k"],
    ).to(device)
    model.eval()
    return model


def load_model_weights(model: ChimeraODIS, weights_path: str, device: torch.device) -> None:
    """Load either a raw state dict or checkpoint payload into the model."""
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)


def create_dummy_input(
    batch_size: int = 1,
    image_size: int = 512,
    device: torch.device | None = None,
) -> Tensor:
    """Create a dummy input tensor for ONNX export."""
    device = device or torch.device("cpu")
    return torch.randn(batch_size, 3, image_size, image_size, device=device)


def get_export_names() -> Tuple[str, list[str]]:
    """Return canonical ONNX input and output tensor names."""
    return EXPORT_INPUT_NAME, EXPORT_OUTPUT_NAMES


def get_dynamic_axes(dynamic_batch: bool = False) -> Dict[str, Dict[int, str]] | None:
    """Return dynamic axis settings for ONNX export."""
    if not dynamic_batch:
        return None
    dynamic_axes = {EXPORT_INPUT_NAME: {0: "batch"}}
    for name in EXPORT_OUTPUT_NAMES:
        dynamic_axes[name] = {0: "batch"}
    return dynamic_axes


def get_export_output_shapes(model: ChimeraODIS, dummy_input: Tensor) -> Dict[str, Tuple[int, ...]]:
    """Run export forward once and return output tensor shapes."""
    with torch.no_grad():
        outputs = model.forward_export(dummy_input)
    _, output_names = get_export_names()
    return {name: tuple(output.shape) for name, output in zip(output_names, outputs)}
