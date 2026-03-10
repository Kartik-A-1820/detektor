from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from utils.export_utils import (
    build_model_from_config,
    create_dummy_input,
    get_dynamic_axes,
    get_export_names,
    get_export_output_shapes,
    load_config,
    load_model_weights,
    resolve_device,
)
from utils.parity import compare_pytorch_onnx


class ExportWrapper(torch.nn.Module):
    """Thin wrapper exposing the tensor-only export forward path."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model.forward_export(images)


def export_onnx(
    config_path: str,
    weights: str,
    output_path: str,
    opset: int = 13,
    dynamic_batch: bool = False,
    check_parity: bool = False,
) -> Dict[str, Any]:
    """Export ChimeraODIS tensor-only outputs to ONNX."""
    cfg = load_config(config_path)
    device = resolve_device(cfg.get("device", "cpu"))
    model = build_model_from_config(cfg, device=device)
    load_model_weights(model, weights, device=device)
    model.eval()

    wrapper = ExportWrapper(model).to(device)
    dummy_input = create_dummy_input(
        batch_size=1,
        image_size=int(cfg["train"].get("img_size", 512)),
        device=device,
    )
    input_name, output_names = get_export_names()
    dynamic_axes = get_dynamic_axes(dynamic_batch=dynamic_batch)
    output_shapes = get_export_output_shapes(model, dummy_input)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path_obj),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    if not output_path_obj.exists():
        raise RuntimeError(f"ONNX export failed; output file was not created: {output_path_obj}")

    print(f"exported: {output_path_obj}")
    for name in output_names:
        print(f"{name}: {output_shapes[name]}")

    parity_summary: Dict[str, Any] | None = None
    if check_parity:
        parity_summary = compare_pytorch_onnx(
            model=model,
            onnx_path=str(output_path_obj),
            sample_input=dummy_input,
            output_names=output_names,
        )

    return {
        "output_path": str(output_path_obj),
        "input_name": input_name,
        "output_names": output_names,
        "output_shapes": output_shapes,
        "dynamic_batch": dynamic_batch,
        "opset": int(opset),
        "parity": parity_summary,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Detektor to ONNX")
    parser.add_argument("--config", type=str, default="configs/chimera_s_512.yaml", help="Path to the base model config YAML")
    parser.add_argument("--weights", type=str, default="runs/chimera/chimera_last.pt", help="Path to model weights or checkpoint")
    parser.add_argument("--output", type=str, default="exports/chimera_odis.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis in the exported graph")
    parser.add_argument("--check-parity", action="store_true", help="Run optional PyTorch vs ONNX parity check after export")
    args = parser.parse_args()

    export_onnx(
        config_path=args.config,
        weights=args.weights,
        output_path=args.output,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        check_parity=args.check_parity,
    )
