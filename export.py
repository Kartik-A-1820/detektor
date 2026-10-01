from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from models.factory import build_model_from_checkpoint, checkpoint_train_img_size, load_model_weights
from utils.export_utils import (
    create_dummy_input,
    get_dynamic_axes,
    get_export_names,
    get_export_output_shapes,
    load_config,
    resolve_device,
    torch_onnx_export,
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
    config_path: Optional[str],
    weights: str,
    output_path: str,
    opset: int = 13,
    dynamic_batch: bool = False,
    check_parity: bool = False,
    image_size: Optional[int] = None,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """Export ChimeraODIS tensor-only outputs to ONNX.

    The architecture (profile, channels, class count) is rebuilt from the metadata embedded in the
    checkpoint, so any profile exports without a matching config file. ``config_path`` is optional
    and only supplies the export size for legacy checkpoints that do not record it. The input size
    defaults to the size the checkpoint was trained at.
    """
    device = resolve_device(device_name)
    checkpoint = torch.load(weights, map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    load_model_weights(model, checkpoint)
    model.eval()

    if image_size is None:
        image_size = checkpoint_train_img_size(checkpoint)
    if image_size is None and config_path:
        image_size = int(load_config(config_path).get("train", {}).get("img_size", 0)) or None
    image_size = int(image_size or 512)

    # Must be eval(): torch.onnx.export would otherwise put the *inner* model into training mode and
    # bake BatchNorm batch statistics into the graph, producing wrong outputs for trained weights.
    wrapper = ExportWrapper(model).to(device).eval()
    dummy_input = create_dummy_input(batch_size=1, image_size=image_size, device=device)
    input_name, output_names = get_export_names()
    dynamic_axes = get_dynamic_axes(dynamic_batch=dynamic_batch)
    output_shapes = get_export_output_shapes(model, dummy_input)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    torch_onnx_export(
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

    parity_ok: Optional[bool] = None
    if parity_summary is not None and parity_summary.get("available"):
        parity_ok = all(item["allclose"] for item in parity_summary["comparisons"])
        if not parity_ok:
            print("WARNING: PyTorch and ONNX outputs differ beyond tolerance - do not deploy this export")

    return {
        "parity_ok": parity_ok,
        "output_path": str(output_path_obj),
        "input_name": input_name,
        "output_names": output_names,
        "output_shapes": output_shapes,
        "dynamic_batch": dynamic_batch,
        "image_size": image_size,
        "opset": int(opset),
        "parity": parity_summary,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Detektor to ONNX")
    parser.add_argument("--weights", type=str, default="runs/chimera/chimera_last.pt", help="Path to model weights or checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML (only used for the export size of legacy checkpoints)")
    parser.add_argument("--output", type=str, default="exports/chimera_odis.onnx", help="Output ONNX file path")
    parser.add_argument("--img-size", type=int, default=None, help="Export input size (default: the size the checkpoint was trained at)")
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"), help="Device used while tracing (default: cpu)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis in the exported graph")
    parser.add_argument("--check-parity", action="store_true", help="Run optional PyTorch vs ONNX parity check after export")
    args = parser.parse_args()

    result = export_onnx(
        config_path=args.config,
        weights=args.weights,
        output_path=args.output,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        check_parity=args.check_parity,
        image_size=args.img_size,
        device_name=args.device,
    )
    if result.get("parity_ok") is False:
        raise SystemExit(2)
