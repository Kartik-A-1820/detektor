from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor

try:
    import onnxruntime as ort
except Exception:
    ort = None


def compare_pytorch_onnx(
    model: Any,
    onnx_path: str,
    sample_input: Tensor,
    output_names: Sequence[str],
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Compare PyTorch export-forward outputs against ONNXRuntime outputs when available."""
    if ort is None:
        message = "onnxruntime is not installed; parity check skipped"
        print(message)
        return {"available": False, "message": message, "comparisons": []}

    onnx_path = str(Path(onnx_path))
    with torch.no_grad():
        torch_outputs = model.forward_export(sample_input)
    torch_outputs_np = [output.detach().cpu().numpy() for output in torch_outputs]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(list(output_names), {input_name: sample_input.detach().cpu().numpy()})

    comparisons: List[Dict[str, Any]] = []
    for name, torch_output, ort_output in zip(output_names, torch_outputs_np, ort_outputs):
        abs_diff = np.abs(torch_output - ort_output)
        max_abs_diff = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
        mean_abs_diff = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0
        close = bool(np.allclose(torch_output, ort_output, rtol=rtol, atol=atol))
        comparison = {
            "name": name,
            "torch_shape": list(torch_output.shape),
            "onnx_shape": list(ort_output.shape),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "allclose": close,
        }
        print(
            f"parity {name}: torch_shape={comparison['torch_shape']} onnx_shape={comparison['onnx_shape']} "
            f"max_abs_diff={max_abs_diff:.6e} mean_abs_diff={mean_abs_diff:.6e} allclose={close}"
        )
        comparisons.append(comparison)

    return {"available": True, "message": "parity check complete", "comparisons": comparisons}
