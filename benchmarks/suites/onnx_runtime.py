"""PyTorch vs ONNX Runtime: export time, file size, parity and latency."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch

from benchmarks.common import BenchContext, build_profile_model, random_batch
from benchmarks.timing import summarize, time_callable
from utils.export_utils import torch_onnx_export

DESCRIPTION = "ONNX export, numerical parity and ONNX Runtime vs PyTorch latency"


class _Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model.forward_export(images)


def _export(wrapper: torch.nn.Module, x: torch.Tensor, path: str) -> None:
    torch_onnx_export(
        wrapper, x, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["cls", "box", "obj", "mask_coeff", "proto"],
    )


def run(ctx: BenchContext) -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        return {"description": DESCRIPTION, "skipped": f"onnxruntime not installed ({exc})", "rows": []}

    scratch = ctx.scratch()
    rows = []
    size = min(ctx.img_size, 320) if ctx.quick else ctx.img_size
    for profile in ctx.profiles:
        row: Dict[str, Any] = {"profile": profile, "img_size": size}
        try:
            model = build_profile_model(profile, ctx.num_classes, "cpu")
            wrapper = _Wrapper(model).eval()
            x = random_batch(1, size, "cpu")
            path = str(scratch / f"{profile}.onnx")
            start = time.perf_counter()
            with torch.no_grad():
                _export(wrapper, x, path)
            row["export_s"] = round(time.perf_counter() - start, 2)
            row["onnx_mb"] = round(__import__("os").path.getsize(path) / 1024**2, 2)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = torch.get_num_threads()
            session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            inp = x.numpy()

            with torch.no_grad():
                ref = [t.numpy() for t in wrapper(x)]
            out = session.run(None, {"images": inp})
            row["max_abs_diff"] = float(max(np.abs(a - b).max() for a, b in zip(ref, out)))

            with torch.no_grad():
                torch_lat = time_callable(lambda: wrapper(x), warmup=ctx.warmup, runs=ctx.runs)
            ort_lat = time_callable(lambda: session.run(None, {"images": inp}), warmup=ctx.warmup, runs=ctx.runs)
            row["pytorch_cpu"] = summarize(torch_lat)
            row["onnxruntime_cpu"] = summarize(ort_lat)
            row["speedup_vs_pytorch"] = round(row["pytorch_cpu"]["p50_ms"] / max(row["onnxruntime_cpu"]["p50_ms"], 1e-9), 2)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        rows.append(row)
    return {"description": DESCRIPTION, "provider": "CPUExecutionProvider", "onnxruntime": ort.__version__, "rows": rows}
