"""Batch-size scaling: images per second as the batch grows."""

from __future__ import annotations

from typing import Any, Dict

import torch

from benchmarks.common import BenchContext, build_profile_model, random_batch
from benchmarks.timing import summarize, time_callable

DESCRIPTION = "Images/second versus batch size (forward pass)"


def run(ctx: BenchContext) -> Dict[str, Any]:
    device = ctx.torch_device
    rows = []
    runs = max(5, ctx.runs // 3)
    for profile in ctx.profiles:
        model = build_profile_model(profile, ctx.num_classes, device)
        for batch in ctx.batch_sizes:
            entry: Dict[str, Any] = {"profile": profile, "img_size": ctx.img_size, "batch": batch}
            try:
                x = random_batch(batch, ctx.img_size, device)
                with torch.no_grad():
                    lat = time_callable(lambda: model(x), device=device, warmup=max(2, ctx.warmup // 2), runs=runs)
                stats = summarize(lat)
                entry.update(
                    {
                        "batch_latency": stats,
                        "images_per_s": round(batch * 1000.0 / stats["mean_ms"], 2),
                        "ms_per_image": round(stats["mean_ms"] / batch, 3),
                    }
                )
            except RuntimeError as exc:  # e.g. CUDA OOM
                entry["error"] = str(exc).splitlines()[0][:200]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(entry)
    return {"description": DESCRIPTION, "rows": rows}
