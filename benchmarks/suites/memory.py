"""Peak memory for inference and a training step."""

from __future__ import annotations

from typing import Any, Dict

import torch

from benchmarks.common import BenchContext, build_profile_model, make_targets, random_batch
from benchmarks.timing import PeakMemory

DESCRIPTION = "Peak RAM (CPU, RSS delta) or VRAM (CUDA, allocated) for inference and a training step"


def run(ctx: BenchContext) -> Dict[str, Any]:
    device = ctx.torch_device
    rows = []
    train_batch = 2 if ctx.quick else 4
    for profile in ctx.profiles:
        row: Dict[str, Any] = {"profile": profile, "img_size": ctx.img_size, "device": ctx.device}

        model = build_profile_model(profile, ctx.num_classes, device)
        x = random_batch(1, ctx.img_size, device)
        with torch.no_grad():
            model(x)  # allocate lazily-initialised buffers outside the measurement
            with PeakMemory(device) as mem:
                for _ in range(3):
                    model.predict(x, original_sizes=[(ctx.img_size, ctx.img_size)], conf_thresh=0.001)
        row["inference_peak_mb"] = round(mem.peak_mb or 0.0, 1)
        row["inference_delta_mb"] = round(mem.delta_mb or 0.0, 1)

        try:
            model.train()
            imgs = random_batch(train_batch, ctx.img_size, device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in make_targets(train_batch, ctx.img_size, ctx.num_classes)]
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            with PeakMemory(device) as mem:
                opt.zero_grad(set_to_none=True)
                loss = model.compute_loss(imgs, targets)
                loss.backward()
                opt.step()
            row["train_batch"] = train_batch
            row["train_peak_mb"] = round(mem.peak_mb or 0.0, 1)
            row["train_delta_mb"] = round(mem.delta_mb or 0.0, 1)
        except RuntimeError as exc:
            row["train_error"] = str(exc).splitlines()[0][:200]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    return {
        "description": DESCRIPTION,
        "metric": "VRAM allocated (MB)" if device.type == "cuda" else "process RSS delta (MB)",
        "rows": rows,
    }
