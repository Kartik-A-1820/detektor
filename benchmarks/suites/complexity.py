"""Model complexity: parameters, FLOPs and on-disk size for every architecture profile."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.flop_counter import FlopCounterMode

from benchmarks.common import BenchContext, build_profile_model, random_batch

DESCRIPTION = "Parameters, GFLOPs and model size per architecture profile and input size"


def run(ctx: BenchContext) -> Dict[str, Any]:
    rows = []
    for profile in ctx.profiles:
        model = build_profile_model(profile, ctx.num_classes, "cpu")
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for size in ctx.img_sizes:
            x = random_batch(1, size, "cpu")
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                model(x)
            flops = counter.get_total_flops()
            rows.append(
                {
                    "profile": profile,
                    "img_size": size,
                    "params_m": round(params / 1e6, 3),
                    "trainable_params_m": round(trainable / 1e6, 3),
                    "gflops": round(flops / 1e9, 3),
                    "gmacs": round(flops / 2e9, 3),
                    "size_fp32_mb": round(params * 4 / 1024**2, 2),
                    "size_fp16_mb": round(params * 2 / 1024**2, 2),
                }
            )
    return {"description": DESCRIPTION, "num_classes": ctx.num_classes, "rows": rows}
