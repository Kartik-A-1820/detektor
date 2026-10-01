"""Cold-start cost: checkpoint load, first-inference penalty, on-disk size."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from api.utils import load_model
from benchmarks.common import BenchContext, random_batch, save_profile_checkpoint
from benchmarks.timing import summarize, sync

DESCRIPTION = "Checkpoint load time, first-request penalty and warm latency"


def run(ctx: BenchContext) -> Dict[str, Any]:
    scratch = ctx.scratch()
    rows = []
    repeats = 3
    for profile in ctx.profiles:
        ckpt = save_profile_checkpoint(profile, ctx.num_classes, scratch / f"{profile}.pt", seed=ctx.seed)
        loads = []
        first = []
        warm = []
        for _ in range(repeats):
            start = time.perf_counter()
            model, dev = load_model(str(ckpt), device_name=ctx.device)
            sync(dev)
            loads.append((time.perf_counter() - start) * 1000.0)
            x = random_batch(1, ctx.img_size, dev)
            with torch.no_grad():
                t0 = time.perf_counter()
                model.predict(x, original_sizes=[(ctx.img_size, ctx.img_size)])
                sync(dev)
                first.append((time.perf_counter() - t0) * 1000.0)
                t1 = time.perf_counter()
                model.predict(x, original_sizes=[(ctx.img_size, ctx.img_size)])
                sync(dev)
                warm.append((time.perf_counter() - t1) * 1000.0)
        rows.append(
            {
                "profile": profile,
                "checkpoint_mb": round(ckpt.stat().st_size / 1024**2, 2),
                "load_ms": summarize(loads),
                "first_inference_ms": round(float(sum(first) / len(first)), 2),
                "second_inference_ms": round(float(sum(warm) / len(warm)), 2),
                "warmup_penalty_ms": round(float(sum(first) / len(first) - sum(warm) / len(warm)), 2),
            }
        )
        ckpt.unlink(missing_ok=True)
    return {"description": DESCRIPTION, "repeats": repeats, "rows": rows}
