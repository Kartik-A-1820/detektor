"""Training-step throughput on synthetic data (forward + loss + backward + optimizer)."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from benchmarks.common import BenchContext, build_profile_model, make_targets, random_batch
from benchmarks.timing import summarize, sync

DESCRIPTION = "Training step time and images/second (synthetic batch, AdamW)"


def run(ctx: BenchContext) -> Dict[str, Any]:
    device = ctx.torch_device
    batch = 2 if ctx.quick else 4
    size = min(ctx.img_size, 320) if device.type == "cpu" else ctx.img_size
    steps = 4 if ctx.quick else 10
    warm = 2
    rows = []
    for profile in ctx.profiles:
        row: Dict[str, Any] = {"profile": profile, "img_size": size, "batch": batch, "device": ctx.device}
        try:
            model = build_profile_model(profile, ctx.num_classes, device).train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            imgs = random_batch(batch, size, device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in make_targets(batch, size, ctx.num_classes)]
            lat = []
            losses = []
            for step in range(warm + steps):
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = model.compute_loss(imgs, targets)
                loss.backward()
                opt.step()
                sync(device)
                elapsed = (time.perf_counter() - start) * 1000.0
                if step >= warm:
                    lat.append(elapsed)
                    losses.append(float(loss.detach()))
            stats = summarize(lat)
            row.update(
                {
                    "step": stats,
                    "images_per_s": round(batch * 1000.0 / stats["mean_ms"], 2),
                    "loss_finite": bool(all(v == v and abs(v) != float("inf") for v in losses)),
                    "final_loss": round(losses[-1], 4),
                }
            )
        except RuntimeError as exc:
            row["error"] = str(exc).splitlines()[0][:200]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    return {"description": DESCRIPTION, "optimizer": "AdamW", "rows": rows}
