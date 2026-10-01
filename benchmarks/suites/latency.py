"""Single-image latency: stage breakdown (decode/preprocess, forward, postprocess)."""

from __future__ import annotations

from typing import Any, Dict

import torch

from api.utils import preprocess_image_bytes
from benchmarks.common import BenchContext, build_profile_model, random_batch, synthetic_jpeg
from benchmarks.timing import summarize, time_callable

DESCRIPTION = "Batch-1 latency with preprocess / forward / postprocess breakdown (p50/p95/p99)"

# A freshly initialised network emits near-uniform scores, so a normal threshold would
# yield zero detections and hide postprocessing cost. The "dense" scenario uses a very
# low threshold so top-k + NMS + mask composition run at their worst-case workload.
SCENARIOS = {
    "default": {"conf_thresh": 0.25},
    "dense": {"conf_thresh": 0.001},  # worst case: top-k + NMS + full-resolution masks
    "dense_boxes": {"conf_thresh": 0.001, "task": "detect"},  # same workload without masks (API default)
}


def run(ctx: BenchContext) -> Dict[str, Any]:
    device = ctx.torch_device
    jpeg = synthetic_jpeg()
    rows = []
    for profile in ctx.profiles:
        model = build_profile_model(profile, ctx.num_classes, device)
        for size in ctx.img_sizes:
            x = random_batch(1, size, device)
            orig = [(720, 1280)]

            pre = time_callable(lambda: preprocess_image_bytes(jpeg, image_size=size), warmup=ctx.warmup, runs=ctx.runs)

            with torch.no_grad():
                fwd = time_callable(lambda: model(x), device=device, warmup=ctx.warmup, runs=ctx.runs)

            row: Dict[str, Any] = {
                "profile": profile,
                "img_size": size,
                "device": ctx.device,
                "preprocess": summarize(pre),
                "forward": summarize(fwd),
            }
            for name, kwargs in SCENARIOS.items():
                n_det = {"v": 0}

                def predict(kwargs=kwargs):
                    out = model.predict(x, original_sizes=orig, **kwargs)
                    n_det["v"] = int(out[0]["boxes"].shape[0])

                total = time_callable(predict, device=device, warmup=ctx.warmup, runs=ctx.runs)
                stats = summarize(total)
                fwd_p50 = row["forward"]["p50_ms"]
                stats["postprocess_p50_ms"] = round(max(stats["p50_ms"] - fwd_p50, 0.0), 3)
                stats["detections"] = n_det["v"]
                row[f"predict_{name}"] = stats
            # End-to-end estimate for a real request: decode+resize -> predict (default thresholds)
            row["e2e_p50_ms"] = round(row["preprocess"]["p50_ms"] + row["predict_default"]["p50_ms"], 3)
            rows.append(row)
    return {
        "description": DESCRIPTION,
        "source_image": "synthetic 1280x720 JPEG",
        "scenarios": SCENARIOS,
        "warmup": ctx.warmup,
        "runs": ctx.runs,
        "rows": rows,
    }
