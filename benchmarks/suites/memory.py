"""Peak memory for inference and a training step, measured in a fresh process per profile.

Process RSS is sticky (allocators rarely return memory to the OS), so measuring several
models in one long-lived process makes later numbers depend on earlier ones. Every profile
is therefore measured in its own short-lived subprocess; CUDA numbers use
``torch.cuda.max_memory_allocated`` (which excludes the CUDA context).
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict

import torch

from benchmarks.common import BenchContext, build_profile_model, make_targets, random_batch
from benchmarks.env import REPO_ROOT
from benchmarks.timing import PeakMemory

DESCRIPTION = "Peak RAM (CPU, RSS) or VRAM (CUDA, allocated) for inference and a training step, one fresh process per profile"

_MARK = "@@MEMORY_JSON@@"


def measure_profile(profile: str, num_classes: int, img_size: int, device_name: str, train_batch: int) -> Dict[str, Any]:
    """Executed inside the child process (also usable in-process for tests)."""
    device = torch.device(device_name)
    row: Dict[str, Any] = {"profile": profile, "img_size": img_size, "device": device_name}
    base = PeakMemory._rss_bytes() / 1024**2
    row["process_baseline_mb"] = round(base, 1)  # interpreter + torch imports

    model = build_profile_model(profile, num_classes, device)
    row["model_weights_mb"] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2, 2)
    x = random_batch(1, img_size, device)
    with torch.no_grad():
        model(x)  # allocate lazily-initialised buffers outside the measurement window
        with PeakMemory(device) as mem:
            for _ in range(3):
                model.predict(x, original_sizes=[(img_size, img_size)], conf_thresh=0.001)
    row["inference_peak_mb"] = round(mem.peak_mb or 0.0, 1)
    row["inference_delta_mb"] = round(max((mem.peak_mb or 0.0) - (mem.baseline_mb or 0.0), 0.0), 1)

    try:
        model.train()
        imgs = random_batch(train_batch, img_size, device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in make_targets(train_batch, img_size, num_classes)]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with PeakMemory(device) as mem:
            opt.zero_grad(set_to_none=True)
            model.compute_loss(imgs, targets).backward()
            opt.step()
        row["train_batch"] = train_batch
        row["train_peak_mb"] = round(mem.peak_mb or 0.0, 1)
        row["train_delta_mb"] = round(max((mem.peak_mb or 0.0) - (mem.baseline_mb or 0.0), 0.0), 1)
    except RuntimeError as exc:
        row["train_error"] = str(exc).splitlines()[0][:200]
    return row


def run(ctx: BenchContext) -> Dict[str, Any]:
    train_batch = 2 if ctx.quick else 4
    rows = []
    for profile in ctx.profiles:
        cmd = [sys.executable, "-m", "benchmarks.suites.memory", profile, str(ctx.num_classes), str(ctx.img_size),
               ctx.device, str(train_batch)]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        payload = next((line[len(_MARK):] for line in proc.stdout.splitlines() if line.startswith(_MARK)), None)
        if proc.returncode != 0 or payload is None:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            rows.append({"profile": profile, "img_size": ctx.img_size, "device": ctx.device, "error": " | ".join(tail)})
            continue
        rows.append(json.loads(payload))
    return {
        "description": DESCRIPTION,
        "metric": "VRAM allocated (MB)" if ctx.device.startswith("cuda") else "process RSS (MB), fresh process per profile",
        "rows": rows,
    }


if __name__ == "__main__":  # child-process entry point
    _profile, _nc, _size, _dev, _tb = sys.argv[1:6]
    print(_MARK + json.dumps(measure_profile(_profile, int(_nc), int(_size), _dev, int(_tb))))
