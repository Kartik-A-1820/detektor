"""End-to-end quality benchmark on a reproducible synthetic dataset.

Generates a seeded synthetic dataset, trains a small profile for a few epochs with the real
``train.py``, validates the best checkpoint with the real ``validate.py`` and records
training time, throughput and detection quality. It proves the *pipeline* (data -> train ->
checkpoint -> validate) end to end; it does not measure performance on real-world data.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any, Dict

from benchmarks.common import BenchContext
from benchmarks.env import REPO_ROOT
from benchmarks.suites.accuracy import validate_checkpoint
from benchmarks.synthetic import CLASS_NAMES, generate_dataset

DESCRIPTION = "Train + validate on a seeded synthetic dataset (pipeline correctness and learning signal)"


def train_synthetic(ctx: BenchContext, workdir, *, profile: str, epochs: int, train_n: int, val_n: int,
                    img_size: int) -> Dict[str, Any]:
    data_yaml = generate_dataset(workdir / "data", train=train_n, val=val_n, img_size=img_size, seed=ctx.seed + 1)
    run_dir = workdir / "run"
    cmd = [
        sys.executable, "train.py",
        "--data-yaml", str(data_yaml),
        "--device", ctx.device,
        "--epochs", str(epochs),
        "--img-size", str(img_size),
        "--batch-size", "8",
        "--model", profile,
        "--out-dir", str(run_dir),
        "--num-workers", "0",
        "--no-auto-tune",
        "--seed", str(ctx.seed),
        "--conf-thresh", "0.05",
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    best = run_dir / "chimera_best.pt"
    last = run_dir / "chimera_last.pt"
    ckpt = best if best.exists() else last
    if proc.returncode != 0 or not ckpt.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise RuntimeError("train.py failed: " + " | ".join(tail))
    return {"data_yaml": data_yaml, "checkpoint": ckpt, "train_wall_s": elapsed, "run_dir": run_dir}


def run(ctx: BenchContext) -> Dict[str, Any]:
    profile = ctx.extras.get("e2e_profile", "firefly")
    epochs = int(ctx.extras.get("e2e_epochs", 4 if ctx.quick else 30))
    train_n, val_n = (64, 24) if ctx.quick else (192, 48)
    img_size = 256
    workdir = ctx.output_dir / "e2e"
    workdir.mkdir(parents=True, exist_ok=True)

    trained = train_synthetic(ctx, workdir, profile=profile, epochs=epochs, train_n=train_n, val_n=val_n, img_size=img_size)
    metrics = validate_checkpoint(str(trained["checkpoint"]), str(trained["data_yaml"]), workdir / "validation", device=ctx.device)
    ctx.extras["e2e_checkpoint"] = str(trained["checkpoint"])
    ctx.extras["e2e_data_yaml"] = str(trained["data_yaml"])
    return {
        "description": DESCRIPTION,
        "profile": profile,
        "device": ctx.device,
        "classes": CLASS_NAMES,
        "train_images": train_n,
        "val_images": val_n,
        "img_size": img_size,
        "epochs": epochs,
        "train_wall_s": round(trained["train_wall_s"], 1),
        "train_images_per_s": round(train_n * epochs / trained["train_wall_s"], 2),
        "metrics": metrics,
        "checkpoint": str(trained["checkpoint"]),
    }
