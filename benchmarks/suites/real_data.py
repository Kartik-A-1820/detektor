"""Real-data baseline: train on a public dataset and report COCO-style metrics.

Unlike ``e2e`` (synthetic shapes) this uses real photographs, so it measures what accuracy changes
actually buy. The dataset is downloaded on demand (see ``benchmarks/datasets.py``); pass your own
``--data-yaml`` to benchmark private data instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from benchmarks import datasets
from benchmarks.common import BenchContext
from benchmarks.suites.accuracy import validate_checkpoint
from benchmarks.suites.e2e_quality import train_on_yaml

DESCRIPTION = "Train on a public (or your own) real dataset and report COCO-style mAP, per-class AP and training time"


def run(ctx: BenchContext) -> Dict[str, Any]:
    name = ctx.extras.get("real_dataset") or "african-wildlife"
    epochs = int(ctx.extras.get("real_epochs", 3 if ctx.quick else 20))
    max_train = ctx.extras.get("real_max_train", 48 if ctx.quick else 400)
    max_val = ctx.extras.get("real_max_val", 24 if ctx.quick else None)
    img_size = int(ctx.extras.get("real_img_size", 320))
    profile = ctx.extras.get("real_profile", "firefly")
    batch = int(ctx.extras.get("real_batch", 8))

    workdir = (ctx.output_dir / "real").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        data_yaml = Path(ctx.data_yaml) if ctx.data_yaml else datasets.fetch(name)
    except Exception as exc:  # noqa: BLE001 - offline machines should skip, not fail the whole run
        return {"description": DESCRIPTION, "skipped": f"dataset '{name}' unavailable ({type(exc).__name__}: {exc})", "rows": []}
    source = str(data_yaml)
    if max_train or max_val:
        data_yaml = datasets.make_subset(data_yaml, workdir / "subset", max_train=max_train, max_val=max_val, seed=ctx.seed)

    trained = train_on_yaml(ctx, workdir, data_yaml=data_yaml, profile=profile, epochs=epochs, img_size=img_size, batch_size=batch)
    metrics = validate_checkpoint(str(trained["checkpoint"]), str(data_yaml), workdir / "validation", device=ctx.device, conf=0.05)
    spec = datasets.DATASETS.get(name)
    return {
        "description": DESCRIPTION,
        "dataset": name if not ctx.data_yaml else "custom",
        "dataset_source": source,
        "dataset_license": spec.license if spec and not ctx.data_yaml else "n/a",
        "profile": profile,
        "device": ctx.device,
        "img_size": img_size,
        "epochs": epochs,
        "train_images": max_train or "all",
        "val_images": metrics.get("num_images"),
        "train_wall_s": round(trained["train_wall_s"], 1),
        "metrics": metrics,
        "note": "trained from scratch (no pretrained weights); a short CPU run, so treat as a relative baseline",
    }
