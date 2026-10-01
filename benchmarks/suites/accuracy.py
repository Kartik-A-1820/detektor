"""Accuracy on a real dataset: wraps ``validate.py`` and reads its metrics artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from benchmarks.common import BenchContext
from benchmarks.env import REPO_ROOT

DESCRIPTION = "Precision/recall/mAP50/mAP50-95/IoU on a dataset split via validate.py (needs --weights and --data-yaml)"


def validate_checkpoint(weights: str, data_yaml: str, out_dir: Path, device: str = "cpu", conf: float = 0.05,
                        batch_size: int = 8) -> Dict[str, Any]:
    """Run ``validate.py`` in a subprocess and return the comprehensive overall metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "validate.py",
        "--weights", str(weights),
        "--data-yaml", str(data_yaml),
        "--conf-thresh", str(conf),
        "--batch-size", str(batch_size),
        "--compute-ap50-95",
        "--output-dir", str(out_dir),
        "--save-json", str(out_dir / "predictions.json"),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    metrics_path = out_dir / "metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise RuntimeError("validate.py failed: " + " | ".join(tail))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    overall = metrics["overall"]
    return {
        "validation_wall_s": round(elapsed, 2),
        "num_images": overall.get("num_images"),
        "precision": round(overall["precision"], 4),
        "recall": round(overall["recall"], 4),
        "f1": round(overall["f1"], 4),
        "map50": round(overall["map50"], 4),
        "ap50_95": round(overall.get("ap50_95", 0.0), 4),
        "mean_box_iou": round(overall.get("mean_box_iou", 0.0), 4),
        "mean_mask_iou": round(overall.get("mean_mask_iou", 0.0), 4),
        "per_class": [
            {"class": c["class_name"], "ap50": round(c["ap50"], 4), "precision": round(c["precision"], 4),
             "recall": round(c["recall"], 4)}
            for c in metrics.get("per_class", [])
        ],
    }


def run(ctx: BenchContext) -> Dict[str, Any]:
    if not ctx.weights or not ctx.data_yaml:
        return {"description": DESCRIPTION, "skipped": "pass --weights and --data-yaml to enable the accuracy suite", "rows": []}
    result = validate_checkpoint(ctx.weights, ctx.data_yaml, ctx.output_dir / "accuracy", device=ctx.device)
    return {"description": DESCRIPTION, "weights": str(ctx.weights), "data_yaml": str(ctx.data_yaml), "metrics": result}
