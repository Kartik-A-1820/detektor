from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml


_PLOT_FILES = {
    "loss_total": "Loss Curve",
    "loss_components": "Loss Components",
    "learning_rate": "Learning Rate",
}


def discover_checkpoint_options(weights_path: str | Path) -> Tuple[Path, Dict[str, Path], str]:
    """Discover sibling checkpoints for a run and default to the best checkpoint when present."""
    weights = Path(weights_path).expanduser().resolve()
    run_dir = weights.parent if weights.suffix.lower() == ".pt" else weights
    checkpoints: Dict[str, Path] = {}

    for key, filename in (("best", "chimera_best.pt"), ("last", "chimera_last.pt")):
        candidate = run_dir / filename
        if candidate.exists():
            checkpoints[key] = candidate.resolve()

    if weights.is_file() and weights.exists() and weights.resolve() not in checkpoints.values():
        checkpoints["custom"] = weights.resolve()

    default_key = "best" if "best" in checkpoints else "last" if "last" in checkpoints else "custom"
    return run_dir.resolve(), checkpoints, default_key


def load_run_artifacts(run_dir: str | Path, checkpoint_path: str | Path, checkpoint_key: str) -> Dict[str, Any]:
    """Collect training metadata and plot paths for the selected run."""
    run_path = Path(run_dir).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_config = _read_yaml(run_path / "resolved_train_config.yaml")
    run_summary = _read_json(run_path / "run_summary.json")
    training_summary = _read_json(run_path / "training_summary.json")
    val_history = _read_jsonl(run_path / "val_metrics.jsonl")
    train_curve = _read_train_curve(run_path / "train_metrics.csv")
    class_names = resolved_config.get("data", {}).get("names", []) or []
    class_map = {str(index): str(name) for index, name in enumerate(class_names)}

    return {
        "run_dir": str(run_path),
        "active_checkpoint_key": checkpoint_key,
        "active_checkpoint_path": str(checkpoint),
        "checkpoint_summary": _read_checkpoint_summary(checkpoint),
        "dataset": _build_dataset_summary(resolved_config, run_summary),
        "runtime": run_summary.get("resolved_runtime", {}),
        "training_summary": training_summary.get("training", {}),
        "validation_summary": training_summary.get("validation", {}),
        "metadata": training_summary.get("metadata", {}),
        "train_curve": train_curve,
        "validation_history": val_history,
        "class_map": class_map,
        "plots": _discover_plots(run_path),
    }


def _build_dataset_summary(resolved_config: Dict[str, Any], run_summary: Dict[str, Any]) -> Dict[str, Any]:
    data = resolved_config.get("data", {})
    runtime = run_summary.get("resolved_runtime", {})
    return {
        "dataset_yaml": runtime.get("dataset_yaml"),
        "train_root": runtime.get("resolved_train_root") or data.get("train"),
        "val_root": runtime.get("resolved_val_root") or data.get("val"),
        "dataset_size": runtime.get("dataset_size"),
        "num_classes": data.get("num_classes"),
        "class_names": data.get("names", []),
    }


def _read_checkpoint_summary(checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        return {}

    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    model_config = payload.get("model_config", {})
    return {
        "format_version": payload.get("format_version"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "best_metric": payload.get("best_metric"),
        "model_class": payload.get("model_class"),
        "model_config": model_config,
        "file_size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
        "modified_time": checkpoint_path.stat().st_mtime,
    }


def _discover_plots(run_dir: Path) -> Dict[str, str]:
    plots_dir = run_dir / "plots"
    plots: Dict[str, str] = {}
    for stem in _PLOT_FILES:
        candidate = plots_dir / f"{stem}.png"
        if candidate.exists():
            plots[stem] = str(candidate.resolve())
    return plots


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_train_curve(path: Path, sample_limit: int = 400) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "epoch": _parse_int(row.get("epoch")),
                    "step": _parse_int(row.get("step")),
                    "loss_total": _parse_float(row.get("loss_total")),
                    "lr": _parse_float(row.get("lr")),
                }
            )

    if len(rows) <= sample_limit:
        return rows

    stride = max(1, len(rows) // sample_limit)
    sampled = rows[::stride]
    if sampled[-1] != rows[-1]:
        sampled.append(rows[-1])
    return sampled


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
