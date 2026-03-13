from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _normalize_split_root(split_path: str | None, base_dir: Path) -> str:
    """Normalize a dataset split path to the split root expected by the internal loader."""
    if not split_path:
        return ""
    path = Path(split_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    normalized_name = path.name.lower()
    if normalized_name == "images":
        return str(path.parent)
    return str(path)


def load_dataset_yaml(data_yaml_path: str) -> Dict[str, Any]:
    """Load and normalize a YOLO-style dataset YAML file for in-memory config overrides."""
    dataset_yaml_path = Path(data_yaml_path).expanduser().resolve()
    with dataset_yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    base_dir = dataset_yaml_path.parent

    names = payload.get("names")
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items())]
    elif isinstance(names, tuple):
        names = list(names)
    if isinstance(names, str):
        names = [names]
    elif names is not None and not isinstance(names, list):
        names = list(names)

    normalized = {
        "data_yaml_path": str(dataset_yaml_path),
        "format": "yolo",
        "train": _normalize_split_root(payload.get("train"), base_dir=base_dir),
        "val": _normalize_split_root(payload.get("val"), base_dir=base_dir),
        "test": _normalize_split_root(payload.get("test"), base_dir=base_dir),
        "num_classes": int(payload.get("nc", 0)),
        "class_names": names,
    }
    return normalized


def apply_dataset_yaml_overrides(cfg: Dict[str, Any], data_yaml_path: str) -> Dict[str, Any]:
    """Apply normalized dataset YAML values onto an existing training or validation config."""
    normalized = load_dataset_yaml(data_yaml_path)
    data_cfg = cfg.setdefault("data", {})
    data_cfg["format"] = normalized["format"]
    if normalized["train"]:
        data_cfg["train"] = normalized["train"]
    if normalized["val"]:
        data_cfg["val"] = normalized["val"]
    if normalized["num_classes"] > 0:
        data_cfg["num_classes"] = normalized["num_classes"]
    if normalized["class_names"] is not None:
        data_cfg["names"] = normalized["class_names"]
    return normalized


def print_resolved_dataset_config(normalized: Dict[str, Any]) -> None:
    """Print the resolved dataset settings for visibility in CLI workflows."""
    print(f"dataset_yaml: {normalized['data_yaml_path']}")
    print(f"resolved_train_root: {normalized['train'] or '<unchanged>'}")
    print(f"resolved_val_root: {normalized['val'] or '<unchanged>'}")
    print(f"num_classes: {normalized['num_classes']}")
    if normalized.get("class_names") is not None:
        print(f"class_names: {normalized['class_names']}")
