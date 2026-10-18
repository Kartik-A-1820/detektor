from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _normalize_split_root(split_path: str | None, base_dir: Path) -> str:
    """Normalize a dataset split path against ``base_dir``.

    Two on-disk layouts are supported (see :func:`resolve_split_dirs`):

    * Roboflow style: ``<split>/images`` + ``<split>/labels`` - the *split root* is returned
      (a trailing ``images`` component is stripped).
    * YOLOv5/v8 style: ``images/<split>`` + ``labels/<split>`` - the images directory is returned as is.
    """
    if not split_path or not isinstance(split_path, (str, Path)):
        return ""
    path = Path(split_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if path.name.lower() == "images":
        return str(path.parent)
    return str(path)


def resolve_split_dirs(split_root: str | Path) -> Tuple[Path, Path]:
    """Return ``(images_dir, labels_dir)`` for a dataset split.

    * ``<root>/images`` exists -> Roboflow layout: ``(<root>/images, <root>/labels)``.
    * otherwise ``root`` is itself an images directory (YOLOv5/v8 layout, e.g. ``.../images/train``) and the
      labels directory is found by swapping the last ``images`` path component for ``labels``
      (``.../labels/train``), falling back to a sibling ``labels`` directory.
    """
    root = Path(split_root)
    if (root / "images").is_dir():
        return root / "images", root / "labels"
    parts = root.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            return root, Path(*parts[:index], "labels", *parts[index + 1:])
    return root, root.parent / "labels"


def _resolve_dataset_base(yaml_dir: Path, payload: Dict[str, Any]) -> Path:
    """Directory that relative split paths are relative to.

    Honors the optional YOLO ``path:`` key. Ultralytics-style YAMLs often name the dataset folder itself
    (``path: brain-tumor`` sitting *inside* that folder), so candidates are checked on disk.
    """
    raw = payload.get("path")
    if not raw:
        return yaml_dir
    declared = Path(str(raw)).expanduser()
    candidates = [declared] if declared.is_absolute() else [(yaml_dir / declared).resolve(), yaml_dir]
    for candidate in candidates:
        for key in ("train", "val"):
            split = payload.get(key)
            if isinstance(split, (str, Path)) and (candidate / split).exists():
                return candidate
    return candidates[0]


def load_dataset_yaml(data_yaml_path: str) -> Dict[str, Any]:
    """Load and normalize a YOLO-style dataset YAML file for in-memory config overrides."""
    dataset_yaml_path = Path(data_yaml_path).expanduser().resolve()
    with dataset_yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    base_dir = _resolve_dataset_base(dataset_yaml_path.parent, payload)

    names = payload.get("names")
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items())]
    elif isinstance(names, tuple):
        names = list(names)
    if isinstance(names, str):
        names = [names]
    elif names is not None and not isinstance(names, list):
        names = list(names)

    num_classes = int(payload.get("nc", 0) or 0)
    if num_classes <= 0 and names:
        num_classes = len(names)  # Ultralytics-style YAMLs list `names` only

    normalized = {
        "data_yaml_path": str(dataset_yaml_path),
        "format": "yolo",
        "train": _normalize_split_root(payload.get("train"), base_dir=base_dir),
        "val": _normalize_split_root(payload.get("val"), base_dir=base_dir),
        "test": _normalize_split_root(payload.get("test"), base_dir=base_dir),
        "num_classes": num_classes,
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
