from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from utils.data_config import apply_dataset_yaml_overrides


DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "device": "auto",
    "seed": 42,
    "deterministic": False,
    "data": {
        "format": "auto",
        "train": "data/train",
        "val": "data/val",
        "coco_train_json": "",
        "coco_val_json": "",
        "images_dir_train": "",
        "images_dir_val": "",
        "num_classes": 1,
        "names": [],
    },
    "train": {
        "img_size": 512,
        "epochs": 5,
        "batch_size": 8,
        "num_workers": 2,
        "lr": 0.002,
        "weight_decay": 0.05,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_epochs": 3,
        "momentum": 0.937,
        "amp": False,
        "grad_accum": 1,
        "vram_cap": 0.80,
        "conf_thresh": 0.25,
        "iou_thresh": 0.6,
        "auto_tune": True,
    },
    "augment": {
        "enabled": True,
        "hsv_h": 0.015,
        "hsv_s": 0.70,
        "hsv_v": 0.40,
        "fliplr": 0.50,
        "flipud": 0.05,
        "translate": 0.10,
        "scale": 0.30,
    },
    "loss": {
        "cls": "bce",
        "box": "ciou",
        "obj": "bce",
        "seg": "bce_dice",
    },
    "model": {
        "proto_k": 24,
    },
    "logging": {
        "out_dir": "runs/chimera_auto",
    },
}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _next_multiple_of_32(value: int) -> int:
    return int(math.ceil(value / 32.0) * 32)


def _safe_yaml_name(path: str | None) -> str:
    if not path:
        return "dataset"
    return Path(path).resolve().stem.replace(" ", "_")


def _build_auto_out_dir(data_yaml: str | None, img_size: int, total_vram_gb: float, device_name: str) -> str:
    dataset_name = _safe_yaml_name(data_yaml)
    vram_label = f"{max(total_vram_gb, 0.0):.1f}".replace(".", "p")
    device_label = "cpu" if device_name == "cpu" else "cuda"
    return str(Path("runs") / f"{dataset_name}_{device_label}_{vram_label}gb_{img_size}")


def detect_hardware_profile() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "gpu_name": "cpu",
            "total_vram_gb": 0.0,
            "total_vram_bytes": 0,
            "supports_amp": False,
            "supports_bf16": False,
        }

    props = torch.cuda.get_device_properties(0)
    major, minor = props.major, props.minor
    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return {
        "device": "cuda",
        "gpu_name": props.name,
        "total_vram_gb": props.total_memory / float(1024 ** 3),
        "total_vram_bytes": int(props.total_memory),
        "compute_capability": f"{major}.{minor}",
        "supports_amp": major >= 7,
        "supports_bf16": supports_bf16,
    }


def _recommend_train_settings(total_vram_gb: float, cpu_count: int) -> Dict[str, Any]:
    if total_vram_gb <= 0:
        return {
            "img_size": 384,
            "batch_size": 2,
            "grad_accum": 4,
            "num_workers": min(max(cpu_count // 2, 0), 2),
            "amp": False,
        }
    if total_vram_gb <= 1.5:
        return {
            "img_size": 320,
            "batch_size": 1,
            "grad_accum": 8,
            "num_workers": 0,
            "amp": False,
        }
    if total_vram_gb <= 2.0:
        return {
            "img_size": 384,
            "batch_size": 2,
            "grad_accum": 4,
            "num_workers": min(max(cpu_count // 2, 0), 2),
            "amp": False,
        }
    if total_vram_gb <= 3.0:
        return {
            "img_size": 416,
            "batch_size": 2,
            "grad_accum": 3,
            "num_workers": min(max(cpu_count // 2, 1), 3),
            "amp": False,
        }
    if total_vram_gb <= 4.0:
        return {
            "img_size": 512,
            "batch_size": 4,
            "grad_accum": 2,
            "num_workers": min(max(cpu_count // 2, 1), 4),
            "amp": False,
        }
    return {
        "img_size": 640,
        "batch_size": 8,
        "grad_accum": 1,
        "num_workers": min(max(cpu_count // 2, 2), 6),
        "amp": False,
    }


def _recommend_augmentation(dataset_size: int, task_names: list[str] | None = None) -> Dict[str, Any]:
    small_dataset = dataset_size < 1500
    names = task_names or []
    likely_small_objects = any("ball" in name.lower() or "helmet" in name.lower() for name in names)
    scale = 0.35 if small_dataset else 0.25
    translate = 0.12 if small_dataset else 0.08
    flipud = 0.02 if likely_small_objects else 0.05
    return {
        "enabled": True,
        "hsv_h": 0.015,
        "hsv_s": 0.70 if small_dataset else 0.50,
        "hsv_v": 0.40 if small_dataset else 0.30,
        "fliplr": 0.50,
        "flipud": flipud,
        "translate": translate,
        "scale": scale,
    }


def _count_split_images(split_root: str | None) -> int:
    if not split_root:
        return 0
    image_dir = Path(split_root) / "images"
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.is_file())


def resolve_training_config(config_path: str | None, data_yaml: str | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded_cfg = yaml.safe_load(handle) or {}
        cfg = _deep_merge(cfg, loaded_cfg)

    dataset_info: Dict[str, Any] = {}
    if data_yaml:
        dataset_info = apply_dataset_yaml_overrides(cfg, data_yaml)

    profile = detect_hardware_profile()
    requested_device = cfg.get("device", "auto")
    resolved_device = profile["device"] if requested_device == "auto" else requested_device
    if resolved_device == "cuda" and profile["device"] != "cuda":
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    cfg["device"] = resolved_device

    cpu_count = os.cpu_count() or 2
    auto_tune_enabled = bool(cfg.get("train", {}).get("auto_tune", True))
    if auto_tune_enabled:
        suggested = _recommend_train_settings(
            total_vram_gb=profile["total_vram_gb"] if resolved_device == "cuda" else 0.0,
            cpu_count=cpu_count,
        )
        cfg["train"].update(suggested)

        class_names = cfg.get("data", {}).get("names") or []
        if isinstance(class_names, dict):
            class_names = [name for _, name in sorted(class_names.items())]
        dataset_size = _count_split_images(cfg.get("data", {}).get("train"))
        cfg["augment"] = _deep_merge(cfg.get("augment", {}), _recommend_augmentation(dataset_size, class_names))

        effective_batch = int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum"])
        base_lr = 0.002
        scaled_lr = base_lr * max(effective_batch, 1) / 8.0
        cfg["train"]["lr"] = round(min(max(scaled_lr, 0.0005), 0.004), 6)

        if "out_dir" not in cfg.get("logging", {}) or not config_path:
            cfg.setdefault("logging", {})
            cfg["logging"]["out_dir"] = _build_auto_out_dir(
                data_yaml=data_yaml,
                img_size=int(cfg["train"]["img_size"]),
                total_vram_gb=profile["total_vram_gb"] if resolved_device == "cuda" else 0.0,
                device_name=resolved_device,
            )

    cfg["train"]["img_size"] = _next_multiple_of_32(int(cfg["train"]["img_size"]))
    cfg["train"]["num_workers"] = max(int(cfg["train"].get("num_workers", 0)), 0)

    resolution = {
        "device": resolved_device,
        "gpu_name": profile["gpu_name"],
        "total_vram_gb": round(profile["total_vram_gb"], 2),
        "auto_tune_enabled": auto_tune_enabled,
        "batch_size": int(cfg["train"]["batch_size"]),
        "grad_accum": int(cfg["train"]["grad_accum"]),
        "effective_batch_size": int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum"]),
        "img_size": int(cfg["train"]["img_size"]),
        "amp": bool(cfg["train"].get("amp", False)),
        "num_workers": int(cfg["train"].get("num_workers", 0)),
        "lr": float(cfg["train"]["lr"]),
        "out_dir": cfg.get("logging", {}).get("out_dir", ""),
        "dataset_yaml": dataset_info.get("data_yaml_path", data_yaml or ""),
        "resolved_train_root": cfg.get("data", {}).get("train", ""),
        "resolved_val_root": cfg.get("data", {}).get("val", ""),
        "num_classes": int(cfg.get("data", {}).get("num_classes", 0)),
        "augment": copy.deepcopy(cfg.get("augment", {})),
    }
    return cfg, resolution


def write_resolved_config(cfg: Dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return output
