from __future__ import annotations

import copy
import ctypes
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import yaml

from models.factory import ARCHITECTURE_PROFILES
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
        "epochs": 20,
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
        "vram_cap": 0.95,
        "conf_thresh": 0.25,
        "iou_thresh": 0.6,
        "auto_tune": True,
        "maximize_batch_size": True,
        "batch_size_multiple": 4,
        "max_batch_probe": 64,
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
        "mosaic": 0.0,
        "cutmix": 0.0,
        "random_cut": 0.0,
        "random_cut_holes": 1,
        "random_cut_scale": 0.25,
    },
    "loss": {
        "cls": "bce",
        "box": "ciou",
        "obj": "bce",
        "seg": "bce_dice",
    },
    "model": {
        "profile": "quasar",
        "display_name": "Quasar",
        "proto_k": 24,
    },
    "logging": {
        "out_dir": "runs/chimera_auto",
    },
    "smart_training": {
        "enabled": True,
        "max_attempts": 4,
        "min_batch_size": 1,
        "min_img_size": 256,
        "min_lr": 0.00025,
        "non_finite_patience": 2,
        "base_out_dir": "",
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


def build_attempt_out_dir(base_out_dir: str | Path, attempt_index: int) -> str:
    base_path = Path(base_out_dir)
    if attempt_index <= 1:
        return str(base_path)
    return str(base_path.parent / f"{base_path.name}_retry{attempt_index:02d}")


def _dataset_size_band(dataset_size: int) -> str:
    if dataset_size <= 300:
        return "micro"
    if dataset_size <= 1200:
        return "small"
    if dataset_size <= 6000:
        return "medium"
    return "large"


def detect_hardware_profile() -> Dict[str, Any]:
    cpu_count = os.cpu_count() or 2
    free_ram_bytes, total_ram_bytes = detect_system_memory()
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "gpu_name": "cpu",
            "total_vram_gb": 0.0,
            "total_vram_bytes": 0,
            "free_vram_gb": 0.0,
            "free_vram_bytes": 0,
            "free_ram_gb": round(free_ram_bytes / float(1024 ** 3), 2),
            "free_ram_bytes": int(free_ram_bytes),
            "total_ram_gb": round(total_ram_bytes / float(1024 ** 3), 2),
            "total_ram_bytes": int(total_ram_bytes),
            "cpu_count": int(cpu_count),
            "supports_amp": False,
            "supports_bf16": False,
        }

    props = torch.cuda.get_device_properties(0)
    major, minor = props.major, props.minor
    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    free_vram_bytes = 0
    try:
        free_vram_bytes, _ = torch.cuda.mem_get_info()
    except Exception:
        free_vram_bytes = int(props.total_memory)
    return {
        "device": "cuda",
        "gpu_name": props.name,
        "total_vram_gb": props.total_memory / float(1024 ** 3),
        "total_vram_bytes": int(props.total_memory),
        "free_vram_gb": free_vram_bytes / float(1024 ** 3),
        "free_vram_bytes": int(free_vram_bytes),
        "free_ram_gb": round(free_ram_bytes / float(1024 ** 3), 2),
        "free_ram_bytes": int(free_ram_bytes),
        "total_ram_gb": round(total_ram_bytes / float(1024 ** 3), 2),
        "total_ram_bytes": int(total_ram_bytes),
        "cpu_count": int(cpu_count),
        "compute_capability": f"{major}.{minor}",
        "supports_amp": major >= 7,
        "supports_bf16": supports_bf16,
    }


def detect_system_memory() -> tuple[int, int]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    try:
        kernel32 = ctypes.windll.kernel32
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys), int(status.ullTotalPhys)
    except Exception:
        pass
    return 0, 0


def _recommend_cpu_train_settings(free_ram_gb: float, cpu_count: int) -> Dict[str, Any]:
    worker_budget = max(cpu_count - 1, 0)
    if free_ram_gb <= 4.0:
        return {
            "img_size": 384,
            "batch_size": 2,
            "grad_accum": 4,
            "num_workers": min(worker_budget, 2),
            "amp": False,
            "device": "cpu",
        }
    if free_ram_gb <= 8.0:
        return {
            "img_size": 512,
            "batch_size": 4,
            "grad_accum": 2,
            "num_workers": min(worker_budget, 4),
            "amp": False,
            "device": "cpu",
        }
    if free_ram_gb <= 16.0:
        return {
            "img_size": 512,
            "batch_size": 8,
            "grad_accum": 1,
            "num_workers": min(worker_budget, 6),
            "amp": False,
            "device": "cpu",
        }
    return {
        "img_size": 640,
        "batch_size": 12,
        "grad_accum": 1,
        "num_workers": min(worker_budget, 8),
        "amp": False,
        "device": "cpu",
    }


def _recommend_train_settings(total_vram_gb: float, free_vram_gb: float, free_ram_gb: float, cpu_count: int) -> Dict[str, Any]:
    if total_vram_gb <= 0:
        return _recommend_cpu_train_settings(free_ram_gb=free_ram_gb, cpu_count=cpu_count)

    usable_vram_gb = max(min(total_vram_gb, free_vram_gb * 0.95), 0.0)
    if usable_vram_gb <= 0.5:
        return _recommend_cpu_train_settings(free_ram_gb=free_ram_gb, cpu_count=cpu_count)

    usable_vram_mb = int(usable_vram_gb * 1024)
    if usable_vram_mb < 512:
        return _recommend_cpu_train_settings(free_ram_gb=free_ram_gb, cpu_count=cpu_count)
    if usable_vram_gb <= 1.5:
        return {
            "img_size": 320,
            "batch_size": 1,
            "grad_accum": 8,
            "num_workers": 0,
            "amp": False,
            "device": "cuda",
        }
    if usable_vram_gb <= 2.0:
        return {
            "img_size": 384,
            "batch_size": 2,
            "grad_accum": 4,
            "num_workers": min(max(cpu_count // 2, 0), 2),
            "amp": False,
            "device": "cuda",
        }
    if usable_vram_gb <= 3.0:
        return {
            "img_size": 416,
            "batch_size": 2,
            "grad_accum": 3,
            "num_workers": min(max(cpu_count // 2, 1), 3),
            "amp": False,
            "device": "cuda",
        }
    if usable_vram_gb <= 4.0:
        return {
            "img_size": 512,
            "batch_size": 16,
            "grad_accum": 1,
            "num_workers": 0,
            "amp": False,
            "device": "cuda",
        }
        return {
            "img_size": 640,
            "batch_size": 8,
            "grad_accum": 1,
            "num_workers": min(max(cpu_count // 2, 2), 6),
            "amp": False,
            "device": "cuda",
        }


def _recommend_augmentation(dataset_size: int, task_names: list[str] | None = None) -> Dict[str, Any]:
    small_dataset = dataset_size < 1500
    names = task_names or []
    likely_small_objects = any("ball" in name.lower() or "helmet" in name.lower() for name in names)
    scale = 0.35 if small_dataset else 0.25
    translate = 0.12 if small_dataset else 0.08
    flipud = 0.02 if likely_small_objects else 0.05
    mosaic = 0.85 if dataset_size < 3000 else 0.35
    cutmix = 0.20 if dataset_size < 3000 else 0.10
    random_cut = 0.40 if small_dataset else 0.20
    return {
        "enabled": True,
        "hsv_h": 0.015,
        "hsv_s": 0.70 if small_dataset else 0.50,
        "hsv_v": 0.40 if small_dataset else 0.30,
        "fliplr": 0.50,
        "flipud": flipud,
        "translate": translate,
        "scale": scale,
        "mosaic": mosaic,
        "cutmix": cutmix,
        "random_cut": random_cut,
        "random_cut_holes": 2 if small_dataset else 1,
        "random_cut_scale": 0.30 if small_dataset else 0.20,
    }


def _count_split_images(split_root: str | None) -> int:
    if not split_root:
        return 0
    image_dir = Path(split_root) / "images"
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.is_file())


def _recommend_architecture_profile(device_name: str, total_vram_gb: float, free_vram_gb: float, dataset_size: int) -> Dict[str, Any]:
    ordered_profiles = ["firefly", "comet", "nova", "pulsar", "quasar", "supernova"]

    if device_name != "cuda":
        hardware_index = 0
    else:
        total_vram_mb = int(max(min(total_vram_gb, free_vram_gb * 0.95), 0.0) * 1024)
        if total_vram_mb < 512:
            hardware_index = 0
        elif total_vram_mb < 1024:
            hardware_index = 1
        elif total_vram_mb < 2048:
            hardware_index = 2
        elif total_vram_mb < 4096:
            hardware_index = 3
        elif total_vram_mb < 6144:
            hardware_index = 4
        else:
            hardware_index = 5

    size_band = _dataset_size_band(dataset_size)
    dataset_bias = {
        "micro": -2,
        "small": -1,
        "medium": 0,
        "large": 0,
    }[size_band]
    if dataset_size >= 12000 and hardware_index < len(ordered_profiles) - 1:
        dataset_bias = 1

    resolved_index = min(max(hardware_index + dataset_bias, 0), hardware_index)
    profile_key = ordered_profiles[resolved_index]
    profile = ARCHITECTURE_PROFILES[profile_key]
    return {
        "profile": profile_key,
        "display_name": profile.display_name,
        "stem_channels": profile.stem_channels,
        "backbone_channels": list(profile.backbone_channels),
        "backbone_depths": list(profile.backbone_depths),
        "neck_channels": list(profile.neck_channels),
        "head_feat_channels": profile.head_feat_channels,
        "proto_k": profile.proto_k,
        "dataset_size_band": size_band,
        "hardware_profile_index": hardware_index,
    }


def _recommend_epoch_count(dataset_size: int, device_name: str, total_vram_gb: float) -> int:
    size_band = _dataset_size_band(dataset_size)
    if device_name == "cpu":
        base_epochs = {
            "micro": 24,
            "small": 20,
            "medium": 16,
            "large": 12,
        }[size_band]
    else:
        base_epochs = {
            "micro": 36,
            "small": 30,
            "medium": 24,
            "large": 18,
        }[size_band]
        if total_vram_gb >= 6.0 and size_band in {"micro", "small"}:
            base_epochs += 4
    return int(base_epochs)


def resolve_training_config(
    config_path: str | None,
    data_yaml: str | None,
    overrides: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    explicit_train_cfg: Dict[str, Any] = {}
    explicit_augment_cfg: Dict[str, Any] = {}
    explicit_model_cfg: Dict[str, Any] = {}
    explicit_logging_cfg: Dict[str, Any] = {}
    explicit_root_cfg: Dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded_cfg = yaml.safe_load(handle) or {}
        explicit_root_cfg = {
            key: copy.deepcopy(value)
            for key, value in loaded_cfg.items()
            if key not in {"train", "augment", "model", "logging"}
        }
        explicit_train_cfg = copy.deepcopy(loaded_cfg.get("train", {}))
        explicit_augment_cfg = copy.deepcopy(loaded_cfg.get("augment", {}))
        explicit_model_cfg = copy.deepcopy(loaded_cfg.get("model", {}))
        explicit_logging_cfg = copy.deepcopy(loaded_cfg.get("logging", {}))
        cfg = _deep_merge(cfg, loaded_cfg)
    if overrides:
        overrides_dict = copy.deepcopy(dict(overrides))
        explicit_root_cfg = _deep_merge(
            explicit_root_cfg,
            {key: value for key, value in overrides_dict.items() if key not in {"train", "augment", "model", "logging"}},
        )
        explicit_train_cfg = _deep_merge(explicit_train_cfg, overrides_dict.get("train", {}))
        explicit_augment_cfg = _deep_merge(explicit_augment_cfg, overrides_dict.get("augment", {}))
        explicit_model_cfg = _deep_merge(explicit_model_cfg, overrides_dict.get("model", {}))
        explicit_logging_cfg = _deep_merge(explicit_logging_cfg, overrides_dict.get("logging", {}))
        cfg = _deep_merge(cfg, overrides_dict)

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
            free_vram_gb=profile["free_vram_gb"] if resolved_device == "cuda" else 0.0,
            free_ram_gb=profile["free_ram_gb"],
            cpu_count=cpu_count,
        )
        cfg["train"].update(suggested)
        if suggested.get("device") == "cpu":
            cfg["device"] = "cpu"
            resolved_device = "cpu"

        class_names = cfg.get("data", {}).get("names") or []
        if isinstance(class_names, dict):
            class_names = [name for _, name in sorted(class_names.items())]
        dataset_size = _count_split_images(cfg.get("data", {}).get("train"))
        cfg["augment"] = _deep_merge(cfg.get("augment", {}), _recommend_augmentation(dataset_size, class_names))
        cfg["model"] = _deep_merge(
            cfg.get("model", {}),
            _recommend_architecture_profile(
                device_name=resolved_device,
                total_vram_gb=profile["total_vram_gb"] if resolved_device == "cuda" else 0.0,
                free_vram_gb=profile["free_vram_gb"] if resolved_device == "cuda" else 0.0,
                dataset_size=dataset_size,
            ),
        )
        cfg["train"]["epochs"] = _recommend_epoch_count(
            dataset_size=dataset_size,
            device_name=resolved_device,
            total_vram_gb=profile["total_vram_gb"] if resolved_device == "cuda" else 0.0,
        )
        cfg["train"]["warmup_epochs"] = min(
            max(1, int(cfg["train"].get("warmup_epochs", 3))),
            max(int(cfg["train"]["epochs"]) // 3, 1),
        )

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

    if explicit_root_cfg:
        cfg = _deep_merge(cfg, explicit_root_cfg)
    if explicit_train_cfg:
        cfg["train"] = _deep_merge(cfg.get("train", {}), explicit_train_cfg)
    if explicit_augment_cfg:
        cfg["augment"] = _deep_merge(cfg.get("augment", {}), explicit_augment_cfg)
    if explicit_model_cfg:
        cfg["model"] = _deep_merge(cfg.get("model", {}), explicit_model_cfg)
    if explicit_logging_cfg:
        cfg["logging"] = _deep_merge(cfg.get("logging", {}), explicit_logging_cfg)

    cfg["train"]["img_size"] = _next_multiple_of_32(int(cfg["train"]["img_size"]))
    cfg["train"]["num_workers"] = max(int(cfg["train"].get("num_workers", 0)), 0)
    cfg["train"]["warmup_epochs"] = min(
        max(1, int(cfg["train"].get("warmup_epochs", 3))),
        max(int(cfg["train"]["epochs"]) // 3, 1),
    )
    cfg.setdefault("smart_training", {})
    cfg["smart_training"] = _deep_merge(DEFAULT_TRAINING_CONFIG["smart_training"], cfg["smart_training"])
    if not cfg["smart_training"].get("base_out_dir"):
        cfg["smart_training"]["base_out_dir"] = cfg.get("logging", {}).get("out_dir", "")
    model_profile_key = str(cfg.get("model", {}).get("profile", "quasar")).lower()
    if model_profile_key in ARCHITECTURE_PROFILES:
        model_profile = ARCHITECTURE_PROFILES[model_profile_key]
        current_model_cfg = copy.deepcopy(cfg.get("model", {}))
        structural_keys = {
            "profile",
            "display_name",
            "stem_channels",
            "backbone_channels",
            "backbone_depths",
            "neck_channels",
            "head_feat_channels",
            "proto_k",
        }
        derived_metadata = {
            key: value
            for key, value in current_model_cfg.items()
            if key not in structural_keys
        }
        explicit_model_overrides = {
            key: value
            for key, value in explicit_model_cfg.items()
            if key in structural_keys
        }
        cfg["model"] = _deep_merge(
            {
                "profile": model_profile.tier_key,
                "display_name": model_profile.display_name,
                "stem_channels": model_profile.stem_channels,
                "backbone_channels": list(model_profile.backbone_channels),
                "backbone_depths": list(model_profile.backbone_depths),
                "neck_channels": list(model_profile.neck_channels),
                "head_feat_channels": model_profile.head_feat_channels,
                "proto_k": model_profile.proto_k,
            },
            derived_metadata,
        )
        if explicit_model_overrides:
            cfg["model"] = _deep_merge(cfg["model"], explicit_model_overrides)

    resolution = {
        "device": resolved_device,
        "gpu_name": profile["gpu_name"],
        "total_vram_gb": round(profile["total_vram_gb"], 2),
        "free_vram_gb": round(profile.get("free_vram_gb", 0.0), 2),
        "free_ram_gb": round(profile.get("free_ram_gb", 0.0), 2),
        "cpu_count": int(profile.get("cpu_count", cpu_count)),
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
        "dataset_size": _count_split_images(cfg.get("data", {}).get("train")),
        "resolved_train_root": cfg.get("data", {}).get("train", ""),
        "resolved_val_root": cfg.get("data", {}).get("val", ""),
        "num_classes": int(cfg.get("data", {}).get("num_classes", 0)),
        "model_profile": str(cfg.get("model", {}).get("profile", "")),
        "model_display_name": str(cfg.get("model", {}).get("display_name", "")),
        "model": copy.deepcopy(cfg.get("model", {})),
        "augment": copy.deepcopy(cfg.get("augment", {})),
        "smart_training": copy.deepcopy(cfg.get("smart_training", {})),
    }
    return cfg, resolution


def summarize_resolved_training(cfg: Dict[str, Any], base_resolution: Dict[str, Any]) -> Dict[str, Any]:
    summary = copy.deepcopy(base_resolution)
    summary.update(
        {
            "device": str(cfg.get("device", summary.get("device", "cpu"))),
            "batch_size": int(cfg["train"]["batch_size"]),
            "grad_accum": int(cfg["train"].get("grad_accum", 1)),
            "effective_batch_size": int(cfg["train"]["batch_size"]) * int(cfg["train"].get("grad_accum", 1)),
            "img_size": int(cfg["train"]["img_size"]),
            "amp": bool(cfg["train"].get("amp", False)),
            "num_workers": int(cfg["train"].get("num_workers", 0)),
            "lr": float(cfg["train"]["lr"]),
            "out_dir": cfg.get("logging", {}).get("out_dir", ""),
            "model_profile": str(cfg.get("model", {}).get("profile", "")),
            "model_display_name": str(cfg.get("model", {}).get("display_name", "")),
            "model": copy.deepcopy(cfg.get("model", {})),
            "augment": copy.deepcopy(cfg.get("augment", {})),
            "smart_training": copy.deepcopy(cfg.get("smart_training", {})),
        }
    )
    return summary


def plan_smart_retry(
    cfg: Dict[str, Any],
    failure_type: str,
    failure_message: str,
    attempt_index: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    policy = cfg.get("smart_training", {})
    if not bool(policy.get("enabled", True)):
        return None

    max_attempts = max(int(policy.get("max_attempts", 1)), 1)
    if attempt_index >= max_attempts:
        return None

    next_cfg = copy.deepcopy(cfg)
    next_policy = next_cfg.setdefault("smart_training", {})
    next_attempt = attempt_index + 1
    changes: list[str] = []
    min_batch_size = max(int(next_policy.get("min_batch_size", 1)), 1)
    min_img_size = max(int(next_policy.get("min_img_size", 256)), 32)
    min_lr = float(next_policy.get("min_lr", 0.00025))

    def reduce_batch_size() -> bool:
        current_batch = int(next_cfg["train"].get("batch_size", 1))
        if current_batch <= min_batch_size:
            return False
        new_batch = max(min_batch_size, current_batch // 2)
        if new_batch == current_batch:
            return False
        accum = max(int(next_cfg["train"].get("grad_accum", 1)), 1)
        scale = max(int(math.ceil(current_batch / max(new_batch, 1))), 1)
        next_cfg["train"]["batch_size"] = new_batch
        next_cfg["train"]["grad_accum"] = accum * scale
        changes.append(f"batch_size {current_batch}->{new_batch}")
        changes.append(f"grad_accum {accum}->{next_cfg['train']['grad_accum']}")
        return True

    def reduce_img_size() -> bool:
        current_img = int(next_cfg["train"].get("img_size", 512))
        if current_img <= min_img_size:
            return False
        proposed = max(min_img_size, current_img - 64)
        next_cfg["train"]["img_size"] = _next_multiple_of_32(proposed)
        if int(next_cfg["train"]["img_size"]) == current_img:
            return False
        changes.append(f"img_size {current_img}->{next_cfg['train']['img_size']}")
        return True

    def disable_amp() -> bool:
        if not bool(next_cfg["train"].get("amp", False)):
            return False
        next_cfg["train"]["amp"] = False
        changes.append("amp true->false")
        return True

    def reduce_lr() -> bool:
        current_lr = float(next_cfg["train"].get("lr", min_lr))
        if current_lr <= min_lr:
            return False
        new_lr = max(min_lr, round(current_lr * 0.5, 6))
        if new_lr == current_lr:
            return False
        next_cfg["train"]["lr"] = new_lr
        changes.append(f"lr {current_lr}->{new_lr}")
        return True

    def soften_augmentation() -> bool:
        augment = next_cfg.setdefault("augment", {})
        adjusted = False
        for key in ("mosaic", "cutmix", "random_cut", "translate", "scale", "hsv_s", "hsv_v"):
            current_value = float(augment.get(key, 0.0))
            new_value = round(current_value * 0.5, 4)
            if new_value != current_value:
                augment[key] = new_value
                changes.append(f"{key} {current_value}->{new_value}")
                adjusted = True
        return adjusted

    changed = False
    if failure_type == "oom":
        next_cfg["train"]["num_workers"] = 0
        changes.append("num_workers ->0")
        changed = reduce_batch_size()
        if not changed:
            changed = reduce_img_size()
    elif failure_type == "amp_instability":
        changed = disable_amp()
        if not changed:
            changed = reduce_lr()
    elif failure_type in {"non_finite_loss", "non_finite_grad"}:
        changed = disable_amp()
        changed = reduce_lr() or changed
        changed = soften_augmentation() or changed
        if failure_type == "non_finite_grad":
            changed = reduce_batch_size() or changed
    else:
        return None

    if not changed:
        return None

    base_out_dir = next_policy.get("base_out_dir") or next_cfg.get("logging", {}).get("out_dir", "")
    next_policy["base_out_dir"] = base_out_dir
    next_policy["last_failure_type"] = failure_type
    next_policy["last_failure_message"] = failure_message
    next_policy["attempt"] = next_attempt
    next_cfg.setdefault("logging", {})
    next_cfg["logging"]["out_dir"] = build_attempt_out_dir(base_out_dir, next_attempt)

    return next_cfg, {
        "failure_type": failure_type,
        "failure_message": failure_message,
        "attempt": attempt_index,
        "next_attempt": next_attempt,
        "changes": changes,
        "out_dir": next_cfg["logging"]["out_dir"],
    }


def write_resolved_config(cfg: Dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return output
