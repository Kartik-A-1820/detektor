from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Sequence

import torch

from models.factory import ARCHITECTURE_PROFILES, build_model_from_model_config
from utils.auto_train_config import detect_hardware_profile, resolve_training_config
from utils.model_info import get_model_info


def _synthetic_targets(batch_size: int, num_classes: int) -> List[Dict[str, torch.Tensor]]:
    targets: List[Dict[str, torch.Tensor]] = []
    for batch_index in range(batch_size):
        label = batch_index % max(num_classes, 1)
        targets.append(
            {
                "boxes": torch.tensor([[0.20, 0.20, 0.70, 0.75]], dtype=torch.float32),
                "labels": torch.tensor([label], dtype=torch.long),
            }
        )
    return targets


def _compatibility_label(max_batch_size: int) -> str:
    if max_batch_size >= 16:
        return "strong"
    if max_batch_size >= 4:
        return "supported"
    if max_batch_size >= 1:
        return "limited"
    return "unsupported"


def _probe_candidates(device_name: str, batch_multiple: int, max_batch_probe: int, cpu_max_batch_probe: int) -> List[int]:
    probe_limit = max_batch_probe if device_name == "cuda" else cpu_max_batch_probe
    candidates = [1]
    start = max(batch_multiple, 1)
    for batch_size in range(start, max(probe_limit, start) + 1, max(batch_multiple, 1)):
        if batch_size not in candidates:
            candidates.append(batch_size)
    return candidates


def probe_profile_training_capacity(
    profile_key: str,
    *,
    device_name: str,
    num_classes: int,
    img_size: int = 512,
    proto_k: int | None = None,
    batch_multiple: int = 4,
    max_batch_probe: int = 64,
    cpu_max_batch_probe: int = 16,
) -> Dict[str, Any]:
    if profile_key not in ARCHITECTURE_PROFILES:
        raise ValueError(f"Unknown architecture profile '{profile_key}'")
    if device_name == "cuda" and not torch.cuda.is_available():
        return {
            "profile": profile_key,
            "device": device_name,
            "compatible": False,
            "compatibility": "unsupported",
            "max_batch_size": 0,
            "error": "CUDA is not available",
        }

    device = torch.device(device_name)
    model_cfg = {"profile": profile_key}
    if proto_k is not None:
        model_cfg["proto_k"] = int(proto_k)
    model = build_model_from_model_config(model_cfg, num_classes=num_classes).to(device)
    info = get_model_info(model)
    total_vram_mb = 0.0
    if device.type == "cuda":
        total_vram_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2

    max_batch_size = 0
    best_peak_memory_mb = 0.0
    last_error = ""
    best_step_time_s = 0.0
    candidates = _probe_candidates(device_name, batch_multiple, max_batch_probe, cpu_max_batch_probe)

    for batch_size in candidates:
        optimizer = None
        try:
            imgs = torch.randn(batch_size, 3, img_size, img_size, device=device)
            targets = _synthetic_targets(batch_size=batch_size, num_classes=num_classes)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            start = time.perf_counter()
            loss = model.compute_loss(imgs, targets, task="detect")
            loss.backward()
            optimizer.step()
            step_time_s = time.perf_counter() - start

            max_batch_size = batch_size
            best_step_time_s = step_time_s
            if device.type == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                best_peak_memory_mb = max(best_peak_memory_mb, min(peak_memory_mb, total_vram_mb))

            del imgs, targets, loss
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except (RuntimeError, MemoryError) as error:
            last_error = str(error)
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if "out of memory" in str(error).lower() or isinstance(error, MemoryError):
                break
            break
        finally:
            del optimizer

    return {
        "profile": profile_key,
        "display_name": ARCHITECTURE_PROFILES[profile_key].display_name,
        "device": device_name,
        "compatible": max_batch_size > 0,
        "compatibility": _compatibility_label(max_batch_size),
        "max_batch_size": int(max_batch_size),
        "peak_memory_mb": round(float(best_peak_memory_mb), 2),
        "step_time_s": round(float(best_step_time_s), 3),
        "model_info": info,
        "error": last_error,
    }


def collect_compatibility_matrix(
    *,
    data_yaml: str | None = None,
    profiles: Sequence[str] | None = None,
    img_size: int = 512,
    batch_multiple: int = 4,
    max_batch_probe: int = 64,
    cpu_max_batch_probe: int = 16,
) -> Dict[str, Any]:
    selected_profiles = list(profiles or ARCHITECTURE_PROFILES.keys())
    base_cfg, summary = resolve_training_config(None, data_yaml, overrides={"train": {"img_size": img_size}})
    hardware = detect_hardware_profile()
    num_classes = int(base_cfg.get("data", {}).get("num_classes", 1))
    available_devices = ["cpu"]
    if hardware.get("device") == "cuda":
        available_devices.append("cuda")

    results: Dict[str, List[Dict[str, Any]]] = {device_name: [] for device_name in available_devices}
    for device_name in available_devices:
        for profile_key in selected_profiles:
            results[device_name].append(
                probe_profile_training_capacity(
                    profile_key,
                    device_name=device_name,
                    num_classes=num_classes,
                    img_size=img_size,
                    batch_multiple=batch_multiple,
                    max_batch_probe=max_batch_probe,
                    cpu_max_batch_probe=cpu_max_batch_probe,
                )
            )

    return {
        "hardware": hardware,
        "resolved_training": summary,
        "recommended_profile": summary.get("model_profile", ""),
        "profiles": selected_profiles,
        "results": results,
    }


def iter_compatible_profiles(matrix: Dict[str, Any], device_name: str) -> Iterable[Dict[str, Any]]:
    for entry in matrix.get("results", {}).get(device_name, []):
        if entry.get("compatible"):
            yield entry
