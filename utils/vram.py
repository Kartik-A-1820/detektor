from __future__ import annotations

import torch


def _resolve_effective_vram_fraction(target_free_fraction: float) -> float:
    if not torch.cuda.is_available():
        return float(target_free_fraction)

    requested = min(max(float(target_free_fraction), 0.0), 1.0)
    if requested <= 0.0:
        return 0.0

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return requested
        # Apply the requested fraction against currently free VRAM, not total VRAM.
        return min(requested, max((free_bytes / total_bytes) * requested, 0.05))
    except Exception:
        return requested


def set_vram_cap(fraction: float = 0.95) -> float:
    if not torch.cuda.is_available():
        return float(fraction)
    effective_fraction = _resolve_effective_vram_fraction(fraction)
    torch.cuda.set_per_process_memory_fraction(effective_fraction)
    return effective_fraction


def vram_report(prefix: str = "") -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    alloc = torch.cuda.memory_allocated() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    try:
        free, _ = torch.cuda.mem_get_info()
        free_mb = free / 1024**2
        return f"{prefix}Allocated: {alloc:.1f}MB / {total:.1f}MB, Free: {free_mb:.1f}MB"
    except Exception:
        return f"{prefix}Allocated: {alloc:.1f}MB / {total:.1f}MB"
