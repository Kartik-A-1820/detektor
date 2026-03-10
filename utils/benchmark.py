from __future__ import annotations

import time
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor, nn


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def benchmark_forward(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: torch.device | None = None,
    warmup: int = 5,
    iterations: int = 20,
) -> Dict[str, Any]:
    """Benchmark average model forward latency and optional CUDA peak memory."""
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.randn(*input_shape, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        _sync_if_cuda(device)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / max(iterations, 1)) * 1000.0
    fps = 1000.0 / max(avg_latency_ms, 1e-6)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "peak_memory_mb": peak_memory_mb,
        "iterations": iterations,
    }


def benchmark_predict(
    model: Any,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: torch.device | None = None,
    warmup: int = 3,
    iterations: int = 10,
) -> Dict[str, Any]:
    """Benchmark average postprocessed prediction latency and optional CUDA peak memory."""
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.randn(*input_shape, device=device)
    original_sizes = [(input_shape[-2], input_shape[-1]) for _ in range(input_shape[0])]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.predict(x, original_sizes=original_sizes)
        _sync_if_cuda(device)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model.predict(x, original_sizes=original_sizes)
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / max(iterations, 1)) * 1000.0
    fps = 1000.0 / max(avg_latency_ms, 1e-6)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "peak_memory_mb": peak_memory_mb,
        "iterations": iterations,
    }
