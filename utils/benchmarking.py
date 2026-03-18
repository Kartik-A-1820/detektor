"""Benchmark utilities for Detektor inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from api.utils import preprocess_image_bytes

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BenchmarkSample:
    """Preprocessed image sample for benchmarking across backends."""

    path: Path
    image_bytes: bytes
    torch_tensor: torch.Tensor
    numpy_array: np.ndarray
    original_size: Tuple[int, int]


def load_benchmark_samples(
    source: str | Path,
    image_size: int = 512,
    max_images: Optional[int] = None,
) -> List[BenchmarkSample]:
    """Load and preprocess images for benchmarking.

    Args:
        source: File or directory containing images
        image_size: Resize target for inference inputs
        max_images: Optional limit on number of images to load

    Returns:
        List of BenchmarkSample objects with tensors ready for inference
    """

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Benchmark source path not found: {source_path}")

    if source_path.is_dir():
        candidates = sorted(
            [p for p in source_path.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        )
    else:
        candidates = [source_path]

    if not candidates:
        raise FileNotFoundError(
            f"No images found in {source_path}. Supported extensions: {sorted(SUPPORTED_IMAGE_EXTS)}"
        )

    if max_images is not None:
        candidates = candidates[: max_images]

    samples: List[BenchmarkSample] = []
    for image_path in candidates:
        image_bytes = image_path.read_bytes()
        tensor, _, original_size = preprocess_image_bytes(image_bytes, image_size=image_size)
        tensor = tensor.contiguous()  # ensure contiguous for .numpy()
        numpy_array = tensor.numpy()
        samples.append(
            BenchmarkSample(
                path=image_path,
                image_bytes=image_bytes,
                torch_tensor=tensor,
                numpy_array=numpy_array,
                original_size=original_size,
            )
        )

    return samples


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "throughput_img_s": 0.0,
            "measurement_time_ms": 0.0,
        }

    total_time_ms = float(sum(latencies_ms))
    count = len(latencies_ms)
    avg_ms = total_time_ms / count
    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    throughput = count / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0

    return {
        "avg_latency_ms": avg_ms,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "throughput_img_s": throughput,
        "measurement_time_ms": total_time_ms,
    }


__all__ = [
    "BenchmarkSample",
    "compute_latency_stats",
    "load_benchmark_samples",
    "percentile",
]
