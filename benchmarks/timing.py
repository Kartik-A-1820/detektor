"""Timing and statistics primitives shared by every suite."""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch


def percentile(values: Sequence[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if len(values) else 0.0


def summarize(latencies_ms: Sequence[float]) -> Dict[str, Any]:
    """Summarise per-iteration latencies (ms) into robust statistics."""
    if not len(latencies_ms):
        return {"n": 0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "n": int(arr.size),
        "mean_ms": round(mean, 3),
        "std_ms": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
        "min_ms": round(float(arr.min()), 3),
        "p50_ms": round(percentile(arr, 50), 3),
        "p90_ms": round(percentile(arr, 90), 3),
        "p95_ms": round(percentile(arr, 95), 3),
        "p99_ms": round(percentile(arr, 99), 3),
        "max_ms": round(float(arr.max()), 3),
        "fps": round(1000.0 / mean, 2) if mean > 0 else 0.0,
    }


def sync(device: torch.device | str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device | str = "cpu",
    warmup: int = 3,
    runs: int = 20,
) -> list[float]:
    """Time ``fn`` ``runs`` times after ``warmup`` untimed calls; returns latencies in ms."""
    for _ in range(max(0, warmup)):
        fn()
    sync(device)
    gc.collect()
    out: list[float] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        fn()
        sync(device)
        out.append((time.perf_counter() - start) * 1000.0)
    return out


class PeakMemory:
    """Context manager measuring peak memory while its body runs.

    CUDA: peak allocated bytes via ``torch.cuda.max_memory_allocated``.
    CPU: peak process RSS above the baseline, sampled every few milliseconds.
    """

    def __init__(self, device: torch.device | str = "cpu", interval_s: float = 0.004) -> None:
        self.device = str(device)
        self.interval_s = interval_s
        self.peak_mb: Optional[float] = None
        self.baseline_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_rss = 0

    @staticmethod
    def _rss_bytes() -> int:
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:  # noqa: BLE001
            try:
                with open("/proc/self/statm", encoding="utf-8") as handle:
                    return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except Exception:  # noqa: BLE001
                return 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, self._rss_bytes())
            time.sleep(self.interval_s)

    def __enter__(self) -> "PeakMemory":
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_mb = torch.cuda.memory_allocated() / 1024**2
        else:
            gc.collect()
            base = self._rss_bytes()
            self.baseline_mb = base / 1024**2
            self._peak_rss = base
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._peak_rss = max(self._peak_rss, self._rss_bytes())
            self.peak_mb = self._peak_rss / 1024**2

    @property
    def delta_mb(self) -> Optional[float]:
        if self.peak_mb is None or self.baseline_mb is None:
            return None
        return max(0.0, self.peak_mb - self.baseline_mb)
