"""Metrics tracking for production API."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List

import numpy as np


@dataclass
class MetricsStore:
    """Thread-safe metrics storage for API performance tracking."""
    
    total_requests: int = 0
    total_predictions: int = 0
    error_count: int = 0
    inference_times: List[float] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    
    # Keep only last N measurements to avoid unbounded memory growth
    max_history: int = 10000

    # Cumulative latency histogram (Prometheus-style, never trimmed)
    bucket_bounds_ms: tuple = (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0)
    bucket_counts: List[int] = field(default_factory=list)
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.bucket_counts:
            self.bucket_counts = [0] * len(self.bucket_bounds_ms)
    
    def record_request(self, inference_time_ms: float, num_predictions: int = 1, error: bool = False) -> None:
        """Record a request with its metrics.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            num_predictions: Number of predictions made
            error: Whether the request resulted in an error
        """
        with self._lock:
            self.total_requests += 1
            if not error:
                self.total_predictions += num_predictions
                self.inference_times.append(inference_time_ms)
                self.latency_sum_ms += inference_time_ms
                self.latency_count += 1
                for index, bound in enumerate(self.bucket_bounds_ms):
                    if inference_time_ms <= bound:
                        self.bucket_counts[index] += 1
                
                # Trim history if needed
                if len(self.inference_times) > self.max_history:
                    self.inference_times = self.inference_times[-self.max_history:]
            else:
                self.error_count += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get current metrics statistics.
        
        Returns:
            Dictionary with metrics statistics
        """
        with self._lock:
            if not self.inference_times:
                return {
                    "total_requests": self.total_requests,
                    "total_predictions": self.total_predictions,
                    "avg_inference_time_ms": 0.0,
                    "p50_inference_time_ms": 0.0,
                    "p95_inference_time_ms": 0.0,
                    "p99_inference_time_ms": 0.0,
                    "error_count": self.error_count,
                }
            
            times_array = np.array(self.inference_times)
            
            return {
                "total_requests": self.total_requests,
                "total_predictions": self.total_predictions,
                "avg_inference_time_ms": float(np.mean(times_array)),
                "p50_inference_time_ms": float(np.percentile(times_array, 50)),
                "p95_inference_time_ms": float(np.percentile(times_array, 95)),
                "p99_inference_time_ms": float(np.percentile(times_array, 99)),
                "error_count": self.error_count,
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_requests = 0
            self.total_predictions = 0
            self.error_count = 0
            self.inference_times.clear()
            self.bucket_counts = [0] * len(self.bucket_bounds_ms)
            self.latency_sum_ms = 0.0
            self.latency_count = 0

    def render_prometheus(self, extra_info: Dict[str, str] | None = None) -> str:
        """Render counters and the latency histogram in Prometheus text exposition format."""
        with self._lock:
            lines = [
                "# HELP detektor_requests_total Total inference requests received.",
                "# TYPE detektor_requests_total counter",
                f"detektor_requests_total {self.total_requests}",
                "# HELP detektor_errors_total Total requests that ended in an error.",
                "# TYPE detektor_errors_total counter",
                f"detektor_errors_total {self.error_count}",
                "# HELP detektor_predictions_total Total detections returned.",
                "# TYPE detektor_predictions_total counter",
                f"detektor_predictions_total {self.total_predictions}",
                "# HELP detektor_inference_latency_ms Model inference latency in milliseconds.",
                "# TYPE detektor_inference_latency_ms histogram",
            ]
            for bound, count in zip(self.bucket_bounds_ms, self.bucket_counts):
                lines.append(f'detektor_inference_latency_ms_bucket{{le="{bound:g}"}} {count}')
            lines.append(f'detektor_inference_latency_ms_bucket{{le="+Inf"}} {self.latency_count}')
            lines.append(f"detektor_inference_latency_ms_sum {self.latency_sum_ms:.6f}")
            lines.append(f"detektor_inference_latency_ms_count {self.latency_count}")
            lines.append("# HELP detektor_uptime_seconds Seconds since the metrics store was created.")
            lines.append("# TYPE detektor_uptime_seconds gauge")
            lines.append(f"detektor_uptime_seconds {time.time() - self.started_at:.1f}")
        if extra_info:
            labels = ",".join(f'{key}="{value}"' for key, value in sorted(extra_info.items()))
            lines.append("# HELP detektor_build_info Static build and runtime information.")
            lines.append("# TYPE detektor_build_info gauge")
            lines.append(f"detektor_build_info{{{labels}}} 1")
        return "\n".join(lines) + "\n"


# Global metrics store instance
METRICS_STORE = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Get the global metrics store instance.
    
    Returns:
        Global MetricsStore instance
    """
    return METRICS_STORE
