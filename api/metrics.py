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


# Global metrics store instance
METRICS_STORE = MetricsStore()


def get_metrics_store() -> MetricsStore:
    """Get the global metrics store instance.
    
    Returns:
        Global MetricsStore instance
    """
    return METRICS_STORE
