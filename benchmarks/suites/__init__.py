"""Suite registry. Each suite module exposes ``DESCRIPTION`` and ``run(ctx) -> dict``."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, List

# name -> module (relative to this package). Order is the default execution order.
SUITES: Dict[str, str] = {
    "complexity": "complexity",
    "latency": "latency",
    "throughput": "throughput",
    "memory": "memory",
    "training": "training",
    "startup": "startup",
    "onnx": "onnx_runtime",
    "api": "api_load",
    "e2e": "e2e_quality",
    "robustness": "robustness",
    "accuracy": "accuracy",
}

# Suites that need a dataset or are slow; excluded from ``--suites fast``.
HEAVY = {"e2e", "robustness", "accuracy", "api"}
FAST = [name for name in SUITES if name not in HEAVY]


def load(name: str):
    if name not in SUITES:
        raise KeyError(f"Unknown suite '{name}'. Available: {', '.join(SUITES)}")
    return import_module(f"benchmarks.suites.{SUITES[name]}")


def resolve(selection: List[str]) -> List[str]:
    """Expand ``all`` / ``fast`` aliases and validate names, preserving order."""
    names: List[str] = []
    for item in selection:
        if item == "all":
            names.extend(SUITES)
        elif item == "fast":
            names.extend(FAST)
        else:
            if item not in SUITES:
                raise KeyError(f"Unknown suite '{item}'. Available: {', '.join(SUITES)} (or 'all', 'fast')")
            names.append(item)
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]
