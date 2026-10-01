"""Capture the hardware/software environment a benchmark ran in."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def collect_environment(device: str, threads: Optional[int] = None) -> Dict[str, Any]:
    """Return a JSON-serialisable description of the host, framework and device."""
    try:
        import psutil

        total_ram_gb: Optional[float] = round(psutil.virtual_memory().total / 1024**3, 2)
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        total_ram_gb, physical_cores, logical_cores = None, None, None

    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": physical_cores,
        "cpu_cores_logical": logical_cores,
        "ram_gb": total_ram_gb,
        "torch": torch.__version__,
        "torch_threads": threads if threads else torch.get_num_threads(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except Exception:  # noqa: BLE001
        info["onnxruntime"] = None
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["cuda_version"] = torch.version.cuda
    return info
