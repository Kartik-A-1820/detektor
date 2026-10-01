"""Run context and model/input helpers shared by the suites."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from models.chimera import ChimeraODIS
from models.factory import ARCHITECTURE_PROFILES, build_model_from_model_config, resolve_model_config

ALL_PROFILES: Tuple[str, ...] = tuple(ARCHITECTURE_PROFILES.keys())


@dataclass
class BenchContext:
    """Settings for one benchmark run. Every suite receives the same context."""

    profiles: List[str] = field(default_factory=lambda: ["firefly", "comet", "nova"])
    device: str = "cpu"
    img_sizes: List[int] = field(default_factory=lambda: [320, 512])
    img_size: int = 512
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_classes: int = 4
    warmup: int = 5
    runs: int = 30
    seed: int = 0
    weights: Optional[str] = None
    data_yaml: Optional[str] = None
    output_dir: Path = Path("runs/benchmarks")
    quick: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def scratch(self) -> Path:
        path = Path(self.output_dir) / "_scratch"
        path.mkdir(parents=True, exist_ok=True)
        return path


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_profile_model(profile: str, num_classes: int, device: torch.device | str = "cpu") -> ChimeraODIS:
    """Randomly-initialised model for ``profile`` (suitable for speed/size benchmarks)."""
    cfg = resolve_model_config({"profile": profile}, num_classes=num_classes)
    model = build_model_from_model_config(cfg, num_classes=num_classes)
    return model.to(device).eval()


def save_profile_checkpoint(profile: str, num_classes: int, path: Path, seed: int = 0) -> Path:
    """Write a loadable checkpoint (state + model config) for a random-init model."""
    torch.manual_seed(seed)
    model = build_profile_model(profile, num_classes, "cpu")
    cfg = resolve_model_config({"profile": profile}, num_classes=num_classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": cfg}, path)
    return path


def random_batch(batch: int, img_size: int, device: torch.device | str, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, img_size, img_size, generator=gen).to(device)


def synthetic_jpeg(width: int = 1280, height: int = 720, seed: int = 0, quality: int = 90) -> bytes:
    """A deterministic, textured JPEG used to benchmark decode/preprocess and the HTTP API."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height // 8, width // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(12):
        x1, y1 = int(rng.integers(0, width - 80)), int(rng.integers(0, height - 80))
        cv2.rectangle(img, (x1, y1), (x1 + int(rng.integers(30, 200)), y1 + int(rng.integers(30, 200))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode synthetic JPEG")
    return buf.tobytes()


def make_targets(batch: int, img_size: int, num_classes: int, seed: int = 0) -> List[Dict[str, torch.Tensor]]:
    """Synthetic normalised-xyxy targets with rectangular masks for loss/backward benchmarks."""
    rng = np.random.default_rng(seed)
    targets: List[Dict[str, torch.Tensor]] = []
    for _ in range(batch):
        n = int(rng.integers(2, 6))
        boxes, labels, masks = [], [], []
        for _ in range(n):
            w, h = rng.uniform(0.1, 0.4, 2)
            x1, y1 = rng.uniform(0.0, 1.0 - w), rng.uniform(0.0, 1.0 - h)
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(int(rng.integers(0, num_classes)))
            m = torch.zeros(img_size, img_size)
            m[int(y1 * img_size): int((y1 + h) * img_size), int(x1 * img_size): int((x1 + w) * img_size)] = 1.0
            masks.append(m)
        targets.append(
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "masks": torch.stack(masks),
            }
        )
    return targets


def make_tempdir(prefix: str = "detektor-bench-") -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix=prefix)
