"""Robustness to common image corruptions (noise, blur, exposure, compression, resolution)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

from api.utils import load_model
from benchmarks.common import BenchContext
from utils.metrics_helpers import compute_per_class_ap

DESCRIPTION = "mAP50 retention under Gaussian noise, blur, brightness/contrast, JPEG and downscaling"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _noise(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    def fn(img: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return fn


def _blur(k: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda img: cv2.GaussianBlur(img, (k, k), 0)


def _brightness(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda img: np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _contrast(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda img: np.clip((img.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)


def _jpeg(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    def fn(img: np.ndarray) -> np.ndarray:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return fn


def _downscale(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    def fn(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return fn


PERTURBATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clean": lambda img: img,
    "gaussian_noise_s10": _noise(10),
    "gaussian_noise_s25": _noise(25),
    "blur_k5": _blur(5),
    "blur_k9": _blur(9),
    "brightness_x0.6": _brightness(0.6),
    "brightness_x1.4": _brightness(1.4),
    "contrast_x0.5": _contrast(0.5),
    "jpeg_q20": _jpeg(20),
    "downscale_x0.5": _downscale(0.5),
}


def _load_split(data_yaml: str, split: str = "val") -> Tuple[List[Path], Path, int]:
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    base = Path(data_yaml).parent
    images_dir = Path(cfg[split])
    if not images_dir.is_absolute():
        images_dir = (base / images_dir).resolve()
    if images_dir.name != "images":
        images_dir = images_dir / "images"
    labels_dir = images_dir.parent / "labels"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return images, labels_dir, int(cfg.get("nc", len(cfg.get("names", []))))


def _read_gt(label_path: Path, size: int) -> Dict[str, torch.Tensor]:
    boxes, labels = [], []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:
                cx, cy, w, h = vals
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            else:
                xs, ys = vals[0::2], vals[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append([x1 * size, y1 * size, x2 * size, y2 * size])
            labels.append(cls)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def run(ctx: BenchContext) -> Dict[str, Any]:
    weights = ctx.weights or ctx.extras.get("e2e_checkpoint")
    data_yaml = ctx.data_yaml or ctx.extras.get("e2e_data_yaml")
    if not weights or not data_yaml:
        return {
            "description": DESCRIPTION,
            "skipped": "needs --weights/--data-yaml, or run together with the 'e2e' suite",
            "rows": [],
        }

    model, device = load_model(str(weights), device_name=ctx.device)
    images, labels_dir, nc = _load_split(str(data_yaml))
    size = int(ctx.extras.get("robustness_img_size", 256))
    if ctx.quick:
        images = images[:16]

    rgb = [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (size, size)), cv2.COLOR_BGR2RGB) for p in images]
    targets = [_read_gt(labels_dir / f"{p.stem}.txt", size) for p in images]
    rows = []
    clean_map = None
    for name, fn in PERTURBATIONS.items():
        preds = []
        for img in rgb:
            tensor = torch.from_numpy(fn(img).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.predict(tensor, original_sizes=[(size, size)], conf_thresh=0.05, task="detect")[0]
            preds.append({k: out[k].detach().cpu() for k in ("boxes", "scores", "labels")})
        per_class = compute_per_class_ap(preds, targets, num_classes=nc)
        gt_classes = {int(c) for t in targets for c in t["labels"].tolist()}
        map50 = float(np.mean([per_class[c] for c in gt_classes])) if gt_classes else 0.0
        if name == "clean":
            clean_map = map50
        rows.append(
            {
                "perturbation": name,
                "map50": round(map50, 4),
                "retention": round(map50 / clean_map, 4) if clean_map else None,
                "mean_detections": round(float(np.mean([p["boxes"].shape[0] for p in preds])), 2),
            }
        )
    return {
        "description": DESCRIPTION,
        "weights": str(weights),
        "images": len(images),
        "img_size": size,
        "note": "mAP50 here is a per-class AP50 mean computed in-process (compute_per_class_ap), not COCO-style.",
        "rows": rows,
    }
