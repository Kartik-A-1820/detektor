"""Deterministic synthetic YOLO-segmentation dataset for reproducible quality benchmarks.

Three visually distinct, easily-learnable classes (rectangle, ellipse, triangle) are drawn
over a noisy textured background with random colour jitter. Labels are written as YOLO
segmentation polygons so both detection and segmentation paths are exercised. The data is
*not* a substitute for a real dataset; it exists so anyone can verify the whole
train -> validate -> serve pipeline end to end, offline, in minutes, with fixed seeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

CLASS_NAMES = ["rectangle", "ellipse", "triangle"]
_BASE_COLORS = [(210, 60, 60), (60, 190, 80), (70, 90, 220)]  # RGB, one hue family per class


def _polygon(kind: int, cx: float, cy: float, w: float, h: float, rng: np.random.Generator) -> np.ndarray:
    if kind == 0:  # rectangle
        return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    if kind == 1:  # ellipse approximated by 24-gon
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return np.stack([cx + (w / 2) * np.cos(t), cy + (h / 2) * np.sin(t)], axis=1)
    return np.array([[cx, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])  # triangle


def _make_image(size: int, rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
    bg = rng.integers(90, 170, size=(size // 16, size // 16, 3), dtype=np.uint8)
    img = cv2.resize(bg, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.int16)
    img += rng.integers(-12, 13, size=img.shape, dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)

    lines: List[str] = []
    occupied: List[Tuple[float, float, float, float]] = []
    for _ in range(int(rng.integers(1, 5))):
        for _attempt in range(20):
            w, h = rng.uniform(0.14, 0.32, 2) * size
            cx = rng.uniform(w / 2 + 2, size - w / 2 - 2)
            cy = rng.uniform(h / 2 + 2, size - h / 2 - 2)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]) for o in occupied):
                continue
            occupied.append(box)
            kind = int(rng.integers(0, len(CLASS_NAMES)))
            color = np.clip(np.array(_BASE_COLORS[kind]) + rng.integers(-30, 31, 3), 0, 255)
            poly = _polygon(kind, cx, cy, w, h, rng)
            cv2.fillPoly(img, [poly.astype(np.int32)], tuple(int(c) for c in color))
            norm = np.clip(poly / size, 0.0, 1.0)
            lines.append(f"{kind} " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)))
            break
    return img, lines


def generate_dataset(root: str | Path, *, train: int = 96, val: int = 32, img_size: int = 256, seed: int = 0) -> Path:
    """Write ``root/{train,val}/{images,labels}`` plus ``root/data.yaml``; returns the YAML path."""
    root = Path(root).resolve()  # absolute: the YAML is later resolved relative to its own folder
    rng = np.random.default_rng(seed)
    for split, count in (("train", train), ("val", val)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for index in range(count):
            img, lines = _make_image(img_size, rng)
            stem = f"{split}_{index:04d}"
            cv2.imwrite(str(root / split / "images" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(root / "train" / "images"),
                "val": str(root / "val" / "images"),
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic shapes dataset (YOLO segmentation format)")
    parser.add_argument("--out", default="data/synthetic", help="Output directory")
    parser.add_argument("--train", type=int, default=192, help="Training images")
    parser.add_argument("--val", type=int, default=48, help="Validation images")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_yaml = generate_dataset(args.out, train=args.train, val=args.val, img_size=args.img_size, seed=args.seed)
    print(f"dataset written to {Path(args.out).resolve()}\ndata yaml: {data_yaml.resolve()}")


if __name__ == "__main__":
    main()
