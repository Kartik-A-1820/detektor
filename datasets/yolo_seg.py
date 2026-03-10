from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLOSegDataset(Dataset):
    """Lightweight YOLO-format dataset loader returning normalized xyxy boxes."""

    def __init__(self, root: str, img_size: int = 512) -> None:
        self.root = Path(root)
        self.img_size = int(img_size)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")

        self.images = sorted(
            path.name
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(
                f"No supported images found in {self.images_dir} "
                f"with extensions {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def _label_path_for_image(self, image_name: str) -> Path:
        return self.labels_dir / f"{Path(image_name).stem}.txt"

    def _parse_yolo_labels(self, label_path: Path) -> Tuple[Tensor, Tensor]:
        if not label_path.exists():
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        raw_text = label_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        boxes: List[List[float]] = []
        labels: List[int] = []

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"warning: skipping malformed label line {line_number} in {label_path}")
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(f"warning: skipping non-numeric label line {line_number} in {label_path}")
                continue

            x1 = max(0.0, min(1.0, x_center - width * 0.5))
            y1 = max(0.0, min(1.0, y_center - height * 0.5))
            x2 = max(0.0, min(1.0, x_center + width * 0.5))
            y2 = max(0.0, min(1.0, y_center + height * 0.5))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        image_name = self.images[idx]
        img_path = self.images_dir / image_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1)).copy()
        img_tensor = torch.from_numpy(img_chw).to(dtype=torch.float32).div(255.0)

        label_path = self._label_path_for_image(image_name)
        boxes, labels = self._parse_yolo_labels(label_path)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.long),
        }
        return img_tensor, target
