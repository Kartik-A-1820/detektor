from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.task_detection import TaskMode, analyze_dataset_task, boxes_from_mask, print_task_detection_summary


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLOSegDataset(Dataset):
    """Task-aware YOLO dataset loader supporting detection and segmentation.
    
    Automatically detects dataset type:
    - Detection only: Returns boxes only, no masks
    - Segmentation: Returns boxes + masks, generates boxes from masks if needed
    """

    def __init__(
        self, 
        root: str, 
        img_size: int = 512,
        task: Optional[str] = None,
        auto_detect_task: bool = True,
    ) -> None:
        """Initialize YOLO dataset with task detection.
        
        Args:
            root: Dataset root directory
            img_size: Image size for resizing
            task: Task mode ('detect' or 'segment'), auto-detected if None
            auto_detect_task: Whether to auto-detect task from dataset
        """
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
        
        # Auto-detect task mode
        if task is None and auto_detect_task:
            self.task_mode, task_stats = analyze_dataset_task(self.labels_dir)
            print_task_detection_summary(self.task_mode, task_stats)
        elif task is not None:
            self.task_mode = TaskMode(task)
            print(f"Task mode set to: {self.task_mode.value}")
        else:
            self.task_mode = TaskMode.SEGMENT  # Default to segment
            print(f"Task mode defaulted to: {self.task_mode.value}")

    def __len__(self) -> int:
        return len(self.images)

    def _label_path_for_image(self, image_name: str) -> Path:
        return self.labels_dir / f"{Path(image_name).stem}.txt"

    def _parse_yolo_labels(self, label_path: Path) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Parse YOLO format labels with task-aware handling.
        
        Returns:
            Tuple of (boxes, labels, masks) where masks can be None for detect mode
        """
        if not label_path.exists():
            empty_masks = None if self.task_mode == TaskMode.DETECT else torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                empty_masks,
            )

        raw_text = label_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            empty_masks = None if self.task_mode == TaskMode.DETECT else torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                empty_masks,
            )

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = [] if self.task_mode == TaskMode.SEGMENT else None

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"warning: skipping malformed label line {line_number} in {label_path}")
                continue

            try:
                class_id = int(float(parts[0]))
                mask = None
                
                # Check if this is segmentation format (polygon points) or bbox format
                if len(parts) > 5:
                    # Segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                    polygon_coords = [float(p) for p in parts[1:]]
                    if len(polygon_coords) % 2 != 0:
                        print(f"warning: odd number of polygon coordinates at line {line_number} in {label_path}")
                        continue
                    
                    # Convert polygon to mask (only if in segment mode)
                    if self.task_mode == TaskMode.SEGMENT:
                        polygon_points = np.array(polygon_coords).reshape(-1, 2)
                        polygon_points[:, 0] = np.clip(polygon_points[:, 0] * self.img_size, 0, self.img_size - 1)
                        polygon_points[:, 1] = np.clip(polygon_points[:, 1] * self.img_size, 0, self.img_size - 1)
                        
                        # Create binary mask from polygon
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                        polygon_points_int = polygon_points.astype(np.int32)
                        cv2.fillPoly(mask, [polygon_points_int], 1.0)
                    
                    # Compute bounding box from polygon
                    x_coords = polygon_coords[0::2]
                    y_coords = polygon_coords[1::2]
                    x1 = max(0.0, min(1.0, min(x_coords)))
                    y1 = max(0.0, min(1.0, min(y_coords)))
                    x2 = max(0.0, min(1.0, max(x_coords)))
                    y2 = max(0.0, min(1.0, max(y_coords)))
                else:
                    # Bounding box format: class_id x_center y_center width height
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    x1 = max(0.0, min(1.0, x_center - width * 0.5))
                    y1 = max(0.0, min(1.0, y_center - height * 0.5))
                    x2 = max(0.0, min(1.0, x_center + width * 0.5))
                    y2 = max(0.0, min(1.0, y_center + height * 0.5))
                    
                    # Create rectangular mask for bbox-only format (only in segment mode)
                    if self.task_mode == TaskMode.SEGMENT:
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                        x1_px = int(x1 * self.img_size)
                        y1_px = int(y1 * self.img_size)
                        x2_px = int(x2 * self.img_size)
                        y2_px = int(y2 * self.img_size)
                        mask[y1_px:y2_px, x1_px:x2_px] = 1.0
                
            except (ValueError, IndexError) as e:
                print(f"warning: error parsing label line {line_number} in {label_path}: {e}")
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            if self.task_mode == TaskMode.SEGMENT and mask is not None:
                masks.append(mask)

        if not boxes:
            empty_masks = None if self.task_mode == TaskMode.DETECT else torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                empty_masks,
            )

        masks_tensor = None
        if self.task_mode == TaskMode.SEGMENT and masks:
            masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(dtype=torch.float32)

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            masks_tensor,
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
        boxes, labels, masks = self._parse_yolo_labels(label_path)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.long),
            "task_mode": self.task_mode.value,
        }
        
        # Only include masks if in segment mode
        if self.task_mode == TaskMode.SEGMENT:
            target["masks"] = masks
        
        return img_tensor, target
