from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        augment: bool = False,
        augment_cfg: Optional[Dict[str, float]] = None,
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
        self.augment = bool(augment)
        self.augment_cfg = augment_cfg or {}
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

    def _empty_masks(self) -> Optional[Tensor]:
        if self.task_mode == TaskMode.DETECT:
            return None
        return torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)

    def _clip_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return boxes.astype(np.float32).reshape(0, 4)
        boxes = boxes.astype(np.float32)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, 1.0)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, 1.0)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths > 1.0 / self.img_size) & (heights > 1.0 / self.img_size)
        return boxes[keep]

    def _resize_masks(self, masks: List[np.ndarray]) -> Optional[Tensor]:
        if self.task_mode != TaskMode.SEGMENT:
            return None
        if not masks:
            return self._empty_masks()
        return torch.from_numpy(np.stack(masks, axis=0)).to(dtype=torch.float32)

    def _apply_hsv_augmentation(self, image_rgb: np.ndarray) -> np.ndarray:
        hsv_h = float(self.augment_cfg.get("hsv_h", 0.0))
        hsv_s = float(self.augment_cfg.get("hsv_s", 0.0))
        hsv_v = float(self.augment_cfg.get("hsv_v", 0.0))
        if hsv_h <= 0.0 and hsv_s <= 0.0 and hsv_v <= 0.0:
            return image_rgb

        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = np.random.uniform(-hsv_h, hsv_h) * 180.0
        sat_scale = 1.0 + np.random.uniform(-hsv_s, hsv_s)
        val_scale = 1.0 + np.random.uniform(-hsv_v, hsv_v)

        image_hsv[..., 0] = (image_hsv[..., 0] + hue_shift) % 180.0
        image_hsv[..., 1] = np.clip(image_hsv[..., 1] * sat_scale, 0.0, 255.0)
        image_hsv[..., 2] = np.clip(image_hsv[..., 2] * val_scale, 0.0, 255.0)
        return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _apply_flip_augmentation(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        masks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if boxes.size == 0 and masks is None:
            return image_rgb, boxes, masks

        fliplr_prob = float(self.augment_cfg.get("fliplr", 0.0))
        flipud_prob = float(self.augment_cfg.get("flipud", 0.0))

        if np.random.rand() < fliplr_prob:
            image_rgb = np.ascontiguousarray(np.flip(image_rgb, axis=1))
            if boxes.size > 0:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = 1.0 - x2
                boxes[:, 2] = 1.0 - x1
            if masks is not None:
                masks = np.ascontiguousarray(np.flip(masks, axis=2))

        if np.random.rand() < flipud_prob:
            image_rgb = np.ascontiguousarray(np.flip(image_rgb, axis=0))
            if boxes.size > 0:
                y1 = boxes[:, 1].copy()
                y2 = boxes[:, 3].copy()
                boxes[:, 1] = 1.0 - y2
                boxes[:, 3] = 1.0 - y1
            if masks is not None:
                masks = np.ascontiguousarray(np.flip(masks, axis=1))

        return image_rgb, self._clip_boxes(boxes), masks

    def _apply_affine_augmentation(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        translate = float(self.augment_cfg.get("translate", 0.0))
        scale = float(self.augment_cfg.get("scale", 0.0))
        if boxes.size == 0 or (translate <= 0.0 and scale <= 0.0):
            return image_rgb, boxes, labels, masks

        scale_factor = np.random.uniform(1.0 - scale, 1.0 + scale)
        tx = np.random.uniform(-translate, translate) * self.img_size
        ty = np.random.uniform(-translate, translate) * self.img_size
        center = self.img_size * 0.5
        matrix = np.array(
            [
                [scale_factor, 0.0, tx + center * (1.0 - scale_factor)],
                [0.0, scale_factor, ty + center * (1.0 - scale_factor)],
            ],
            dtype=np.float32,
        )

        warped_image = cv2.warpAffine(
            image_rgb,
            matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )

        boxes_px = boxes.copy() * float(self.img_size)
        corners = np.stack(
            [
                np.stack([boxes_px[:, 0], boxes_px[:, 1], np.ones(len(boxes_px))], axis=1),
                np.stack([boxes_px[:, 2], boxes_px[:, 1], np.ones(len(boxes_px))], axis=1),
                np.stack([boxes_px[:, 2], boxes_px[:, 3], np.ones(len(boxes_px))], axis=1),
                np.stack([boxes_px[:, 0], boxes_px[:, 3], np.ones(len(boxes_px))], axis=1),
            ],
            axis=1,
        )
        warped_corners = corners @ matrix.T
        min_xy = warped_corners.min(axis=1)
        max_xy = warped_corners.max(axis=1)
        warped_boxes = np.concatenate([min_xy, max_xy], axis=1) / float(self.img_size)
        warped_boxes = self._clip_boxes(warped_boxes)

        if warped_boxes.shape[0] != boxes.shape[0]:
            valid = []
            for idx, box in enumerate(np.concatenate([min_xy, max_xy], axis=1) / float(self.img_size)):
                clipped = np.clip(box, 0.0, 1.0)
                if (clipped[2] - clipped[0]) > 1.0 / self.img_size and (clipped[3] - clipped[1]) > 1.0 / self.img_size:
                    valid.append(idx)
            labels = labels[np.array(valid, dtype=np.int64)] if valid else labels[:0]
            if masks is not None:
                masks = masks[np.array(valid, dtype=np.int64)] if valid else masks[:0]
        if masks is not None and len(masks) > 0:
            warped_masks = []
            for mask in masks:
                warped_mask = cv2.warpAffine(
                    mask.astype(np.float32),
                    matrix,
                    (self.img_size, self.img_size),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                warped_masks.append((warped_mask > 0.5).astype(np.float32))
            masks = np.stack(warped_masks, axis=0) if warped_masks else masks[:0]

        return warped_image, warped_boxes, labels, masks

    def _apply_augmentations(
        self,
        image_rgb: np.ndarray,
        boxes: Tensor,
        labels: Tensor,
        masks: Optional[Tensor],
    ) -> Tuple[np.ndarray, Tensor, Tensor, Optional[Tensor]]:
        if not self.augment:
            return image_rgb, boxes, labels, masks

        boxes_np = boxes.cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy().astype(np.int64)
        masks_np = None if masks is None else masks.cpu().numpy().astype(np.float32)

        image_rgb = self._apply_hsv_augmentation(image_rgb)
        image_rgb, boxes_np, masks_np = self._apply_flip_augmentation(image_rgb, boxes_np, masks_np)
        image_rgb, boxes_np, labels_np, masks_np = self._apply_affine_augmentation(
            image_rgb,
            boxes_np,
            labels_np,
            masks_np,
        )

        boxes_tensor = torch.from_numpy(boxes_np.astype(np.float32)).reshape(-1, 4)
        labels_tensor = torch.from_numpy(labels_np.astype(np.int64)).reshape(-1)
        masks_tensor = None
        if self.task_mode == TaskMode.SEGMENT:
            if masks_np is None or len(masks_np) == 0:
                masks_tensor = self._empty_masks()
            else:
                masks_tensor = torch.from_numpy(masks_np.astype(np.float32))
        return image_rgb, boxes_tensor, labels_tensor, masks_tensor

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
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                self._empty_masks(),
            )

        raw_text = label_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                self._empty_masks(),
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
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                self._empty_masks(),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            self._resize_masks(masks or []),
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        image_name = self.images[idx]
        img_path = self.images_dir / image_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = self._label_path_for_image(image_name)
        boxes, labels, masks = self._parse_yolo_labels(label_path)
        img_rgb, boxes, labels, masks = self._apply_augmentations(img_rgb, boxes, labels, masks)
        img_chw = np.transpose(img_rgb, (2, 0, 1)).copy()
        img_tensor = torch.from_numpy(img_chw).to(dtype=torch.float32).div(255.0)
        
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
