from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.task_detection import TaskMode, analyze_dataset_task, print_task_detection_summary


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLOSegDataset(Dataset):
    """Task-aware YOLO dataset loader supporting detection and segmentation."""

    def __init__(
        self,
        root: str,
        img_size: int = 512,
        task: Optional[str] = None,
        auto_detect_task: bool = True,
        augment: bool = False,
        augment_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
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

        if task is None and auto_detect_task:
            self.task_mode, task_stats = analyze_dataset_task(self.labels_dir)
            print_task_detection_summary(self.task_mode, task_stats)
        elif task is not None:
            self.task_mode = TaskMode(task)
            print(f"Task mode set to: {self.task_mode.value}")
        else:
            self.task_mode = TaskMode.SEGMENT
            print(f"Task mode defaulted to: {self.task_mode.value}")

    def __len__(self) -> int:
        return len(self.images)

    def _empty_masks(self) -> Optional[Tensor]:
        if self.task_mode == TaskMode.DETECT:
            return None
        return torch.zeros((0, self.img_size, self.img_size), dtype=torch.float32)

    def _label_path_for_image(self, image_name: str) -> Path:
        return self.labels_dir / f"{Path(image_name).stem}.txt"

    def _clip_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return boxes.astype(np.float32).reshape(0, 4)
        clipped = boxes.astype(np.float32).copy()
        clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0.0, 1.0)
        clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0.0, 1.0)
        widths = clipped[:, 2] - clipped[:, 0]
        heights = clipped[:, 3] - clipped[:, 1]
        keep = (widths > 1.0 / self.img_size) & (heights > 1.0 / self.img_size)
        return clipped[keep]

    def _filter_instances(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if boxes.size == 0:
            empty_masks = None if masks is None else masks[:0]
            return boxes.reshape(0, 4), labels[:0], empty_masks

        clipped = boxes.astype(np.float32).copy()
        clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0.0, 1.0)
        clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0.0, 1.0)
        widths = clipped[:, 2] - clipped[:, 0]
        heights = clipped[:, 3] - clipped[:, 1]
        keep = (widths > 1.0 / self.img_size) & (heights > 1.0 / self.img_size)
        filtered_masks = None if masks is None else masks[keep]
        return clipped[keep], labels[keep], filtered_masks

    def _resize_masks(self, masks: List[np.ndarray]) -> Optional[Tensor]:
        if self.task_mode != TaskMode.SEGMENT:
            return None
        if not masks:
            return self._empty_masks()
        return torch.from_numpy(np.stack(masks, axis=0)).to(dtype=torch.float32)

    def _parse_yolo_labels(self, label_path: Path) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
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
                if len(parts) > 5:
                    polygon_coords = [float(p) for p in parts[1:]]
                    if len(polygon_coords) % 2 != 0:
                        print(f"warning: odd number of polygon coordinates at line {line_number} in {label_path}")
                        continue
                    if self.task_mode == TaskMode.SEGMENT:
                        polygon_points = np.array(polygon_coords, dtype=np.float32).reshape(-1, 2)
                        polygon_points[:, 0] = np.clip(polygon_points[:, 0] * self.img_size, 0, self.img_size - 1)
                        polygon_points[:, 1] = np.clip(polygon_points[:, 1] * self.img_size, 0, self.img_size - 1)
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                        cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 1.0)
                    x_coords = polygon_coords[0::2]
                    y_coords = polygon_coords[1::2]
                    x1 = max(0.0, min(1.0, min(x_coords)))
                    y1 = max(0.0, min(1.0, min(y_coords)))
                    x2 = max(0.0, min(1.0, max(x_coords)))
                    y2 = max(0.0, min(1.0, max(y_coords)))
                else:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    x1 = max(0.0, min(1.0, x_center - width * 0.5))
                    y1 = max(0.0, min(1.0, y_center - height * 0.5))
                    x2 = max(0.0, min(1.0, x_center + width * 0.5))
                    y2 = max(0.0, min(1.0, y_center + height * 0.5))
                    if self.task_mode == TaskMode.SEGMENT:
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                        x1_px = int(x1 * self.img_size)
                        y1_px = int(y1 * self.img_size)
                        x2_px = int(x2 * self.img_size)
                        y2_px = int(y2 * self.img_size)
                        mask[y1_px:y2_px, x1_px:x2_px] = 1.0
            except (ValueError, IndexError) as exc:
                print(f"warning: error parsing label line {line_number} in {label_path}: {exc}")
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

    def _load_image_tensor(self, image_name: str) -> np.ndarray:
        img_path = self.images_dir / image_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        image_name = self.images[idx]
        image_rgb = self._load_image_tensor(image_name)
        label_path = self._label_path_for_image(image_name)
        boxes, labels, masks = self._parse_yolo_labels(label_path)
        boxes_np = boxes.cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy().astype(np.int64)
        masks_np = None if masks is None else masks.cpu().numpy().astype(np.float32)
        return image_rgb, boxes_np, labels_np, masks_np

    def _build_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.augment and np.random.rand() < float(self.augment_cfg.get("mosaic", 0.0)) and len(self.images) >= 4:
            image_rgb, boxes, labels, masks = self._build_mosaic_sample(idx)
        else:
            image_rgb, boxes, labels, masks = self._load_raw_sample(idx)

        if self.augment and np.random.rand() < float(self.augment_cfg.get("cutmix", 0.0)):
            image_rgb, boxes, labels, masks = self._apply_cutmix_augmentation(image_rgb, boxes, labels, masks)
        if self.augment and np.random.rand() < float(self.augment_cfg.get("random_cut", 0.0)):
            image_rgb = self._apply_random_cut_augmentation(image_rgb)
        return image_rgb, boxes, labels, masks

    def _build_mosaic_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        indices = [idx] + np.random.choice(len(self.images), size=3, replace=len(self.images) < 4).tolist()
        size = self.img_size
        mosaic_image = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
        mosaic_masks: List[np.ndarray] = []
        mosaic_boxes: List[np.ndarray] = []
        mosaic_labels: List[np.ndarray] = []
        placements = [(0, 0), (0, size), (size, 0), (size, size)]

        for sample_index, (top, left) in zip(indices, placements):
            tile_image, tile_boxes, tile_labels, tile_masks = self._load_raw_sample(int(sample_index))
            mosaic_image[top : top + size, left : left + size] = tile_image
            if tile_boxes.size > 0:
                tile_boxes_px = tile_boxes.copy()
                tile_boxes_px[:, [0, 2]] = tile_boxes_px[:, [0, 2]] * size + left
                tile_boxes_px[:, [1, 3]] = tile_boxes_px[:, [1, 3]] * size + top
                mosaic_boxes.append(tile_boxes_px)
                mosaic_labels.append(tile_labels)
            if tile_masks is not None and len(tile_masks) > 0:
                for mask in tile_masks:
                    mask_canvas = np.zeros((size * 2, size * 2), dtype=np.float32)
                    mask_canvas[top : top + size, left : left + size] = mask
                    mosaic_masks.append(mask_canvas)

        crop_x = int(np.random.randint(0, size + 1))
        crop_y = int(np.random.randint(0, size + 1))
        cropped_image = mosaic_image[crop_y : crop_y + size, crop_x : crop_x + size]

        if mosaic_boxes:
            boxes_px = np.concatenate(mosaic_boxes, axis=0)
            labels = np.concatenate(mosaic_labels, axis=0)
            boxes_px[:, [0, 2]] -= crop_x
            boxes_px[:, [1, 3]] -= crop_y
            boxes = boxes_px / float(size)
            masks = None
            if self.task_mode == TaskMode.SEGMENT:
                cropped_masks = [mask[crop_y : crop_y + size, crop_x : crop_x + size] for mask in mosaic_masks]
                masks = np.stack(cropped_masks, axis=0).astype(np.float32) if cropped_masks else np.zeros((0, size, size), dtype=np.float32)
            boxes, labels, masks = self._filter_instances(boxes, labels, masks)
            return cropped_image, boxes, labels, masks

        empty_masks = None if self.task_mode == TaskMode.DETECT else np.zeros((0, size, size), dtype=np.float32)
        return cropped_image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), empty_masks

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

        warped_masks = None
        if masks is not None and len(masks) > 0:
            warped_mask_list = []
            for mask in masks:
                warped_mask = cv2.warpAffine(
                    mask.astype(np.float32),
                    matrix,
                    (self.img_size, self.img_size),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                warped_mask_list.append((warped_mask > 0.5).astype(np.float32))
            warped_masks = np.stack(warped_mask_list, axis=0) if warped_mask_list else masks[:0]

        return (warped_image, *self._filter_instances(warped_boxes, labels, warped_masks))

    def _apply_cutmix_augmentation(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        donor_index = int(np.random.randint(0, len(self.images)))
        donor_image, donor_boxes, donor_labels, donor_masks = self._load_raw_sample(donor_index)
        patch_scale = float(np.random.uniform(0.25, 0.60))
        patch_w = max(1, int(self.img_size * patch_scale))
        patch_h = max(1, int(self.img_size * patch_scale))
        left = int(np.random.randint(0, max(self.img_size - patch_w + 1, 1)))
        top = int(np.random.randint(0, max(self.img_size - patch_h + 1, 1)))
        right = left + patch_w
        bottom = top + patch_h

        mixed_image = image_rgb.copy()
        mixed_image[top:bottom, left:right] = donor_image[top:bottom, left:right]

        def center_inside(current_boxes: np.ndarray) -> np.ndarray:
            if current_boxes.size == 0:
                return np.zeros((0,), dtype=bool)
            centers_x = ((current_boxes[:, 0] + current_boxes[:, 2]) * 0.5) * self.img_size
            centers_y = ((current_boxes[:, 1] + current_boxes[:, 3]) * 0.5) * self.img_size
            return (centers_x >= left) & (centers_x <= right) & (centers_y >= top) & (centers_y <= bottom)

        keep_base = ~center_inside(boxes)
        keep_donor = center_inside(donor_boxes)

        mixed_boxes = np.concatenate([boxes[keep_base], donor_boxes[keep_donor]], axis=0) if (keep_base.any() or keep_donor.any()) else np.zeros((0, 4), dtype=np.float32)
        mixed_labels = np.concatenate([labels[keep_base], donor_labels[keep_donor]], axis=0) if (keep_base.any() or keep_donor.any()) else np.zeros((0,), dtype=np.int64)

        mixed_masks = None
        if self.task_mode == TaskMode.SEGMENT:
            base_masks = masks[keep_base] if masks is not None and len(masks) > 0 else np.zeros((0, self.img_size, self.img_size), dtype=np.float32)
            donor_masks_kept = donor_masks[keep_donor] if donor_masks is not None and len(donor_masks) > 0 else np.zeros((0, self.img_size, self.img_size), dtype=np.float32)
            mixed_masks = np.concatenate([base_masks, donor_masks_kept], axis=0) if (len(base_masks) or len(donor_masks_kept)) else np.zeros((0, self.img_size, self.img_size), dtype=np.float32)

        return mixed_image, mixed_boxes, mixed_labels, mixed_masks

    def _apply_random_cut_augmentation(self, image_rgb: np.ndarray) -> np.ndarray:
        holes = max(int(self.augment_cfg.get("random_cut_holes", 1)), 1)
        scale = float(self.augment_cfg.get("random_cut_scale", 0.25))
        cut_image = image_rgb.copy()
        for _ in range(holes):
            cut_w = max(1, int(self.img_size * np.random.uniform(0.05, scale)))
            cut_h = max(1, int(self.img_size * np.random.uniform(0.05, scale)))
            left = int(np.random.randint(0, max(self.img_size - cut_w + 1, 1)))
            top = int(np.random.randint(0, max(self.img_size - cut_h + 1, 1)))
            color = np.random.randint(32, 160, size=(3,), dtype=np.uint8)
            cut_image[top : top + cut_h, left : left + cut_w] = color
        return cut_image

    def _apply_augmentations(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Tensor, Tensor, Optional[Tensor]]:
        if self.augment:
            image_rgb = self._apply_hsv_augmentation(image_rgb)
            image_rgb, boxes, masks = self._apply_flip_augmentation(image_rgb, boxes, masks)
            image_rgb, boxes, labels, masks = self._apply_affine_augmentation(image_rgb, boxes, labels, masks)

        boxes, labels, masks = self._filter_instances(boxes, labels, masks)
        boxes_tensor = torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4)
        labels_tensor = torch.from_numpy(labels.astype(np.int64)).reshape(-1)
        if self.task_mode == TaskMode.SEGMENT:
            if masks is None or len(masks) == 0:
                masks_tensor = self._empty_masks()
            else:
                masks_tensor = torch.from_numpy(masks.astype(np.float32))
        else:
            masks_tensor = None
        return image_rgb, boxes_tensor, labels_tensor, masks_tensor

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        image_rgb, boxes_np, labels_np, masks_np = self._build_raw_sample(idx)
        image_rgb, boxes, labels, masks = self._apply_augmentations(image_rgb, boxes_np, labels_np, masks_np)
        img_chw = np.transpose(image_rgb, (2, 0, 1)).copy()
        img_tensor = torch.from_numpy(img_chw).to(dtype=torch.float32).div(255.0)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.long),
            "task_mode": self.task_mode.value,
        }
        if self.task_mode == TaskMode.SEGMENT:
            target["masks"] = masks
        return img_tensor, target
