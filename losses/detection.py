from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.box_ops import box_area, box_iou, ciou_loss
from utils.robust_loss import sanitize_tensor, safe_divide, clamp_loss


def _cast_like(src: Tensor, ref: Tensor, *, dtype_override: torch.dtype | None = None) -> Tensor:
    target_dtype = dtype_override if dtype_override is not None else ref.dtype
    if src.device == ref.device and src.dtype == target_dtype:
        return src
    return src.to(device=ref.device, dtype=target_dtype)


def _one_hot(labels: Tensor, num_classes: int, dtype: torch.dtype) -> Tensor:
    target = torch.zeros((labels.shape[0], num_classes), device=labels.device, dtype=dtype)
    if labels.numel() > 0:
        target.scatter_(1, labels.view(-1, 1), 1.0)
    return target


class CenterPriorAssigner:
    """Lightweight anchor-free assigner using inside-box and center-prior constraints.

    Points are considered positive if they lie inside a ground-truth box and also fall
    within a center region proportional to the feature stride. If multiple boxes match,
    the smallest-area box is selected to reduce ambiguity for crowded scenes.
    """

    def __init__(self, center_radius: float = 2.5) -> None:
        self.center_radius = center_radius

    def assign(
        self,
        points: Tensor,
        strides: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ) -> Dict[str, Tensor]:
        """Assign ground-truth boxes and labels to points for one image."""
        num_points = points.shape[0]
        device = points.device
        dtype = points.dtype

        fg_mask = torch.zeros(num_points, device=device, dtype=torch.bool)
        assigned_boxes = torch.zeros((num_points, 4), device=device, dtype=dtype)
        assigned_labels = torch.zeros((num_points,), device=device, dtype=torch.long)
        quality = torch.zeros((num_points,), device=device, dtype=dtype)
        matched_gt_indices = torch.full((num_points,), -1, device=device, dtype=torch.long)

        if gt_boxes.numel() == 0:
            return {
                "fg_mask": fg_mask,
                "assigned_boxes": assigned_boxes,
                "assigned_labels": assigned_labels,
                "quality": quality,
                "matched_gt_indices": matched_gt_indices,
            }

        px = points[:, 0:1]
        py = points[:, 1:2]

        left = px - gt_boxes[:, 0]
        top = py - gt_boxes[:, 1]
        right = gt_boxes[:, 2] - px
        bottom = gt_boxes[:, 3] - py
        deltas = torch.stack((left, top, right, bottom), dim=-1)
        inside_box = deltas.amin(dim=-1) > 0

        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        radii = strides[:, None] * self.center_radius
        center_left = px - (centers[:, 0] - radii)
        center_top = py - (centers[:, 1] - radii)
        center_right = (centers[:, 0] + radii) - px
        center_bottom = (centers[:, 1] + radii) - py
        center_deltas = torch.stack((center_left, center_top, center_right, center_bottom), dim=-1)
        inside_center = center_deltas.amin(dim=-1) > 0

        gt_areas = box_area(gt_boxes)
        max_scale = strides[:, None] * 8.0
        fits_scale = deltas.amax(dim=-1) <= max_scale

        match_matrix = inside_box & inside_center & fits_scale
        fallback_matrix = inside_box & inside_center
        has_match = match_matrix.any(dim=1)
        match_matrix = torch.where(has_match[:, None], match_matrix, fallback_matrix)

        if not match_matrix.any():
            return {
                "fg_mask": fg_mask,
                "assigned_boxes": assigned_boxes,
                "assigned_labels": assigned_labels,
                "quality": quality,
                "matched_gt_indices": matched_gt_indices,
            }

        areas = gt_areas.unsqueeze(0).repeat(num_points, 1)
        inf = torch.full_like(areas, float("inf"))
        candidate_areas = torch.where(match_matrix, areas, inf)
        min_area, matched_gt_idx = candidate_areas.min(dim=1)
        fg_mask = torch.isfinite(min_area)

        if fg_mask.any():
            matched_boxes = _cast_like(gt_boxes[matched_gt_idx[fg_mask]], assigned_boxes)
            assigned_boxes[fg_mask] = matched_boxes
            assigned_labels[fg_mask] = gt_labels[matched_gt_idx[fg_mask]]
            matched_gt_indices[fg_mask] = matched_gt_idx[fg_mask]
            point_boxes = torch.cat((points[fg_mask], points[fg_mask]), dim=1)
            quality_values = box_iou(point_boxes, matched_boxes).diag().clamp(min=0.0, max=1.0)
            quality[fg_mask] = _cast_like(quality_values, quality)

        return {
            "fg_mask": fg_mask,
            "assigned_boxes": assigned_boxes,
            "assigned_labels": assigned_labels,
            "quality": quality,
            "matched_gt_indices": matched_gt_indices,
        }


class DetectionLoss(nn.Module):
    """Detection losses for anchor-free classification, box regression, and objectness.
    
    Implements Ultralytics-style training stability improvements:
    - Warmup for box loss weight (gradual increase over first 3 epochs)
    - Label smoothing for classification (reduces overconfidence)
    - Balanced loss weights optimized for stability
    """

    def __init__(
        self,
        num_classes: int,
        cls_weight: float = 0.5,  # Reduced from 1.0 for stability
        box_weight: float = 7.5,  # Increased from 5.0 (Ultralytics uses 7.5)
        obj_weight: float = 1.0,
        center_radius: float = 2.5,
        label_smoothing: float = 0.0,  # Label smoothing epsilon
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.label_smoothing = label_smoothing
        self.assigner = CenterPriorAssigner(center_radius=center_radius)
        
        # Warmup tracking
        self.warmup_epochs = 3
        self.current_epoch = 0

    def forward(
        self,
        pred_cls: Tensor,
        pred_box: Tensor,
        pred_obj: Tensor,
        decoded_boxes: Tensor,
        points: Tensor,
        strides: Tensor,
        targets: Sequence[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute detection losses from flattened predictions and per-image targets."""
        batch_size, total_points, _ = pred_cls.shape
        device = pred_cls.device
        dtype = pred_cls.dtype

        # Validate batch consistency
        if len(targets) != batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} images vs {len(targets)} targets")

        cls_target = torch.zeros((batch_size, total_points, self.num_classes), device=device, dtype=dtype)
        obj_target = torch.zeros((batch_size, total_points), device=device, dtype=dtype)
        assignment_info: List[Dict[str, Tensor]] = []
        box_losses = []
        total_fg = pred_cls.new_tensor(0.0)

        for batch_index, target in enumerate(targets):
            # Validate target has required keys
            if "boxes" not in target or "labels" not in target:
                raise ValueError(f"Target at index {batch_index} missing 'boxes' or 'labels'")
            
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            
            # Validate box-label consistency
            if gt_boxes.shape[0] != gt_labels.shape[0]:
                raise ValueError(
                    f"Box-label mismatch at index {batch_index}: "
                    f"{gt_boxes.shape[0]} boxes vs {gt_labels.shape[0]} labels"
                )
            
            assignment = self.assigner.assign(
                points=points,
                strides=strides,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )
            fg_mask = assignment["fg_mask"]
            num_fg = fg_mask.sum().to(dtype)
            total_fg = total_fg + num_fg
            matched_boxes = assignment["assigned_boxes"][fg_mask]
            matched_labels = assignment["assigned_labels"][fg_mask]

            assignment_info.append(
                {
                    "fg_mask": fg_mask,
                    "assigned_boxes": assignment["assigned_boxes"],
                    "assigned_labels": assignment["assigned_labels"],
                    "matched_gt_indices": assignment["matched_gt_indices"],
                }
            )

            if fg_mask.any():
                # Apply label smoothing (Ultralytics-style)
                one_hot_labels = _one_hot(matched_labels, self.num_classes, dtype=dtype)
                if self.label_smoothing > 0.0:
                    one_hot_labels = one_hot_labels * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
                cls_target[batch_index, fg_mask] = one_hot_labels
                pred_boxes_pos = decoded_boxes[batch_index, fg_mask]
                matched_boxes = _cast_like(matched_boxes, pred_boxes_pos)
                
                # Sanitize predicted boxes BEFORE CIoU to prevent NaN
                pred_boxes_pos = sanitize_tensor(pred_boxes_pos, nan_value=0.0, posinf_value=1000.0, neginf_value=0.0, name="pred_boxes")
                pred_boxes_pos = pred_boxes_pos.clamp(min=-1000.0, max=1000.0)
                
                # Ensure boxes have valid geometry (x2 > x1, y2 > y1)
                px1, py1, px2, py2 = pred_boxes_pos.unbind(dim=-1)
                px2 = torch.maximum(px1 + 1e-3, px2)
                py2 = torch.maximum(py1 + 1e-3, py2)
                pred_boxes_pos = torch.stack([px1, py1, px2, py2], dim=-1)
                
                # Compute CIoU loss with robust handling
                ciou_values = ciou_loss(pred_boxes_pos, matched_boxes)
                ciou_values = sanitize_tensor(ciou_values, nan_value=1.0, posinf_value=1.0, neginf_value=1.0, name="ciou")
                ciou_values = ciou_values.clamp(min=0.0, max=2.0)  # CIoU range is [0, 2]
                
                ious_per_box = (1.0 - ciou_values).clamp(min=0.0, max=1.0)
                obj_target_values = ious_per_box.detach()
                obj_target[batch_index, fg_mask] = _cast_like(obj_target_values, obj_target)
                box_losses.append(ciou_values.sum())

        # Robust normalization - use max(total_fg, 1) to avoid division by zero
        normalizer = total_fg.clamp(min=1.0)
        
        # Warmup for box loss weight (Ultralytics-style)
        # Gradually increase box weight over first 3 epochs to stabilize training
        warmup_factor = min(1.0, (self.current_epoch + 1) / self.warmup_epochs)
        box_weight_current = self.box_weight * warmup_factor
        
        # Compute classification loss with sanitization
        # Use mean instead of sum for better scaling
        loss_cls_raw = F.binary_cross_entropy_with_logits(pred_cls, cls_target, reduction="mean")
        loss_cls = sanitize_tensor(loss_cls_raw, name="loss_cls")
        loss_cls = clamp_loss(loss_cls, max_value=10.0, name="loss_cls")  # Lower clamp for mean
        
        # Compute objectness loss with sanitization
        loss_obj_raw = F.binary_cross_entropy_with_logits(pred_obj.squeeze(-1), obj_target, reduction="mean")
        loss_obj = sanitize_tensor(loss_obj_raw, name="loss_obj")
        loss_obj = clamp_loss(loss_obj, max_value=10.0, name="loss_obj")  # Lower clamp for mean

        # Compute box loss with sanitization
        if box_losses:
            loss_box_raw = torch.stack(box_losses).sum() / normalizer
            loss_box = sanitize_tensor(loss_box_raw, name="loss_box")
            loss_box = clamp_loss(loss_box, max_value=10.0, name="loss_box")  # Lower clamp
        else:
            loss_box = pred_box.sum() * 0.0

        # Compute total loss with warmup-adjusted box weight
        loss_total = self.cls_weight * loss_cls + box_weight_current * loss_box + self.obj_weight * loss_obj
        loss_total = sanitize_tensor(loss_total, name="loss_total")
        loss_total = clamp_loss(loss_total, max_value=100.0, name="loss_total")
        
        return {
            "loss_total": loss_total,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_obj": loss_obj,
            "num_fg": total_fg,
            "assignments": assignment_info,
        }
