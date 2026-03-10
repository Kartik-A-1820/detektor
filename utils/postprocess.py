from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from .box_ops import box_iou

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:
    torchvision_nms = None


def select_topk_candidates(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    mask_coeff: Tensor,
    topk_pre_nms: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select top-k scoring candidates for one image before NMS."""
    if boxes.numel() == 0:
        return boxes, scores, labels, mask_coeff
    k = min(int(topk_pre_nms), scores.shape[0])
    topk_scores, topk_indices = torch.topk(scores, k=k, dim=0, largest=True, sorted=True)
    return boxes[topk_indices], topk_scores, labels[topk_indices], mask_coeff[topk_indices]


def _pure_torch_nms(boxes: Tensor, scores: Tensor, iou_thresh: float) -> Tensor:
    """Fallback NMS implementation when torchvision is unavailable."""
    if boxes.numel() == 0:
        return torch.zeros((0,), device=boxes.device, dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        current_box = boxes[current].unsqueeze(0)
        remaining = order[1:]
        ious = box_iou(current_box, boxes[remaining]).squeeze(0)
        order = remaining[ious <= iou_thresh]
    return torch.stack(keep) if keep else torch.zeros((0,), device=boxes.device, dtype=torch.long)


def class_aware_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_thresh: float,
    max_det: int,
) -> Tensor:
    """Run class-aware NMS using torchvision when available, else a pure PyTorch fallback."""
    if boxes.numel() == 0:
        return torch.zeros((0,), device=boxes.device, dtype=torch.long)

    keep_indices = []
    unique_labels = labels.unique(sorted=True)
    for class_id in unique_labels:
        class_mask = labels == class_id
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        if torchvision_nms is not None:
            kept_local = torchvision_nms(class_boxes, class_scores, iou_thresh)
        else:
            kept_local = _pure_torch_nms(class_boxes, class_scores, iou_thresh)
        keep_indices.append(class_indices[kept_local])

    if not keep_indices:
        return torch.zeros((0,), device=boxes.device, dtype=torch.long)

    keep = torch.cat(keep_indices, dim=0)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep[:max_det]


def rescale_boxes(
    boxes: Tensor,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tensor:
    """Rescale xyxy boxes from model image size to original image size."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    from_h, from_w = from_size
    to_h, to_w = to_size
    scale = boxes.new_tensor([
        float(to_w) / max(float(from_w), 1.0),
        float(to_h) / max(float(from_h), 1.0),
        float(to_w) / max(float(from_w), 1.0),
        float(to_h) / max(float(from_h), 1.0),
    ])
    scaled = boxes * scale
    scaled[:, 0::2] = scaled[:, 0::2].clamp(min=0.0, max=float(to_w))
    scaled[:, 1::2] = scaled[:, 1::2].clamp(min=0.0, max=float(to_h))
    return scaled


def build_empty_prediction(
    image_size: Tuple[int, int],
    device: torch.device,
    score_dtype: torch.dtype,
    mask_dtype: torch.dtype,
) -> Dict[str, Tensor]:
    """Create an empty per-image prediction dictionary."""
    image_h, image_w = image_size
    return {
        "boxes": torch.zeros((0, 4), device=device, dtype=score_dtype),
        "scores": torch.zeros((0,), device=device, dtype=score_dtype),
        "labels": torch.zeros((0,), device=device, dtype=torch.long),
        "masks": torch.zeros((0, image_h, image_w), device=device, dtype=mask_dtype),
    }
