from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-7


def flatten_prediction_map(pred: Tensor) -> Tensor:
    """Flatten a prediction tensor from ``[B, C, H, W]`` to ``[B, H*W, C]``."""
    if pred.ndim != 4:
        raise AssertionError(f"Expected [B, C, H, W], got {tuple(pred.shape)}")
    return pred.permute(0, 2, 3, 1).reshape(pred.shape[0], -1, pred.shape[1])


def flatten_prediction_levels(predictions: Sequence[Tensor]) -> Tensor:
    """Flatten and concatenate a sequence of multi-level prediction maps."""
    return torch.cat([flatten_prediction_map(level) for level in predictions], dim=1)


BOX_DECODE_PIXELS = "pixels"  # legacy: relu(raw) is the distance in pixels
BOX_DECODE_STRIDE = "stride"  # softplus(raw) * stride: distances in stride units (default for new models)


def distances_to_boxes(
    points: Tensor,
    distances: Tensor,
    strides: Tensor | None = None,
    mode: str = BOX_DECODE_PIXELS,
) -> Tensor:
    """Decode ltrb distances into absolute xyxy boxes.

    ``mode="pixels"`` (legacy, the default here so old checkpoints decode identically) applies a ``relu``
    and treats the result as pixels. That makes the head regress large pixel values from a near-zero
    initialisation and leaves a dead zone (about half the outputs sit at <= 0 with zero gradient), which
    stalls training on datasets with large objects.

    ``mode="stride"`` predicts distances in units of the feature-map stride through a smooth ``softplus``
    (``distance_px = softplus(raw) * stride``), as in FCOS/YOLOX-style heads: values are O(1-10) at every
    pyramid level and every output receives gradient. Requires ``strides`` with one entry per point.
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise AssertionError(f"Expected points shape [N, 2], got {tuple(points.shape)}")
    if distances.ndim != 3 or distances.shape[-1] != 4:
        raise AssertionError(f"Expected distances shape [B, N, 4], got {tuple(distances.shape)}")
    if distances.shape[1] != points.shape[0]:
        raise AssertionError("Point count and distance count must match")

    # Sanitize distances to prevent NaN from model initialization
    distances = torch.nan_to_num(distances, nan=0.0, posinf=1000.0, neginf=0.0)
    distances = distances.clamp(min=-1000.0, max=1000.0)

    if mode == BOX_DECODE_STRIDE:
        if strides is None:
            raise AssertionError("box_decode='stride' requires per-point strides")
        positive = F.softplus(distances) * strides.reshape(1, -1, 1).to(dtype=distances.dtype)
    elif mode == BOX_DECODE_PIXELS:
        positive = F.relu(distances)
    else:
        raise ValueError(f"Unknown box decode mode '{mode}' (expected 'pixels' or 'stride')")
    px = points[:, 0].view(1, -1)
    py = points[:, 1].view(1, -1)

    left = positive[..., 0]
    top = positive[..., 1]
    right = positive[..., 2]
    bottom = positive[..., 3]

    x1 = px - left
    y1 = py - top
    x2 = px + right
    y2 = py + bottom

    # Sanitize final boxes
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1000.0, neginf=0.0)
    boxes = boxes.clamp(min=-1000.0, max=1000.0)

    return boxes


def box_area(boxes: Tensor) -> Tensor:
    """Compute box areas for boxes in xyxy format."""
    widths = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    heights = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return widths * heights


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=EPS)


def ciou_loss(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Compute CIoU loss for aligned predicted and target boxes in xyxy format."""
    if pred_boxes.numel() == 0:
        return pred_boxes.new_zeros((0,))

    px1, py1, px2, py2 = pred_boxes.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=-1)

    # Ensure boxes are valid (x2 > x1, y2 > y1)
    px1, px2 = torch.minimum(px1, px2), torch.maximum(px1, px2)
    py1, py2 = torch.minimum(py1, py2), torch.maximum(py1, py2)
    tx1, tx2 = torch.minimum(tx1, tx2), torch.maximum(tx1, tx2)
    ty1, ty2 = torch.minimum(ty1, ty2), torch.maximum(ty1, ty2)

    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = ((px2 - px1).clamp(min=EPS) * (py2 - py1).clamp(min=EPS))
    area_t = ((tx2 - tx1).clamp(min=EPS) * (ty2 - ty1).clamp(min=EPS))
    union = (area_p + area_t - inter).clamp(min=EPS)
    iou = (inter / union).clamp(min=0.0, max=1.0)

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5
    center_dist = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

    enc_x1 = torch.minimum(px1, tx1)
    enc_y1 = torch.minimum(py1, ty1)
    enc_x2 = torch.maximum(px2, tx2)
    enc_y2 = torch.maximum(py2, ty2)
    enc_w = (enc_x2 - enc_x1).clamp(min=EPS)
    enc_h = (enc_y2 - enc_y1).clamp(min=EPS)
    enc_diag = enc_w.pow(2) + enc_h.pow(2)

    pw = (px2 - px1).clamp(min=EPS)
    ph = (py2 - py1).clamp(min=EPS)
    tw = (tx2 - tx1).clamp(min=EPS)
    th = (ty2 - ty1).clamp(min=EPS)

    # Robust aspect ratio computation
    v = (4.0 / (torch.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    v = v.clamp(min=0.0, max=4.0)  # Clamp to reasonable range

    # Robust alpha computation
    alpha_denom = (1.0 - iou + v).clamp(min=EPS)
    alpha = (v / alpha_denom).clamp(min=0.0, max=1.0)

    # Robust distance ratio
    dist_ratio = (center_dist / enc_diag.clamp(min=EPS)).clamp(min=0.0, max=1.0)

    # Compute CIoU with clamping
    ciou = iou - dist_ratio - alpha * v
    ciou = ciou.clamp(min=-1.0, max=1.0)

    return (1.0 - ciou).clamp(min=0.0, max=2.0)
