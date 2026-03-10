from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


EPS = 1e-6


def flatten_mask_coefficients(mask_coeff_levels: Tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    """Flatten per-level mask coefficients from ``[B, K, H, W]`` to ``[B, H*W, K]`` and concatenate."""
    if len(mask_coeff_levels) == 0:
        raise AssertionError("mask_coeff_levels must not be empty")
    return torch.cat([level.permute(0, 2, 3, 1).reshape(level.shape[0], -1, level.shape[1]) for level in mask_coeff_levels], dim=1)


def compose_instance_masks(prototypes: Tensor, coefficients: Tensor) -> Tensor:
    """Compose instance mask logits from prototypes and per-instance coefficients.

    Args:
        prototypes: ``[K, Hp, Wp]`` tensor for a single image.
        coefficients: ``[N, K]`` tensor for positive instances.

    Returns:
        Raw mask logits with shape ``[N, Hp, Wp]``.
    """
    if coefficients.numel() == 0:
        return prototypes.new_zeros((0, prototypes.shape[1], prototypes.shape[2]))
    return torch.einsum("nk,khw->nhw", coefficients, prototypes)


def resize_instance_masks(masks: Tensor, proto_size: Tuple[int, int]) -> Tensor:
    """Resize GT instance masks to prototype resolution using nearest-neighbor interpolation."""
    if masks.numel() == 0:
        return masks.reshape(0, proto_size[0], proto_size[1]).to(dtype=torch.float32)
    masks = masks.to(dtype=torch.float32)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    resized = F.interpolate(masks, size=proto_size, mode="nearest")
    return resized.squeeze(1)


def boxes_to_proto_coordinates(
    boxes_xyxy: Tensor,
    image_size: Tuple[int, int],
    proto_size: Tuple[int, int],
) -> Tensor:
    """Map absolute-image xyxy boxes to prototype-grid xyxy coordinates."""
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)
    image_h, image_w = image_size
    proto_h, proto_w = proto_size
    scale = boxes_xyxy.new_tensor([
        proto_w / max(float(image_w), 1.0),
        proto_h / max(float(image_h), 1.0),
        proto_w / max(float(image_w), 1.0),
        proto_h / max(float(image_h), 1.0),
    ])
    proto_boxes = boxes_xyxy * scale
    proto_boxes[:, 0::2] = proto_boxes[:, 0::2].clamp(min=0.0, max=float(proto_w))
    proto_boxes[:, 1::2] = proto_boxes[:, 1::2].clamp(min=0.0, max=float(proto_h))
    return proto_boxes


def crop_mask_region(masks: Tensor, boxes_xyxy: Tensor) -> Tensor:
    """Apply a lightweight box crop mask on prototype-resolution masks.

    Args:
        masks: ``[N, Hp, Wp]`` tensor.
        boxes_xyxy: ``[N, 4]`` tensor in prototype coordinates.
    """
    if masks.numel() == 0:
        return masks
    num_instances, proto_h, proto_w = masks.shape
    ys = torch.arange(proto_h, device=masks.device, dtype=masks.dtype).view(1, proto_h, 1)
    xs = torch.arange(proto_w, device=masks.device, dtype=masks.dtype).view(1, 1, proto_w)

    x1 = boxes_xyxy[:, 0].floor().view(-1, 1, 1)
    y1 = boxes_xyxy[:, 1].floor().view(-1, 1, 1)
    x2 = boxes_xyxy[:, 2].ceil().view(-1, 1, 1)
    y2 = boxes_xyxy[:, 3].ceil().view(-1, 1, 1)

    crop = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return masks * crop.to(dtype=masks.dtype)


def upsample_masks_to_image(masks: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """Upsample prototype-resolution masks to image resolution using bilinear interpolation."""
    if masks.numel() == 0:
        return masks.reshape(0, image_size[0], image_size[1])
    masks = masks.unsqueeze(1)
    upsampled = F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
    return upsampled.squeeze(1)


def threshold_masks(mask_probs: Tensor, threshold: float = 0.5) -> Tensor:
    """Threshold probabilistic masks into binary masks."""
    if mask_probs.numel() == 0:
        return mask_probs > threshold
    return mask_probs >= threshold
