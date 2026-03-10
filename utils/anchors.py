from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor


def generate_level_points(
    feature_height: int,
    feature_width: int,
    image_height: int,
    image_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, int]:
    """Generate anchor-free point centers for one feature level.

    Returns point centers in absolute image coordinates with shape ``[num_points, 2]``
    in ``(x, y)`` order, together with the integer stride for that level.
    """
    if feature_height <= 0 or feature_width <= 0:
        raise AssertionError("Feature map dimensions must be positive")

    stride_h = image_height // feature_height
    stride_w = image_width // feature_width
    if stride_h != stride_w:
        raise AssertionError(f"Expected square stride, got ({stride_h}, {stride_w})")
    stride = int(stride_h)

    shift_x = (torch.arange(feature_width, device=device, dtype=dtype) + 0.5) * stride
    shift_y = (torch.arange(feature_height, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    points = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    return points, stride


def generate_points_for_features(
    feature_maps: Sequence[Tensor],
    image_size: Tuple[int, int],
) -> Tuple[List[Tensor], List[int]]:
    """Generate point centers and strides for a list of feature maps."""
    if len(feature_maps) == 0:
        raise AssertionError("feature_maps must not be empty")

    image_height, image_width = image_size
    device = feature_maps[0].device
    dtype = feature_maps[0].dtype

    all_points: List[Tensor] = []
    all_strides: List[int] = []
    for feature in feature_maps:
        _, _, height, width = feature.shape
        points, stride = generate_level_points(
            feature_height=height,
            feature_width=width,
            image_height=image_height,
            image_width=image_width,
            device=device,
            dtype=dtype,
        )
        all_points.append(points)
        all_strides.append(stride)
    return all_points, all_strides


def concatenate_points_and_strides(
    points_per_level: Sequence[Tensor],
    strides_per_level: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    """Concatenate per-level points and create a stride tensor per point."""
    if len(points_per_level) != len(strides_per_level):
        raise AssertionError("points_per_level and strides_per_level must have equal length")

    if len(points_per_level) == 0:
        raise AssertionError("At least one feature level is required")

    points = torch.cat(list(points_per_level), dim=0)
    stride_tensors = [
        torch.full((level_points.shape[0],), float(stride), device=level_points.device, dtype=level_points.dtype)
        for level_points, stride in zip(points_per_level, strides_per_level)
    ]
    strides = torch.cat(stride_tensors, dim=0)
    return points, strides
