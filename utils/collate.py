from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor


TargetType = Dict[str, Any]


def detection_segmentation_collate_fn(batch: Sequence[Tuple[Tensor, TargetType]]) -> Tuple[Tensor, List[TargetType]]:
    """Collate detection or segmentation samples with variable-length targets.

    Images are stacked into ``[B, C, H, W]`` while targets remain a list of
    per-image dictionaries. Known target keys such as ``boxes``, ``labels``,
    ``masks``, and ``image_id`` are preserved as-is.
    """
    if len(batch) == 0:
        raise ValueError("Batch must contain at least one sample")

    images: List[Tensor] = []
    targets: List[TargetType] = []
    for image, target in batch:
        if not isinstance(image, Tensor):
            raise TypeError("Each batch image must be a torch.Tensor")
        if not isinstance(target, dict):
            raise TypeError("Each batch target must be a dictionary")

        images.append(image)
        collated_target: TargetType = {}
        for key, value in target.items():
            if isinstance(value, Tensor):
                collated_target[key] = value
            else:
                collated_target[key] = value
        if "boxes" not in collated_target:
            collated_target["boxes"] = torch.zeros((0, 4), dtype=image.dtype)
        if "labels" not in collated_target:
            collated_target["labels"] = torch.zeros((0,), dtype=torch.long)
        targets.append(collated_target)

    return torch.stack(images, dim=0), targets
