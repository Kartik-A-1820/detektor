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
    
    for idx, (image, target) in enumerate(batch):
        if not isinstance(image, Tensor):
            raise TypeError(f"Sample {idx}: image must be a torch.Tensor, got {type(image)}")
        if not isinstance(target, dict):
            raise TypeError(f"Sample {idx}: target must be a dictionary, got {type(target)}")
        
        # Validate image shape
        if image.ndim != 3:
            raise ValueError(f"Sample {idx}: image must be [C, H, W], got shape {tuple(image.shape)}")
        
        # Validate target contains required keys
        if "boxes" not in target or "labels" not in target:
            raise ValueError(f"Sample {idx}: target missing 'boxes' or 'labels'")
        
        # Validate boxes and labels
        boxes = target["boxes"]
        labels = target["labels"]
        
        if not isinstance(boxes, Tensor) or not isinstance(labels, Tensor):
            raise TypeError(f"Sample {idx}: boxes and labels must be tensors")
        
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"Sample {idx}: boxes must be [N, 4], got {tuple(boxes.shape)}")
        
        if labels.ndim != 1:
            raise ValueError(f"Sample {idx}: labels must be [N], got {tuple(labels.shape)}")
        
        if boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Sample {idx}: box-label count mismatch: "
                f"{boxes.shape[0]} boxes vs {labels.shape[0]} labels"
            )
        
        # Validate box values are in valid range [0, 1] for normalized coordinates
        if boxes.numel() > 0:
            if boxes.min() < 0.0 or boxes.max() > 1.0:
                print(f"warning: sample {idx} has boxes outside [0, 1] range, clamping")
                boxes = boxes.clamp(min=0.0, max=1.0)
                target["boxes"] = boxes
        
        images.append(image)
        collated_target: TargetType = {}
        for key, value in target.items():
            if isinstance(value, Tensor):
                collated_target[key] = value
            else:
                collated_target[key] = value
        
        # Ensure required keys exist
        if "boxes" not in collated_target:
            collated_target["boxes"] = torch.zeros((0, 4), dtype=image.dtype)
        if "labels" not in collated_target:
            collated_target["labels"] = torch.zeros((0,), dtype=torch.long)
        
        targets.append(collated_target)
    
    # Validate all images have same shape
    first_shape = images[0].shape
    for idx, img in enumerate(images[1:], start=1):
        if img.shape != first_shape:
            raise ValueError(
                f"Image shape mismatch: image 0 has shape {tuple(first_shape)}, "
                f"but image {idx} has shape {tuple(img.shape)}"
            )

    return torch.stack(images, dim=0), targets
