from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor


def prediction_to_serializable(prediction: Dict[str, Tensor]) -> Dict[str, Any]:
    """Convert a prediction dict into JSON-friendly Python objects."""
    return {
        "boxes": prediction["boxes"].detach().cpu().tolist(),
        "scores": prediction["scores"].detach().cpu().tolist(),
        "labels": prediction["labels"].detach().cpu().tolist(),
        "masks_shape": list(prediction["masks"].shape),
        "mask_count": int(prediction["masks"].shape[0]),
    }


def predictions_to_coco_like(predictions: Sequence[Dict[str, Tensor]], image_ids: Sequence[int] | None = None) -> List[Dict[str, Any]]:
    """Convert predictions to a lightweight COCO-export-friendly detection list."""
    results: List[Dict[str, Any]] = []
    if image_ids is None:
        image_ids = list(range(len(predictions)))

    for image_id, prediction in zip(image_ids, predictions):
        boxes = prediction["boxes"].detach().cpu()
        scores = prediction["scores"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item()),
                }
            )
    return results


def summarize_image_result(prediction: Dict[str, Tensor], target: Dict[str, Tensor] | None = None) -> Dict[str, Any]:
    """Create a concise per-image summary for debugging or logging."""
    summary = {
        "num_predictions": int(prediction["boxes"].shape[0]),
        "num_masks": int(prediction["masks"].shape[0]),
        "mean_score": float(prediction["scores"].mean().item()) if prediction["scores"].numel() > 0 else 0.0,
    }
    if target is not None:
        summary["num_gt"] = int(target["boxes"].shape[0])
        summary["num_gt_masks"] = int(target["masks"].shape[0]) if "masks" in target else 0
    return summary
