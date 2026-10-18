"""COCO-style detection evaluation backed by ``pycocotools`` (the reference implementation).

Detektor's older helpers average *per-image* AP and skip images with no predictions or no
ground truth, which overstates AP50-95 (about +6 % on a small test, more on harder data).
COCO evaluation pools detections across the whole dataset, uses 101-point interpolated
precision with the monotone envelope, and counts every ground-truth object, so numbers are
comparable with published results and other tools.

Size buckets follow COCO (small < 32², medium 32²-96², large > 96² pixels of box area) and are
therefore relative to the *image resolution* you evaluate at.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# COCOeval stats vector indices
_AP, _AP50, _AP75, _APS, _APM, _APL, _AR1, _AR10, _AR100, _ARS, _ARM, _ARL = range(12)


def _empty_result(num_classes: int) -> Dict[str, Any]:
    return {
        "map50": 0.0, "map50_95": 0.0, "map75": 0.0,
        "ap_small": None, "ap_medium": None, "ap_large": None,
        "ar_1": 0.0, "ar_10": 0.0, "ar_100": 0.0,
        "per_class_ap50": {c: 0.0 for c in range(num_classes)},
        "per_class_ap50_95": {c: 0.0 for c in range(num_classes)},
        "num_images": 0, "num_gt": 0, "num_detections": 0,
        "recall_thresholds": [], "precision_at_iou50": {},
    }


def _clean(value: float) -> Optional[float]:
    """pycocotools reports -1 for 'no data'; surface that as ``None``."""
    return None if value is None or value < 0 else float(value)


def _tolist(value: Any) -> List[float]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().reshape(-1).tolist()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def coco_evaluate(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    num_classes: int,
    max_dets: Sequence[int] = (1, 10, 100),
) -> Dict[str, Any]:
    """Evaluate per-image predictions against per-image targets with ``pycocotools``.

    Args:
        predictions: one dict per image with ``boxes`` [N, 4] xyxy, ``scores`` [N], ``labels`` [N].
        targets: one dict per image with ``boxes`` [M, 4] xyxy (same coordinate space) and ``labels`` [M].
        num_classes: number of classes (labels are ``0 .. num_classes-1``).
        max_dets: COCO ``maxDets`` (the last entry is the per-image detection cap).

    Returns:
        ``map50``, ``map50_95``, ``map75``, size-bucketed AP (``None`` when the bucket has no objects),
        average recall, per-class AP50 and AP50-95, and the per-class precision-recall curve at IoU 0.5.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(predictions) != len(targets):
        raise ValueError(f"predictions ({len(predictions)}) and targets ({len(targets)}) must have equal length")

    images, annotations, detections = [], [], []
    annotation_id = 1
    for image_id, (pred, target) in enumerate(zip(predictions, targets), start=1):
        images.append({"id": image_id})
        gt_boxes = torch.as_tensor(target["boxes"]).reshape(-1, 4).double()
        gt_labels = _tolist(target["labels"])
        for box, label in zip(gt_boxes.tolist(), gt_labels):
            width, height = max(box[2] - box[0], 0.0), max(box[3] - box[1], 0.0)
            annotations.append({
                "id": annotation_id, "image_id": image_id, "category_id": int(label),
                "bbox": [box[0], box[1], width, height], "area": width * height, "iscrowd": 0,
            })
            annotation_id += 1
        pred_boxes = torch.as_tensor(pred["boxes"]).reshape(-1, 4).double()
        scores, labels = _tolist(pred["scores"]), _tolist(pred["labels"])
        for box, score, label in zip(pred_boxes.tolist(), scores, labels):
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            detections.append({
                "image_id": image_id, "category_id": int(label),
                "bbox": [box[0], box[1], max(box[2] - box[0], 0.0), max(box[3] - box[1], 0.0)],
                "score": float(score),
            })

    result = _empty_result(num_classes)
    result.update({"num_images": len(images), "num_gt": len(annotations), "num_detections": len(detections)})
    if not annotations or not detections:
        return result

    dataset = {"images": images, "annotations": annotations, "categories": [{"id": c} for c in range(num_classes)]}
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):  # pycocotools prints progress to stdout
        gt = COCO()
        gt.dataset = dataset
        gt.createIndex()
        dt = gt.loadRes(detections)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = list(max_dets)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    result.update({
        "map50_95": float(max(stats[_AP], 0.0)),
        "map50": float(max(stats[_AP50], 0.0)),
        "map75": float(max(stats[_AP75], 0.0)),
        "ap_small": _clean(stats[_APS]), "ap_medium": _clean(stats[_APM]), "ap_large": _clean(stats[_APL]),
        "ar_1": float(max(stats[_AR1], 0.0)), "ar_10": float(max(stats[_AR10], 0.0)), "ar_100": float(max(stats[_AR100], 0.0)),
    })

    # precision[T, R, K, A, M]: IoU thresholds x recall thresholds x classes x area ranges x max dets
    precision = evaluator.eval["precision"]
    result["recall_thresholds"] = [float(r) for r in evaluator.params.recThrs]
    for class_index in range(num_classes):
        overall = precision[:, :, class_index, 0, -1]
        valid = overall[overall > -1]
        result["per_class_ap50_95"][class_index] = float(valid.mean()) if valid.size else 0.0
        at_50 = precision[0, :, class_index, 0, -1]  # IoU = 0.50
        valid50 = at_50[at_50 > -1]
        result["per_class_ap50"][class_index] = float(valid50.mean()) if valid50.size else 0.0
        result["precision_at_iou50"][class_index] = [float(max(p, 0.0)) for p in at_50]
    return result
