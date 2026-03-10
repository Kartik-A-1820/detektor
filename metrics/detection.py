from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor

from utils.box_ops import box_iou


EPS = 1e-6


def _safe_boxes(boxes: Tensor, image_size: tuple[int, int] | None = None) -> Tensor:
    """Clip boxes to valid image bounds when image size is available."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.clone()
    if image_size is not None:
        image_h, image_w = image_size
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0.0, max=float(image_w))
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0.0, max=float(image_h))
    return boxes


def _match_detections(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_threshold: float,
) -> Dict[str, Any]:
    """Match predictions to ground truth boxes using class-aware greedy IoU matching."""
    if pred_boxes.numel() == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": int(gt_boxes.shape[0]),
            "ious": [],
            "scores": [],
            "tp_flags": [],
            "per_class_tp": {},
            "per_class_pred": {},
            "per_class_gt": {},
        }

    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]

    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    ious: List[float] = []
    tp_flags: List[int] = []
    scores: List[float] = []
    per_class_tp: Dict[int, int] = {}
    per_class_pred: Dict[int, int] = {}
    per_class_gt: Dict[int, int] = {}

    for class_id in gt_labels.tolist():
        per_class_gt[class_id] = per_class_gt.get(class_id, 0) + 1

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        label_int = int(label.item())
        per_class_pred[label_int] = per_class_pred.get(label_int, 0) + 1
        scores.append(float(score.item()))

        candidate_mask = (gt_labels == label) & (~matched_gt)
        if not candidate_mask.any():
            tp_flags.append(0)
            continue

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        pair_iou = box_iou(box.unsqueeze(0), gt_boxes[candidate_indices]).squeeze(0)
        best_iou, best_idx_local = pair_iou.max(dim=0)
        if float(best_iou.item()) >= iou_threshold:
            gt_index = candidate_indices[int(best_idx_local.item())]
            matched_gt[gt_index] = True
            tp_flags.append(1)
            ious.append(float(best_iou.item()))
            per_class_tp[label_int] = per_class_tp.get(label_int, 0) + 1
        else:
            tp_flags.append(0)

    tp = int(sum(tp_flags))
    fp = int(len(tp_flags) - tp)
    fn = int(gt_boxes.shape[0] - matched_gt.sum().item())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ious": ious,
        "scores": scores,
        "tp_flags": tp_flags,
        "per_class_tp": per_class_tp,
        "per_class_pred": per_class_pred,
        "per_class_gt": per_class_gt,
    }


def compute_detection_metrics(
    predictions: Sequence[Dict[str, Tensor]],
    targets: Sequence[Dict[str, Tensor]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute lightweight detection metrics across a validation set."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: List[float] = []
    all_scores: List[float] = []
    all_tp_flags: List[int] = []
    per_class_tp: Dict[int, int] = {}
    per_class_pred: Dict[int, int] = {}
    per_class_gt: Dict[int, int] = {}
    total_predictions = 0
    images_with_gt = 0

    for prediction, target in zip(predictions, targets):
        image_size = None
        if prediction["masks"].ndim == 3 and prediction["masks"].shape[0] >= 0:
            image_size = (prediction["masks"].shape[-2], prediction["masks"].shape[-1])

        pred_boxes = _safe_boxes(prediction["boxes"].detach().cpu(), image_size=image_size)
        pred_scores = prediction["scores"].detach().cpu()
        pred_labels = prediction["labels"].detach().cpu().to(dtype=torch.long)
        gt_boxes = _safe_boxes(target["boxes"].detach().cpu(), image_size=image_size)
        gt_labels = target["labels"].detach().cpu().to(dtype=torch.long)

        if gt_boxes.shape[0] > 0:
            images_with_gt += 1
        total_predictions += int(pred_boxes.shape[0])

        matched = _match_detections(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=iou_threshold,
        )
        total_tp += matched["tp"]
        total_fp += matched["fp"]
        total_fn += matched["fn"]
        all_ious.extend(matched["ious"])
        all_scores.extend(matched["scores"])
        all_tp_flags.extend(matched["tp_flags"])

        for mapping, aggregate in (
            (matched["per_class_tp"], per_class_tp),
            (matched["per_class_pred"], per_class_pred),
            (matched["per_class_gt"], per_class_gt),
        ):
            for key, value in mapping.items():
                aggregate[key] = aggregate.get(key, 0) + value

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    mean_iou = sum(all_ious) / max(len(all_ious), 1)

    if all_scores:
        sorted_pairs = sorted(zip(all_scores, all_tp_flags), key=lambda item: item[0], reverse=True)
        running_tp = 0
        running_fp = 0
        precisions = []
        recalls = []
        total_gt = total_tp + total_fn
        for _, tp_flag in sorted_pairs:
            running_tp += tp_flag
            running_fp += 1 - tp_flag
            precisions.append(running_tp / max(running_tp + running_fp, 1))
            recalls.append(running_tp / max(total_gt, 1))
        ap50 = 0.0
        prev_recall = 0.0
        for precision_i, recall_i in zip(precisions, recalls):
            ap50 += precision_i * max(recall_i - prev_recall, 0.0)
            prev_recall = recall_i
    else:
        ap50 = 0.0

    per_class = {}
    all_class_ids = sorted(set(per_class_tp) | set(per_class_pred) | set(per_class_gt))
    for class_id in all_class_ids:
        cls_tp = per_class_tp.get(class_id, 0)
        cls_pred = per_class_pred.get(class_id, 0)
        cls_gt = per_class_gt.get(class_id, 0)
        per_class[class_id] = {
            "tp": cls_tp,
            "pred": cls_pred,
            "gt": cls_gt,
            "precision": cls_tp / max(cls_pred, 1),
            "recall": cls_tp / max(cls_gt, 1),
        }

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "ap50": ap50,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_predictions": total_predictions,
        "images_with_gt": images_with_gt,
        "per_class": per_class,
    }


def smoke_test_detection_metrics() -> Dict[str, Any]:
    """Run a small synthetic smoke test for detection metrics."""
    predictions = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [100.0, 100.0, 140.0, 140.0]]),
            "scores": torch.tensor([0.95, 0.20]),
            "labels": torch.tensor([0, 0]),
            "masks": torch.zeros((2, 64, 64), dtype=torch.bool),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[12.0, 12.0, 28.0, 28.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.zeros((1, 64, 64), dtype=torch.bool),
        }
    ]
    metrics = compute_detection_metrics(predictions, targets)
    print(f"det_precision: {metrics['precision']:.4f}")
    print(f"det_recall: {metrics['recall']:.4f}")
    print(f"det_ap50: {metrics['ap50']:.4f}")
    print(f"det_mean_iou: {metrics['mean_iou']:.4f}")
    return metrics
