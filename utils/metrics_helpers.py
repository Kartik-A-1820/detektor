"""Production-grade metrics computation helpers for detection and segmentation.

This module provides comprehensive metrics computation including:
- Precision, Recall, F1, AP50, AP50-95
- Per-class metrics
- Confusion matrices
- Threshold sweeps
- Segmentation metrics (IoU, Dice)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from utils.box_ops import box_iou


EPS = 1e-6


def compute_precision_recall_f1(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, EPS)
    return precision, recall, f1


def compute_ap_from_pr_curve(
    precisions: List[float],
    recalls: List[float],
    method: str = "interp",
) -> float:
    """Compute Average Precision from precision-recall curve.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        method: Interpolation method ('interp' or 'continuous')
        
    Returns:
        Average Precision score
    """
    if not precisions or not recalls:
        return 0.0
    
    if method == "interp":
        # 11-point interpolation (PASCAL VOC style)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p_interp = max([p for p, r in zip(precisions, recalls) if r >= t] or [0])
            ap += p_interp / 11
        return ap
    else:
        # Continuous integration
        ap = 0.0
        prev_recall = 0.0
        for precision, recall in zip(precisions, recalls):
            ap += precision * max(recall - prev_recall, 0.0)
            prev_recall = recall
        return ap


def compute_ap50_95(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
) -> float:
    """Compute AP averaged over IoU thresholds 0.5:0.05:0.95 (COCO-style).
    
    Args:
        pred_boxes: Predicted boxes [N, 4]
        pred_scores: Prediction scores [N]
        pred_labels: Prediction labels [N]
        gt_boxes: Ground truth boxes [M, 4]
        gt_labels: Ground truth labels [M]
        
    Returns:
        AP50-95 score
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0.0
    
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    
    for iou_thresh in iou_thresholds:
        # Sort by score
        order = pred_scores.argsort(descending=True)
        pred_boxes_sorted = pred_boxes[order]
        pred_scores_sorted = pred_scores[order]
        pred_labels_sorted = pred_labels[order]
        
        matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        tp_flags = []
        
        for box, label in zip(pred_boxes_sorted, pred_labels_sorted):
            candidate_mask = (gt_labels == label) & (~matched_gt)
            if not candidate_mask.any():
                tp_flags.append(0)
                continue
            
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
            pair_iou = box_iou(box.unsqueeze(0), gt_boxes[candidate_indices]).squeeze(0)
            best_iou, best_idx_local = pair_iou.max(dim=0)
            
            if float(best_iou.item()) >= iou_thresh:
                gt_index = candidate_indices[int(best_idx_local.item())]
                matched_gt[gt_index] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)
        
        # Compute AP for this IoU threshold
        running_tp = 0
        running_fp = 0
        precisions = []
        recalls = []
        total_gt = gt_boxes.shape[0]
        
        for tp_flag in tp_flags:
            running_tp += tp_flag
            running_fp += 1 - tp_flag
            precisions.append(running_tp / max(running_tp + running_fp, 1))
            recalls.append(running_tp / max(total_gt, 1))
        
        ap = compute_ap_from_pr_curve(precisions, recalls, method="continuous")
        aps.append(ap)
    
    return float(np.mean(aps))


def compute_confusion_matrix(
    pred_labels: Tensor,
    gt_labels: Tensor,
    num_classes: int,
    matched_indices: Optional[Tensor] = None,
) -> np.ndarray:
    """Compute confusion matrix for matched predictions.
    
    Args:
        pred_labels: Predicted labels [N]
        gt_labels: Ground truth labels [M]
        num_classes: Number of classes
        matched_indices: Indices of matched GT for each prediction [N], -1 if unmatched
        
    Returns:
        Confusion matrix [num_classes+1, num_classes+1] (last row/col for background)
    """
    # +1 for background class
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    
    if matched_indices is None:
        # Simple confusion matrix without matching
        for pred_label, gt_label in zip(pred_labels.tolist(), gt_labels.tolist()):
            cm[gt_label, pred_label] += 1
    else:
        # Matched predictions
        for pred_label, matched_idx in zip(pred_labels.tolist(), matched_indices.tolist()):
            if matched_idx >= 0:
                gt_label = gt_labels[matched_idx].item()
                cm[gt_label, pred_label] += 1
            else:
                # False positive (background)
                cm[num_classes, pred_label] += 1
        
        # False negatives (unmatched GT)
        matched_gt_set = set(matched_indices[matched_indices >= 0].tolist())
        for idx, gt_label in enumerate(gt_labels.tolist()):
            if idx not in matched_gt_set:
                cm[gt_label, num_classes] += 1
    
    return cm


def compute_mask_iou(
    pred_mask: Tensor,
    gt_mask: Tensor,
) -> float:
    """Compute IoU between two binary masks.
    
    Args:
        pred_mask: Predicted mask [H, W]
        gt_mask: Ground truth mask [H, W]
        
    Returns:
        IoU score
    """
    if pred_mask.numel() == 0 or gt_mask.numel() == 0:
        return 0.0
    
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    intersection = (pred_mask & gt_mask).sum().item()
    union = (pred_mask | gt_mask).sum().item()
    
    return intersection / max(union, 1)


def compute_dice_score(
    pred_mask: Tensor,
    gt_mask: Tensor,
) -> float:
    """Compute Dice coefficient between two binary masks.
    
    Args:
        pred_mask: Predicted mask [H, W]
        gt_mask: Ground truth mask [H, W]
        
    Returns:
        Dice score
    """
    if pred_mask.numel() == 0 or gt_mask.numel() == 0:
        return 0.0
    
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    intersection = (pred_mask & gt_mask).sum().item()
    pred_sum = pred_mask.sum().item()
    gt_sum = gt_mask.sum().item()
    
    return (2 * intersection) / max(pred_sum + gt_sum, 1)


def threshold_sweep(
    scores: List[float],
    tp_flags: List[int],
    total_gt: int,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Perform threshold sweep to find optimal confidence threshold.
    
    Args:
        scores: Prediction confidence scores
        tp_flags: True positive flags (1 for TP, 0 for FP)
        total_gt: Total number of ground truth instances
        thresholds: List of thresholds to evaluate (default: 0.1 to 0.9 step 0.1)
        
    Returns:
        Dictionary with threshold sweep results
    """
    if thresholds is None:
        thresholds = [i * 0.1 for i in range(1, 10)]
    
    results = []
    
    for thresh in thresholds:
        # Filter predictions by threshold
        filtered_tp = sum(tp for score, tp in zip(scores, tp_flags) if score >= thresh)
        filtered_fp = sum(1 - tp for score, tp in zip(scores, tp_flags) if score >= thresh)
        filtered_fn = total_gt - filtered_tp
        
        precision, recall, f1 = compute_precision_recall_f1(filtered_tp, filtered_fp, filtered_fn)
        
        results.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": filtered_tp,
            "fp": filtered_fp,
            "fn": filtered_fn,
        })
    
    # Find best threshold by F1
    best_result = max(results, key=lambda x: x["f1"])
    
    return {
        "sweep": results,
        "best_threshold": best_result["threshold"],
        "best_f1": best_result["f1"],
        "best_precision": best_result["precision"],
        "best_recall": best_result["recall"],
    }


def compute_per_class_ap(
    predictions: List[Dict[str, Tensor]],
    targets: List[Dict[str, Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    """Compute per-class Average Precision.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary mapping class_id to AP score
    """
    per_class_scores = {i: [] for i in range(num_classes)}
    per_class_tp_flags = {i: [] for i in range(num_classes)}
    per_class_gt_count = {i: 0 for i in range(num_classes)}
    
    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction["boxes"]
        pred_scores = prediction["scores"]
        pred_labels = prediction["labels"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        
        # Count GT per class
        for label in gt_labels.tolist():
            per_class_gt_count[label] += 1
        
        if pred_boxes.numel() == 0:
            continue
        
        # Sort by score
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]
        
        matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            label_int = int(label.item())
            per_class_scores[label_int].append(float(score.item()))
            
            candidate_mask = (gt_labels == label) & (~matched_gt)
            if not candidate_mask.any():
                per_class_tp_flags[label_int].append(0)
                continue
            
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
            pair_iou = box_iou(box.unsqueeze(0), gt_boxes[candidate_indices]).squeeze(0)
            best_iou, best_idx_local = pair_iou.max(dim=0)
            
            if float(best_iou.item()) >= iou_threshold:
                gt_index = candidate_indices[int(best_idx_local.item())]
                matched_gt[gt_index] = True
                per_class_tp_flags[label_int].append(1)
            else:
                per_class_tp_flags[label_int].append(0)
    
    # Compute AP per class
    per_class_ap = {}
    for class_id in range(num_classes):
        scores = per_class_scores[class_id]
        tp_flags = per_class_tp_flags[class_id]
        total_gt = per_class_gt_count[class_id]
        
        if not scores or total_gt == 0:
            per_class_ap[class_id] = 0.0
            continue
        
        # Sort by score
        sorted_pairs = sorted(zip(scores, tp_flags), key=lambda x: x[0], reverse=True)
        
        running_tp = 0
        running_fp = 0
        precisions = []
        recalls = []
        
        for score, tp_flag in sorted_pairs:
            running_tp += tp_flag
            running_fp += 1 - tp_flag
            precisions.append(running_tp / max(running_tp + running_fp, 1))
            recalls.append(running_tp / max(total_gt, 1))
        
        ap = compute_ap_from_pr_curve(precisions, recalls, method="continuous")
        per_class_ap[class_id] = ap
    
    return per_class_ap
