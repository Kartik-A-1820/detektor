from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor


EPS = 1e-6


def mask_iou(masks1: Tensor, masks2: Tensor) -> Tensor:
    """Compute pairwise IoU between binary instance masks."""
    masks1 = masks1.to(dtype=torch.bool)
    masks2 = masks2.to(dtype=torch.bool)
    inter = (masks1[:, None] & masks2[None, :]).flatten(2).sum(dim=-1).float()
    union = (masks1[:, None] | masks2[None, :]).flatten(2).sum(dim=-1).float()
    return inter / union.clamp(min=EPS)


def _match_masks(
    pred_masks: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_masks: Tensor,
    gt_labels: Tensor,
    iou_threshold: float,
) -> Dict[str, Any]:
    """Match predicted instance masks to GT masks using greedy class-aware IoU matching."""
    if pred_masks.numel() == 0:
        return {"tp": 0, "fp": 0, "fn": int(gt_masks.shape[0]), "ious": [], "scores": [], "tp_flags": []}

    order = pred_scores.argsort(descending=True)
    pred_masks = pred_masks[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]

    matched_gt = torch.zeros(gt_masks.shape[0], dtype=torch.bool, device=gt_masks.device)
    ious: List[float] = []
    scores: List[float] = []
    tp_flags: List[int] = []

    for mask, score, label in zip(pred_masks, pred_scores, pred_labels):
        scores.append(float(score.item()))
        candidate_mask = (gt_labels == label) & (~matched_gt)
        if not candidate_mask.any():
            tp_flags.append(0)
            continue
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        pair_iou = mask_iou(mask.unsqueeze(0), gt_masks[candidate_indices]).squeeze(0)
        best_iou, best_idx_local = pair_iou.max(dim=0)
        if float(best_iou.item()) >= iou_threshold:
            gt_index = candidate_indices[int(best_idx_local.item())]
            matched_gt[gt_index] = True
            tp_flags.append(1)
            ious.append(float(best_iou.item()))
        else:
            tp_flags.append(0)

    tp = int(sum(tp_flags))
    fp = int(len(tp_flags) - tp)
    fn = int(gt_masks.shape[0] - matched_gt.sum().item())
    return {"tp": tp, "fp": fp, "fn": fn, "ious": ious, "scores": scores, "tp_flags": tp_flags}


def compute_segmentation_metrics(
    predictions: Sequence[Dict[str, Tensor]],
    targets: Sequence[Dict[str, Tensor]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute lightweight instance-segmentation metrics across a validation set."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predictions = 0
    all_ious: List[float] = []
    all_scores: List[float] = []
    all_tp_flags: List[int] = []

    for prediction, target in zip(predictions, targets):
        pred_masks = prediction["masks"].detach().cpu().to(dtype=torch.bool)
        pred_scores = prediction["scores"].detach().cpu()
        pred_labels = prediction["labels"].detach().cpu().to(dtype=torch.long)
        gt_masks = target["masks"].detach().cpu().to(dtype=torch.bool)
        gt_labels = target["labels"].detach().cpu().to(dtype=torch.long)

        total_predictions += int(pred_masks.shape[0])
        matched = _match_masks(
            pred_masks=pred_masks,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            iou_threshold=iou_threshold,
        )
        total_tp += matched["tp"]
        total_fp += matched["fp"]
        total_fn += matched["fn"]
        all_ious.extend(matched["ious"])
        all_scores.extend(matched["scores"])
        all_tp_flags.extend(matched["tp_flags"])

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    mean_mask_iou = sum(all_ious) / max(len(all_ious), 1)

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

    return {
        "precision": precision,
        "recall": recall,
        "mean_mask_iou": mean_mask_iou,
        "ap50": ap50,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_predictions": total_predictions,
    }


def smoke_test_segmentation_metrics() -> Dict[str, Any]:
    """Run a small synthetic smoke test for segmentation metrics."""
    pred_mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    pred_mask[:, 10:30, 10:30] = True
    gt_mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    gt_mask[:, 12:28, 12:28] = True
    predictions = [{"boxes": torch.zeros((1, 4)), "scores": torch.tensor([0.9]), "labels": torch.tensor([0]), "masks": pred_mask}]
    targets = [{"boxes": torch.zeros((1, 4)), "labels": torch.tensor([0]), "masks": gt_mask}]
    metrics = compute_segmentation_metrics(predictions, targets)
    print(f"mask_precision: {metrics['precision']:.4f}")
    print(f"mask_recall: {metrics['recall']:.4f}")
    print(f"mask_ap50: {metrics['ap50']:.4f}")
    print(f"mask_mean_iou: {metrics['mean_mask_iou']:.4f}")
    return metrics
