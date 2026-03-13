"""Production-grade validation with comprehensive metrics and artifacts.

This module provides enhanced validation capabilities including:
- Precision, Recall, F1, AP50, AP50-95
- Per-class metrics and confusion matrix
- Threshold sweep analysis
- Segmentation metrics (IoU, Dice)
- Qualitative outputs (annotated images)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from models.factory import build_model_from_config, load_model_weights
from utils.box_ops import box_iou
from utils.collate import detection_segmentation_collate_fn
from utils.data_config import apply_dataset_yaml_overrides, print_resolved_dataset_config
from utils.metrics_helpers import (
    compute_ap50_95,
    compute_confusion_matrix,
    compute_dice_score,
    compute_mask_iou,
    compute_per_class_ap,
    compute_precision_recall_f1,
    threshold_sweep,
)
from utils.visualize import draw_boxes


def _normalize_validation_targets(
    imgs: Tensor,
    targets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Tensor]]:
    """Convert validation targets to absolute-image coordinates."""
    image_height, image_width = imgs.shape[-2:]
    device = imgs.device
    dtype = imgs.dtype
    scale = torch.tensor(
        [float(image_width), float(image_height), float(image_width), float(image_height)],
        device=device,
        dtype=dtype,
    )

    normalized: List[Dict[str, Tensor]] = []
    for target in targets:
        boxes = target.get("boxes", torch.zeros((0, 4), dtype=dtype)).to(device=device, dtype=dtype).reshape(-1, 4)
        labels = target.get("labels", torch.zeros((0,), dtype=torch.long)).to(device=device, dtype=torch.long).reshape(-1)
        masks = target.get("masks")
        
        if isinstance(masks, Tensor):
            masks = masks.to(device=device)
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            masks = masks.to(dtype=torch.bool)
        else:
            masks = torch.zeros((boxes.shape[0], image_height, image_width), device=device, dtype=torch.bool)

        if boxes.numel() > 0:
            boxes = boxes * scale
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0.0, max=float(image_width))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0.0, max=float(image_height))

        normalized.append({"boxes": boxes, "labels": labels, "masks": masks})
    return normalized


def _match_predictions_to_targets(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    pred_masks: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    gt_masks: Tensor,
    iou_threshold: float,
) -> Dict[str, Any]:
    """Match predictions to ground truth with detailed metrics."""
    if pred_boxes.numel() == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": int(gt_boxes.shape[0]),
            "matched_indices": torch.tensor([], dtype=torch.long),
            "box_ious": [],
            "mask_ious": [],
            "dice_scores": [],
            "scores": [],
            "tp_flags": [],
        }

    # Sort by score
    order = pred_scores.argsort(descending=True)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]
    pred_masks = pred_masks[order]

    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    matched_indices = torch.full((pred_boxes.shape[0],), -1, dtype=torch.long)
    box_ious: List[float] = []
    mask_ious: List[float] = []
    dice_scores: List[float] = []
    scores: List[float] = []
    tp_flags: List[int] = []

    for idx, (box, score, label, mask) in enumerate(zip(pred_boxes, pred_scores, pred_labels, pred_masks)):
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
            matched_indices[idx] = gt_index
            tp_flags.append(1)
            box_ious.append(float(best_iou.item()))
            
            # Compute mask metrics if available
            if mask.numel() > 0 and gt_masks.shape[0] > gt_index:
                gt_mask = gt_masks[gt_index]
                mask_iou = compute_mask_iou(mask, gt_mask)
                dice = compute_dice_score(mask, gt_mask)
                mask_ious.append(mask_iou)
                dice_scores.append(dice)
        else:
            tp_flags.append(0)

    tp = int(sum(tp_flags))
    fp = int(len(tp_flags) - tp)
    fn = int(gt_boxes.shape[0] - matched_gt.sum().item())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_indices": matched_indices,
        "box_ious": box_ious,
        "mask_ious": mask_ious,
        "dice_scores": dice_scores,
        "scores": scores,
        "tp_flags": tp_flags,
    }


def validate_comprehensive(
    config_path: str,
    data_yaml: Optional[str] = None,
    weights: str = "chimera_last.pt",
    batch_size: int = 4,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    topk_pre_nms: int = 300,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    output_dir: Optional[str] = None,
    save_images: bool = False,
    max_images_to_save: int = 20,
    compute_ap50_95_metric: bool = False,
) -> Dict[str, Any]:
    """Run comprehensive validation with production-grade metrics.
    
    Args:
        config_path: Path to config YAML
        data_yaml: Optional dataset YAML for overrides
        weights: Path to model weights
        batch_size: Validation batch size
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold for matching
        topk_pre_nms: Max candidates before NMS
        max_det: Max detections per image
        mask_thresh: Mask binarization threshold
        output_dir: Output directory for results
        save_images: Whether to save annotated images
        max_images_to_save: Max number of images to save
        compute_ap50_95_metric: Whether to compute AP50-95 (slower)
        
    Returns:
        Comprehensive metrics dictionary
    """
    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if data_yaml:
        resolved_dataset = apply_dataset_yaml_overrides(cfg, data_yaml)
        print_resolved_dataset_config(resolved_dataset)
        class_names = resolved_dataset.get("class_names", [f"class_{i}" for i in range(cfg["data"]["num_classes"])])
    else:
        class_names = [f"class_{i}" for i in range(cfg["data"]["num_classes"])]
    
    num_classes = cfg["data"]["num_classes"]
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)
    
    checkpoint = torch.load(weights, map_location=device)
    load_model_weights(model, checkpoint, strict=True)
    model.eval()
    
    # Setup dataset
    dataset = build_dataset(cfg, split="val")
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_segmentation_collate_fn,
    )
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"runs/validate/{Path(weights).stem}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_images:
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
    
    # Collect predictions and targets
    all_predictions: List[Dict[str, Tensor]] = []
    all_targets: List[Dict[str, Tensor]] = []
    all_matches: List[Dict[str, Any]] = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_box_ious: List[float] = []
    all_mask_ious: List[float] = []
    all_dice_scores: List[float] = []
    all_scores: List[float] = []
    all_tp_flags: List[int] = []
    
    per_class_tp = {i: 0 for i in range(num_classes)}
    per_class_fp = {i: 0 for i in range(num_classes)}
    per_class_fn = {i: 0 for i in range(num_classes)}
    
    images_saved = 0
    
    print(f"Running validation on {len(dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(tqdm(loader, desc="Validating")):
            imgs = imgs.to(device)
            normalized_targets = _normalize_validation_targets(imgs, targets)
            original_sizes = [(imgs.shape[-2], imgs.shape[-1]) for _ in range(imgs.shape[0])]

            predictions = model.predict(
                imgs,
                original_sizes=original_sizes,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                topk_pre_nms=topk_pre_nms,
                max_det=max_det,
                mask_thresh=mask_thresh,
            )

            for img_idx, (prediction, target) in enumerate(zip(predictions, normalized_targets)):
                # Move to CPU
                pred_cpu = {
                    "boxes": prediction["boxes"].detach().cpu(),
                    "scores": prediction["scores"].detach().cpu(),
                    "labels": prediction["labels"].detach().cpu(),
                    "masks": prediction["masks"].detach().cpu().to(dtype=torch.bool),
                }
                target_cpu = {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                    "masks": target["masks"].detach().cpu().to(dtype=torch.bool),
                }
                
                all_predictions.append(pred_cpu)
                all_targets.append(target_cpu)
                
                # Match predictions to targets
                match_result = _match_predictions_to_targets(
                    pred_cpu["boxes"],
                    pred_cpu["scores"],
                    pred_cpu["labels"],
                    pred_cpu["masks"],
                    target_cpu["boxes"],
                    target_cpu["labels"],
                    target_cpu["masks"],
                    iou_thresh,
                )
                
                all_matches.append(match_result)
                total_tp += match_result["tp"]
                total_fp += match_result["fp"]
                total_fn += match_result["fn"]
                all_box_ious.extend(match_result["box_ious"])
                all_mask_ious.extend(match_result["mask_ious"])
                all_dice_scores.extend(match_result["dice_scores"])
                all_scores.extend(match_result["scores"])
                all_tp_flags.extend(match_result["tp_flags"])
                
                # Per-class counts
                for label in pred_cpu["labels"].tolist():
                    if label in per_class_fp:
                        per_class_fp[label] += 1
                for label in target_cpu["labels"].tolist():
                    if label in per_class_fn:
                        per_class_fn[label] += 1
                for idx, tp_flag in enumerate(match_result["tp_flags"]):
                    if tp_flag == 1:
                        label = pred_cpu["labels"][idx].item()
                        if label in per_class_tp:
                            per_class_tp[label] += 1
                            per_class_fp[label] -= 1
                            per_class_fn[label] -= 1
                
                # Save annotated images
                if save_images and images_saved < max_images_to_save:
                    img_np = imgs[img_idx].detach().cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    
                    # Draw predictions
                    if pred_cpu["boxes"].numel() > 0:
                        img_np = draw_boxes(img_np, pred_cpu["boxes"].numpy())
                    
                    output_file = images_dir / f"val_{batch_idx}_{img_idx}.jpg"
                    cv2.imwrite(str(output_file), img_np)
                    images_saved += 1
    
    # Compute overall metrics
    precision, recall, f1 = compute_precision_recall_f1(total_tp, total_fp, total_fn)
    mean_box_iou = np.mean(all_box_ious) if all_box_ious else 0.0
    mean_mask_iou = np.mean(all_mask_ious) if all_mask_ious else 0.0
    mean_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
    
    # Compute AP50
    if all_scores:
        sorted_pairs = sorted(zip(all_scores, all_tp_flags), key=lambda x: x[0], reverse=True)
        running_tp = 0
        running_fp = 0
        precisions = []
        recalls = []
        total_gt = total_tp + total_fn
        
        for score, tp_flag in sorted_pairs:
            running_tp += tp_flag
            running_fp += 1 - tp_flag
            precisions.append(running_tp / max(running_tp + running_fp, 1))
            recalls.append(running_tp / max(total_gt, 1))
        
        ap50 = 0.0
        prev_recall = 0.0
        for p, r in zip(precisions, recalls):
            ap50 += p * max(r - prev_recall, 0.0)
            prev_recall = r
    else:
        ap50 = 0.0
        precisions = []
        recalls = []
    
    # Compute per-class AP
    print("Computing per-class AP...")
    per_class_ap = compute_per_class_ap(all_predictions, all_targets, num_classes, iou_thresh)
    map50 = np.mean(list(per_class_ap.values()))
    
    # Compute AP50-95 if requested
    ap50_95 = 0.0
    if compute_ap50_95_metric:
        print("Computing AP50-95 (this may take a while)...")
        ap50_95_values = []
        for pred, target in zip(all_predictions, all_targets):
            if pred["boxes"].numel() > 0 and target["boxes"].numel() > 0:
                ap = compute_ap50_95(
                    pred["boxes"],
                    pred["scores"],
                    pred["labels"],
                    target["boxes"],
                    target["labels"],
                )
                ap50_95_values.append(ap)
        ap50_95 = np.mean(ap50_95_values) if ap50_95_values else 0.0
    
    # Threshold sweep
    print("Performing threshold sweep...")
    sweep_result = threshold_sweep(all_scores, all_tp_flags, total_tp + total_fn)
    
    # Confusion matrix
    print("Computing confusion matrix...")
    all_pred_labels = torch.cat([p["labels"] for p in all_predictions if p["labels"].numel() > 0])
    all_gt_labels = torch.cat([t["labels"] for t in all_targets if t["labels"].numel() > 0])
    all_matched_indices = torch.cat([m["matched_indices"] for m in all_matches])
    
    confusion_matrix = compute_confusion_matrix(
        all_pred_labels,
        all_gt_labels,
        num_classes,
        all_matched_indices,
    )
    
    # Per-class metrics
    per_class_metrics = []
    for class_id in range(num_classes):
        cls_precision, cls_recall, cls_f1 = compute_precision_recall_f1(
            per_class_tp[class_id],
            per_class_fp[class_id],
            per_class_fn[class_id],
        )
        per_class_metrics.append({
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
            "ap50": per_class_ap.get(class_id, 0.0),
            "tp": per_class_tp[class_id],
            "fp": per_class_fp[class_id],
            "fn": per_class_fn[class_id],
        })
    
    # Compile results
    results = {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap50,
            "map50": map50,
            "ap50_95": ap50_95,
            "mean_box_iou": mean_box_iou,
            "mean_mask_iou": mean_mask_iou,
            "mean_dice": mean_dice,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "num_images": len(all_predictions),
        },
        "per_class": per_class_metrics,
        "threshold_sweep": sweep_result,
        "class_names": class_names,
        "confusion_matrix": confusion_matrix.tolist(),
        "precision_recall_curve": {
            "precision": precisions,
            "recall": recalls,
        },
    }
    
    # Save results
    print(f"Saving results to {output_path}...")
    
    # Save metrics.json
    with open(output_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save per_class_metrics.csv
    df_per_class = pd.DataFrame(per_class_metrics)
    df_per_class.to_csv(output_path / "per_class_metrics.csv", index=False)
    
    # Save confusion_matrix.csv
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names + ["background"],
        columns=class_names + ["background"],
    )
    df_cm.to_csv(output_path / "confusion_matrix.csv")
    
    # Save threshold sweep
    df_sweep = pd.DataFrame(sweep_result["sweep"])
    df_sweep.to_csv(output_path / "threshold_sweep.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Images evaluated: {len(all_predictions)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AP50: {ap50:.4f}")
    print(f"mAP50: {map50:.4f}")
    if compute_ap50_95_metric:
        print(f"AP50-95: {ap50_95:.4f}")
    print(f"Mean Box IoU: {mean_box_iou:.4f}")
    print(f"Mean Mask IoU: {mean_mask_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"\nBest threshold: {sweep_result['best_threshold']:.2f} (F1={sweep_result['best_f1']:.4f})")
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production-grade validation for Detektor")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data-yaml", type=str, default="", help="Dataset YAML for overrides")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--save-images", action="store_true", help="Save annotated images")
    parser.add_argument("--max-images", type=int, default=20, help="Max images to save")
    parser.add_argument("--compute-ap50-95", action="store_true", help="Compute AP50-95 (slower)")
    args = parser.parse_args()

    validate_comprehensive(
        config_path=args.config,
        data_yaml=args.data_yaml or None,
        weights=args.weights,
        batch_size=args.batch_size,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        output_dir=args.output_dir or None,
        save_images=args.save_images,
        max_images_to_save=args.max_images,
        compute_ap50_95_metric=args.compute_ap50_95,
    )
