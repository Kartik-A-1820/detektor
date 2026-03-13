from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import build_dataset
from metrics import (
    compute_detection_metrics,
    compute_segmentation_metrics,
    smoke_test_detection_metrics,
    smoke_test_segmentation_metrics,
)
from models.factory import build_model_from_checkpoint, build_model_from_config, load_model_weights
from utils.benchmark import benchmark_forward, benchmark_predict
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
from utils.results import prediction_to_serializable, predictions_to_coco_like, summarize_image_result
from utils.visualize import draw_boxes


def _normalize_validation_targets(imgs: Tensor, targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
    """Convert validation targets to absolute-image coordinates and normalized mask dtypes."""
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


def _has_invalid_prediction(prediction: Dict[str, Tensor]) -> bool:
    """Check whether a prediction dict contains NaN or Inf values."""
    for key in ("boxes", "scores"):
        tensor = prediction[key]
        if tensor.numel() > 0 and not torch.isfinite(tensor).all():
            return True
    return False


def _resolve_task_mode(dataset: Any) -> str:
    task_mode = getattr(dataset, "task_mode", None)
    if task_mode is not None:
        return task_mode.value if hasattr(task_mode, "value") else str(task_mode)
    return "segment"


def _ensure_prediction_schema(
    prediction: Dict[str, Tensor],
    *,
    image_size: tuple[int, int],
    device: torch.device,
) -> Dict[str, Tensor]:
    if "masks" not in prediction:
        prediction = dict(prediction)
        prediction["masks"] = torch.zeros((prediction["boxes"].shape[0], image_size[0], image_size[1]), device=device, dtype=torch.bool)
    return prediction


def _resolve_class_names(cfg: Dict[str, Any], resolved_dataset: Dict[str, Any] | None) -> List[str]:
    if isinstance(resolved_dataset, dict):
        names = resolved_dataset.get("class_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    data_names = cfg.get("data", {}).get("class_names")
    if isinstance(data_names, list) and data_names:
        return [str(name) for name in data_names]
    num_classes = int(cfg.get("data", {}).get("num_classes", 1))
    return [f"class_{i}" for i in range(num_classes)]


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
    if pred_boxes.numel() == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": int(gt_boxes.shape[0]),
            "matched_indices": torch.empty((0,), dtype=torch.long),
            "box_ious": [],
            "mask_ious": [],
            "dice_scores": [],
            "scores": [],
            "tp_flags": [],
            "sorted_labels": torch.empty((0,), dtype=torch.long),
        }

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
            if mask.numel() > 0 and gt_masks.shape[0] > gt_index:
                gt_mask = gt_masks[gt_index]
                mask_ious.append(compute_mask_iou(mask, gt_mask))
                dice_scores.append(compute_dice_score(mask, gt_mask))
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
        "sorted_labels": pred_labels,
    }


def _save_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate(
    config_path: str,
    data_yaml: str | None = None,
    weights: str = "chimera_last.pt",
    batch_size: int = 4,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    topk_pre_nms: int = 300,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    save_json: str | None = None,
    run_benchmark: bool = False,
    output_dir: str | None = None,
    save_images: bool = False,
    max_images_to_save: int = 20,
    compute_ap50_95_metric: bool = False,
) -> Dict[str, Any]:
    """Run validation with optional comprehensive metrics and artifacts."""
    if not (0.0 <= conf_thresh <= 1.0):
        raise ValueError(f"conf_thresh must be in [0, 1], got {conf_thresh}")

    with open(config_path, "r", encoding="utf-8") as handle:
        fallback_cfg = yaml.safe_load(handle)

    checkpoint = torch.load(weights, map_location="cpu")
    checkpoint_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    cfg = checkpoint_cfg if isinstance(checkpoint_cfg, dict) else fallback_cfg

    resolved_dataset: Dict[str, Any] | None = None
    if data_yaml:
        resolved_dataset = apply_dataset_yaml_overrides(cfg, data_yaml)
        print_resolved_dataset_config(resolved_dataset)
    class_names = _resolve_class_names(cfg, resolved_dataset)
    num_classes = int(cfg["data"]["num_classes"])

    requested_device = cfg.get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in the config, but CUDA is not available on this machine")
    device = torch.device("cuda" if requested_device == "cuda" else "cpu")
    if isinstance(checkpoint, dict):
        model = build_model_from_checkpoint(checkpoint).to(device)
    else:
        model = build_model_from_config(cfg).to(device)
    checkpoint = torch.load(weights, map_location=device)
    load_model_weights(model, checkpoint, strict=True)
    model.eval()

    dataset = build_dataset(cfg, split="val")
    task_mode = _resolve_task_mode(dataset)
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_segmentation_collate_fn,
    )

    all_predictions: List[Dict[str, Tensor]] = []
    all_targets: List[Dict[str, Tensor]] = []
    serializable_results: List[Dict[str, Any]] = []
    image_summaries: List[Dict[str, Any]] = []
    all_matches: List[Dict[str, Any]] = []
    total_images = 0
    total_predictions = 0
    invalid_batches = 0
    any_gt = False
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_box_ious: List[float] = []
    all_mask_ious: List[float] = []
    all_dice_scores: List[float] = []
    all_scores: List[float] = []
    all_tp_flags: List[int] = []
    per_class_tp = {class_id: 0 for class_id in range(num_classes)}
    per_class_fp = {class_id: 0 for class_id in range(num_classes)}
    per_class_fn = {class_id: 0 for class_id in range(num_classes)}

    comprehensive_requested = bool(output_dir or save_images or compute_ap50_95_metric)
    output_path = Path(output_dir) if output_dir else Path("runs") / "validate" / Path(weights).stem
    images_dir = output_path / "images"
    images_saved = 0
    if comprehensive_requested:
        output_path.mkdir(parents=True, exist_ok=True)
        if save_images:
            images_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader):
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
                task=task_mode,
            )

            for image_idx, (prediction, target, image_size) in enumerate(zip(predictions, normalized_targets, original_sizes)):
                prediction = _ensure_prediction_schema(
                    prediction,
                    image_size=image_size,
                    device=imgs.device,
                )
                total_images += 1
                total_predictions += int(prediction["boxes"].shape[0])
                any_gt = any_gt or int(target["boxes"].shape[0]) > 0

                if _has_invalid_prediction(prediction):
                    invalid_batches += 1
                    prediction = {
                        "boxes": torch.nan_to_num(prediction["boxes"], nan=0.0, posinf=0.0, neginf=0.0),
                        "scores": torch.nan_to_num(prediction["scores"], nan=0.0, posinf=0.0, neginf=0.0),
                        "labels": prediction["labels"],
                        "masks": prediction["masks"].to(dtype=torch.bool),
                    }

                prediction = {
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

                all_predictions.append(prediction)
                all_targets.append(target_cpu)
                serializable_results.append(prediction_to_serializable(prediction))
                image_summaries.append(summarize_image_result(prediction, target_cpu))

                match_result = _match_predictions_to_targets(
                    prediction["boxes"],
                    prediction["scores"],
                    prediction["labels"],
                    prediction["masks"],
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

                for label in match_result["sorted_labels"].tolist():
                    if label in per_class_fp:
                        per_class_fp[label] += 1
                for label in target_cpu["labels"].tolist():
                    if label in per_class_fn:
                        per_class_fn[label] += 1
                for idx, tp_flag in enumerate(match_result["tp_flags"]):
                    if tp_flag == 1:
                        label = int(match_result["sorted_labels"][idx].item())
                        if label in per_class_tp:
                            per_class_tp[label] += 1
                            per_class_fp[label] -= 1
                        gt_index = int(match_result["matched_indices"][idx].item())
                        if gt_index >= 0:
                            gt_label = int(target_cpu["labels"][gt_index].item())
                            if gt_label in per_class_fn:
                                per_class_fn[gt_label] -= 1

                if save_images and images_saved < max_images_to_save:
                    image_np = imgs[image_idx]
                    image_np = image_np.detach().cpu().permute(1, 2, 0).numpy()
                    image_np = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    if prediction["boxes"].numel() > 0:
                        image_np = draw_boxes(image_np, prediction["boxes"].numpy())
                    output_file = images_dir / f"val_{batch_idx}_{images_saved}.jpg"
                    cv2.imwrite(str(output_file), image_np)
                    images_saved += 1

    det_metrics = compute_detection_metrics(all_predictions, all_targets, iou_threshold=iou_thresh)
    seg_metrics = compute_segmentation_metrics(all_predictions, all_targets, iou_threshold=iou_thresh)

    summary: Dict[str, Any] = {
        "num_images": total_images,
        "total_predictions": total_predictions,
        "invalid_prediction_batches": invalid_batches,
        "detection": det_metrics,
        "segmentation": seg_metrics,
        "image_summaries": image_summaries,
    }

    if run_benchmark:
        summary["benchmark_forward"] = benchmark_forward(model, device=device)
        summary["benchmark_predict"] = benchmark_predict(model, device=device)

    if comprehensive_requested:
        precision, recall, f1 = compute_precision_recall_f1(total_tp, total_fp, total_fn)
        mean_box_iou = float(np.mean(all_box_ious)) if all_box_ious else 0.0
        mean_mask_iou = float(np.mean(all_mask_ious)) if all_mask_ious else 0.0
        mean_dice = float(np.mean(all_dice_scores)) if all_dice_scores else 0.0

        sorted_pairs = sorted(zip(all_scores, all_tp_flags), key=lambda item: item[0], reverse=True)
        running_tp = 0
        running_fp = 0
        precisions: List[float] = []
        recalls: List[float] = []
        total_gt = total_tp + total_fn
        for _, tp_flag in sorted_pairs:
            running_tp += tp_flag
            running_fp += 1 - tp_flag
            precisions.append(running_tp / max(running_tp + running_fp, 1))
            recalls.append(running_tp / max(total_gt, 1))
        ap50 = det_metrics.get("ap50", 0.0)

        per_class_ap = compute_per_class_ap(all_predictions, all_targets, num_classes, iou_thresh)
        map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
        ap50_95 = 0.0
        if compute_ap50_95_metric:
            ap50_95_values = []
            for prediction, target in zip(all_predictions, all_targets):
                if prediction["boxes"].numel() > 0 and target["boxes"].numel() > 0:
                    ap50_95_values.append(
                        compute_ap50_95(
                            prediction["boxes"],
                            prediction["scores"],
                            prediction["labels"],
                            target["boxes"],
                            target["labels"],
                        )
                    )
            ap50_95 = float(np.mean(ap50_95_values)) if ap50_95_values else 0.0

        sweep_result = threshold_sweep(all_scores, all_tp_flags, total_gt)
        pred_label_tensors = [match["sorted_labels"] for match in all_matches if match["sorted_labels"].numel() > 0]
        gt_label_tensors = [target["labels"] for target in all_targets if target["labels"].numel() > 0]
        matched_index_tensors = [match["matched_indices"] for match in all_matches if match["matched_indices"].numel() > 0]
        all_pred_labels = torch.cat(pred_label_tensors) if pred_label_tensors else torch.empty((0,), dtype=torch.long)
        all_gt_labels = torch.cat(gt_label_tensors) if gt_label_tensors else torch.empty((0,), dtype=torch.long)
        all_matched_indices = torch.cat(matched_index_tensors) if matched_index_tensors else torch.empty((0,), dtype=torch.long)
        confusion_matrix = compute_confusion_matrix(
            all_pred_labels,
            all_gt_labels,
            num_classes,
            all_matched_indices,
        )

        per_class_metrics: List[Dict[str, Any]] = []
        for class_id in range(num_classes):
            cls_precision, cls_recall, cls_f1 = compute_precision_recall_f1(
                per_class_tp[class_id],
                per_class_fp[class_id],
                per_class_fn[class_id],
            )
            per_class_metrics.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                    "precision": cls_precision,
                    "recall": cls_recall,
                    "f1": cls_f1,
                    "ap50": per_class_ap.get(class_id, 0.0),
                    "tp": per_class_tp[class_id],
                    "fp": per_class_fp[class_id],
                    "fn": per_class_fn[class_id],
                }
            )

        comprehensive = {
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
                "num_images": total_images,
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
        summary["comprehensive"] = comprehensive

        with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(comprehensive, handle, indent=2)
        _save_csv_rows(
            output_path / "per_class_metrics.csv",
            ["class_id", "class_name", "precision", "recall", "f1", "ap50", "tp", "fp", "fn"],
            per_class_metrics,
        )
        cm_rows = []
        cm_labels = class_names + ["background"]
        for label, row in zip(cm_labels, confusion_matrix.tolist()):
            cm_row = {"label": label}
            for column_label, value in zip(cm_labels, row):
                cm_row[column_label] = value
            cm_rows.append(cm_row)
        _save_csv_rows(output_path / "confusion_matrix.csv", ["label", *cm_labels], cm_rows)
        _save_csv_rows(
            output_path / "threshold_sweep.csv",
            ["threshold", "precision", "recall", "f1", "tp", "fp", "fn"],
            sweep_result.get("sweep", []),
        )

    if save_json is not None:
        output_path = Path(save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "num_images": total_images,
                "total_predictions": total_predictions,
                "invalid_prediction_batches": invalid_batches,
                "detection": det_metrics,
                "segmentation": seg_metrics,
            },
            "predictions": serializable_results,
            "coco_like": predictions_to_coco_like(all_predictions),
            "image_summaries": image_summaries,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print(f"images_evaluated: {total_images}")
    print(f"total_predictions: {total_predictions}")
    print(f"detection_precision: {det_metrics['precision']:.4f}")
    print(f"detection_recall: {det_metrics['recall']:.4f}")
    print(f"detection_ap50: {det_metrics['ap50']:.4f}")
    print(f"mean_box_iou: {det_metrics['mean_iou']:.4f}")
    print(f"mask_precision: {seg_metrics['precision']:.4f}")
    print(f"mask_recall: {seg_metrics['recall']:.4f}")
    print(f"mask_ap50: {seg_metrics['ap50']:.4f}")
    print(f"mean_mask_iou: {seg_metrics['mean_mask_iou']:.4f}")
    if comprehensive_requested:
        comprehensive_overall = summary["comprehensive"]["overall"]
        print(f"comprehensive_f1: {comprehensive_overall['f1']:.4f}")
        print(f"comprehensive_map50: {comprehensive_overall['map50']:.4f}")
        if compute_ap50_95_metric:
            print(f"comprehensive_ap50_95: {comprehensive_overall['ap50_95']:.4f}")
        print(f"validation_artifacts: {output_path}")

    if total_predictions == 0:
        print("warning: no predictions were produced during validation")
    if not any_gt:
        print("warning: no ground-truth instances were found in the validation set")
    if invalid_batches > 0:
        print(f"warning: {invalid_batches} prediction entries contained NaN/Inf and were sanitized")

    return summary


def smoke_test_validation_metrics() -> Dict[str, Any]:
    """Run lightweight synthetic smoke tests for validation metrics."""
    det_metrics = smoke_test_detection_metrics()
    seg_metrics = smoke_test_segmentation_metrics()
    summary = {"detection": det_metrics, "segmentation": seg_metrics}
    print("validation_smoke_test_complete: true")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Detektor weights on a configured validation split")
    parser.add_argument("--config", type=str, default="configs/chimera_s_512.yaml", help="Path to the base validation config YAML")
    parser.add_argument("--data-yaml", type=str, default="", help="Optional YOLO/Roboflow dataset YAML used to override validation roots and class metadata")
    parser.add_argument("--weights", type=str, default="chimera_last.pt", help="Path to model weights or checkpoint")
    parser.add_argument("--batch-size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used before NMS")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold used for NMS and metrics")
    parser.add_argument("--topk-pre-nms", type=int, default=300, help="Maximum number of candidates kept before NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image after NMS")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Mask threshold for binary mask generation")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save validation results JSON")
    parser.add_argument("--output-dir", type=str, default="", help="Optional directory for comprehensive validation artifacts")
    parser.add_argument("--save-images", action="store_true", help="Save annotated validation images into the output directory")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum number of annotated validation images to save")
    parser.add_argument("--compute-ap50-95", action="store_true", help="Compute AP50-95 and emit comprehensive validation artifacts")
    parser.add_argument("--benchmark", action="store_true", help="Run lightweight forward and predict benchmarks during validation")
    parser.add_argument("--smoke-test", action="store_true", help="Run only the lightweight synthetic validation smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test_validation_metrics()
    else:
        validate(
            config_path=args.config,
            data_yaml=args.data_yaml or None,
            weights=args.weights,
            batch_size=args.batch_size,
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            topk_pre_nms=args.topk_pre_nms,
            max_det=args.max_det,
            mask_thresh=args.mask_thresh,
            save_json=args.save_json or None,
            run_benchmark=args.benchmark,
            output_dir=args.output_dir or None,
            save_images=args.save_images,
            max_images_to_save=args.max_images,
            compute_ap50_95_metric=args.compute_ap50_95,
        )
