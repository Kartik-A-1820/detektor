from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from models.chimera import ChimeraODIS
from utils.benchmark import benchmark_forward, benchmark_predict
from utils.results import prediction_to_serializable, predictions_to_coco_like, summarize_image_result


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


def validate(
    config_path: str,
    weights: str = "chimera_last.pt",
    batch_size: int = 4,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    topk_pre_nms: int = 300,
    max_det: int = 100,
    mask_thresh: float = 0.5,
    save_json: str | None = None,
    run_benchmark: bool = False,
) -> Dict[str, Any]:
    """Run lightweight validation with detection and segmentation metrics."""
    if not (0.0 <= conf_thresh <= 1.0):
        raise ValueError(f"conf_thresh must be in [0, 1], got {conf_thresh}")

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = ChimeraODIS(
        num_classes=cfg["data"]["num_classes"],
        proto_k=cfg["model"]["proto_k"],
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    dataset = build_dataset(cfg, split="val")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_predictions: List[Dict[str, Tensor]] = []
    all_targets: List[Dict[str, Tensor]] = []
    serializable_results: List[Dict[str, Any]] = []
    image_summaries: List[Dict[str, Any]] = []
    total_images = 0
    total_predictions = 0
    invalid_batches = 0
    any_gt = False

    for imgs, targets in loader:
        imgs = imgs.to(device)
        target_list = list(targets) if isinstance(targets, Sequence) and not isinstance(targets, dict) else [targets]
        normalized_targets = _normalize_validation_targets(imgs, target_list)
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

        for prediction, target in zip(predictions, normalized_targets):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/chimera_s_512.yaml")
    parser.add_argument("--weights", type=str, default="chimera_last.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--topk-pre-nms", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test_validation_metrics()
    else:
        validate(
            config_path=args.config,
            weights=args.weights,
            batch_size=args.batch_size,
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            topk_pre_nms=args.topk_pre_nms,
            max_det=args.max_det,
            mask_thresh=args.mask_thresh,
            save_json=args.save_json or None,
            run_benchmark=args.benchmark,
        )
