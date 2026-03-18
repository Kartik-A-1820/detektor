#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.detection import CenterPriorAssigner
from utils.anchors import concatenate_points_and_strides, generate_level_points


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_dataset_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "train" not in payload or "val" not in payload:
        raise ValueError(f"dataset yaml must define train and val splits: {path}")
    if "names" not in payload:
        raise ValueError(f"dataset yaml must define class names: {path}")
    return payload


def _normalize_class_names(names: Any) -> List[str]:
    if isinstance(names, dict):
        return [str(name) for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"unsupported class names payload: {type(names)!r}")


def _parse_label_file(label_path: Path, img_size: int) -> Tuple[List[List[float]], List[int]]:
    if not label_path.exists():
        return [], []
    raw_text = label_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return [], []

    boxes: List[List[float]] = []
    labels: List[int] = []
    for line in raw_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if len(parts) > 5:
            coords = [float(value) for value in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            x1 = max(0.0, min(1.0, min(xs)))
            y1 = max(0.0, min(1.0, min(ys)))
            x2 = max(0.0, min(1.0, max(xs)))
            y2 = max(0.0, min(1.0, max(ys)))
        else:
            xc, yc, width, height = map(float, parts[1:5])
            x1 = max(0.0, min(1.0, xc - width * 0.5))
            y1 = max(0.0, min(1.0, yc - height * 0.5))
            x2 = max(0.0, min(1.0, xc + width * 0.5))
            y2 = max(0.0, min(1.0, yc + height * 0.5))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1 * img_size, y1 * img_size, x2 * img_size, y2 * img_size])
        labels.append(class_id)
    return boxes, labels


def _build_points(img_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    points_per_level = []
    strides_per_level = []
    for feature_size in (img_size // 8, img_size // 16, img_size // 32):
        level_points, stride = generate_level_points(
            feature_height=feature_size,
            feature_width=feature_size,
            image_height=img_size,
            image_width=img_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        points_per_level.append(level_points)
        strides_per_level.append(stride)
    points, strides = concatenate_points_and_strides(points_per_level, strides_per_level)
    return points, strides, strides_per_level


def _quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return round(float(ordered[index]), 2)


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part / float(total) * 100.0, 2)


def _collect_split_summary(
    split_images_dir: Path,
    *,
    class_names: List[str],
    img_size: int,
    assigner: CenterPriorAssigner,
    points: torch.Tensor,
    strides: torch.Tensor,
) -> Dict[str, Any]:
    labels_dir = split_images_dir.parent / "labels"
    summary: Dict[str, Any] = {
        "images": 0,
        "annotations": 0,
        "class_counts": Counter(),
        "width_px": defaultdict(list),
        "height_px": defaultdict(list),
        "area_px": defaultdict(list),
        "assigned_points": defaultdict(list),
        "zero_assignment_gt": Counter(),
        "level_hits": defaultdict(Counter),
    }

    for image_path in sorted(split_images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        summary["images"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes, labels = _parse_label_file(label_path, img_size)
        if not boxes:
            continue

        gt_boxes = torch.tensor(boxes, dtype=torch.float32)
        match_matrix = assigner.build_match_matrix(
            points=points,
            strides=strides,
            gt_boxes=gt_boxes,
        )

        for gt_index, class_id in enumerate(labels):
            width_px = float(gt_boxes[gt_index, 2] - gt_boxes[gt_index, 0])
            height_px = float(gt_boxes[gt_index, 3] - gt_boxes[gt_index, 1])
            area_px = width_px * height_px
            summary["annotations"] += 1
            summary["class_counts"][class_id] += 1
            summary["width_px"][class_id].append(width_px)
            summary["height_px"][class_id].append(height_px)
            summary["area_px"][class_id].append(area_px)

            gt_matches = match_matrix[:, gt_index]
            assigned_points = int(gt_matches.sum().item())
            summary["assigned_points"][class_id].append(assigned_points)
            if assigned_points == 0:
                summary["zero_assignment_gt"][class_id] += 1
            else:
                for stride in (8, 16, 32):
                    stride_hits = int(((strides == float(stride)) & gt_matches).sum().item())
                    if stride_hits > 0:
                        summary["level_hits"][class_id][stride] += stride_hits

    class_summary: Dict[str, Any] = {}
    for class_id, class_name in enumerate(class_names):
        count = int(summary["class_counts"][class_id])
        class_summary[class_name] = {
            "count": count,
            "share_pct": _pct(count, int(summary["annotations"])),
            "median_w_px": _quantile(summary["width_px"][class_id], 0.5),
            "median_h_px": _quantile(summary["height_px"][class_id], 0.5),
            "median_area_px": _quantile(summary["area_px"][class_id], 0.5),
            "p10_area_px": _quantile(summary["area_px"][class_id], 0.1),
            "p90_area_px": _quantile(summary["area_px"][class_id], 0.9),
            "median_assigned_points": _quantile(summary["assigned_points"][class_id], 0.5),
            "p10_assigned_points": _quantile(summary["assigned_points"][class_id], 0.1),
            "p90_assigned_points": _quantile(summary["assigned_points"][class_id], 0.9),
            "zero_assignment_gt": int(summary["zero_assignment_gt"][class_id]),
            "zero_assignment_pct": _pct(int(summary["zero_assignment_gt"][class_id]), count),
            "level_hits": {str(stride): hits for stride, hits in sorted(summary["level_hits"][class_id].items())},
        }

    return {
        "images": int(summary["images"]),
        "annotations": int(summary["annotations"]),
        "classes": class_summary,
    }


def build_phase3_diagnostics(data_yaml: Path, *, img_size: int, center_radius: float) -> Dict[str, Any]:
    dataset = _load_dataset_yaml(data_yaml)
    class_names = _normalize_class_names(dataset["names"])
    points, strides, stride_levels = _build_points(img_size)
    assigner = CenterPriorAssigner(center_radius=center_radius)

    train_images_dir = Path(dataset["train"])
    val_images_dir = Path(dataset["val"])

    return {
        "data_yaml": str(data_yaml.resolve()),
        "img_size": int(img_size),
        "assigner": {
            "center_radius": float(center_radius),
            "min_effective_box_size": float(assigner.min_effective_box_size),
            "strides": stride_levels,
        },
        "train": _collect_split_summary(
            train_images_dir,
            class_names=class_names,
            img_size=img_size,
            assigner=assigner,
            points=points,
            strides=strides,
        ),
        "val": _collect_split_summary(
            val_images_dir,
            class_names=class_names,
            img_size=img_size,
            assigner=assigner,
            points=points,
            strides=strides,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 training-signal diagnostics for Detektor")
    parser.add_argument("--data-yaml", required=True, type=str, help="Dataset yaml to audit")
    parser.add_argument("--img-size", type=int, default=512, help="Training image size used for assignment analysis")
    parser.add_argument("--center-radius", type=float, default=2.5, help="Center-prior radius to audit")
    parser.add_argument(
        "--output",
        type=str,
        default="runs/phase3_diagnostics.json",
        help="JSON output path for the diagnostic summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = build_phase3_diagnostics(
        Path(args.data_yaml),
        img_size=int(args.img_size),
        center_radius=float(args.center_radius),
    )
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"phase3_diagnostics: {output_path}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
