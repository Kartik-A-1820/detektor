"""Automatic task detection for detection vs segmentation datasets.

This module analyzes datasets to determine if they contain:
- Detection only (bounding boxes)
- Segmentation only (masks/polygons)
- Both detection and segmentation
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class TaskMode(str, Enum):
    """Task mode enumeration."""
    DETECT = "detect"  # Bounding box detection only
    SEGMENT = "segment"  # Instance segmentation (masks + boxes)
    

def detect_yolo_format(label_path: Path) -> str:
    """Detect if a YOLO label file contains bbox or segmentation format.
    
    Args:
        label_path: Path to YOLO label file
        
    Returns:
        'bbox' for detection format, 'segment' for segmentation format
    """
    if not label_path.exists():
        return 'bbox'
    
    raw_text = label_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return 'bbox'
    
    for line in raw_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        # Segmentation format has more than 5 values (class_id + polygon points)
        # Detection format has exactly 5 values (class_id + bbox)
        if len(parts) > 5:
            return 'segment'
    
    return 'bbox'


def analyze_dataset_task(
    labels_dir: Path,
    sample_size: int = 50,
) -> Tuple[TaskMode, dict]:
    """Analyze a dataset to determine the task type.
    
    Args:
        labels_dir: Directory containing label files
        sample_size: Number of files to sample for analysis
        
    Returns:
        Tuple of (task_mode, stats_dict)
    """
    if not labels_dir.exists():
        return TaskMode.DETECT, {"error": "labels directory not found"}
    
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        return TaskMode.DETECT, {"error": "no label files found"}
    
    # Sample files
    sample_files = label_files[:min(sample_size, len(label_files))]
    
    bbox_count = 0
    segment_count = 0
    
    for label_file in sample_files:
        format_type = detect_yolo_format(label_file)
        if format_type == 'bbox':
            bbox_count += 1
        else:
            segment_count += 1
    
    stats = {
        "total_sampled": len(sample_files),
        "bbox_format": bbox_count,
        "segment_format": segment_count,
        "total_files": len(label_files),
    }
    
    # Determine task mode
    if segment_count > 0:
        # If any segmentation annotations found, use segment mode
        task_mode = TaskMode.SEGMENT
    else:
        # All bbox format, use detect mode
        task_mode = TaskMode.DETECT
    
    return task_mode, stats


def print_task_detection_summary(task_mode: TaskMode, stats: dict) -> None:
    """Print task detection summary.
    
    Args:
        task_mode: Detected task mode
        stats: Statistics dictionary
    """
    print("=" * 60)
    print("TASK DETECTION SUMMARY")
    print("=" * 60)
    print(f"Detected task mode: {task_mode.value}")
    print(f"Total label files: {stats.get('total_files', 0)}")
    print(f"Sampled files: {stats.get('total_sampled', 0)}")
    print(f"  - Bbox format: {stats.get('bbox_format', 0)}")
    print(f"  - Segment format: {stats.get('segment_format', 0)}")
    
    if task_mode == TaskMode.DETECT:
        print("\nMode: DETECTION ONLY")
        print("  - Training: Box detection losses only")
        print("  - Inference: Returns bounding boxes")
    else:
        print("\nMode: INSTANCE SEGMENTATION")
        print("  - Training: Box + mask losses")
        print("  - Inference: Returns bounding boxes + masks")
        print("  - Boxes auto-generated from masks if needed")
    
    print("=" * 60)


def boxes_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask [H, W]
        
    Returns:
        Bounding box [x1, y1, x2, y2] in normalized coordinates or None
    """
    if mask.sum() == 0:
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    h, w = mask.shape
    
    # Normalize to [0, 1]
    box = np.array([
        x1 / w,
        y1 / h,
        (x2 + 1) / w,  # +1 to include the last pixel
        (y2 + 1) / h,
    ], dtype=np.float32)
    
    # Clip to valid range
    box = np.clip(box, 0.0, 1.0)
    
    return box
