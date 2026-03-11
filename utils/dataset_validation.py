"""Dataset validation utilities for YOLO-format datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    severity: str  # "error" or "warning"
    category: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class DatasetStats:
    """Statistics about the dataset."""

    total_images: int = 0
    total_labels: int = 0
    total_annotations: int = 0
    class_distribution: Dict[int, int] = field(default_factory=dict)
    image_size_distribution: Dict[Tuple[int, int], int] = field(default_factory=dict)
    duplicate_filenames: Set[str] = field(default_factory=set)
    empty_labels: int = 0
    corrupt_images: int = 0


@dataclass
class ValidationResult:
    """Complete validation result."""

    issues: List[ValidationIssue] = field(default_factory=list)
    stats: DatasetStats = field(default_factory=DatasetStats)
    has_errors: bool = False
    has_warnings: bool = False


def validate_image_file(image_path: Path) -> Optional[ValidationIssue]:
    """Validate that an image file exists and is readable.

    Args:
        image_path: Path to image file

    Returns:
        ValidationIssue if there's a problem, None otherwise
    """
    if not image_path.exists():
        return ValidationIssue(
            severity="error",
            category="missing_image",
            message=f"Image file not found",
            file_path=str(image_path),
        )

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return ValidationIssue(
                severity="error",
                category="corrupt_image",
                message=f"Image file is corrupt or unreadable",
                file_path=str(image_path),
            )
        if img.shape[0] == 0 or img.shape[1] == 0:
            return ValidationIssue(
                severity="error",
                category="invalid_image",
                message=f"Image has zero dimensions: {img.shape}",
                file_path=str(image_path),
            )
    except Exception as e:  # noqa: BLE001
        return ValidationIssue(
            severity="error",
            category="corrupt_image",
            message=f"Failed to read image: {str(e)}",
            file_path=str(image_path),
        )

    return None


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions (height, width).

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (height, width) or None if image cannot be read
    """
    try:
        img = cv2.imread(str(image_path))
        if img is not None:
            return (img.shape[0], img.shape[1])
    except Exception:  # noqa: BLE001, S110
        pass
    return None


def validate_label_file(
    label_path: Path,
    num_classes: int,
    image_path: Optional[Path] = None,
) -> List[ValidationIssue]:
    """Validate a YOLO format label file.

    Args:
        label_path: Path to label file
        num_classes: Expected number of classes
        image_path: Optional path to corresponding image for existence check

    Returns:
        List of ValidationIssue objects
    """
    issues: List[ValidationIssue] = []

    if not label_path.exists():
        if image_path and image_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="missing_label",
                    message=f"Label file not found for image",
                    file_path=str(label_path),
                )
            )
        return issues

    try:
        with label_path.open("r") as f:
            lines = f.readlines()

        if not lines:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="empty_label",
                    message=f"Label file is empty",
                    file_path=str(label_path),
                )
            )
            return issues

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="malformed_label",
                        message=f"Expected at least 5 values (class x y w h), got {len(parts)}",
                        file_path=str(label_path),
                        line_number=line_num,
                    )
                )
                continue

            # Validate class ID
            try:
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="invalid_class_id",
                            message=f"Class ID {class_id} out of range [0, {num_classes-1}]",
                            file_path=str(label_path),
                            line_number=line_num,
                        )
                    )
            except ValueError:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="malformed_label",
                        message=f"Class ID must be an integer, got '{parts[0]}'",
                        file_path=str(label_path),
                        line_number=line_num,
                    )
                )
                continue

            # Validate coordinates
            try:
                x_center, y_center, width, height = map(float, parts[1:5])

                # Check normalization (should be in [0, 1])
                for coord_name, coord_val in [
                    ("x_center", x_center),
                    ("y_center", y_center),
                    ("width", width),
                    ("height", height),
                ]:
                    if coord_val < 0.0 or coord_val > 1.0:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                category="invalid_coordinates",
                                message=f"{coord_name}={coord_val} not in normalized range [0, 1]",
                                file_path=str(label_path),
                                line_number=line_num,
                            )
                        )

                # Check for zero or negative width/height
                if width <= 0.0:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="invalid_coordinates",
                            message=f"width={width} must be positive",
                            file_path=str(label_path),
                            line_number=line_num,
                        )
                    )
                if height <= 0.0:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="invalid_coordinates",
                            message=f"height={height} must be positive",
                            file_path=str(label_path),
                            line_number=line_num,
                        )
                    )

            except ValueError as e:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="malformed_label",
                        message=f"Invalid coordinate values: {str(e)}",
                        file_path=str(label_path),
                        line_number=line_num,
                    )
                )

    except Exception as e:  # noqa: BLE001
        issues.append(
            ValidationIssue(
                severity="error",
                category="corrupt_label",
                message=f"Failed to read label file: {str(e)}",
                file_path=str(label_path),
            )
        )

    return issues


def collect_class_distribution(label_path: Path) -> Dict[int, int]:
    """Collect class distribution from a label file.

    Args:
        label_path: Path to label file

    Returns:
        Dictionary mapping class_id to count
    """
    distribution: Dict[int, int] = defaultdict(int)

    if not label_path.exists():
        return distribution

    try:
        with label_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        distribution[class_id] += 1
                    except ValueError:  # noqa: S110, PERF203
                        pass
    except Exception:  # noqa: BLE001, S110
        pass

    return distribution


def find_duplicate_filenames(image_paths: List[Path]) -> Set[str]:
    """Find duplicate filenames (not full paths).

    Args:
        image_paths: List of image paths

    Returns:
        Set of duplicate filenames
    """
    filename_counts = Counter(p.name for p in image_paths)
    return {name for name, count in filename_counts.items() if count > 1}


def write_validation_summary(
    result: ValidationResult,
    output_dir: Path,
    dataset_name: str = "dataset",
) -> None:
    """Write validation summary to JSON and CSV files.

    Args:
        result: ValidationResult object
        output_dir: Directory to write outputs
        dataset_name: Name of the dataset for file naming
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON summary
    json_path = output_dir / f"{dataset_name}_check.json"
    summary = {
        "has_errors": result.has_errors,
        "has_warnings": result.has_warnings,
        "total_issues": len(result.issues),
        "error_count": sum(1 for issue in result.issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in result.issues if issue.severity == "warning"),
        "stats": {
            "total_images": result.stats.total_images,
            "total_labels": result.stats.total_labels,
            "total_annotations": result.stats.total_annotations,
            "empty_labels": result.stats.empty_labels,
            "corrupt_images": result.stats.corrupt_images,
            "class_distribution": {str(k): v for k, v in result.stats.class_distribution.items()},
            "image_size_distribution": {
                f"{h}x{w}": count for (h, w), count in result.stats.image_size_distribution.items()
            },
            "duplicate_filenames": sorted(result.stats.duplicate_filenames),
        },
        "issues": [
            {
                "severity": issue.severity,
                "category": issue.category,
                "message": issue.message,
                "file_path": issue.file_path,
                "line_number": issue.line_number,
            }
            for issue in result.issues
        ],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write CSV summary
    csv_path = output_dir / f"{dataset_name}_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["severity", "category", "message", "file_path", "line_number"])
        for issue in result.issues:
            writer.writerow(
                [
                    issue.severity,
                    issue.category,
                    issue.message,
                    issue.file_path or "",
                    issue.line_number or "",
                ]
            )

    print(f"Validation summary written to {json_path}")
    print(f"Validation issues written to {csv_path}")


__all__ = [
    "DatasetStats",
    "ValidationIssue",
    "ValidationResult",
    "collect_class_distribution",
    "find_duplicate_filenames",
    "get_image_size",
    "validate_image_file",
    "validate_label_file",
    "write_validation_summary",
]
