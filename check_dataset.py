#!/usr/bin/env python
"""Dataset validation tool for YOLO-format datasets."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List

import yaml

from utils.dataset_validation import (
    DatasetStats,
    ValidationIssue,
    ValidationResult,
    collect_class_distribution,
    find_duplicate_filenames,
    get_image_size,
    validate_image_file,
    validate_label_file,
    write_validation_summary,
)


def resolve_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
) -> List[tuple[Path, Path]]:
    """Resolve image and label file pairs.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels

    Returns:
        List of (image_path, label_path) tuples
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []

    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    pairs = []
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path))

    return pairs


def validate_dataset_split(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    split_name: str,
) -> ValidationResult:
    """Validate a single dataset split (train/val/test).

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        num_classes: Number of classes in dataset
        split_name: Name of the split for logging

    Returns:
        ValidationResult object
    """
    result = ValidationResult()

    if not images_dir.exists():
        result.issues.append(
            ValidationIssue(
                severity="error",
                category="missing_directory",
                message=f"{split_name} images directory not found: {images_dir}",
                file_path=str(images_dir),
            )
        )
        result.has_errors = True
        return result

    if not labels_dir.exists():
        result.issues.append(
            ValidationIssue(
                severity="error",
                category="missing_directory",
                message=f"{split_name} labels directory not found: {labels_dir}",
                file_path=str(labels_dir),
            )
        )
        result.has_errors = True
        return result

    print(f"\nValidating {split_name} split...")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    pairs = resolve_image_label_pairs(images_dir, labels_dir)
    result.stats.total_images = len(pairs)

    if not pairs:
        result.issues.append(
            ValidationIssue(
                severity="warning",
                category="empty_split",
                message=f"No images found in {split_name} split",
                file_path=str(images_dir),
            )
        )
        result.has_warnings = True
        return result

    # Find duplicate filenames
    image_paths = [img for img, _ in pairs]
    duplicates = find_duplicate_filenames(image_paths)
    if duplicates:
        result.stats.duplicate_filenames.update(duplicates)
        result.issues.append(
            ValidationIssue(
                severity="warning",
                category="duplicate_filenames",
                message=f"Found {len(duplicates)} duplicate filenames: {', '.join(sorted(duplicates)[:5])}...",
                file_path=None,
            )
        )
        result.has_warnings = True

    # Validate each image-label pair
    for image_path, label_path in pairs:
        # Validate image
        image_issue = validate_image_file(image_path)
        if image_issue:
            result.issues.append(image_issue)
            if image_issue.severity == "error":
                result.has_errors = True
                if image_issue.category == "corrupt_image":
                    result.stats.corrupt_images += 1
            else:
                result.has_warnings = True
            continue

        # Get image size for stats
        img_size = get_image_size(image_path)
        if img_size:
            result.stats.image_size_distribution[img_size] = (
                result.stats.image_size_distribution.get(img_size, 0) + 1
            )

        # Validate label
        label_issues = validate_label_file(label_path, num_classes, image_path)
        for issue in label_issues:
            result.issues.append(issue)
            if issue.severity == "error":
                result.has_errors = True
            else:
                result.has_warnings = True
            if issue.category == "empty_label":
                result.stats.empty_labels += 1

        # Collect class distribution
        if label_path.exists():
            result.stats.total_labels += 1
            class_dist = collect_class_distribution(label_path)
            for class_id, count in class_dist.items():
                result.stats.class_distribution[class_id] = (
                    result.stats.class_distribution.get(class_id, 0) + count
                )
                result.stats.total_annotations += count

    return result


def merge_results(results: List[ValidationResult]) -> ValidationResult:
    """Merge multiple ValidationResult objects.

    Args:
        results: List of ValidationResult objects

    Returns:
        Merged ValidationResult
    """
    merged = ValidationResult()

    for result in results:
        merged.issues.extend(result.issues)
        merged.has_errors = merged.has_errors or result.has_errors
        merged.has_warnings = merged.has_warnings or result.has_warnings

        merged.stats.total_images += result.stats.total_images
        merged.stats.total_labels += result.stats.total_labels
        merged.stats.total_annotations += result.stats.total_annotations
        merged.stats.empty_labels += result.stats.empty_labels
        merged.stats.corrupt_images += result.stats.corrupt_images

        for class_id, count in result.stats.class_distribution.items():
            merged.stats.class_distribution[class_id] = (
                merged.stats.class_distribution.get(class_id, 0) + count
            )

        for size, count in result.stats.image_size_distribution.items():
            merged.stats.image_size_distribution[size] = (
                merged.stats.image_size_distribution.get(size, 0) + count
            )

        merged.stats.duplicate_filenames.update(result.stats.duplicate_filenames)

    return merged


def print_summary(result: ValidationResult) -> None:
    """Print validation summary to console.

    Args:
        result: ValidationResult object
    """
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nDataset Statistics:")
    print(f"  Total images: {result.stats.total_images}")
    print(f"  Total labels: {result.stats.total_labels}")
    print(f"  Total annotations: {result.stats.total_annotations}")
    print(f"  Empty labels: {result.stats.empty_labels}")
    print(f"  Corrupt images: {result.stats.corrupt_images}")

    if result.stats.class_distribution:
        print(f"\nClass Distribution:")
        for class_id in sorted(result.stats.class_distribution.keys()):
            count = result.stats.class_distribution[class_id]
            percentage = (count / result.stats.total_annotations * 100) if result.stats.total_annotations > 0 else 0
            print(f"  Class {class_id}: {count} annotations ({percentage:.1f}%)")

    if result.stats.image_size_distribution:
        print(f"\nImage Size Distribution (top 5):")
        sorted_sizes = sorted(
            result.stats.image_size_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for (h, w), count in sorted_sizes:
            percentage = (count / result.stats.total_images * 100) if result.stats.total_images > 0 else 0
            print(f"  {h}x{w}: {count} images ({percentage:.1f}%)")

    if result.stats.duplicate_filenames:
        print(f"\nDuplicate Filenames: {len(result.stats.duplicate_filenames)}")
        for filename in sorted(result.stats.duplicate_filenames)[:5]:
            print(f"  - {filename}")
        if len(result.stats.duplicate_filenames) > 5:
            print(f"  ... and {len(result.stats.duplicate_filenames) - 5} more")

    print(f"\nValidation Issues:")
    print(f"  Errors: {sum(1 for issue in result.issues if issue.severity == 'error')}")
    print(f"  Warnings: {sum(1 for issue in result.issues if issue.severity == 'warning')}")

    if result.has_errors:
        status = "[FAIL] VALIDATION FAILED - Errors found"
    elif result.has_warnings:
        status = "[WARN] VALIDATION PASSED - Warnings found"
    else:
        status = "[OK] VALIDATION PASSED - No issues found"

    print(f"\n{status}")


def configure_stdout() -> None:
    """Prefer UTF-8 output, but fall back safely when stdout cannot be reconfigured."""
    stdout = sys.stdout
    if not hasattr(stdout, "reconfigure"):
        return

    try:
        stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation, ValueError):
        return


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate YOLO-format dataset")
    parser.add_argument(
        "--data-yaml",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to write validation reports",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to validate (default: train val)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    configure_stdout()
    args = parse_args()

    data_yaml_path = Path(args.data_yaml)
    if not data_yaml_path.exists():
        print(f"Error: Dataset YAML not found: {data_yaml_path}")
        return 1

    # Load dataset YAML
    with data_yaml_path.open("r") as f:
        data_config = yaml.safe_load(f)

    num_classes = data_config.get("nc")
    if num_classes is None:
        print("Error: 'nc' (number of classes) not found in dataset YAML")
        return 1

    print(f"Dataset: {data_yaml_path}")
    print(f"Number of classes: {num_classes}")

    # Validate each split
    split_results = []
    for split in args.splits:
        split_path = data_config.get(split)
        if not split_path:
            print(f"Warning: '{split}' split not found in dataset YAML, skipping")
            continue

        # Convert images path to split root
        split_path = Path(split_path)
        if split_path.name == "images":
            split_root = split_path.parent
        else:
            split_root = split_path

        images_dir = split_root / "images"
        labels_dir = split_root / "labels"

        result = validate_dataset_split(images_dir, labels_dir, num_classes, split)
        split_results.append(result)

    if not split_results:
        print("Error: No splits were validated")
        return 1

    # Merge results
    merged_result = merge_results(split_results)

    # Print summary
    print_summary(merged_result)

    # Write outputs
    output_dir = Path(args.output_dir)
    write_validation_summary(merged_result, output_dir, dataset_name="dataset")

    # Return exit code
    if merged_result.has_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
