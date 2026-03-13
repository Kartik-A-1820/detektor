"""Tests for dataset validation utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from check_dataset import print_summary
from utils.dataset_validation import (
    DatasetStats,
    ValidationIssue,
    collect_class_distribution,
    find_duplicate_filenames,
    get_image_size,
    validate_image_file,
    validate_label_file,
    write_validation_summary,
    ValidationResult,
)


class TestDatasetValidation(unittest.TestCase):
    """Tests for dataset validation functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        temp_root = Path(__file__).resolve().parents[1] / "reports" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(dir=str(temp_root))
        self.temp_path = Path(self.temp_dir)

    def create_test_image(self, path: Path, width: int = 100, height: int = 100) -> None:
        """Create a test image."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[10:90, 10:90] = 255
        cv2.imwrite(str(path), img)

    def create_test_label(self, path: Path, content: str) -> None:
        """Create a test label file."""
        path.write_text(content)

    def test_validate_image_file_exists(self) -> None:
        """Test validation of existing valid image."""
        img_path = self.temp_path / "test.jpg"
        self.create_test_image(img_path)

        issue = validate_image_file(img_path)

        self.assertIsNone(issue)

    def test_validate_image_file_missing(self) -> None:
        """Test validation of missing image."""
        img_path = self.temp_path / "missing.jpg"

        issue = validate_image_file(img_path)

        self.assertIsNotNone(issue)
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.category, "missing_image")

    def test_validate_image_file_corrupt(self) -> None:
        """Test validation of corrupt image."""
        img_path = self.temp_path / "corrupt.jpg"
        img_path.write_bytes(b"not an image")

        issue = validate_image_file(img_path)

        self.assertIsNotNone(issue)
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.category, "corrupt_image")

    def test_get_image_size(self) -> None:
        """Test getting image dimensions."""
        img_path = self.temp_path / "test.jpg"
        self.create_test_image(img_path, width=200, height=150)

        size = get_image_size(img_path)

        self.assertEqual(size, (150, 200))

    def test_get_image_size_missing(self) -> None:
        """Test getting size of missing image."""
        img_path = self.temp_path / "missing.jpg"

        size = get_image_size(img_path)

        self.assertIsNone(size)

    def test_validate_label_file_valid(self) -> None:
        """Test validation of valid label file."""
        label_path = self.temp_path / "test.txt"
        self.create_test_label(label_path, "0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.15\n")

        issues = validate_label_file(label_path, num_classes=4)

        self.assertEqual(len(issues), 0)

    def test_validate_label_file_missing(self) -> None:
        """Test validation of missing label file."""
        label_path = self.temp_path / "missing.txt"
        img_path = self.temp_path / "test.jpg"
        self.create_test_image(img_path)

        issues = validate_label_file(label_path, num_classes=4, image_path=img_path)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")
        self.assertEqual(issues[0].category, "missing_label")

    def test_validate_label_file_empty(self) -> None:
        """Test validation of empty label file."""
        label_path = self.temp_path / "empty.txt"
        self.create_test_label(label_path, "")

        issues = validate_label_file(label_path, num_classes=4)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")
        self.assertEqual(issues[0].category, "empty_label")

    def test_validate_label_file_malformed(self) -> None:
        """Test validation of malformed label file."""
        label_path = self.temp_path / "malformed.txt"
        self.create_test_label(label_path, "0 0.5 0.5\n")  # Missing width and height

        issues = validate_label_file(label_path, num_classes=4)

        self.assertGreater(len(issues), 0)
        self.assertTrue(any(issue.category == "malformed_label" for issue in issues))

    def test_validate_label_file_invalid_class_id(self) -> None:
        """Test validation with invalid class ID."""
        label_path = self.temp_path / "invalid_class.txt"
        self.create_test_label(label_path, "5 0.5 0.5 0.2 0.3\n")  # Class 5 out of range

        issues = validate_label_file(label_path, num_classes=4)

        self.assertGreater(len(issues), 0)
        self.assertTrue(any(issue.category == "invalid_class_id" for issue in issues))

    def test_validate_label_file_invalid_coordinates(self) -> None:
        """Test validation with invalid coordinates."""
        label_path = self.temp_path / "invalid_coords.txt"
        self.create_test_label(
            label_path,
            "0 1.5 0.5 0.2 0.3\n"  # x_center > 1.0
            "1 0.5 -0.1 0.2 0.3\n"  # y_center < 0.0
            "2 0.5 0.5 0.0 0.3\n"  # width = 0
            "3 0.5 0.5 0.2 -0.1\n",  # height < 0
        )

        issues = validate_label_file(label_path, num_classes=4)

        self.assertGreater(len(issues), 0)
        self.assertTrue(any(issue.category == "invalid_coordinates" for issue in issues))

    def test_validate_label_file_non_numeric_class(self) -> None:
        """Test validation with non-numeric class ID."""
        label_path = self.temp_path / "non_numeric.txt"
        self.create_test_label(label_path, "abc 0.5 0.5 0.2 0.3\n")

        issues = validate_label_file(label_path, num_classes=4)

        self.assertGreater(len(issues), 0)
        self.assertTrue(any(issue.category == "malformed_label" for issue in issues))

    def test_collect_class_distribution(self) -> None:
        """Test collecting class distribution."""
        label_path = self.temp_path / "test.txt"
        self.create_test_label(
            label_path,
            "0 0.5 0.5 0.2 0.3\n"
            "1 0.3 0.4 0.1 0.15\n"
            "0 0.7 0.6 0.15 0.2\n"
            "2 0.2 0.3 0.1 0.1\n",
        )

        distribution = collect_class_distribution(label_path)

        self.assertEqual(distribution[0], 2)
        self.assertEqual(distribution[1], 1)
        self.assertEqual(distribution[2], 1)

    def test_collect_class_distribution_missing_file(self) -> None:
        """Test collecting distribution from missing file."""
        label_path = self.temp_path / "missing.txt"

        distribution = collect_class_distribution(label_path)

        self.assertEqual(len(distribution), 0)

    def test_find_duplicate_filenames(self) -> None:
        """Test finding duplicate filenames."""
        paths = [
            Path("/path1/image1.jpg"),
            Path("/path2/image1.jpg"),  # Duplicate filename
            Path("/path1/image2.jpg"),
            Path("/path3/image3.jpg"),
        ]

        duplicates = find_duplicate_filenames(paths)

        self.assertIn("image1.jpg", duplicates)
        self.assertNotIn("image2.jpg", duplicates)
        self.assertNotIn("image3.jpg", duplicates)

    def test_find_duplicate_filenames_no_duplicates(self) -> None:
        """Test finding duplicates when there are none."""
        paths = [
            Path("/path1/image1.jpg"),
            Path("/path1/image2.jpg"),
            Path("/path1/image3.jpg"),
        ]

        duplicates = find_duplicate_filenames(paths)

        self.assertEqual(len(duplicates), 0)

    def test_write_validation_summary(self) -> None:
        """Test writing validation summary to files."""
        result = ValidationResult()
        result.has_errors = True
        result.has_warnings = True
        result.issues.append(
            ValidationIssue(
                severity="error",
                category="missing_image",
                message="Image not found",
                file_path="/path/to/image.jpg",
            )
        )
        result.issues.append(
            ValidationIssue(
                severity="warning",
                category="empty_label",
                message="Label is empty",
                file_path="/path/to/label.txt",
            )
        )
        result.stats.total_images = 10
        result.stats.total_labels = 9
        result.stats.total_annotations = 45
        result.stats.class_distribution = {0: 20, 1: 15, 2: 10}
        result.stats.image_size_distribution = {(640, 480): 8, (1280, 720): 2}

        output_dir = self.temp_path / "reports"
        write_validation_summary(result, output_dir, dataset_name="test")

        # Check JSON file
        json_path = output_dir / "test_check.json"
        self.assertTrue(json_path.exists())

        import json

        with json_path.open("r") as f:
            data = json.load(f)

        self.assertTrue(data["has_errors"])
        self.assertTrue(data["has_warnings"])
        self.assertEqual(data["total_issues"], 2)
        self.assertEqual(data["error_count"], 1)
        self.assertEqual(data["warning_count"], 1)
        self.assertEqual(data["stats"]["total_images"], 10)

        # Check CSV file
        csv_path = output_dir / "test_check.csv"
        self.assertTrue(csv_path.exists())

    def test_validation_issue_dataclass(self) -> None:
        """Test ValidationIssue dataclass."""
        issue = ValidationIssue(
            severity="error",
            category="test_category",
            message="Test message",
            file_path="/path/to/file",
            line_number=42,
        )

        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.category, "test_category")
        self.assertEqual(issue.message, "Test message")
        self.assertEqual(issue.file_path, "/path/to/file")
        self.assertEqual(issue.line_number, 42)

    def test_dataset_stats_dataclass(self) -> None:
        """Test DatasetStats dataclass."""
        stats = DatasetStats()

        self.assertEqual(stats.total_images, 0)
        self.assertEqual(stats.total_labels, 0)
        self.assertEqual(stats.total_annotations, 0)
        self.assertEqual(len(stats.class_distribution), 0)
        self.assertEqual(len(stats.image_size_distribution), 0)
        self.assertEqual(len(stats.duplicate_filenames), 0)

    def test_validation_result_dataclass(self) -> None:
        """Test ValidationResult dataclass."""
        result = ValidationResult()

        self.assertEqual(len(result.issues), 0)
        self.assertFalse(result.has_errors)
        self.assertFalse(result.has_warnings)
        self.assertIsInstance(result.stats, DatasetStats)

    def test_print_summary_uses_ascii_safe_status_text(self) -> None:
        """Test summary output uses console-safe ASCII status markers."""
        result = ValidationResult()
        result.has_warnings = True
        result.stats.total_images = 1
        result.stats.total_labels = 1

        with mock.patch("sys.stdout") as fake_stdout:
            print_summary(result)

        output = "".join(call.args[0] for call in fake_stdout.write.call_args_list)
        self.assertIn("[WARN] VALIDATION PASSED - Warnings found", output)


class TestDatasetValidationIntegration(unittest.TestCase):
    """Integration tests for dataset validation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        temp_root = Path(__file__).resolve().parents[1] / "reports" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(dir=str(temp_root))
        self.temp_path = Path(self.temp_dir)

    def create_fake_dataset(self) -> Path:
        """Create a fake dataset for testing."""
        dataset_root = self.temp_path / "fake_dataset"
        train_images = dataset_root / "train" / "images"
        train_labels = dataset_root / "train" / "labels"
        val_images = dataset_root / "val" / "images"
        val_labels = dataset_root / "val" / "labels"

        for dir_path in [train_images, train_labels, val_images, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create valid train images and labels
        for i in range(3):
            img_path = train_images / f"train_{i}.jpg"
            label_path = train_labels / f"train_{i}.txt"

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)

            label_path.write_text(f"0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.15\n")

        # Create valid val images and labels
        for i in range(2):
            img_path = val_images / f"val_{i}.jpg"
            label_path = val_labels / f"val_{i}.txt"

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)

            label_path.write_text(f"1 0.6 0.7 0.15 0.2\n")

        # Create one image with missing label
        missing_label_img = train_images / "missing_label.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(missing_label_img), img)

        # Create one corrupt image
        corrupt_img = train_images / "corrupt.jpg"
        corrupt_img.write_bytes(b"not an image")

        # Create one label with invalid class
        invalid_class_img = train_images / "invalid_class.jpg"
        invalid_class_label = train_labels / "invalid_class.txt"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(invalid_class_img), img)
        invalid_class_label.write_text("5 0.5 0.5 0.2 0.3\n")  # Class 5 out of range

        return dataset_root

    def test_full_dataset_validation(self) -> None:
        """Test full dataset validation workflow."""
        dataset_root = self.create_fake_dataset()

        # This would normally be called from check_dataset.py
        # Here we just verify the dataset structure is correct
        train_images = dataset_root / "train" / "images"
        train_labels = dataset_root / "train" / "labels"

        self.assertTrue(train_images.exists())
        self.assertTrue(train_labels.exists())

        # Verify we have the expected files
        image_files = list(train_images.glob("*.jpg"))
        self.assertEqual(len(image_files), 5)  # 3 valid + 1 missing label + 1 corrupt

        label_files = list(train_labels.glob("*.txt"))
        self.assertEqual(len(label_files), 4)  # 3 valid + 1 invalid class


if __name__ == "__main__":
    unittest.main()
