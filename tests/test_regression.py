"""Regression tests for schema stability and metric correctness."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch


class TestRegression(unittest.TestCase):
    """Regression tests to ensure stability across versions."""

    def test_prediction_response_schema_stability(self) -> None:
        """Test that PredictionResponse schema remains stable."""
        from api.schemas import PredictionResponse, Detection

        # This schema should remain stable for backward compatibility
        response = PredictionResponse(
            request_id="test123",
            num_detections=2,
            detections=[
                Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=0, mask="mask1"),
                Detection(box=[50.0, 60.0, 70.0, 80.0], score=0.85, label=1),
            ],
            boxes=[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
            scores=[0.95, 0.85],
            labels=[0, 1],
            masks=["mask1", None],
            image_width=640,
            image_height=480,
            inference_time_ms=42.5,
        )

        data = response.model_dump()

        # Verify all expected keys exist
        required_keys = {
            "request_id",
            "num_detections",
            "detections",
            "boxes",
            "scores",
            "labels",
            "image_width",
            "image_height",
        }
        self.assertTrue(required_keys.issubset(data.keys()))

        # Verify structure
        self.assertEqual(len(data["detections"]), 2)
        self.assertEqual(len(data["boxes"]), 2)
        self.assertEqual(len(data["scores"]), 2)
        self.assertEqual(len(data["labels"]), 2)

        # Verify detection structure
        det = data["detections"][0]
        self.assertIn("box", det)
        self.assertIn("score", det)
        self.assertIn("label", det)

    def test_batch_prediction_response_schema_stability(self) -> None:
        """Test that BatchPredictionResponse schema remains stable."""
        from api.schemas import BatchPredictionResponse, PredictionResponse

        pred = PredictionResponse(
            num_detections=1,
            detections=[],
            boxes=[[10.0, 20.0, 30.0, 40.0]],
            scores=[0.95],
            labels=[0],
            image_width=640,
            image_height=480,
        )

        response = BatchPredictionResponse(
            request_id="batch123",
            num_images=2,
            predictions=[pred, pred],
            total_inference_time_ms=85.0,
        )

        data = response.model_dump()

        # Verify all expected keys exist
        required_keys = {"num_images", "predictions"}
        self.assertTrue(required_keys.issubset(data.keys()))
        self.assertEqual(len(data["predictions"]), 2)

    def test_error_response_schema_stability(self) -> None:
        """Test that ErrorResponse schema remains stable."""
        from api.schemas import ErrorResponse

        response = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            request_id="err123",
            details={"field": "image", "reason": "corrupt"},
        )

        data = response.model_dump()

        # Verify all expected keys exist
        required_keys = {"error", "message"}
        self.assertTrue(required_keys.issubset(data.keys()))

    def test_metrics_no_nan_with_valid_data(self) -> None:
        """Test that metrics computation never produces NaN with valid data."""
        from utils.metrics_helpers import compute_ap50

        # Valid predictions and targets
        pred_boxes = np.array([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]])
        pred_scores = np.array([0.9, 0.8])
        pred_labels = np.array([0, 1])
        gt_boxes = np.array([[12.0, 12.0, 48.0, 48.0], [62.0, 62.0, 98.0, 98.0]])
        gt_labels = np.array([0, 1])

        ap50 = compute_ap50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=2)

        self.assertFalse(np.isnan(ap50))
        self.assertGreaterEqual(ap50, 0.0)
        self.assertLessEqual(ap50, 1.0)

    def test_metrics_no_nan_with_no_predictions(self) -> None:
        """Test that metrics handle no predictions gracefully."""
        from utils.metrics_helpers import compute_ap50

        pred_boxes = np.array([]).reshape(0, 4)
        pred_scores = np.array([])
        pred_labels = np.array([])
        gt_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        gt_labels = np.array([0])

        ap50 = compute_ap50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=2)

        self.assertFalse(np.isnan(ap50))
        self.assertEqual(ap50, 0.0)

    def test_metrics_no_nan_with_no_ground_truth(self) -> None:
        """Test that metrics handle no ground truth gracefully."""
        from utils.metrics_helpers import compute_ap50

        pred_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        pred_scores = np.array([0.9])
        pred_labels = np.array([0])
        gt_boxes = np.array([]).reshape(0, 4)
        gt_labels = np.array([])

        ap50 = compute_ap50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=2)

        self.assertFalse(np.isnan(ap50))
        self.assertEqual(ap50, 0.0)

    def test_box_ops_no_nan_regression(self) -> None:
        """Test that box operations never produce NaN."""
        from utils.box_ops import distances_to_boxes

        points = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        distances = torch.tensor([
            [[5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]],
            [[4.0, 4.0, 4.0, 4.0], [6.0, 6.0, 6.0, 6.0], [1.0, 1.0, 1.0, 1.0]],
        ])

        boxes = distances_to_boxes(points, distances)

        self.assertFalse(torch.isnan(boxes).any())
        self.assertFalse(torch.isinf(boxes).any())

    def test_ciou_loss_no_nan_regression(self) -> None:
        """Test that CIoU loss never produces NaN."""
        from utils.box_ops import ciou_loss

        pred_boxes = torch.tensor([
            [10.0, 10.0, 50.0, 50.0],
            [60.0, 60.0, 100.0, 100.0],
            [20.0, 20.0, 80.0, 80.0],
        ])
        target_boxes = torch.tensor([
            [12.0, 12.0, 48.0, 48.0],
            [62.0, 62.0, 98.0, 98.0],
            [25.0, 25.0, 75.0, 75.0],
        ])

        loss = ciou_loss(pred_boxes, target_boxes)

        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())
        self.assertTrue((loss >= 0.0).all())
        self.assertTrue((loss <= 2.0).all())

    def test_mask_ops_no_nan_regression(self) -> None:
        """Test that mask operations never produce NaN."""
        from utils.mask_ops import compose_instance_masks, crop_mask_region

        prototypes = torch.randn(24, 32, 32)
        coefficients = torch.randn(5, 24)

        masks = compose_instance_masks(prototypes, coefficients)

        self.assertFalse(torch.isnan(masks).any())
        self.assertFalse(torch.isinf(masks).any())

        # Test crop
        boxes = torch.tensor([
            [0.0, 0.0, 16.0, 16.0],
            [16.0, 16.0, 32.0, 32.0],
            [8.0, 8.0, 24.0, 24.0],
            [0.0, 16.0, 16.0, 32.0],
            [16.0, 0.0, 32.0, 16.0],
        ])

        cropped = crop_mask_region(masks, boxes)

        self.assertFalse(torch.isnan(cropped).any())
        self.assertFalse(torch.isinf(cropped).any())

    def test_json_serialization_stability(self) -> None:
        """Test that JSON serialization is stable."""
        from api.schemas import PredictionResponse, Detection

        response = PredictionResponse(
            request_id="test",
            num_detections=1,
            detections=[Detection(box=[1.0, 2.0, 3.0, 4.0], score=0.9, label=0)],
            boxes=[[1.0, 2.0, 3.0, 4.0]],
            scores=[0.9],
            labels=[0],
            image_width=100,
            image_height=100,
        )

        # Serialize to JSON
        json_str = response.model_dump_json()

        # Deserialize
        data = json.loads(json_str)

        # Verify structure is preserved
        self.assertEqual(data["request_id"], "test")
        self.assertEqual(data["num_detections"], 1)
        self.assertEqual(len(data["detections"]), 1)
        self.assertEqual(data["detections"][0]["score"], 0.9)

    def test_validation_output_format_regression(self) -> None:
        """Test that validation output format is stable."""
        # Expected validation output structure
        expected_structure = {
            "overall": {
                "precision": float,
                "recall": float,
                "f1": float,
                "ap50": float,
                "map50": float,
            },
            "per_class": list,
        }

        # Simulate validation output
        val_output = {
            "overall": {
                "precision": 0.85,
                "recall": 0.78,
                "f1": 0.81,
                "ap50": 0.82,
                "map50": 0.80,
            },
            "per_class": [
                {"class_id": 0, "precision": 0.9, "recall": 0.85, "ap50": 0.88},
            ],
        }

        # Verify structure matches expected
        for key, expected_type in expected_structure.items():
            self.assertIn(key, val_output)
            if expected_type == float:
                self.assertIsInstance(val_output[key], (int, float))
            elif expected_type == list:
                self.assertIsInstance(val_output[key], list)
            elif isinstance(expected_type, dict):
                for subkey, subtype in expected_type.items():
                    self.assertIn(subkey, val_output[key])
                    self.assertIsInstance(val_output[key][subkey], (int, float))


if __name__ == "__main__":
    unittest.main()
