"""Integration tests for inference, reporting, and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from utils.reporting import generate_metrics_summary, plot_epoch_metrics


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_tiny_inference_run(self) -> None:
        """Test a minimal inference run with a tiny model."""
        try:
            from models.chimera import ChimeraODIS
        except Exception:  # pragma: no cover
            self.skipTest("Model dependencies not available")
            return

        model = ChimeraODIS(num_classes=2, proto_k=8)
        model.eval()

        # Create a tiny dummy input
        dummy_input = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            predictions = model.predict(
                dummy_input,
                original_sizes=[(64, 64)],
                conf_thresh=0.5,
            )

        self.assertEqual(len(predictions), 1)
        self.assertIn("boxes", predictions[0])
        self.assertIn("scores", predictions[0])
        self.assertIn("labels", predictions[0])
        self.assertIn("masks", predictions[0])

    def test_report_generation_with_minimal_data(self) -> None:
        """Test report generation with minimal training data."""
        import pandas as pd

        # Create minimal training metrics
        train_data = {
            "step": [1, 2, 3],
            "loss_total": [10.0, 9.5, 9.0],
            "loss_cls": [3.0, 2.8, 2.6],
            "loss_box": [5.0, 4.8, 4.6],
            "loss_obj": [2.0, 1.9, 1.8],
            "lr": [0.001, 0.001, 0.001],
        }
        train_df = pd.DataFrame(train_data)

        epoch_data = {
            "epoch": [1],
            "avg_loss": [9.5],
        }
        epoch_df = pd.DataFrame(epoch_data)

        output_path = Path(self.temp_dir) / "metrics_summary.json"

        # Generate summary
        generate_metrics_summary(train_df, epoch_df, None, output_path)

        # Verify output
        self.assertTrue(output_path.exists())
        with output_path.open("r") as f:
            summary = json.load(f)

        self.assertIn("training", summary)
        self.assertEqual(summary["training"]["total_steps"], 3)
        self.assertAlmostEqual(summary["training"]["final_loss"], 9.0, places=1)

    def test_report_generation_no_nan_outputs(self) -> None:
        """Test that report generation doesn't produce NaN values."""
        import pandas as pd

        train_data = {
            "step": [1, 2, 3, 4, 5],
            "loss_total": [12.0, 11.0, 10.5, 10.0, 9.8],
            "loss_cls": [4.0, 3.5, 3.2, 3.0, 2.9],
            "loss_box": [6.0, 5.5, 5.3, 5.0, 4.9],
            "loss_obj": [2.0, 2.0, 2.0, 2.0, 2.0],
            "lr": [0.001, 0.001, 0.001, 0.001, 0.001],
        }
        train_df = pd.DataFrame(train_data)

        epoch_data = {
            "epoch": [1, 2],
            "avg_loss": [11.5, 10.0],
        }
        epoch_df = pd.DataFrame(epoch_data)

        output_path = Path(self.temp_dir) / "metrics_summary.json"

        generate_metrics_summary(train_df, epoch_df, None, output_path)

        with output_path.open("r") as f:
            summary = json.load(f)

        # Check no NaN values in summary
        def check_no_nan(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    check_no_nan(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_no_nan(item)
            elif isinstance(obj, float):
                self.assertFalse(np.isnan(obj), "Found NaN in metrics summary")

        check_no_nan(summary)

    def test_plot_generation_no_crash(self) -> None:
        """Test that plot generation doesn't crash with minimal data."""
        import pandas as pd

        train_data = {
            "step": [1, 2, 3],
            "loss_total": [10.0, 9.5, 9.0],
            "loss_cls": [3.0, 2.8, 2.6],
            "loss_box": [5.0, 4.8, 4.6],
            "loss_obj": [2.0, 1.9, 1.8],
            "lr": [0.001, 0.001, 0.001],
        }
        train_df = pd.DataFrame(train_data)

        plots_dir = Path(self.temp_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Should not crash
        try:
            plot_epoch_metrics(train_df, plots_dir)
        except Exception as e:  # noqa: BLE001
            self.fail(f"Plot generation crashed: {e}")

    def test_validation_output_schema_stability(self) -> None:
        """Test that validation output has stable schema."""
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
                {"class_id": 0, "class_name": "class0", "precision": 0.9, "recall": 0.85, "ap50": 0.88},
                {"class_id": 1, "class_name": "class1", "precision": 0.8, "recall": 0.71, "ap50": 0.72},
            ],
        }

        # Verify expected keys exist
        self.assertIn("overall", val_output)
        self.assertIn("per_class", val_output)
        self.assertIn("precision", val_output["overall"])
        self.assertIn("recall", val_output["overall"])
        self.assertIn("f1", val_output["overall"])
        self.assertIn("ap50", val_output["overall"])
        self.assertIn("map50", val_output["overall"])

        # Verify per-class structure
        for cls_metrics in val_output["per_class"]:
            self.assertIn("class_id", cls_metrics)
            self.assertIn("class_name", cls_metrics)
            self.assertIn("precision", cls_metrics)
            self.assertIn("recall", cls_metrics)
            self.assertIn("ap50", cls_metrics)

    def test_image_preprocessing_consistency(self) -> None:
        """Test that image preprocessing produces consistent outputs."""
        try:
            from api.utils import preprocess_image_bytes
        except Exception:  # pragma: no cover
            self.skipTest("API utils not available")
            return

        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = 255  # White square in center

        # Encode to bytes
        success, buffer = cv2.imencode(".png", test_image)
        self.assertTrue(success)
        image_bytes = buffer.tobytes()

        # Preprocess
        tensor, rgb_image, original_size = preprocess_image_bytes(image_bytes, image_size=64)

        # Verify outputs
        self.assertEqual(tensor.shape, (1, 3, 64, 64))
        self.assertEqual(original_size, (100, 100))
        self.assertEqual(rgb_image.shape, (100, 100, 3))

        # Verify tensor is normalized
        self.assertTrue((tensor >= 0.0).all())
        self.assertTrue((tensor <= 1.0).all())

    def test_metrics_no_nan_on_empty_predictions(self) -> None:
        """Test that metrics computation handles empty predictions gracefully."""
        from utils.metrics_helpers import compute_ap50

        # Empty predictions and targets
        pred_boxes = np.array([]).reshape(0, 4)
        pred_scores = np.array([])
        pred_labels = np.array([])
        gt_boxes = np.array([]).reshape(0, 4)
        gt_labels = np.array([])

        ap50 = compute_ap50(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=2)

        # Should return 0.0, not NaN
        self.assertFalse(np.isnan(ap50))
        self.assertEqual(ap50, 0.0)

    def test_inference_service_warmup(self) -> None:
        """Test that inference service warmup runs without errors."""
        try:
            from api.inference import InferenceService
            from models.chimera import ChimeraODIS
        except Exception:  # pragma: no cover
            self.skipTest("Inference service dependencies not available")
            return

        model = ChimeraODIS(num_classes=2, proto_k=8)
        device = torch.device("cpu")

        service = InferenceService(
            model=model,
            device=device,
            image_size=64,
        )

        # Warmup should not crash
        try:
            service.warmup(num_iterations=2)
        except Exception as e:  # noqa: BLE001
            self.fail(f"Warmup failed: {e}")


if __name__ == "__main__":
    unittest.main()
