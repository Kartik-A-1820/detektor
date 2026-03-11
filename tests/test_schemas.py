"""Unit tests for API schema serialization."""

from __future__ import annotations

import unittest

from api.schemas import (
    BatchPredictionResponse,
    Detection,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    ReadyResponse,
    VersionResponse,
)


class TestSchemas(unittest.TestCase):
    """Unit tests for Pydantic schema serialization."""

    def test_health_response_serialization(self) -> None:
        """Test HealthResponse schema."""
        response = HealthResponse(status="ok", device="cuda", model_loaded=True)
        
        data = response.model_dump()
        
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["device"], "cuda")
        self.assertTrue(data["model_loaded"])

    def test_ready_response_serialization(self) -> None:
        """Test ReadyResponse schema."""
        response = ReadyResponse(ready=True, model_loaded=True, device="cpu")
        
        data = response.model_dump()
        
        self.assertTrue(data["ready"])
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["device"], "cpu")

    def test_version_response_serialization(self) -> None:
        """Test VersionResponse schema."""
        response = VersionResponse(version="1.0.0", model_type="ChimeraODIS", num_classes=4)
        
        data = response.model_dump()
        
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["model_type"], "ChimeraODIS")
        self.assertEqual(data["num_classes"], 4)

    def test_version_response_optional_num_classes(self) -> None:
        """Test VersionResponse with optional num_classes."""
        response = VersionResponse(version="1.0.0", model_type="ChimeraODIS")
        
        data = response.model_dump()
        
        self.assertIsNone(data["num_classes"])

    def test_error_response_serialization(self) -> None:
        """Test ErrorResponse schema."""
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            request_id="abc123",
            details={"field": "image"},
        )
        
        data = response.model_dump()
        
        self.assertEqual(data["error"], "ValidationError")
        self.assertEqual(data["message"], "Invalid input")
        self.assertEqual(data["request_id"], "abc123")
        self.assertEqual(data["details"]["field"], "image")

    def test_error_response_optional_fields(self) -> None:
        """Test ErrorResponse with optional fields."""
        response = ErrorResponse(error="Error", message="Something went wrong")
        
        data = response.model_dump()
        
        self.assertIsNone(data["request_id"])
        self.assertIsNone(data["details"])

    def test_detection_serialization(self) -> None:
        """Test Detection schema."""
        detection = Detection(
            box=[10.0, 20.0, 30.0, 40.0],
            score=0.95,
            label=0,
            mask="base64encodedmask",
        )
        
        data = detection.model_dump()
        
        self.assertEqual(data["box"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(data["score"], 0.95)
        self.assertEqual(data["label"], 0)
        self.assertEqual(data["mask"], "base64encodedmask")

    def test_detection_optional_mask(self) -> None:
        """Test Detection without mask."""
        detection = Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=0)
        
        data = detection.model_dump()
        
        self.assertIsNone(data["mask"])

    def test_prediction_response_serialization(self) -> None:
        """Test PredictionResponse schema."""
        response = PredictionResponse(
            request_id="req123",
            num_detections=2,
            detections=[
                Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=0),
                Detection(box=[50.0, 60.0, 70.0, 80.0], score=0.85, label=1),
            ],
            boxes=[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
            scores=[0.95, 0.85],
            labels=[0, 1],
            image_width=640,
            image_height=480,
            inference_time_ms=42.5,
        )
        
        data = response.model_dump()
        
        self.assertEqual(data["request_id"], "req123")
        self.assertEqual(data["num_detections"], 2)
        self.assertEqual(len(data["detections"]), 2)
        self.assertEqual(data["inference_time_ms"], 42.5)

    def test_prediction_response_optional_fields(self) -> None:
        """Test PredictionResponse with optional fields."""
        response = PredictionResponse(
            num_detections=0,
            detections=[],
            boxes=[],
            scores=[],
            labels=[],
            image_width=640,
            image_height=480,
        )
        
        data = response.model_dump()
        
        self.assertIsNone(data["request_id"])
        self.assertIsNone(data["inference_time_ms"])
        self.assertIsNone(data["masks"])

    def test_batch_prediction_response_serialization(self) -> None:
        """Test BatchPredictionResponse schema."""
        pred1 = PredictionResponse(
            num_detections=1,
            detections=[Detection(box=[10.0, 20.0, 30.0, 40.0], score=0.95, label=0)],
            boxes=[[10.0, 20.0, 30.0, 40.0]],
            scores=[0.95],
            labels=[0],
            image_width=640,
            image_height=480,
        )
        pred2 = PredictionResponse(
            num_detections=0,
            detections=[],
            boxes=[],
            scores=[],
            labels=[],
            image_width=640,
            image_height=480,
        )
        
        response = BatchPredictionResponse(
            request_id="batch123",
            num_images=2,
            predictions=[pred1, pred2],
            total_inference_time_ms=85.0,
        )
        
        data = response.model_dump()
        
        self.assertEqual(data["request_id"], "batch123")
        self.assertEqual(data["num_images"], 2)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["total_inference_time_ms"], 85.0)

    def test_batch_prediction_response_optional_fields(self) -> None:
        """Test BatchPredictionResponse with optional fields."""
        response = BatchPredictionResponse(num_images=0, predictions=[])
        
        data = response.model_dump()
        
        self.assertIsNone(data["request_id"])
        self.assertIsNone(data["total_inference_time_ms"])

    def test_metrics_response_serialization(self) -> None:
        """Test MetricsResponse schema."""
        response = MetricsResponse(
            total_requests=100,
            total_predictions=250,
            total_errors=5,
            avg_latency_ms=45.2,
            p50_latency_ms=42.0,
            p95_latency_ms=78.5,
        )
        
        data = response.model_dump()
        
        self.assertEqual(data["total_requests"], 100)
        self.assertEqual(data["total_predictions"], 250)
        self.assertEqual(data["total_errors"], 5)
        self.assertEqual(data["avg_latency_ms"], 45.2)

    def test_schema_json_serialization(self) -> None:
        """Test that schemas can be serialized to JSON."""
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
        
        json_str = response.model_dump_json()
        
        self.assertIsInstance(json_str, str)
        self.assertIn("request_id", json_str)
        self.assertIn("test", json_str)


if __name__ == "__main__":
    unittest.main()
