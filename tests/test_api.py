from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from api.metrics import get_metrics_store
from serve import ServiceConfig, create_app


class DummyModel:
    """Minimal stand-in for the real model to keep tests lightweight."""

    def __init__(self) -> None:
        self.num_classes = 2
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> DummyModel:
        self.device = device
        return self

    def eval(self) -> DummyModel:
        return self

    def predict(self, image_tensor: torch.Tensor, original_sizes, **kwargs):
        batch_size = image_tensor.shape[0]
        predictions = []
        for orig_h, orig_w in original_sizes:
            boxes = torch.tensor([[0.1, 0.1, 0.9, 0.9]], dtype=torch.float32, device=self.device)
            scores = torch.tensor([0.95], dtype=torch.float32, device=self.device)
            labels = torch.tensor([0], dtype=torch.int64, device=self.device)
            masks = torch.zeros((1, orig_h, orig_w), dtype=torch.float32, device=self.device)
            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "masks": masks,
            })
        return predictions


def _create_test_image_bytes() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode test image")
    return buffer.tobytes()


class ProductionAPITests(unittest.TestCase):
    """Production-grade API tests using a patched inference stack."""

    def setUp(self) -> None:  # noqa: D401
        self.image_bytes = _create_test_image_bytes()

    @contextmanager
    def _test_client(self):
        config = ServiceConfig(weights="dummy.pt", enable_warmup=False)
        get_metrics_store().reset()
        with patch("serve.load_model", return_value=(DummyModel(), torch.device("cpu"))):
            app = create_app(config)
            with TestClient(app) as client:
                yield client

    def test_health_ready_version_endpoints(self) -> None:
        with self._test_client() as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["model_loaded"])

            ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertTrue(ready.json()["ready"])

            version = client.get("/version")
            self.assertEqual(version.status_code, 200)
            self.assertIn("version", version.json())

    def test_predict_v1_endpoint_success(self) -> None:
        with self._test_client() as client:
            files = {"image": ("test.png", self.image_bytes, "image/png")}
            response = client.post("/v1/predict", files=files)
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["num_detections"], 1)
            self.assertIn("request_id", payload)
            self.assertIn("detections", payload)

    def test_legacy_predict_endpoint_alias(self) -> None:
        with self._test_client() as client:
            files = {"image": ("test.png", self.image_bytes, "image/png")}
            response = client.post("/predict", files=files)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["num_detections"], 1)

    def test_predict_rejects_invalid_mime_type(self) -> None:
        with self._test_client() as client:
            files = {"image": ("bad.txt", b"not an image", "text/plain")}
            response = client.post("/v1/predict", files=files)
            self.assertEqual(response.status_code, 400)

    def test_batch_prediction_endpoint(self) -> None:
        with self._test_client() as client:
            files = [
                ("images", ("img1.png", self.image_bytes, "image/png")),
                ("images", ("img2.png", self.image_bytes, "image/png")),
            ]
            response = client.post("/v1/predict_batch", files=files)
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["num_images"], 2)
            self.assertEqual(len(payload["predictions"]), 2)

    def test_metrics_endpoint_reports_activity(self) -> None:
        with self._test_client() as client:
            files = {"image": ("test.png", self.image_bytes, "image/png")}
            client.post("/v1/predict", files=files)

            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            payload = metrics.json()
            self.assertGreaterEqual(payload["total_requests"], 1)
            self.assertGreaterEqual(payload["total_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
