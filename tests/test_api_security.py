"""Tests for optional API-key auth, CORS, payload limits and Prometheus metrics."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from api.metrics import MetricsStore, get_metrics_store
from api.security import extract_api_key, sanitize_request_id
from serve import ServiceConfig, create_app
from tests.test_api import DummyModel, _create_test_image_bytes


@contextmanager
def _client(**overrides):
    config = ServiceConfig(weights="dummy.pt", enable_warmup=False, **overrides)
    get_metrics_store().reset()
    with patch("serve.load_model", return_value=(DummyModel(), torch.device("cpu"))):
        with TestClient(create_app(config)) as client:
            yield client


class ApiKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = _create_test_image_bytes()

    def _predict(self, client, **kwargs):
        return client.post("/v1/predict", files={"image": ("a.png", self.image, "image/png")}, **kwargs)

    def test_open_by_default(self) -> None:
        with _client() as client:
            self.assertEqual(self._predict(client).status_code, 200)

    def test_rejects_missing_and_wrong_key(self) -> None:
        with _client(api_key="s3cret") as client:
            self.assertEqual(self._predict(client).status_code, 401)
            self.assertEqual(self._predict(client, headers={"X-API-Key": "nope"}).status_code, 401)
            self.assertEqual(client.get("/metrics").status_code, 401)
            self.assertEqual(client.get("/runtime").status_code, 401)

    def test_accepts_header_and_bearer(self) -> None:
        with _client(api_key="s3cret") as client:
            self.assertEqual(self._predict(client, headers={"X-API-Key": "s3cret"}).status_code, 200)
            self.assertEqual(self._predict(client, headers={"Authorization": "Bearer s3cret"}).status_code, 200)

    def test_probes_stay_open(self) -> None:
        with _client(api_key="s3cret") as client:
            for path in ("/health", "/ready", "/version"):
                self.assertEqual(client.get(path).status_code, 200, path)

    def test_extract_api_key_helper(self) -> None:
        class Req:
            def __init__(self, headers):
                self.headers = headers

        self.assertEqual(extract_api_key(Req({"x-api-key": " k "})), "k")
        self.assertEqual(extract_api_key(Req({"authorization": "Bearer abc"})), "abc")
        self.assertIsNone(extract_api_key(Req({"authorization": "Basic abc"})))
        self.assertIsNone(extract_api_key(Req({})))


class RequestHygieneTests(unittest.TestCase):
    def test_request_id_roundtrip_and_sanitising(self) -> None:
        with _client() as client:
            ok = client.get("/health", headers={"X-Request-ID": "trace-123"})
            self.assertEqual(ok.headers["X-Request-ID"], "trace-123")
            bad = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"})
            self.assertNotEqual(bad.headers["X-Request-ID"], "bad id\twith spaces")
            self.assertEqual(ok.headers["X-Content-Type-Options"], "nosniff")
        self.assertIsNone(sanitize_request_id("x" * 200))

    def test_oversized_body_is_rejected_with_413(self) -> None:
        with _client(max_upload_size_mb=1, max_batch_size=1) as client:
            response = client.post(
                "/v1/predict",
                content=b"0" * (3 * 1024 * 1024),
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["error"], "PayloadTooLarge")

    def test_cors_only_when_configured(self) -> None:
        with _client() as client:
            r = client.get("/health", headers={"Origin": "https://app.example"})
            self.assertNotIn("access-control-allow-origin", r.headers)
        with _client(cors_origins=["https://app.example"]) as client:
            r = client.get("/health", headers={"Origin": "https://app.example"})
            self.assertEqual(r.headers["access-control-allow-origin"], "https://app.example")


class PrometheusTests(unittest.TestCase):
    def test_histogram_is_cumulative_and_exposed(self) -> None:
        store = MetricsStore()
        for value in (3.0, 20.0, 700.0):
            store.record_request(value, 2)
        store.record_request(0.0, 0, error=True)
        text = store.render_prometheus({"version": "x"})
        self.assertIn("detektor_requests_total 4", text)
        self.assertIn("detektor_errors_total 1", text)
        self.assertIn('detektor_inference_latency_ms_bucket{le="5"} 1', text)
        self.assertIn('detektor_inference_latency_ms_bucket{le="25"} 2', text)
        self.assertIn('detektor_inference_latency_ms_bucket{le="+Inf"} 3', text)
        self.assertIn('detektor_build_info{version="x"} 1', text)

    def test_endpoint_serves_text(self) -> None:
        with _client() as client:
            client.post("/v1/predict", files={"image": ("a.png", _create_test_image_bytes(), "image/png")})
            response = client.get("/metrics/prometheus")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/plain"))
            self.assertIn("detektor_requests_total", response.text)
            self.assertIn("detektor_build_info", response.text)


if __name__ == "__main__":
    unittest.main()
