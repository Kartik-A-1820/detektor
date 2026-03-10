from __future__ import annotations

import unittest


class APISmokeTests(unittest.TestCase):
    """Lightweight FastAPI import and route smoke tests."""

    def test_fastapi_app_import_and_health_route(self) -> None:
        try:
            from serve import ServiceConfig, create_app
        except Exception as exc:
            self.skipTest(f"FastAPI service dependencies unavailable: {exc}")
            return

        app = create_app(ServiceConfig(weights="dummy.pt"))
        routes = {route.path for route in app.routes}
        self.assertIn("/health", routes)
        self.assertIn("/predict", routes)

    def test_health_response_schema(self) -> None:
        try:
            from api.schemas import HealthResponse
        except Exception as exc:
            self.skipTest(f"API schema dependencies unavailable: {exc}")
            return

        payload = HealthResponse(status="ok", device="cpu", model_loaded=False)
        self.assertEqual(payload.status, "ok")
        self.assertEqual(payload.device, "cpu")
        self.assertFalse(payload.model_loaded)


if __name__ == "__main__":
    unittest.main()
