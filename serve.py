from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api.schemas import HealthResponse, PredictionResponse
from api.utils import load_model, prediction_to_service_response, preprocess_image_bytes


LOGGER = logging.getLogger("detektor.serve")


@dataclass
class ServiceConfig:
    weights: str
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "auto"
    conf_thresh: float = 0.25
    iou_thresh: float = 0.6
    max_det: int = 100
    topk_pre_nms: int = 300
    mask_thresh: float = 0.5
    include_masks_default: bool = False
    num_classes: int = 1
    proto_k: int = 24
    image_size: int = 512


class ModelStore:
    """Singleton-like holder for the loaded inference model and runtime config."""

    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.config: Optional[ServiceConfig] = None


MODEL_STORE = ModelStore()


def create_app(config: ServiceConfig) -> FastAPI:
    """Create the FastAPI app for local detektor inference serving."""
    app = FastAPI(title="detektor local inference service", version="1.0.0")
    MODEL_STORE.config = config

    @app.on_event("startup")
    async def startup_event() -> None:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        LOGGER.info("loading model from %s", config.weights)
        model, device = load_model(
            weights=config.weights,
            num_classes=config.num_classes,
            proto_k=config.proto_k,
            device_name=config.device,
        )
        MODEL_STORE.model = model
        MODEL_STORE.device = device
        LOGGER.info("model ready on %s", device)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            device=str(MODEL_STORE.device) if MODEL_STORE.device is not None else "uninitialized",
            model_loaded=MODEL_STORE.model is not None,
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        image: UploadFile = File(...),
        include_masks: bool = Query(default=config.include_masks_default),
    ) -> PredictionResponse:
        if MODEL_STORE.model is None or MODEL_STORE.device is None or MODEL_STORE.config is None:
            raise HTTPException(status_code=503, detail="Model is not initialized")
        if image.content_type is not None and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        try:
            image_bytes = await image.read()
            image_tensor, _, original_size = preprocess_image_bytes(image_bytes, image_size=MODEL_STORE.config.image_size)
            image_tensor = image_tensor.to(MODEL_STORE.device)
            prediction = MODEL_STORE.model.predict(
                image_tensor,
                original_sizes=[original_size],
                conf_thresh=MODEL_STORE.config.conf_thresh,
                iou_thresh=MODEL_STORE.config.iou_thresh,
                topk_pre_nms=MODEL_STORE.config.topk_pre_nms,
                max_det=MODEL_STORE.config.max_det,
                mask_thresh=MODEL_STORE.config.mask_thresh,
            )[0]
            payload = prediction_to_service_response(
                prediction=prediction,
                image_size=original_size,
                include_masks=include_masks,
            )
            return PredictionResponse(**payload)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    """Parse CLI args and launch the local FastAPI service."""
    parser = argparse.ArgumentParser(description="Start the Detektor local FastAPI inference service")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights or checkpoint")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind the API server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server to")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto, cpu, or cuda")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used before NMS")
    parser.add_argument("--iou-thresh", type=float, default=0.6, help="IoU threshold used by NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections returned per image")
    parser.add_argument("--topk-pre-nms", type=int, default=300, help="Maximum candidates kept before NMS")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Mask threshold for binary mask generation")
    parser.add_argument("--include-masks", action="store_true", help="Include base64-encoded masks in API responses by default")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes expected by the checkpoint")
    parser.add_argument("--proto-k", type=int, default=24, help="Number of prototype channels expected by the checkpoint")
    parser.add_argument("--img-size", type=int, default=512, help="Square model input size used for preprocessing")
    args = parser.parse_args()

    config = ServiceConfig(
        weights=args.weights,
        host=args.host,
        port=args.port,
        device=args.device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        max_det=args.max_det,
        topk_pre_nms=args.topk_pre_nms,
        mask_thresh=args.mask_thresh,
        include_masks_default=args.include_masks,
        num_classes=args.num_classes,
        proto_k=args.proto_k,
        image_size=args.img_size,
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
