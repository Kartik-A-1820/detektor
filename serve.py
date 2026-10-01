from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from api import __version__ as API_PACKAGE_VERSION
from api.inference import InferenceService
from api.logging_utils import (
    RequestTimer,
    generate_request_id,
    get_request_id,
    log_error,
    log_request,
    log_response,
    set_request_id,
    setup_logging,
)
from api.metrics import get_metrics_store
from api.run_artifacts import discover_checkpoint_options, load_run_artifacts
from api.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    ReadyResponse,
    VersionResponse,
)
from api.security import make_api_key_dependency, sanitize_request_id
from api.utils import load_model
from api.validation import ImageValidationError, validate_uploaded_image

LOGGER = logging.getLogger("detektor.serve")

# API version
API_VERSION = API_PACKAGE_VERSION


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
    num_classes: Optional[int] = None
    proto_k: int = 24
    image_size: int = 512
    max_upload_size_mb: int = 10
    max_batch_size: int = 16
    enable_warmup: bool = True
    warmup_iterations: int = 3
    log_level: str = "INFO"
    ui_enabled: bool = False
    ui_path: str = "/ui"
    ui_auth: Optional[str] = None  # "user:password" for the Gradio UI
    api_key: Optional[str] = None  # when set, protects inference/runtime/metrics endpoints
    cors_origins: List[str] = field(default_factory=list)
    max_concurrency: int = 1  # simultaneous model executions (protects small GPUs)


class ModelStore:
    """Singleton-like holder for the loaded inference service and runtime config."""

    def __init__(self) -> None:
        self.inference_service: Optional[InferenceService] = None
        self.config: Optional[ServiceConfig] = None
        self.num_classes: Optional[int] = None
        self.active_weights: Optional[str] = None
        self.active_checkpoint_key: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.checkpoint_paths: dict[str, str] = {}
        self.runtime_state: dict = {}
        self.lock = threading.RLock()


MODEL_STORE = ModelStore()


def _load_service(
    config: ServiceConfig,
    weights_path: str,
    checkpoint_key: Optional[str] = None,
    *,
    warmup_iterations: Optional[int] = None,
) -> None:
    run_dir, checkpoint_options, default_key = discover_checkpoint_options(weights_path)
    selected_key = checkpoint_key if checkpoint_key in checkpoint_options else default_key
    selected_path = checkpoint_options.get(selected_key, Path(weights_path).expanduser().resolve())

    LOGGER.info("Loading checkpoint '%s' from %s", selected_key, selected_path)
    model, device = load_model(
        weights=str(selected_path),
        num_classes=config.num_classes,
        proto_k=config.proto_k,
        device_name=config.device,
    )
    inference_service = InferenceService(
        model=model,
        device=device,
        image_size=config.image_size,
        default_conf_thresh=config.conf_thresh,
        default_iou_thresh=config.iou_thresh,
        default_max_det=config.max_det,
        default_topk_pre_nms=config.topk_pre_nms,
        default_mask_thresh=config.mask_thresh,
        default_include_masks=config.include_masks_default,
    )

    if config.enable_warmup and warmup_iterations and warmup_iterations > 0:
        LOGGER.info("Warming up checkpoint '%s' with %d iteration(s)", selected_key, warmup_iterations)
        inference_service.warmup(num_iterations=warmup_iterations)

    runtime_state = load_run_artifacts(run_dir, selected_path, selected_key)
    runtime_state["available_checkpoints"] = {
        key: str(path) for key, path in checkpoint_options.items()
    }
    runtime_state["device"] = str(device)

    with MODEL_STORE.lock:
        MODEL_STORE.inference_service = inference_service
        MODEL_STORE.config = config
        MODEL_STORE.num_classes = model.num_classes
        MODEL_STORE.active_weights = str(selected_path)
        MODEL_STORE.active_checkpoint_key = selected_key
        MODEL_STORE.run_dir = str(run_dir)
        MODEL_STORE.checkpoint_paths = runtime_state["available_checkpoints"]
        MODEL_STORE.runtime_state = runtime_state


def get_runtime_state() -> dict:
    with MODEL_STORE.lock:
        return copy.deepcopy(MODEL_STORE.runtime_state)


def get_service_snapshot() -> tuple[InferenceService, ServiceConfig]:
    with MODEL_STORE.lock:
        service = MODEL_STORE.inference_service
        config = MODEL_STORE.config
    if service is None or config is None:
        raise RuntimeError("Model is not initialized")
    return service, config


def select_active_checkpoint(checkpoint_key: str) -> dict:
    with MODEL_STORE.lock:
        config = MODEL_STORE.config
        current_weights = MODEL_STORE.active_weights or (MODEL_STORE.config.weights if MODEL_STORE.config else None)
    if config is None or current_weights is None:
        raise RuntimeError("Model is not initialized")
    _load_service(config, current_weights, checkpoint_key=checkpoint_key, warmup_iterations=1)
    return get_runtime_state()


def create_app(config: ServiceConfig) -> FastAPI:
    """Create the production-ready FastAPI app for detektor inference serving."""
    with MODEL_STORE.lock:
        MODEL_STORE.config = config

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Setup structured logging
        setup_logging(level=config.log_level)

        LOGGER.info("Starting Detektor API v%s", API_VERSION)
        LOGGER.info("Requested weights: %s", config.weights)
        _load_service(config, config.weights, warmup_iterations=config.warmup_iterations)
        LOGGER.info("Service ready to accept requests")
        yield
        LOGGER.info("Shutting down service")

    app = FastAPI(
        title="Detektor Production Inference API",
        version=API_VERSION,
        description="Production-ready object detection and instance segmentation API",
        lifespan=lifespan,
    )

    require_api_key = make_api_key_dependency(config.api_key)
    protected = [Depends(require_api_key)]
    inference_slots = asyncio.Semaphore(max(1, int(config.max_concurrency)))
    # Generous bound for a full batch of max-size uploads plus multipart overhead.
    max_body_bytes = (max(1, config.max_batch_size) * config.max_upload_size_mb + 1) * 1024 * 1024

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Response-Time"],
        )

    # Request ID, payload limit and security headers middleware
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        request_id = sanitize_request_id(request.headers.get("x-request-id")) or generate_request_id()
        set_request_id(request_id)

        declared_length = request.headers.get("content-length")
        if declared_length and declared_length.isdigit() and int(declared_length) > max_body_bytes:
            get_metrics_store().record_request(0.0, 0, error=True)
            response = JSONResponse(
                status_code=413,
                content=ErrorResponse(
                    error="PayloadTooLarge",
                    message=f"Request body exceeds the {max_body_bytes // (1024 * 1024)} MB limit",
                    request_id=request_id,
                ).model_dump(),
            )
        else:
            with RequestTimer() as timer:
                response = await call_next(request)
            response.headers["X-Response-Time"] = f"{timer.duration_ms:.2f}ms"

        response.headers["X-Request-ID"] = request_id
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    # Exception handlers
    @app.exception_handler(ImageValidationError)
    async def validation_error_handler(request: Request, exc: ImageValidationError):
        request_id = get_request_id()
        log_error(LOGGER, exc, request_id=request_id)
        get_metrics_store().record_request(0.0, 0, error=True)

        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="ValidationError",
                message=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        request_id = get_request_id()
        log_error(LOGGER, exc, request_id=request_id)
        get_metrics_store().record_request(0.0, 0, error=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An internal error occurred during processing",
                request_id=request_id,
                details={"error_type": type(exc).__name__},
            ).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        """Health check endpoint - always returns OK if service is running."""
        with MODEL_STORE.lock:
            service = MODEL_STORE.inference_service
        return HealthResponse(
            status="ok",
            device=str(service.device) if service is not None else "uninitialized",
            model_loaded=service is not None,
        )

    @app.get("/ready", response_model=ReadyResponse, tags=["Health"])
    async def ready() -> ReadyResponse:
        """Readiness check endpoint - returns ready only if model is loaded."""
        with MODEL_STORE.lock:
            service = MODEL_STORE.inference_service
        is_ready = service is not None

        return ReadyResponse(
            ready=is_ready,
            model_loaded=is_ready,
            device=str(service.device) if service is not None else "uninitialized",
        )

    @app.get("/version", response_model=VersionResponse, tags=["Health"])
    async def version() -> VersionResponse:
        """Version information endpoint."""
        with MODEL_STORE.lock:
            num_classes = MODEL_STORE.num_classes
        return VersionResponse(
            version=API_VERSION,
            model_type="ChimeraODIS",
            num_classes=num_classes,
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"], dependencies=protected)
    async def metrics() -> MetricsResponse:
        """Service metrics endpoint."""
        stats = get_metrics_store().get_stats()
        return MetricsResponse(**stats)

    @app.get("/metrics/prometheus", response_class=PlainTextResponse, tags=["Monitoring"], dependencies=protected)
    async def metrics_prometheus() -> PlainTextResponse:
        """Metrics in Prometheus text exposition format (scrape target)."""
        with MODEL_STORE.lock:
            service = MODEL_STORE.inference_service
            checkpoint = MODEL_STORE.active_checkpoint_key or "unknown"
        info = {
            "version": API_VERSION,
            "device": str(service.device) if service is not None else "uninitialized",
            "checkpoint": checkpoint,
        }
        return PlainTextResponse(
            get_metrics_store().render_prometheus(info),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/runtime", tags=["Runtime"], dependencies=protected)
    async def runtime() -> dict:
        """Runtime metadata for the active run and checkpoint."""
        return get_runtime_state()

    @app.post("/runtime/select_model", tags=["Runtime"], dependencies=protected)
    async def select_model(model_key: str = Query(..., description="Checkpoint key: best, last, or custom")) -> dict:
        """Switch the active checkpoint without restarting the service."""
        state = get_runtime_state()
        if model_key not in state.get("available_checkpoints", {}):
            raise HTTPException(status_code=404, detail=f"Unknown checkpoint key: {model_key}")
        return await run_in_threadpool(select_active_checkpoint, model_key)

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=protected)
    async def predict_v1(
        image: UploadFile = File(..., description="Image file to run inference on"),
        conf_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence threshold"),
        iou_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="IoU threshold for NMS"),
        max_det: Optional[int] = Query(None, ge=1, le=1000, description="Maximum detections"),
        include_masks: Optional[bool] = Query(None, description="Include segmentation masks"),
    ) -> PredictionResponse:
        """Run object detection and instance segmentation on a single image (v1 endpoint)."""
        try:
            service, active_config = get_service_snapshot()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        request_id = get_request_id()
        log_request(LOGGER, "POST", "/v1/predict", request_id or "-")

        # Read and validate image
        image_bytes = await image.read()
        max_size_bytes = active_config.max_upload_size_mb * 1024 * 1024
        validate_uploaded_image(
            image_bytes,
            image.content_type,
            max_file_size=max_size_bytes,
        )

        # Run inference
        async with inference_slots:
            response_data, inference_time = await run_in_threadpool(
                service.predict_single,
                image_bytes=image_bytes,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                max_det=max_det,
                include_masks=include_masks,
            )

        # Add request ID
        response_data["request_id"] = request_id

        # Record metrics
        get_metrics_store().record_request(inference_time, response_data["num_detections"])

        log_response(
            LOGGER,
            "POST",
            "/v1/predict",
            200,
            inference_time,
            request_id or "-",
            detections=response_data["num_detections"],
        )

        return PredictionResponse(**response_data)

    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], deprecated=True, dependencies=protected)
    async def predict_legacy(
        image: UploadFile = File(...),
        include_masks: bool = Query(default=config.include_masks_default),
    ) -> PredictionResponse:
        """Legacy prediction endpoint - maintained for backward compatibility.

        Use /v1/predict for new integrations.
        """
        return await predict_v1(
            image=image,
            conf_thresh=None,
            iou_thresh=None,
            max_det=None,
            include_masks=include_masks,
        )

    @app.post("/v1/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"], dependencies=protected)
    async def predict_batch(
        images: List[UploadFile] = File(..., description="List of image files"),
        conf_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence threshold"),
        iou_thresh: Optional[float] = Query(None, ge=0.0, le=1.0, description="IoU threshold for NMS"),
        max_det: Optional[int] = Query(None, ge=1, le=1000, description="Maximum detections per image"),
        include_masks: Optional[bool] = Query(None, description="Include segmentation masks"),
    ) -> BatchPredictionResponse:
        """Run object detection and instance segmentation on a batch of images."""
        try:
            service, active_config = get_service_snapshot()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        request_id = get_request_id()
        log_request(LOGGER, "POST", "/v1/predict_batch", request_id or "-", num_images=len(images))

        # Validate batch size
        if len(images) > active_config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(images)} exceeds maximum {active_config.max_batch_size}",
            )

        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No images provided")

        # Read and validate all images
        images_bytes = []
        max_size_bytes = active_config.max_upload_size_mb * 1024 * 1024

        for idx, img in enumerate(images):
            img_bytes = await img.read()
            try:
                validate_uploaded_image(
                    img_bytes,
                    img.content_type,
                    max_file_size=max_size_bytes,
                )
                images_bytes.append(img_bytes)
            except ImageValidationError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image {idx} validation failed: {str(e)}",
                )

        # Run batch inference
        async with inference_slots:
            predictions, total_time = await run_in_threadpool(
                service.predict_batch,
                images_bytes=images_bytes,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                max_det=max_det,
                include_masks=include_masks,
            )

        # Add request IDs to individual predictions
        for pred in predictions:
            pred["request_id"] = request_id

        # Record metrics
        total_detections = sum(p["num_detections"] for p in predictions)
        get_metrics_store().record_request(total_time, total_detections)

        log_response(
            LOGGER,
            "POST",
            "/v1/predict_batch",
            200,
            total_time,
            request_id or "-",
            num_images=len(predictions),
            total_detections=total_detections,
        )

        response = BatchPredictionResponse(
            request_id=request_id,
            num_images=len(predictions),
            predictions=[PredictionResponse(**p) for p in predictions],
            total_inference_time_ms=total_time,
        )

        return response

    return app


def main() -> None:
    """Parse CLI args and launch the production FastAPI service."""
    parser = argparse.ArgumentParser(
        description="Start the Detektor Production FastAPI Inference Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--weights",
        type=str,
        default=os.getenv("DETEKTOR_WEIGHTS"),
        required=os.getenv("DETEKTOR_WEIGHTS") is None,
        help="Path to model weights or checkpoint (env: DETEKTOR_WEIGHTS)",
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("DETEKTOR_HOST", "127.0.0.1"),
        help="Host interface to bind the API server to (env: DETEKTOR_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DETEKTOR_PORT", "8000")),
        help="Port to bind the API server to (env: DETEKTOR_PORT)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DETEKTOR_DEVICE", "auto"),
        help="Inference device: auto, cpu, or cuda (env: DETEKTOR_DEVICE)",
    )

    # Model configuration
    parser.add_argument(
        "--num-classes",
        type=int,
        default=int(os.getenv("DETEKTOR_NUM_CLASSES")) if os.getenv("DETEKTOR_NUM_CLASSES") else None,
        help="Number of classes (auto-detected if not provided) (env: DETEKTOR_NUM_CLASSES)",
    )
    parser.add_argument(
        "--proto-k",
        type=int,
        default=int(os.getenv("DETEKTOR_PROTO_K", "24")),
        help="Number of prototype channels (env: DETEKTOR_PROTO_K)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=int(os.getenv("DETEKTOR_IMG_SIZE", "512")),
        help="Square model input size (env: DETEKTOR_IMG_SIZE)",
    )

    # Inference defaults
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=float(os.getenv("DETEKTOR_CONF_THRESH", "0.25")),
        help="Default confidence threshold (env: DETEKTOR_CONF_THRESH)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=float(os.getenv("DETEKTOR_IOU_THRESH", "0.6")),
        help="Default IoU threshold for NMS (env: DETEKTOR_IOU_THRESH)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=int(os.getenv("DETEKTOR_MAX_DET", "100")),
        help="Default maximum detections per image (env: DETEKTOR_MAX_DET)",
    )
    parser.add_argument(
        "--topk-pre-nms",
        type=int,
        default=int(os.getenv("DETEKTOR_TOPK_PRE_NMS", "300")),
        help="Maximum candidates before NMS (env: DETEKTOR_TOPK_PRE_NMS)",
    )
    parser.add_argument(
        "--mask-thresh",
        type=float,
        default=float(os.getenv("DETEKTOR_MASK_THRESH", "0.5")),
        help="Mask threshold for binary mask generation (env: DETEKTOR_MASK_THRESH)",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        default=os.getenv("DETEKTOR_INCLUDE_MASKS", "false").lower() == "true",
        help="Include masks by default (env: DETEKTOR_INCLUDE_MASKS)",
    )

    # Service configuration
    parser.add_argument(
        "--max-upload-size-mb",
        type=int,
        default=int(os.getenv("DETEKTOR_MAX_UPLOAD_SIZE_MB", "10")),
        help="Maximum upload size in MB (env: DETEKTOR_MAX_UPLOAD_SIZE_MB)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.getenv("DETEKTOR_MAX_BATCH_SIZE", "16")),
        help="Maximum batch size for batch prediction (env: DETEKTOR_MAX_BATCH_SIZE)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=os.getenv("DETEKTOR_NO_WARMUP", "false").lower() == "true",
        help="Disable model warmup on startup (env: DETEKTOR_NO_WARMUP)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=int(os.getenv("DETEKTOR_WARMUP_ITERATIONS", "3")),
        help="Number of warmup iterations (env: DETEKTOR_WARMUP_ITERATIONS)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("DETEKTOR_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (env: DETEKTOR_LOG_LEVEL)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        default=os.getenv("DETEKTOR_UI", "false").lower() == "true",
        help="Mount the built-in GUI at the configured UI path (env: DETEKTOR_UI)",
    )
    parser.add_argument(
        "--ui-path",
        type=str,
        default=os.getenv("DETEKTOR_UI_PATH", "/ui"),
        help="Path where the GUI will be mounted when --ui is enabled (env: DETEKTOR_UI_PATH)",
    )

    parser.add_argument(
        "--ui-auth",
        type=str,
        default=os.getenv("DETEKTOR_UI_AUTH") or None,
        help="Protect the GUI with HTTP login, formatted as user:password (env: DETEKTOR_UI_AUTH)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DETEKTOR_API_KEY") or None,
        help="Require this API key (X-API-Key or Bearer token) on inference, runtime and metrics "
        "endpoints (env: DETEKTOR_API_KEY)",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        default=os.getenv("DETEKTOR_CORS_ORIGINS", ""),
        help="Comma-separated list of allowed CORS origins; empty disables CORS (env: DETEKTOR_CORS_ORIGINS)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("DETEKTOR_MAX_CONCURRENCY", "1")),
        help="Maximum simultaneous model executions (env: DETEKTOR_MAX_CONCURRENCY)",
    )

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
        max_upload_size_mb=args.max_upload_size_mb,
        max_batch_size=args.max_batch_size,
        enable_warmup=not args.no_warmup,
        warmup_iterations=args.warmup_iterations,
        log_level=args.log_level,
        ui_enabled=args.ui,
        ui_path=args.ui_path,
        ui_auth=args.ui_auth,
        api_key=args.api_key,
        cors_origins=[origin.strip() for origin in args.cors_origins.split(",") if origin.strip()],
        max_concurrency=args.max_concurrency,
    )

    app = create_app(config)
    if config.ui_enabled:
        import gradio as gr

        from ui.app import DetektorUIRuntime, build_interface

        runtime = DetektorUIRuntime(
            get_runtime_state=get_runtime_state,
            get_service_snapshot=get_service_snapshot,
            select_checkpoint=select_active_checkpoint,
        )
        app = gr.mount_gradio_app(
            app,
            build_interface(runtime=runtime),
            path=config.ui_path,
            allowed_paths=[str(Path.cwd())],
            show_error=True,
            auth=tuple(config.ui_auth.split(":", 1)) if config.ui_auth and ":" in config.ui_auth else None,
        )
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_config=None,  # Use our custom logging
    )


if __name__ == "__main__":
    main()
