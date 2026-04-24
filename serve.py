from __future__ import annotations

import argparse
import copy
import io
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from api.inference import InferenceService
from api.logging_utils import (
    generate_request_id,
    get_request_id,
    log_error,
    log_request,
    log_response,
    set_request_id,
    setup_logging,
    RequestTimer,
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
    RuntimeStateResponse,
    VersionResponse,
)
from api.utils import load_model
from api.validation import ImageValidationError, validate_uploaded_image


LOGGER = logging.getLogger("detektor.serve")

# API version
API_VERSION = "1.0.0"

# Training job storage
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
TRAINING_JOBS_LOCK = threading.RLock()


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


def build_runtime_state_response() -> RuntimeStateResponse:
    return RuntimeStateResponse(**get_runtime_state())


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
    
    # Request ID middleware
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        request_id = generate_request_id()
        set_request_id(request_id)
        
        with RequestTimer() as timer:
            response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{timer.duration_ms:.2f}ms"
        
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
    
    @app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
    async def metrics() -> MetricsResponse:
        """Service metrics endpoint."""
        stats = get_metrics_store().get_stats()
        return MetricsResponse(**stats)

    @app.get("/runtime", response_model=RuntimeStateResponse, tags=["Runtime"])
    async def runtime() -> RuntimeStateResponse:
        """Runtime metadata for the active run and checkpoint."""
        return build_runtime_state_response()

    @app.post("/runtime/select_model", response_model=RuntimeStateResponse, tags=["Runtime"])
    async def select_model(
        model_key: str = Query(..., description="Checkpoint key: best, last, or custom"),
    ) -> RuntimeStateResponse:
        """Switch the active checkpoint without restarting the service."""
        state = get_runtime_state()
        if model_key not in state.get("available_checkpoints", {}):
            raise HTTPException(status_code=404, detail=f"Unknown checkpoint key: {model_key}")
        return RuntimeStateResponse(**select_active_checkpoint(model_key))

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
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
        response_data, inference_time = service.predict_single(
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
    
    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"], deprecated=True)
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
    
    @app.post("/v1/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
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
        predictions, total_time = service.predict_batch(
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

    # ------------------------------------------------------------------ #
    #  Training endpoints                                                  #
    # ------------------------------------------------------------------ #

    @app.post("/v1/train/start", tags=["Training"])
    async def start_training(request: Request) -> JSONResponse:
        """Start a training run in a background thread. Returns job_id."""
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON", "message": str(exc)})

        data_yaml: Optional[str] = body.get("data_yaml")
        config_path: Optional[str] = body.get("config_path")
        epochs: int = int(body.get("epochs", 10))
        batch_size: int = int(body.get("batch_size", 4))
        lr: float = float(body.get("lr", 0.002))
        model_profile: Optional[str] = body.get("model_profile")
        focal_loss_gamma: float = float(body.get("focal_loss_gamma", 0.0))
        out_dir: Optional[str] = body.get("out_dir")
        run_val: bool = bool(body.get("run_val", False))

        if not data_yaml:
            return JSONResponse(status_code=400, content={"error": "Missing field", "message": "data_yaml is required"})

        job_id = str(uuid.uuid4())
        log_buffer: deque = deque(maxlen=500)

        job: Dict[str, Any] = {
            "job_id": job_id,
            "status": "starting",
            "log_buffer": log_buffer,
            "metrics": {},
            "out_dir": out_dir or "",
            "started_at": time.time(),
            "stop_requested": False,
            "thread": None,
        }

        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id] = job

        def _run_training() -> None:
            try:
                # Lazy import to avoid circular imports
                from train import train as _train  # noqa: PLC0415

                overrides: Dict[str, Any] = {
                    "train": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                    }
                }
                if model_profile:
                    overrides["model"] = {"profile": model_profile}
                if focal_loss_gamma != 0.0:
                    overrides.setdefault("loss", {})["focal_loss_gamma"] = focal_loss_gamma
                if out_dir:
                    overrides.setdefault("train", {})["out_dir"] = out_dir

                log_buffer.append(f"[INFO] Training job {job_id} started")
                log_buffer.append(f"[INFO] data_yaml={data_yaml}, epochs={epochs}, batch_size={batch_size}, lr={lr}")

                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[job_id]["status"] = "running"

                summary = _train(
                    config_path=config_path,
                    data_yaml=data_yaml,
                    overrides=overrides,
                    run_val=run_val,
                )

                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[job_id]["status"] = "done"
                    TRAINING_JOBS[job_id]["metrics"] = summary if isinstance(summary, dict) else {}
                log_buffer.append("[INFO] Training completed successfully")

            except Exception as exc:  # noqa: BLE001
                with TRAINING_JOBS_LOCK:
                    TRAINING_JOBS[job_id]["status"] = "failed"
                    TRAINING_JOBS[job_id]["error"] = str(exc)
                log_buffer.append(f"[ERROR] Training failed: {exc}")
                LOGGER.exception("Training job %s failed", job_id)

        thread = threading.Thread(target=_run_training, daemon=True, name=f"train-{job_id[:8]}")
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id]["thread"] = thread
        thread.start()

        return JSONResponse(content={"job_id": job_id, "status": "started", "out_dir": out_dir or ""})

    @app.get("/v1/train/status/{job_id}", tags=["Training"])
    async def get_training_status(job_id: str) -> JSONResponse:
        """Get training job status and latest log lines."""
        with TRAINING_JOBS_LOCK:
            job = TRAINING_JOBS.get(job_id)

        if job is None:
            return JSONResponse(status_code=404, content={"error": "Not found", "message": f"Job {job_id} not found"})

        log_tail = list(job["log_buffer"])[-20:]
        return JSONResponse(content={
            "job_id": job_id,
            "status": job["status"],
            "log_tail": log_tail,
            "metrics": job.get("metrics", {}),
            "out_dir": job.get("out_dir", ""),
            "error": job.get("error"),
        })

    @app.post("/v1/train/stop/{job_id}", tags=["Training"])
    async def stop_training(job_id: str) -> JSONResponse:
        """Request cancellation of a running training job."""
        with TRAINING_JOBS_LOCK:
            job = TRAINING_JOBS.get(job_id)

        if job is None:
            return JSONResponse(status_code=404, content={"error": "Not found", "message": f"Job {job_id} not found"})

        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[job_id]["stop_requested"] = True
            current_status = TRAINING_JOBS[job_id]["status"]

        if current_status not in ("running", "starting"):
            return JSONResponse(content={"job_id": job_id, "status": current_status, "message": "Job is not running"})

        return JSONResponse(content={"job_id": job_id, "status": "stop_requested", "message": "Stop signal sent"})

    # ------------------------------------------------------------------ #
    #  Validation endpoint                                                 #
    # ------------------------------------------------------------------ #

    @app.post("/v1/validate/run", tags=["Validation"])
    async def run_validation(request: Request) -> JSONResponse:
        """Run validation synchronously and return metrics."""
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON", "message": str(exc)})

        weights: Optional[str] = body.get("weights")
        data_yaml: Optional[str] = body.get("data_yaml")
        conf_thresh: float = float(body.get("conf_thresh", 0.25))
        iou_thresh: float = float(body.get("iou_thresh", 0.5))
        output_dir: Optional[str] = body.get("output_dir")

        # Fall back to active checkpoint if weights not provided
        if not weights:
            with MODEL_STORE.lock:
                weights = MODEL_STORE.active_weights
        if not weights:
            return JSONResponse(status_code=400, content={"error": "Missing field", "message": "weights is required (no active checkpoint)"})

        # Resolve a config path from the checkpoint or run dir
        config_path: Optional[str] = None
        with MODEL_STORE.lock:
            run_dir = MODEL_STORE.run_dir
        if run_dir:
            candidate = Path(run_dir) / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)

        if not config_path:
            # Try to find any config yaml near the weights file
            weights_path = Path(weights)
            for candidate in [
                weights_path.parent / "config.yaml",
                weights_path.parent.parent / "config.yaml",
            ]:
                if candidate.exists():
                    config_path = str(candidate)
                    break

        if not config_path:
            return JSONResponse(status_code=400, content={"error": "Missing config", "message": "Could not resolve config.yaml for validation"})

        try:
            from validate import validate as _validate  # noqa: PLC0415

            metrics = _validate(
                config_path=config_path,
                data_yaml=data_yaml,
                weights=weights,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                output_dir=output_dir,
            )
            # Ensure JSON-serialisable
            def _to_serialisable(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _to_serialisable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_serialisable(v) for v in obj]
                try:
                    import torch  # noqa: PLC0415
                    if isinstance(obj, torch.Tensor):
                        return obj.item() if obj.numel() == 1 else obj.tolist()
                except ImportError:
                    pass
                try:
                    import numpy as np  # noqa: PLC0415
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                except ImportError:
                    pass
                return obj

            return JSONResponse(content={"status": "ok", "metrics": _to_serialisable(metrics)})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Validation failed")
            return JSONResponse(status_code=500, content={"error": "ValidationFailed", "message": str(exc)})

    # ------------------------------------------------------------------ #
    #  Dataset check endpoint                                              #
    # ------------------------------------------------------------------ #

    @app.post("/v1/dataset/check", tags=["Dataset"])
    async def check_dataset_endpoint(request: Request) -> JSONResponse:
        """Run dataset validation and return results."""
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON", "message": str(exc)})

        data_yaml: Optional[str] = body.get("data_yaml")
        output_dir: str = body.get("output_dir", "reports")

        if not data_yaml:
            return JSONResponse(status_code=400, content={"error": "Missing field", "message": "data_yaml is required"})

        data_yaml_path = Path(data_yaml)
        if not data_yaml_path.exists():
            return JSONResponse(status_code=400, content={"error": "Not found", "message": f"data_yaml not found: {data_yaml}"})

        try:
            # Lazy import
            import yaml as _yaml  # noqa: PLC0415
            from check_dataset import merge_results, validate_dataset_split  # noqa: PLC0415
            from utils.dataset_validation import write_validation_summary  # noqa: PLC0415

            with data_yaml_path.open("r") as fh:
                data_config = _yaml.safe_load(fh)

            num_classes = data_config.get("nc")
            if num_classes is None:
                return JSONResponse(status_code=400, content={"error": "Invalid YAML", "message": "'nc' not found in data_yaml"})

            split_results = []
            for split in ("train", "val"):
                split_path_str = data_config.get(split)
                if not split_path_str:
                    continue
                split_path = Path(split_path_str)
                if split_path.name == "images":
                    split_root = split_path.parent
                else:
                    split_root = split_path
                images_dir = split_root / "images"
                labels_dir = split_root / "labels"
                result = validate_dataset_split(images_dir, labels_dir, num_classes, split)
                split_results.append(result)

            if not split_results:
                return JSONResponse(status_code=400, content={"error": "No splits", "message": "No train/val splits found in data_yaml"})

            merged = merge_results(split_results)

            # Write reports
            out_path = Path(output_dir)
            write_validation_summary(merged, out_path, dataset_name="dataset")

            # Serialise issues
            issues = [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                    "file": issue.file_path or "",
                }
                for issue in merged.issues
            ]

            summary = {
                "total_images": merged.stats.total_images,
                "total_labels": merged.stats.total_labels,
                "total_annotations": merged.stats.total_annotations,
                "empty_labels": merged.stats.empty_labels,
                "corrupt_images": merged.stats.corrupt_images,
                "has_errors": merged.has_errors,
                "has_warnings": merged.has_warnings,
                "num_issues": len(merged.issues),
                "class_distribution": {str(k): v for k, v in merged.stats.class_distribution.items()},
            }

            return JSONResponse(content={"status": "ok", "summary": summary, "issues": issues})

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Dataset check failed")
            return JSONResponse(status_code=500, content={"error": "DatasetCheckFailed", "message": str(exc)})

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
    )
    
    app = create_app(config)
    if config.ui_enabled:
        import gradio as gr

        from ui.app import DetektorUIRuntime, build_interface

        runtime = DetektorUIRuntime(
            get_runtime_state=get_runtime_state,
            get_service_snapshot=get_service_snapshot,
            select_checkpoint=select_active_checkpoint,
            backend_port=config.port,
        )
        app = gr.mount_gradio_app(
            app,
            build_interface(runtime=runtime),
            path=config.ui_path,
            allowed_paths=[str(Path.cwd())],
            show_error=True,
        )
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_config=None,  # Use our custom logging
    )


if __name__ == "__main__":
    main()
