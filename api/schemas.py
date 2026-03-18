from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


LOCAL_API_CONTRACT_VERSION = "v1"


class HealthResponse(BaseModel):
    """Health status payload for the local inference service."""

    status: str = Field(..., description="Service status string")
    device: str = Field(..., description="Active inference device")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


class ReadyResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether the service is ready to accept requests")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Active inference device")


class VersionResponse(BaseModel):
    """Version information response."""

    version: str = Field(..., description="API version")
    contract_version: str = Field(
        default=LOCAL_API_CONTRACT_VERSION,
        description="Stable local API contract version for client integrations",
    )
    model_type: str = Field(default="ChimeraODIS", description="Model architecture")
    num_classes: Optional[int] = Field(None, description="Number of object classes")


class PredictionRequest(BaseModel):
    """Request parameters for prediction endpoint."""

    conf_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence threshold (0.0-1.0)")
    iou_thresh: Optional[float] = Field(None, ge=0.0, le=1.0, description="IoU threshold for NMS (0.0-1.0)")
    max_det: Optional[int] = Field(None, ge=1, le=1000, description="Maximum detections per image")
    include_masks: Optional[bool] = Field(None, description="Include segmentation masks in response")


class Detection(BaseModel):
    """Single detection result."""

    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    label: int = Field(..., ge=0, description="Class label ID")
    mask: Optional[str] = Field(None, description="Base64-encoded PNG mask")

    @field_validator('box')
    @classmethod
    def validate_box(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError('Box must have exactly 4 coordinates [x1, y1, x2, y2]')
        return v


class PredictionResponse(BaseModel):
    """Single-image prediction response schema."""

    request_id: Optional[str] = Field(None, description="Unique request identifier")
    num_detections: int = Field(..., description="Number of detections")
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")

    # Legacy fields for backward compatibility
    boxes: Optional[List[List[float]]] = Field(None, description="Legacy: list of boxes")
    scores: Optional[List[float]] = Field(None, description="Legacy: list of scores")
    labels: Optional[List[int]] = Field(None, description="Legacy: list of labels")
    masks: Optional[List[Optional[str]]] = Field(None, description="Legacy: list of masks")

    @model_validator(mode="before")
    @classmethod
    def populate_detections_from_legacy_fields(cls, data):
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        detections = payload.get("detections")
        boxes = payload.get("boxes")
        scores = payload.get("scores")
        labels = payload.get("labels")
        masks = payload.get("masks")

        if (detections is None or len(detections) == 0) and boxes is not None and scores is not None and labels is not None:
            built_detections = []
            for index, box in enumerate(boxes):
                detection = {
                    "box": box,
                    "score": scores[index],
                    "label": labels[index],
                }
                if masks is not None and index < len(masks) and masks[index] is not None:
                    detection["mask"] = masks[index]
                built_detections.append(detection)
            payload["detections"] = built_detections

        return payload

    @model_validator(mode="after")
    def sync_legacy_prediction_fields(self) -> "PredictionResponse":
        self.num_detections = len(self.detections)
        self.boxes = [list(det.box) for det in self.detections]
        self.scores = [float(det.score) for det in self.detections]
        self.labels = [int(det.label) for det in self.detections]

        masks = [det.mask for det in self.detections]
        self.masks = masks if any(mask is not None for mask in masks) else None
        return self


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    request_id: Optional[str] = Field(None, description="Unique request identifier")
    num_images: int = Field(..., description="Number of images processed")
    predictions: List[PredictionResponse] = Field(..., description="Predictions for each image")
    total_inference_time_ms: Optional[float] = Field(None, description="Total inference time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request identifier if available")
    details: Optional[dict] = Field(None, description="Additional error details")


class MetricsResponse(BaseModel):
    """Service metrics response."""

    total_requests: int = Field(..., description="Total number of requests processed")
    total_predictions: int = Field(..., description="Total number of predictions made")
    avg_inference_time_ms: float = Field(0.0, description="Average inference time in milliseconds")
    p50_inference_time_ms: float = Field(0.0, description="50th percentile inference time")
    p95_inference_time_ms: float = Field(0.0, description="95th percentile inference time")
    p99_inference_time_ms: float = Field(0.0, description="99th percentile inference time")
    error_count: int = Field(0, description="Total number of errors")

    # Legacy contract fields retained for compatibility.
    total_errors: int = Field(0, description="Legacy alias for error_count")
    avg_latency_ms: float = Field(0.0, description="Legacy alias for avg_inference_time_ms")
    p50_latency_ms: float = Field(0.0, description="Legacy alias for p50_inference_time_ms")
    p95_latency_ms: float = Field(0.0, description="Legacy alias for p95_inference_time_ms")
    p99_latency_ms: float = Field(0.0, description="Legacy alias for p99_inference_time_ms")

    @model_validator(mode="before")
    @classmethod
    def populate_metric_aliases(cls, data):
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        alias_pairs = (
            ("error_count", "total_errors"),
            ("avg_inference_time_ms", "avg_latency_ms"),
            ("p50_inference_time_ms", "p50_latency_ms"),
            ("p95_inference_time_ms", "p95_latency_ms"),
            ("p99_inference_time_ms", "p99_latency_ms"),
        )
        for canonical, legacy in alias_pairs:
            if canonical not in payload and legacy in payload:
                payload[canonical] = payload[legacy]
            if legacy not in payload and canonical in payload:
                payload[legacy] = payload[canonical]
        return payload

    @model_validator(mode="after")
    def sync_metric_aliases(self) -> "MetricsResponse":
        self.total_errors = self.error_count
        self.avg_latency_ms = self.avg_inference_time_ms
        self.p50_latency_ms = self.p50_inference_time_ms
        self.p95_latency_ms = self.p95_inference_time_ms
        self.p99_latency_ms = self.p99_inference_time_ms
        return self


class RuntimeStateResponse(BaseModel):
    """Runtime metadata contract exposed by the local serving API."""

    contract_version: str = Field(
        default=LOCAL_API_CONTRACT_VERSION,
        description="Stable local API contract version for client integrations",
    )
    run_dir: Optional[str] = Field(None, description="Resolved training run directory")
    active_checkpoint_key: Optional[str] = Field(None, description="Active checkpoint selector key")
    active_checkpoint_path: Optional[str] = Field(None, description="Resolved path to the active checkpoint")
    available_checkpoints: Dict[str, str] = Field(
        default_factory=dict,
        description="Checkpoint options keyed by selector name such as best, last, or custom",
    )
    checkpoint_summary: Dict[str, Any] = Field(default_factory=dict, description="Loaded checkpoint metadata")
    dataset: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata for the active run")
    runtime: Dict[str, Any] = Field(default_factory=dict, description="Resolved runtime configuration")
    training_summary: Dict[str, Any] = Field(default_factory=dict, description="Training summary fields")
    validation_summary: Dict[str, Any] = Field(default_factory=dict, description="Validation summary fields")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional run metadata")
    train_curve: List[Dict[str, Any]] = Field(default_factory=list, description="Sampled training curve rows")
    validation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Validation history rows")
    class_map: Dict[str, str] = Field(default_factory=dict, description="Stringified class id to class name map")
    plots: Dict[str, str] = Field(default_factory=dict, description="Resolved paths to discovered plot images")
