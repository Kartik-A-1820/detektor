from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


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
    masks: Optional[List[str]] = Field(None, description="Legacy: list of masks")


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
    avg_inference_time_ms: float = Field(..., description="Average inference time in milliseconds")
    p50_inference_time_ms: float = Field(..., description="50th percentile inference time")
    p95_inference_time_ms: float = Field(..., description="95th percentile inference time")
    p99_inference_time_ms: float = Field(..., description="99th percentile inference time")
    error_count: int = Field(..., description="Total number of errors")
