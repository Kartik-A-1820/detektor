from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health status payload for the local inference service."""

    status: str = Field(..., description="Service status string")
    device: str = Field(..., description="Active inference device")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


class PredictionResponse(BaseModel):
    """Single-image prediction response schema."""

    num_detections: int
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]
    masks: Optional[List[str]] = None
    image_width: int
    image_height: int
