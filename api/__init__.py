"""Detektor serving API package."""

from .schemas import HealthResponse, PredictionResponse

__version__ = "1.1.0"

__all__ = ["HealthResponse", "PredictionResponse", "__version__"]
