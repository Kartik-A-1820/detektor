"""Image validation utilities for production API."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# Allowed MIME types for image uploads
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Maximum file size in bytes (default 10MB)
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


def validate_mime_type(content_type: str | None) -> None:
    """Validate that the uploaded file has an allowed MIME type.
    
    Args:
        content_type: MIME type from upload
        
    Raises:
        ImageValidationError: If MIME type is invalid or not allowed
    """
    if content_type is None:
        raise ImageValidationError("No content type provided")
    
    if content_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(
            f"Invalid content type: {content_type}. "
            f"Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )


def validate_file_size(file_bytes: bytes, max_size: int = DEFAULT_MAX_FILE_SIZE) -> None:
    """Validate that the file size is within limits.
    
    Args:
        file_bytes: Raw file bytes
        max_size: Maximum allowed file size in bytes
        
    Raises:
        ImageValidationError: If file is too large
    """
    file_size = len(file_bytes)
    if file_size > max_size:
        raise ImageValidationError(
            f"File size {file_size} bytes exceeds maximum allowed size {max_size} bytes"
        )
    if file_size == 0:
        raise ImageValidationError("File is empty")


def validate_image_integrity(file_bytes: bytes) -> Tuple[int, int]:
    """Validate that the file is a valid, non-corrupt image.
    
    Args:
        file_bytes: Raw image bytes
        
    Returns:
        Tuple of (width, height) of the image
        
    Raises:
        ImageValidationError: If image is corrupt or cannot be decoded
    """
    np_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Could not decode uploaded image")
    
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ImageValidationError(f"Invalid image dimensions: {width}x{height}")
    
    return width, height


def validate_image_dimensions(
    width: int,
    height: int,
    min_size: int = 32,
    max_size: int = 8192,
) -> None:
    """Validate image dimensions are within acceptable ranges.
    
    Args:
        width: Image width
        height: Image height
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
        
    Raises:
        ImageValidationError: If dimensions are out of range
    """
    if width < min_size or height < min_size:
        raise ImageValidationError(
            f"Image dimensions {width}x{height} are too small. "
            f"Minimum size: {min_size}x{min_size}"
        )
    
    if width > max_size or height > max_size:
        raise ImageValidationError(
            f"Image dimensions {width}x{height} are too large. "
            f"Maximum size: {max_size}x{max_size}"
        )


def validate_uploaded_image(
    file_bytes: bytes,
    content_type: str | None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    min_dimension: int = 32,
    max_dimension: int = 8192,
) -> Tuple[int, int]:
    """Comprehensive validation of uploaded image.
    
    Args:
        file_bytes: Raw image bytes
        content_type: MIME type from upload
        max_file_size: Maximum file size in bytes
        min_dimension: Minimum image dimension
        max_dimension: Maximum image dimension
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ImageValidationError: If any validation check fails
    """
    validate_mime_type(content_type)
    validate_file_size(file_bytes, max_file_size)
    width, height = validate_image_integrity(file_bytes)
    validate_image_dimensions(width, height, min_dimension, max_dimension)
    
    return width, height
