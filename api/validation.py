"""Image validation utilities for production API."""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

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

# Hard cap on decoded pixel count (8192 x 8192) to stop decompression bombs.
MAX_PIXELS = 8192 * 8192


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


def validate_mime_type(content_type: str | None) -> None:
    """Validate the declared MIME type of an upload.

    Many HTTP clients omit the part ``Content-Type`` or send the generic
    ``application/octet-stream``; those are accepted here and the real check is the
    decode step in :func:`validate_image_integrity`. An explicitly *wrong* type
    (e.g. ``text/plain``) is still rejected.

    Raises:
        ImageValidationError: If an explicit MIME type is not an allowed image type
    """
    if not content_type:
        return
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in ("", "application/octet-stream"):
        return
    if media_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(
            f"Invalid content type: {content_type}. "
            f"Allowed types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
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


def peek_image_size(file_bytes: bytes) -> Tuple[int, int] | None:
    """Read (width, height) from the image header without decoding pixels; ``None`` if unreadable."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:  # noqa: BLE001 - any failure falls through to the full decode check
        return None


def validate_image_integrity(file_bytes: bytes) -> Tuple[int, int]:
    """Validate that the file is a valid, non-corrupt image.

    Args:
        file_bytes: Raw image bytes

    Returns:
        Tuple of (width, height) of the image

    Raises:
        ImageValidationError: If image is corrupt or cannot be decoded
    """
    peeked = peek_image_size(file_bytes)
    if peeked is not None and peeked[0] * peeked[1] > MAX_PIXELS:
        # Decompression-bomb guard: refuse before allocating the decoded pixel buffer.
        raise ImageValidationError(
            f"Image dimensions {peeked[0]}x{peeked[1]} exceed the {MAX_PIXELS:,} pixel limit"
        )

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
