"""Structured logging utilities for production API."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variable to store request ID across async calls
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def generate_request_id() -> str:
    """Generate a unique request ID.
    
    Returns:
        UUID-based request ID string
    """
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """Set the request ID in context.
    
    Args:
        request_id: Request ID to set
    """
    request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context.
    
    Returns:
        Current request ID or None
    """
    return request_id_ctx.get()


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured fields to log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured fields.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            record.request_id = request_id
        else:
            record.request_id = "-"
        
        # Add timestamp
        record.timestamp = self.formatTime(record, self.datefmt)
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> None:
    """Setup structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (uses default if None)
    """
    if format_string is None:
        format_string = (
            "[%(timestamp)s] [%(levelname)s] [%(request_id)s] "
            "%(name)s - %(message)s"
        )
    
    formatter = StructuredFormatter(
        fmt=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # Reduce noise from uvicorn
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    request_id: str,
    **kwargs: Any,
) -> None:
    """Log an incoming request with structured data.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        request_id: Request ID
        **kwargs: Additional fields to log
    """
    extra_fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(
        f"Request: {method} {path} {extra_fields}",
        extra={"request_id": request_id},
    )


def log_response(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str,
    **kwargs: Any,
) -> None:
    """Log a response with structured data.
    
    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        request_id: Request ID
        **kwargs: Additional fields to log
    """
    extra_fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(
        f"Response: {method} {path} status={status_code} duration={duration_ms:.2f}ms {extra_fields}",
        extra={"request_id": request_id},
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Log an error with structured data.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        request_id: Request ID if available
        **kwargs: Additional fields to log
    """
    extra_fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.error(
        f"Error: {type(error).__name__}: {str(error)} {extra_fields}",
        extra={"request_id": request_id or "-"},
        exc_info=True,
    )


class RequestTimer:
    """Context manager for timing requests."""
    
    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
    
    def __enter__(self) -> RequestTimer:
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop the timer and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        if self.end_time > 0:
            return self.duration_ms
        return (time.perf_counter() - self.start_time) * 1000.0
