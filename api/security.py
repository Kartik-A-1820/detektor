"""Optional authentication helpers for the Detektor inference API.

Authentication is opt-in: when no API key is configured the service behaves exactly as
before (suitable for localhost / private-network use). When ``DETEKTOR_API_KEY`` (or
``--api-key``) is set, every inference, runtime and metrics endpoint requires the key via
either ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``. Liveness/readiness probes
(`/health`, `/ready`, `/version`) always stay open so orchestrators keep working.
"""

from __future__ import annotations

import hmac
import re
from typing import Callable, Optional

from fastapi import HTTPException, Request, status

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


def extract_api_key(request: Request) -> Optional[str]:
    """Return the credential presented by the client, if any."""
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return None


def make_api_key_dependency(expected_key: Optional[str]) -> Callable[[Request], None]:
    """Build a FastAPI dependency enforcing ``expected_key`` (no-op when unset)."""

    def dependency(request: Request) -> None:
        if not expected_key:
            return
        presented = extract_api_key(request)
        if presented is None or not hmac.compare_digest(presented.encode("utf-8"), expected_key.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency


def sanitize_request_id(value: Optional[str]) -> Optional[str]:
    """Accept a client-supplied request id only if it is short and log-safe."""
    if value and _REQUEST_ID_RE.match(value):
        return value
    return None
