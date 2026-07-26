"""HS256 service-token JWT auth dependency for FastAPI.

Validates ``Authorization: Bearer <jwt>`` against the shared
``SERVICE_TOKEN_SECRET`` env var. ``/healthz``, ``/readyz`` and ``/metrics``
bypass auth. In ``DEV_MODE=1`` with an unset secret the dependency is a no-op.
Mirrors the orchestrator-tx ``internal/authtoken`` middleware shape.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

from fastapi import HTTPException, Request, status

log = logging.getLogger("fraud_engine.authtoken")

SKIP_PATHS = {"/healthz", "/readyz", "/metrics"}


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _verify(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed token")
    signing_input = f"{parts[0]}.{parts[1]}".encode()
    expected = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    if not hmac.compare_digest(expected, parts[2]):
        raise ValueError("invalid signature")
    claims = json.loads(_b64url_decode(parts[1]))
    return claims


def secret_from_env() -> tuple[str, bool]:
    s = os.environ.get("SERVICE_TOKEN_SECRET", "")
    if s:
        return s, False
    if os.environ.get("DEV_MODE") == "1":
        log.warning(
            "warn: SERVICE_TOKEN_SECRET unset and DEV_MODE=1; service-token auth disabled (NOT FOR PRODUCTION)"
        )
        return "", True
    raise RuntimeError(
        "SERVICE_TOKEN_SECRET not set and DEV_MODE!=1; refusing to start in production mode"
    )


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return ""
    return auth[len("Bearer ") :]


async def require_token(request: Request) -> dict[str, Any] | None:
    """FastAPI dependency: validates the Bearer token, returns the claims
    (or ``None`` when bypassed in DEV_MODE). Raises 401 on failure."""
    if request.url.path in SKIP_PATHS:
        return None
    secret, bypass = secret_from_env()
    if bypass:
        return None
    token = _bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "unauthorized", "message": "missing or malformed Authorization header"}},
        )
    try:
        claims = _verify(token, secret)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "unauthorized", "message": str(exc)}},
        ) from exc
    if isinstance(claims.get("exp"), (int, float)) and time.time() > claims["exp"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"code": "unauthorized", "message": "token expired"}},
        )
    return claims


def issue(service_name: str, secret: str) -> str:
    if not secret:
        raise ValueError("authtoken: secret is required to issue a token")
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    claims = {"sub": service_name, "iat": now, "exp": now + 24 * 60 * 60}
    hb = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode()).rstrip(b"=")
    cb = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode()).rstrip(b"=")
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), f"{hb.decode()}.{cb.decode()}".encode(), hashlib.sha256).digest()
    ).rstrip(b"=")
    return f"{hb.decode()}.{cb.decode()}.{sig.decode()}"
