"""Tests for the service-token JWT auth dependency."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException

from fraud_engine.authtoken import _verify, issue, require_token, secret_from_env


class _FakeRequest:
    def __init__(self, path: str = "/v1/foo", headers: dict[str, str] | None = None) -> None:
        self.url = type("U", (), {"path": path})()
        self.headers = headers or {}


def test_secret_from_env_dev_mode_bypass(monkeypatch):
    monkeypatch.setenv("DEV_MODE", "1")
    monkeypatch.delenv("SERVICE_TOKEN_SECRET", raising=False)
    secret, bypass = secret_from_env()
    assert secret == ""
    assert bypass is True


def test_secret_from_env_prod_unset_raises(monkeypatch):
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.delenv("SERVICE_TOKEN_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        secret_from_env()


def test_secret_from_env_prod_set(monkeypatch):
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.setenv("SERVICE_TOKEN_SECRET", "s3cr3t")
    secret, bypass = secret_from_env()
    assert secret == "s3cr3t"
    assert bypass is False


async def test_require_token_skips_healthz(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN_SECRET", "s3cr3t")
    req = _FakeRequest(path="/healthz")
    assert await require_token(req) is None


async def test_require_token_bypass_in_dev_mode(monkeypatch):
    monkeypatch.setenv("DEV_MODE", "1")
    monkeypatch.delenv("SERVICE_TOKEN_SECRET", raising=False)
    req = _FakeRequest(path="/v1/foo")
    assert await require_token(req) is None


async def test_require_token_missing_header(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN_SECRET", "s3cr3t")
    monkeypatch.delenv("DEV_MODE", raising=False)
    req = _FakeRequest(path="/v1/foo")
    with pytest.raises(HTTPException) as exc:
        await require_token(req)
    assert exc.value.status_code == 401
    detail = exc.value.detail
    assert "missing or malformed" in detail["error"]["message"]


async def test_require_token_invalid_signature(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN_SECRET", "s3cr3t")
    monkeypatch.delenv("DEV_MODE", raising=False)
    req = _FakeRequest(path="/v1/foo", headers={"Authorization": "Bearer a.b.c"})
    with pytest.raises(HTTPException) as exc:
        await require_token(req)
    assert exc.value.status_code == 401


async def test_require_token_valid(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN_SECRET", "s3cr3t")
    monkeypatch.delenv("DEV_MODE", raising=False)
    token = issue("engine-fraud", "s3cr3t")
    req = _FakeRequest(path="/v1/foo", headers={"Authorization": f"Bearer {token}"})
    claims = await require_token(req)
    assert claims is not None
    assert claims["sub"] == "engine-fraud"


async def test_require_token_expired(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN_SECRET", "s3cr3t")
    monkeypatch.delenv("DEV_MODE", raising=False)
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    claims = {"sub": "engine-fraud", "iat": now - 100, "exp": now - 10}
    hb = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode()).rstrip(b"=")
    cb = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode()).rstrip(b"=")
    sig = base64.urlsafe_b64encode(
        hmac.new(b"s3cr3t", f"{hb.decode()}.{cb.decode()}".encode(), hashlib.sha256).digest()
    ).rstrip(b"=")
    token = f"{hb.decode()}.{cb.decode()}.{sig.decode()}"
    req = _FakeRequest(path="/v1/foo", headers={"Authorization": f"Bearer {token}"})
    with pytest.raises(HTTPException) as exc:
        await require_token(req)
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail["error"]["message"]


def test_issue_requires_secret():
    with pytest.raises(ValueError):
        issue("engine-fraud", "")


def test_verify_roundtrip():
    token = issue("engine-fraud", "s3cr3t")
    claims = _verify(token, "s3cr3t")
    assert claims["sub"] == "engine-fraud"


def test_verify_wrong_secret():
    token = issue("engine-fraud", "s3cr3t")
    with pytest.raises(ValueError):
        _verify(token, "wrong")
