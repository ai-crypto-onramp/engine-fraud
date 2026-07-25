from __future__ import annotations

import hashlib
import json
import threading
import uuid
from typing import Any

from .db import PostgresStore
from .models.schemas import ScoreResponse

try:
    from aiokafka import AIOKafkaProducer
    _HAVE_AIOKAFKA = True
except Exception:  # pragma: no cover - optional in tests
    AIOKafkaProducer = None  # type: ignore
    _HAVE_AIOKAFKA = False


AUDIT_FIELDS = [
    "tx_id", "user_id", "score", "risk_band", "model_version", "variant",
    "top_features", "feature_snapshot_uri", "scored_at",
]


def build_audit_payload(
    req, resp: ScoreResponse, feature_snapshot_uri: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "fraud.audit/v1",
        "tx_id": req.tx_id or req.payment_id,
        "user_id": req.user_id,
        "score": resp.score,
        "risk_band": resp.risk_band,
        "model_version": resp.model_version,
        "variant": resp.variant,
        "top_features": [f.model_dump() for f in resp.top_features],
        "feature_snapshot_uri": feature_snapshot_uri,
        "scored_at": resp.scored_at,
    }


def build_envelope(payload: dict[str, Any], scored_at: str) -> dict[str, Any]:
    payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_hash = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
    target_id = payload.get("tx_id") or ""
    return {
        "schema_version": "1",
        "id": str(uuid.uuid4()),
        "ts": scored_at,
        "source_service": "fraud-detection",
        "actor_id": "fraud-detection",
        "action": "fraud.score",
        "target_type": "transaction",
        "target_id": target_id,
        "payload_hash": payload_hash,
        "payload": payload,
    }


class AuditEmitter:
    def __init__(self, db: PostgresStore | None = None, producer: Any = None,
                 audit_topic: str = "audit.v1") -> None:
        self.db = db
        self.producer = producer
        self.audit_topic = audit_topic
        self._lock = threading.Lock()
        self._emitted: list[dict[str, Any]] = []
        self._emitted_keys: set[tuple[str, str]] = set()

    def emit(self, req, resp: ScoreResponse, feature_snapshot_uri: str | None = None) -> dict[str, Any]:
        payload = build_audit_payload(req, resp, feature_snapshot_uri=feature_snapshot_uri)
        tx_id = payload["tx_id"]
        key = (tx_id, resp.scored_at)
        with self._lock:
            if key in self._emitted_keys:
                return payload
            self._emitted_keys.add(key)
            self._emitted.append(payload)
        if self.db is not None:
            try:
                self.db.insert_score(
                    tx_id=tx_id, user_id=payload["user_id"], score=resp.score,
                    risk_band=resp.risk_band, model_version=resp.model_version,
                    variant=resp.variant, top_features=payload["top_features"],
                    scored_at=resp.scored_at,
                )
            except Exception:
                pass
        if self.producer is not None and _HAVE_AIOKAFKA:
            envelope = build_envelope(payload, resp.scored_at)
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self.producer.send_and_wait(self.audit_topic, value=json.dumps(envelope).encode("utf-8"))
                    )
                else:
                    loop.run_until_complete(
                        self.producer.send_and_wait(self.audit_topic, value=json.dumps(envelope).encode("utf-8"))
                    )
            except Exception:
                pass
        return payload

    @property
    def emitted(self) -> list[dict[str, Any]]:
        return list(self._emitted)

    def reset(self) -> None:
        with self._lock:
            self._emitted.clear()
            self._emitted_keys.clear()


async def make_kafka_producer(brokers: list[str]) -> Any:
    if not _HAVE_AIOKAFKA or not brokers:
        return None
    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    await producer.start()
    return producer
