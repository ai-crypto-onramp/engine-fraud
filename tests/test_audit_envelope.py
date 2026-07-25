import json
from unittest.mock import AsyncMock, MagicMock

from fraud_engine.audit import (
    AuditEmitter,
    build_audit_payload,
    build_envelope,
    make_kafka_producer,
)


def test_build_envelope_canonical_shape() -> None:
    payload = {"tx_id": "t1", "user_id": "u1", "score": 0.9}
    env = build_envelope(payload, "2026-01-01T00:00:00Z")
    assert env["schema_version"] == "1"
    assert env["source_service"] == "fraud-detection"
    assert env["action"] == "fraud.score"
    assert env["target_type"] == "transaction"
    assert env["target_id"] == "t1"
    assert env["payload_hash"].startswith("sha256:")
    assert env["payload"] == payload


def test_audit_emitter_dedups_same_tx_and_timestamp() -> None:
    em = AuditEmitter()
    req = MagicMock()
    req.tx_id = "t1"
    req.payment_id = None
    req.user_id = "u1"
    resp = MagicMock()
    resp.score = 0.9
    resp.risk_band = "high"
    resp.model_version = "v1"
    resp.variant = "champion"
    resp.top_features = []
    resp.scored_at = "2026-01-01T00:00:00Z"
    em.emit(req, resp)
    em.emit(req, resp)
    assert len(em.emitted) == 1


def test_audit_emitter_db_insert_swallows_exception() -> None:
    db = MagicMock()
    db.insert_score.side_effect = RuntimeError("db down")
    em = AuditEmitter(db=db)
    req = MagicMock()
    req.tx_id = "t2"
    req.payment_id = None
    req.user_id = "u2"
    resp = MagicMock()
    resp.score = 0.5
    resp.risk_band = "medium"
    resp.model_version = "v1"
    resp.variant = "champion"
    resp.top_features = []
    resp.scored_at = "2026-01-01T00:00:00Z"
    em.emit(req, resp)
    assert len(em.emitted) == 1


def test_build_audit_payload_uses_payment_id_when_no_tx_id() -> None:
    req = MagicMock()
    req.tx_id = None
    req.payment_id = "p1"
    req.user_id = "u1"
    resp = MagicMock()
    resp.score = 0.1
    resp.risk_band = "low"
    resp.model_version = "v1"
    resp.variant = "champion"
    resp.top_features = []
    resp.scored_at = "2026-01-01T00:00:00Z"
    payload = build_audit_payload(req, resp)
    assert payload["tx_id"] == "p1"


async def test_make_kafka_producer_empty_brokers_returns_none() -> None:
    assert await make_kafka_producer([]) is None


def test_audit_emitter_with_mock_producer_swallows_exception() -> None:
    producer = MagicMock()
    fut = MagicMock()
    fut.exception.return_value = RuntimeError("kafka down")
    producer.send_and_wait.return_value = fut
    em = AuditEmitter(producer=producer, audit_topic="audit.v1")
    req = MagicMock()
    req.tx_id = "t3"
    req.payment_id = None
    req.user_id = "u3"
    resp = MagicMock()
    resp.score = 0.5
    resp.risk_band = "medium"
    resp.model_version = "v1"
    resp.variant = "champion"
    resp.top_features = []
    resp.scored_at = "2026-01-01T00:00:00Z"
    em.emit(req, resp)
    assert len(em.emitted) == 1


async def test_audit_emitter_with_producer_in_running_loop() -> None:
    producer = MagicMock()
    producer.send_and_wait = AsyncMock(return_value=None)
    em = AuditEmitter(producer=producer, audit_topic="audit.v1")
    req = MagicMock()
    req.tx_id = "t4"
    req.payment_id = None
    req.user_id = "u4"
    resp = MagicMock()
    resp.score = 0.5
    resp.risk_band = "medium"
    resp.model_version = "v1"
    resp.variant = "champion"
    resp.top_features = []
    resp.scored_at = "2026-01-01T00:00:00Z"
    em.emit(req, resp)
    assert len(em.emitted) == 1
