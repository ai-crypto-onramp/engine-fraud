import asyncio
import json

from fraud_engine.config import Settings
from fraud_engine.consumers.kafka import (
    FraudConsumer,
    event_to_score_request,
    parse_payment_event,
)


def test_parse_payment_event() -> None:
    e = parse_payment_event(b'{"user_id":"u1","payment_id":"p1","amount":{"currency":"USD","minor_units":100},"device":{"fingerprint":"fp1"},"ip":"1.2.3.4"}')
    assert e["user_id"] == "u1"


def test_event_to_score_request() -> None:
    req = event_to_score_request({
        "user_id": "u1", "payment_id": "p1", "tx_id": "t1",
        "amount": {"currency": "USD", "minor_units": 100},
        "device": {"fingerprint": "fp1", "type": "mobile_ios"},
        "ip": "1.2.3.4",
        "behavioral_features": {"session_duration_ms": 1, "keystroke_entropy": 0.5, "tap_variance": 0.2},
    })
    assert req.user_id == "u1"
    assert req.amount.minor_units == 100
    assert req.device.fingerprint == "fp1"
    assert req.tx_id == "t1"


async def test_consumer_handles_payment_and_chargeback() -> None:
    settings = Settings()
    processed_scores: list = []
    feedback_events: list = []

    async def handler(req):
        processed_scores.append(req.tx_id)
        return {"tx_id": req.tx_id}

    async def feedback_handler(event):
        feedback_events.append(event)

    consumer = FraudConsumer(settings, handler=handler, feedback_handler=feedback_handler)
    await consumer.handle_message("payment.authorized", json.dumps({
        "user_id": "u1", "payment_id": "p1", "tx_id": "t1",
        "amount": {"currency": "USD", "minor_units": 100},
        "device": {"fingerprint": "fp1"}, "ip": "1.2.3.4",
    }).encode("utf-8"))
    await consumer.handle_message("chargeback.received", json.dumps({
        "tx_id": "t1", "outcome": "chargeback",
        "reported_at": "2026-07-12T09:00:00Z",
    }).encode("utf-8"))
    assert processed_scores == ["t1"]
    assert feedback_events and feedback_events[0]["outcome"] == "chargeback"


async def test_consumer_run_noop_without_brokers() -> None:
    settings = Settings()
    consumer = FraudConsumer(settings)
    await consumer.run()
    assert consumer.processed == 0


async def test_consumer_scoring_path_emits_audit() -> None:
    from fraud_engine.app import _do_score
    from fraud_engine.audit import AuditEmitter
    from fraud_engine.models.schemas import BehavioralFeatures, DeviceInfo, Money, ScoreRequest

    audit = AuditEmitter(db=None)

    async def handler(req: ScoreRequest):
        return _do_score(req, audit=audit)

    settings = Settings()
    consumer = FraudConsumer(settings, handler=handler)
    await consumer.handle_message("payment.authorized", json.dumps({
        "user_id": "u1", "payment_id": "p1", "tx_id": "t_audit",
        "amount": {"currency": "USD", "minor_units": 100},
        "device": {"fingerprint": "fp1"}, "ip": "1.2.3.4",
    }).encode("utf-8"))
    assert audit.emitted
    assert audit.emitted[0]["tx_id"] == "t_audit"
    assert audit.emitted[0]["schema"] == "fraud.audit/v1"


async def test_make_kafka_producer_returns_none_without_brokers() -> None:
    from fraud_engine.audit import make_kafka_producer
    producer = await make_kafka_producer([])
    assert producer is None


async def test_consumer_start_stop_noop_without_brokers() -> None:
    consumer = FraudConsumer(Settings())
    await consumer.start()
    assert consumer._consumer is None
    await consumer.stop()


async def test_consumer_handle_message_no_handler() -> None:
    consumer = FraudConsumer(Settings(), handler=None)
    result = await consumer.handle_message(
        "payment.authorized",
        json.dumps({"user_id": "u", "payment_id": "p", "tx_id": "t",
                    "amount": {"currency": "USD", "minor_units": 1},
                    "device": {"fingerprint": "fp"}, "ip": "1.2.3.4"}).encode("utf-8"),
    )
    assert result["tx_id"] == "t" and result["skipped"] is True


async def test_app_starts_and_stops_consumer_without_brokers(monkeypatch) -> None:
    from fraud_engine import app as app_module

    monkeypatch.delenv("KAFKA_BROKERS", raising=False)
    await app_module._start_kafka_consumer()
    assert app_module._consumer_task is None
    assert app_module._consumer is None
    await app_module._stop_kafka_consumer()


async def test_app_starts_consumer_with_fake_broker(monkeypatch) -> None:
    import asyncio

    from fraud_engine import app as app_module
    from fraud_engine.consumers.kafka import FraudConsumer

    monkeypatch.setenv("KAFKA_BROKERS", "127.0.0.1:9092")

    started = {"stop_called": False}

    class FakeConsumer(FraudConsumer):
        async def run(self) -> None:
            await asyncio.sleep(0.05)

        async def stop(self) -> None:
            started["stop_called"] = True

    monkeypatch.setattr(app_module, "FraudConsumer", FakeConsumer)
    await app_module._start_kafka_consumer()
    assert app_module._consumer_task is not None
    await app_module._consumer_task
    await app_module._stop_kafka_consumer()
    assert started["stop_called"] is True
    assert app_module._consumer_task is None
