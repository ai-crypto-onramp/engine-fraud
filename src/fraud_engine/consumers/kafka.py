from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..config import Settings
from ..models.schemas import BehavioralFeatures, DeviceInfo, Money, ScoreRequest
from ..observability.monitoring import CONSUMER_LAG, CONSUMER_THROUGHPUT

try:
    from aiokafka import AIOKafkaConsumer
    _HAVE_AIOKAFKA = True
except Exception:  # pragma: no cover - optional in tests
    AIOKafkaConsumer = None  # type: ignore
    _HAVE_AIOKAFKA = False


log = logging.getLogger("fraud_engine.consumer")


def parse_payment_event(raw: bytes | str) -> dict[str, Any]:
    return json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))


def event_to_score_request(event: Mapping[str, Any]) -> ScoreRequest:
    amount = event.get("amount", {}) or {}
    dev = event.get("device", {}) or {}
    beh = event.get("behavioral_features", {}) or {}
    return ScoreRequest(
        user_id=str(event["user_id"]),
        payment_id=str(event.get("payment_id", event.get("tx_id", ""))),
        tx_id=str(event.get("tx_id") or event.get("payment_id")),
        amount=Money(
            currency=str(amount.get("currency", "USD")),
            minor_units=int(amount.get("minor_units", 0)),
        ),
        device=DeviceInfo(
            fingerprint=str(dev.get("fingerprint", "")),
            type=dev.get("type"),
            rooted=bool(dev.get("rooted", False)),
            emulator=bool(dev.get("emulator", False)),
        ),
        ip=str(event.get("ip", "")),
        behavioral_features=BehavioralFeatures(
            session_duration_ms=int(beh.get("session_duration_ms", 0)),
            keystroke_entropy=float(beh.get("keystroke_entropy", 0.0)),
            tap_variance=float(beh.get("tap_variance", 0.0)),
        ),
    )


class FraudConsumer:
    def __init__(
        self, settings: Settings,
        handler: Callable[[ScoreRequest], Awaitable[Any]] | None = None,
        feedback_handler: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.handler = handler
        self.feedback_handler = feedback_handler
        self._consumer: Any = None
        self._running = asyncio.Event()
        self._processed = 0

    async def start(self) -> None:
        if not _HAVE_AIOKAFKA or not self.settings.kafka_brokers:
            return
        self._consumer = AIOKafkaConsumer(
            "payment.authorized", "payment.captured", "chargeback.received",
            bootstrap_servers=self.settings.kafka_brokers,
            group_id=self.settings.kafka_consumer_group,
            enable_auto_commit=False,
            value_deserializer=lambda v: v,
        )
        await self._consumer.start()
        self._running.set()

    async def stop(self) -> None:
        self._running.clear()
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

    async def handle_message(self, topic: str, raw: bytes | str) -> dict[str, Any] | None:
        CONSUMER_THROUGHPUT.labels(topic=topic).inc()
        event = parse_payment_event(raw)
        if topic.startswith("chargeback."):
            if self.feedback_handler is not None:
                await self.feedback_handler(event)
            return None
        req = event_to_score_request(event)
        if self.handler is None:
            return {"tx_id": req.tx_id or req.payment_id, "skipped": True}
        result = await self.handler(req)
        return result

    async def run(self) -> None:
        await self.start()
        if self._consumer is None:
            return
        try:
            async for msg in self._consumer:
                if not self._running.is_set():
                    break
                await self.handle_message(msg.topic, msg.value)
                try:
                    partitions = self._consumer.assignment()
                    if partitions:
                        lag = sum(
                            self._consumer.position(p) - (self._consumer.highwater(p) or 0)
                            for p in partitions
                        )
                        CONSUMER_LAG.labels(topic=msg.topic).set(max(0, lag))
                except Exception:
                    pass
                await self._consumer.commit()
                self._processed += 1
        finally:
            await self.stop()

    @property
    def processed(self) -> int:
        return self._processed
