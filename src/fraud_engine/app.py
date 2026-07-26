from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .audit import AuditEmitter, make_kafka_producer
from .authtoken import require_token
from .config import Settings, get_settings
from .consumers.kafka import FraudConsumer
from .db import PostgresStore, RedisStore
from .feature_store import FeatureStoreClient, InMemoryFeatureStore, build_feature_store
from .models.schemas import (
    FeedbackRequest,
    ModelInfo,
    ModelsResponse,
    ScoreRequest,
    ScoreResponse,
)
from .observability.monitoring import VARIANT_TRAFFIC, AlertSink, DriftMonitor
from .registry import ModelRegistry
from .scoring import ModelLoader, score_request
from .tracing import init_tracing, instrument_app

init_tracing()

app = FastAPI(title="Fraud Detection")
instrument_app(app)


@app.middleware("http")
async def _service_token_auth(request: Request, call_next):
    try:
        await require_token(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return await call_next(request)

log = logging.getLogger("fraud_engine.app")

_DEFAULT_SETTINGS = get_settings()
_DEV_MODE = _DEFAULT_SETTINGS.dev_mode

if _DEV_MODE:
    log.warning("DEV_MODE=1: using StubModel / in-memory defaults — NOT FOR PRODUCTION")


@app.on_event("startup")
async def _enforce_prod_requirements() -> None:
    if os.environ.get("DEV_MODE") == "1":
        return
    missing: list[str] = []
    if not os.environ.get("DB_URL"):
        missing.append("DB_URL")
    if not os.environ.get("REDIS_URL"):
        missing.append("REDIS_URL")
    if not os.environ.get("KAFKA_BROKERS"):
        missing.append("KAFKA_BROKERS")
    if not (os.environ.get("MODEL_REGISTRY_URL") or os.environ.get("MODEL_PATH")):
        missing.append("FRAUD_MODEL_URL (MODEL_REGISTRY_URL) or MODEL_PATH")
    if missing:
        log.error(
            "required env vars missing in production mode: %s — set DEV_MODE=1 for local dev",
            ", ".join(missing),
        )
        raise SystemExit(1)


_DEFAULT_DB = PostgresStore(_DEFAULT_SETTINGS.db_url)
_DEFAULT_REDIS = RedisStore(_DEFAULT_SETTINGS.redis_url)
if _DEV_MODE:
    _DEFAULT_FEATURE_STORE: FeatureStoreClient = InMemoryFeatureStore(_DEFAULT_SETTINGS)
else:
    _DEFAULT_FEATURE_STORE = build_feature_store(_DEFAULT_SETTINGS)
_DEFAULT_LOADER = ModelLoader(_DEFAULT_SETTINGS.model_registry_url)
_DEFAULT_REGISTRY = ModelRegistry(_DEFAULT_SETTINGS, db=None, loader=_DEFAULT_LOADER)
_DEFAULT_AUDIT = AuditEmitter(db=None, audit_topic=_DEFAULT_SETTINGS.audit_topic)
_DEFAULT_ALERTS = AlertSink()
_DEFAULT_DRIFT = DriftMonitor(_DEFAULT_SETTINGS, db=None, alert_sink=_DEFAULT_ALERTS)

_kafka_producer: Any = None

# Kafka consumer task handle (started on startup when KAFKA_BROKERS is set).
_consumer_task: asyncio.Task[Any] | None = None
_consumer: FraudConsumer | None = None


def _scoring_model_is_real() -> bool:
    try:
        model = _DEFAULT_REGISTRY.get_model("chargeback-xgb", "v3.2.0-stub")
    except Exception:
        return False
    return type(model).__name__ != "StubModel" and type(model).__name__ != "StubVelocityModel"


def db_ready() -> bool:
    if _DEV_MODE:
        return True
    return _DEFAULT_DB.ping()


def mq_ready() -> bool:
    if _DEV_MODE:
        return True
    return bool(_DEFAULT_SETTINGS.kafka_brokers) and _kafka_producer is not None


def rules_ready() -> bool:
    if _DEV_MODE:
        return True
    return True


def features_ready() -> bool:
    if _DEV_MODE:
        return True
    return _DEFAULT_FEATURE_STORE.ping()


def scoring_ready() -> bool:
    if _DEV_MODE:
        return True
    return _scoring_model_is_real()


def kyt_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def ledger_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def pricing_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def identity_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def policy_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def notification_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def audit_ready() -> bool:
    if _DEV_MODE:
        return True
    return bool(_DEFAULT_SETTINGS.kafka_brokers) and _kafka_producer is not None


def rail_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def exchange_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def blockchain_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def mpc_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def wallet_ready() -> bool:
    if _DEV_MODE:
        return True
    return False


def onboarding_ready() -> bool:
    if _DEV_MODE:
        return True
    return False

READINESS_CHECKS = [
    ("db", db_ready),
    ("mq", mq_ready),
    ("rules", rules_ready),
    ("features", features_ready),
    ("scoring", scoring_ready),
    ("kyt", kyt_ready),
    ("ledger", ledger_ready),
    ("pricing", pricing_ready),
    ("identity", identity_ready),
    ("policy", policy_ready),
    ("notification", notification_ready),
    ("audit", audit_ready),
    ("rail", rail_ready),
    ("exchange", exchange_ready),
    ("blockchain", blockchain_ready),
    ("mpc", mpc_ready),
    ("wallet", wallet_ready),
    ("onboarding", onboarding_ready),
]


def readiness_report() -> tuple[dict[str, str], int, int]:
    results: dict[str, str] = {}
    failed = 0
    total = 0
    for name, fn in READINESS_CHECKS:
        total += 1
        if fn():
            results[name] = "ok"
        else:
            results[name] = "down"
            failed += 1
    return results, failed, total


def classify_readiness(failed: int, total: int) -> tuple[int, str]:
    if failed == total and total > 0:
        return 503, "not ready"
    if failed > 0:
        return 200, "degraded"
    return 200, "ready"


@app.on_event("startup")
async def _apply_migrations_on_startup() -> None:
    if _DEFAULT_SETTINGS.db_url and _DEFAULT_DB.ping():
        _DEFAULT_DB.apply_migrations()


@app.on_event("startup")
async def _start_kafka_consumer() -> None:
    """Start the FraudConsumer background task when KAFKA_BROKERS is set."""
    global _consumer_task, _consumer
    brokers = [b for b in (os.environ.get("KAFKA_BROKERS", "") or "").split(",") if b.strip()]
    if not brokers:
        return
    _consumer = FraudConsumer(
        _DEFAULT_SETTINGS,
        handler=_async_score_handler,
        feedback_handler=_async_feedback_handler,
    )
    _consumer_task = asyncio.create_task(_consumer.run())
    log.info(
        "fraud kafka consumer started: brokers=%s group=%s",
        ",".join(brokers),
        _DEFAULT_SETTINGS.kafka_consumer_group,
    )


@app.on_event("startup")
async def _start_audit_kafka_producer() -> None:
    """Start the Kafka audit producer when KAFKA_BROKERS is set.

    In production the audit envelope must be published to the audit.v1 topic;
    without a broker the AuditEmitter silently keeps events in memory (which
    is only acceptable in DEV_MODE).
    """
    global _kafka_producer
    brokers = _DEFAULT_SETTINGS.kafka_brokers
    if not brokers:
        if not _DEV_MODE:
            log.error("KAFKA_BROKERS unset in production: audit events will NOT be published")
        return
    try:
        _kafka_producer = await make_kafka_producer(brokers)
        _DEFAULT_AUDIT.producer = _kafka_producer
        log.info("fraud audit kafka producer started: brokers=%s topic=%s",
                 ",".join(brokers), _DEFAULT_SETTINGS.audit_topic)
    except Exception as exc:
        if _DEV_MODE:
            log.warning("audit kafka producer failed to start (DEV_MODE=1): %s", exc)
        else:
            log.error("audit kafka producer failed to start in production: %s", exc)


@app.on_event("shutdown")
async def _stop_kafka_consumer() -> None:
    global _consumer_task, _consumer, _kafka_producer
    task = _consumer_task
    _consumer_task = None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    if _consumer is not None:
        await _consumer.stop()
        _consumer = None
    if _kafka_producer is not None:
        try:
            await _kafka_producer.stop()
        except Exception:
            pass
        _kafka_producer = None


async def _async_score_handler(req: ScoreRequest) -> ScoreResponse:
    return _do_score(req)


async def _async_feedback_handler(event: Any) -> None:
    log.info("feedback event received: tx_id=%s", event.get("tx_id"))


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    results, failed, total = readiness_report()
    code, status = classify_readiness(failed, total)
    results["status"] = status
    results["healthy"] = str(total - failed)
    results["failed"] = str(failed)
    results["total"] = str(total)
    return JSONResponse(status_code=code, content=results)


def _store_for(score_req: ScoreRequest) -> ScoreRequest:
    return score_req


def _features_for(
    feature_store: FeatureStoreClient, req: ScoreRequest,
) -> dict[str, Any]:
    feats: dict[str, Any] = {}
    feats.update(feature_store.get_features(req.user_id, "user_velocity") or {})
    feats.update(feature_store.get_features(req.user_id, "payment_history") or {})
    feats.update(feature_store.get_features(req.device.fingerprint, "device") or {})
    feats.update(feature_store.get_features(req.user_id, "geolocation") or {})
    feats["session_duration_ms"] = req.behavioral_features.session_duration_ms
    feats["keystroke_entropy"] = req.behavioral_features.keystroke_entropy
    feats["tap_variance"] = req.behavioral_features.tap_variance
    return feats


def _do_score(
    req: ScoreRequest,
    *,
    settings: Settings | None = None,
    feature_store: FeatureStoreClient | None = None,
    registry: ModelRegistry | None = None,
    audit: AuditEmitter | None = None,
) -> ScoreResponse:
    settings = settings or _DEFAULT_SETTINGS
    feature_store = feature_store or _DEFAULT_FEATURE_STORE
    registry = registry or _DEFAULT_REGISTRY
    audit = audit or _DEFAULT_AUDIT
    feats = _features_for(feature_store, req)
    model_name = "chargeback-xgb"
    tx_id = req.tx_id or req.payment_id
    split = registry.traffic_split_for(model_name)
    from .models.routing import pick_variant, resolve_split
    variant = pick_variant(
        tx_id,
        challenger_fraction=resolve_split(split, default_challenger_fraction=settings.challenger_traffic_fraction),
        force_variant=settings.force_variant,
    )
    version = registry.resolve_version(model_name, variant) or "v3.2.0-stub"
    model = registry.get_model(model_name, version)
    resp = score_request(
        req, features=feats, model=model, model_version=f"{model_name}@{version}",
        threshold_high=settings.score_threshold_high,
        threshold_medium=settings.score_threshold_medium,
        challenger_fraction=settings.challenger_traffic_fraction,
        traffic_split=split, force_variant=settings.force_variant,
    )
    VARIANT_TRAFFIC.labels(variant=resp.variant).inc()
    audit.emit(req, resp, feature_snapshot_uri=f"feature_values:{tx_id}")
    return resp


@app.post("/v1/fraud/score", response_model=ScoreResponse)
async def score(req: ScoreRequest) -> ScoreResponse:
    return _do_score(req)


@app.get("/v1/fraud/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    models = _DEFAULT_REGISTRY.list_models()
    out: list[ModelInfo] = []
    for m in models:
        breaches = _DEFAULT_DRIFT.breaches_for(m["name"])
        out.append(ModelInfo(
            name=m["name"], champion=m["champion"], challenger=m["challenger"],
            traffic_split=m["traffic_split"], updated_at=m["updated_at"],
            drift_breaches=breaches,
        ))
    return ModelsResponse(models=out)


@app.post("/v1/fraud/feedback")
async def feedback(req: FeedbackRequest) -> Response:
    if req.outcome not in {"CHARGEBACK", "FRAUD", "CLEAN"}:
        raise HTTPException(status_code=422, detail="invalid outcome")
    inserted = _DEFAULT_DB.upsert_chargeback(
        tx_id=req.tx_id, outcome=req.outcome, reason_code=req.reason_code,
        source=req.source, reported_at=req.reported_at,
    )
    if not inserted:
        return Response(status_code=204)
    return Response(status_code=204)
