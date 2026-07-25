from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        if required:
            raise RuntimeError(f"Required env var {name!r} is not set")
        return default
    return value


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    port: int = field(default_factory=lambda: _int("PORT", 8080))
    db_url: str | None = field(default_factory=lambda: _env("DB_URL"))
    redis_url: str | None = field(default_factory=lambda: _env("REDIS_URL"))
    feature_store_url: str = field(
        default_factory=lambda: _env("FEATURE_STORE_URL", "redis://redis:6379") or "redis://redis:6379"
    )
    kafka_brokers: list[str] = field(
        default_factory=lambda: _list("KAFKA_BROKERS", [])
    )
    kafka_consumer_group: str = field(
        default_factory=lambda: _env("KAFKA_CONSUMER_GROUP", "fraud-detection") or "fraud-detection"
    )
    model_registry_url: str | None = field(default_factory=lambda: _env("MODEL_REGISTRY_URL"))
    score_threshold_high: float = field(
        default_factory=lambda: _float("SCORE_THRESHOLD_HIGH", 0.75)
    )
    score_threshold_medium: float = field(
        default_factory=lambda: _float("SCORE_THRESHOLD_MEDIUM", 0.40)
    )
    challenger_traffic_fraction: float = field(
        default_factory=lambda: _float("CHALLENGER_TRAFFIC_FRACTION", 0.10)
    )
    retrain_velocity_cron: str = field(
        default_factory=lambda: _env("RETRAIN_VELOCITY_CRON", "0 3 * * *") or "0 3 * * *"
    )
    retrain_chargeback_cron: str = field(
        default_factory=lambda: _env("RETRAIN_CHARGEBACK_CRON", "0 4 * * 1") or "0 4 * * 1"
    )
    drift_psi_threshold: float = field(
        default_factory=lambda: _float("DRIFT_PSI_THRESHOLD", 0.2)
    )
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "info") or "info")
    otel_endpoint: str | None = field(
        default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    known_bad_devices_path: str = field(
        default_factory=lambda: _env("KNOWN_BAD_DEVICES_PATH", "config/known_bad_devices.txt")
        or "config/known_bad_devices.txt"
    )
    force_variant: str | None = field(
        default_factory=lambda: _env("FORCE_VARIANT")
    )
    audit_topic: str = field(default_factory=lambda: _env("AUDIT_TOPIC", "audit.v1") or "audit.v1")
    scored_topic: str = field(
        default_factory=lambda: _env("SCORED_TOPIC", "fraud.scored") or "fraud.scored"
    )
    alert_topic: str = field(
        default_factory=lambda: _env("ALERT_TOPIC", "fraud.alert.raised") or "fraud.alert.raised"
    )
    offline_store_dir: str = field(
        default_factory=lambda: _env("OFFLINE_STORE_DIR", "data/offline") or "data/offline"
    )
    model_path: str | None = field(default_factory=lambda: _env("MODEL_PATH"))
    dev_mode: bool = field(default_factory=lambda: os.environ.get("DEV_MODE") == "1")

    def kafka_brokers_str(self) -> str:
        return ",".join(self.kafka_brokers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "db_url": self.db_url,
            "redis_url": self.redis_url,
            "feature_store_url": self.feature_store_url,
            "kafka_brokers": self.kafka_brokers,
            "kafka_consumer_group": self.kafka_consumer_group,
            "model_registry_url": self.model_registry_url,
            "score_threshold_high": self.score_threshold_high,
            "score_threshold_medium": self.score_threshold_medium,
            "challenger_traffic_fraction": self.challenger_traffic_fraction,
            "retrain_velocity_cron": self.retrain_velocity_cron,
            "retrain_chargeback_cron": self.retrain_chargeback_cron,
            "drift_psi_threshold": self.drift_psi_threshold,
            "log_level": self.log_level,
            "otel_endpoint": self.otel_endpoint,
            "known_bad_devices_path": self.known_bad_devices_path,
            "force_variant": self.force_variant,
            "audit_topic": self.audit_topic,
            "scored_topic": self.scored_topic,
            "alert_topic": self.alert_topic,
            "offline_store_dir": self.offline_store_dir,
        }


def get_settings() -> Settings:
    return Settings()


def require_runtime_settings() -> Settings:
    return Settings(
        db_url=_env("DB_URL", required=True),
        redis_url=_env("REDIS_URL", required=True),
        kafka_brokers=_list("KAFKA_BROKERS", ["localhost:9092"]),
        model_registry_url=_env("MODEL_REGISTRY_URL", required=True),
    )
