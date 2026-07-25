from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from ..config import Settings
from ..db import PostgresStore
from .drift import breached_features, drift_report

SCORE_REQUESTS = Counter(
    "fraud_score_requests_total", "Total scoring requests", ["variant", "risk_band"]
)
SCORE_LATENCY = Histogram(
    "fraud_score_latency_seconds", "Score latency in seconds", ["variant"]
)
ALERTS_RAISED = Counter(
    "fraud_alerts_raised_total", "fraud.alert.raised emitted", ["model_version"]
)
CONSUMER_LAG = Gauge(
    "fraud_consumer_lag_records", "Kafka consumer lag in records", ["topic"]
)
CONSUMER_THROUGHPUT = Counter(
    "fraud_consumer_messages_total", "Kafka messages consumed", ["topic"]
)
VARIANT_TRAFFIC = Counter(
    "fraud_variant_traffic_total", "Traffic routed to a variant", ["variant"]
)


_METRICS_INITIALIZED = True


class AlertSink:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.alerts: list[dict[str, Any]] = []

    def emit_alert(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.alerts.append(dict(payload))
        ALERTS_RAISED.labels(model_version=str(payload.get("model_version", ""))).inc()
        return dict(payload)

    def reset(self) -> None:
        with self._lock:
            self.alerts.clear()


class DriftMonitor:
    def __init__(self, settings: Settings, db: PostgresStore | None = None,
                 alert_sink: AlertSink | None = None) -> None:
        self.settings = settings
        self.db = db
        self.alert_sink = alert_sink or AlertSink()
        self._breaches_by_model: dict[str, list[str]] = {}

    def run(
        self, model_name: str, baseline: Mapping[str, Sequence[float]],
        current: Mapping[str, Sequence[float]],
    ) -> list[dict[str, object]]:
        report = drift_report(
            baseline, current,
            psi_threshold=self.settings.drift_psi_threshold,
        )
        feats = breached_features(report)
        self._breaches_by_model[model_name] = feats
        if self.db is not None:
            for r in report:
                try:
                    self.db.insert_drift_metric(
                        model_name=model_name, feature_name=str(r["feature"]),
                        psi=float(r["psi"]), ks=float(r["ks"]),
                        breached=bool(r["breached"]),
                    )
                except Exception:
                    pass
        if feats:
            self.alert_sink.emit_alert({
                "schema": "fraud.alert.raised/v1",
                "model_name": model_name,
                "breached_features": feats,
                "psi_threshold": self.settings.drift_psi_threshold,
                "report": report,
            })
        return report

    def breaches_for(self, model_name: str) -> list[str]:
        return list(self._breaches_by_model.get(model_name, []))

    def maybe_trigger_retraining(self, model_name: str, *, repeat_threshold: int = 3) -> bool:
        return len(self._breaches_by_model.get(model_name, [])) >= repeat_threshold
