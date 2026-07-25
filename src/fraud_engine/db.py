from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from .config import Settings


def _new_id() -> str:
    # Python 3.14+ exposes uuid.uuid7; the service's venv currently runs 3.13
    # (see Dockerfile / .venv), so fall back to uuid4 on older runtimes.
    gen = getattr(uuid, "uuid7", None)
    return str(gen() if gen is not None else uuid.uuid4())

try:
    import psycopg  # type: ignore
    _HAVE_PSYCOPG = True
except Exception:  # pragma: no cover - optional in tests
    psycopg = None  # type: ignore
    _HAVE_PSYCOPG = False

try:
    import redis  # type: ignore
    _HAVE_REDIS = True
except Exception:  # pragma: no cover - optional in tests
    redis = None  # type: ignore
    _HAVE_REDIS = False


SCHEMA_SQL_PATH = "migrations/001_init.sql"
SCHEMA_VERSION = 1


def _loads(v: Any) -> Any:
    if v is None:
        return {}
    if isinstance(v, (dict, list)):
        return v
    return json.loads(v)


class PostgresStore:
    def __init__(self, dsn: str | None) -> None:
        self.dsn = dsn
        self._local = threading.local()

    def _connect(self) -> Any:
        if not _HAVE_PSYCOPG or not self.dsn:
            raise RuntimeError("psycopg not available or DB_URL unset")
        conn = psycopg.connect(self.dsn, autocommit=True)
        return conn

    @property
    def conn(self) -> Any:
        if getattr(self._local, "conn", None) is None:
            self._local.conn = self._connect()
        return self._local.conn

    def ping(self) -> bool:
        if not _HAVE_PSYCOPG or not self.dsn:
            return False
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            return False

    def apply_migrations(self, sql: str | None = None) -> None:
        """Apply the schema migration with idempotent tracking.

        Maintains a ``schema_migrations`` table (matching the convention used
        across the other services) so re-runs after ``make reset-db`` are a
        no-op instead of re-executing the DDL. When ``sql`` is provided it is
        applied verbatim (used by the ``make migrate`` entrypoint for ad-hoc
        scripts); otherwise the canonical embedded migration identified by
        ``SCHEMA_VERSION`` is applied once and recorded.
        """
        if sql is not None:
            with self.conn.cursor() as cur:
                cur.execute(sql)
            return
        with self.conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "    version    INTEGER PRIMARY KEY,"
                "    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
            cur.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (SCHEMA_VERSION,),
            )
            if cur.fetchone() is not None:
                return
            with open(SCHEMA_SQL_PATH, encoding="utf-8") as fh:
                cur.execute(fh.read())
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (SCHEMA_VERSION,),
            )

    def _json(self, value: Any) -> Any:
        return json.dumps(value)

    def insert_score(
        self,
        tx_id: str,
        user_id: str,
        score: float,
        risk_band: str,
        model_version: str,
        variant: str,
        top_features: list[dict[str, Any]],
        scored_at: str,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fraud_scores (id, tx_id, user_id, score, risk_band, model_version, "
                "variant, top_features, scored_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (_new_id(), tx_id, user_id, score, risk_band, model_version, variant,
                 self._json(top_features), scored_at),
            )

    def insert_feature_values(self, tx_id: str, group: str, payload: Mapping[str, Any]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feature_values (id, tx_id, feature_group, payload) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (tx_id, feature_group) DO UPDATE SET "
                "payload = EXCLUDED.payload, updated_at = now()",
                (_new_id(), tx_id, group, self._json(dict(payload))),
            )

    def insert_model_version(
        self, name: str, version: str, stage: str, metrics: Mapping[str, Any],
        traffic_split: Mapping[str, float], trained_at: str,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_versions (id, name, version, stage, metrics, traffic_split, trained_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (name, version) DO UPDATE SET "
                "stage=EXCLUDED.stage, metrics=EXCLUDED.metrics, traffic_split=EXCLUDED.traffic_split, "
                "updated_at=now()",
                (_new_id(), name, version, stage, self._json(dict(metrics)),
                 self._json(dict(traffic_split)), trained_at),
            )

    def list_model_versions(self) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT name, version, stage, metrics, traffic_split, trained_at, updated_at "
                "FROM model_versions ORDER BY name, version"
            )
            rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "name": r[0], "version": r[1], "stage": r[2],
                "metrics": _loads(r[3]),
                "traffic_split": _loads(r[4]),
                "trained_at": r[5].isoformat() if r[5] else None,
                "updated_at": r[6].isoformat() if r[6] else None,
            })
        return out

    def upsert_chargeback(
        self, tx_id: str, outcome: str, reason_code: str | None,
        source: str, reported_at: str,
    ) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chargeback_events (id, tx_id, outcome, reason_code, source, reported_at) "
                "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (tx_id, reported_at) DO NOTHING",
                (_new_id(), tx_id, outcome, reason_code, source, reported_at),
            )
            return cur.rowcount > 0

    def fetch_labeled_since(self, watermark: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT tx_id, outcome, reason_code, source, reported_at "
                "FROM chargeback_events WHERE reported_at >= %s ORDER BY reported_at",
                (watermark,),
            )
            rows = cur.fetchall()
        return [
            {"tx_id": r[0], "outcome": r[1], "reason_code": r[2], "source": r[3],
             "reported_at": r[4].isoformat() if r[4] else None}
            for r in rows
        ]

    def insert_drift_metric(
        self, model_name: str, feature_name: str, psi: float, ks: float, breached: bool,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO drift_metrics (id, model_name, feature_name, psi, ks, breached) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (_new_id(), model_name, feature_name, psi, ks, breached),
            )

    def fetch_scores_since(self, since: str, limit: int = 10000) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT tx_id, user_id, score, risk_band, model_version, variant, scored_at "
                "FROM fraud_scores WHERE scored_at >= %s ORDER BY scored_at LIMIT %s",
                (since, limit),
            )
            rows = cur.fetchall()
        return [
            {"tx_id": r[0], "user_id": r[1], "score": float(r[2]), "risk_band": r[3],
             "model_version": r[4], "variant": r[5], "scored_at": r[6].isoformat() if r[6] else None}
            for r in rows
        ]

    def fetch_score(self, tx_id: str) -> dict[str, Any] | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT tx_id, user_id, score, risk_band, model_version, variant, top_features, scored_at "
                "FROM fraud_scores WHERE tx_id = %s ORDER BY scored_at DESC LIMIT 1",
                (tx_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "tx_id": row[0], "user_id": row[1], "score": float(row[2]),
            "risk_band": row[3], "model_version": row[4], "variant": row[5],
            "top_features": _loads(row[6]),
            "scored_at": row[7].isoformat() if row[7] else None,
        }


class RedisStore:
    def __init__(self, url: str | None) -> None:
        self.url = url
        self._client: Any = None

    @property
    def client(self) -> Any:
        if not _HAVE_REDIS or not self.url:
            raise RuntimeError("redis not available or REDIS_URL unset")
        if self._client is None:
            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def ping(self) -> bool:
        if not _HAVE_REDIS or not self.url:
            return False
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def set_features(self, entity: str, group: str, values: Mapping[str, Any]) -> None:
        self.client.hset(f"features:{group}:{entity}", mapping={k: json.dumps(v) for k, v in values.items()})

    def get_features(self, entity: str, group: str) -> dict[str, Any]:
        raw = self.client.hgetall(f"features:{group}:{entity}") or {}
        return {k: json.loads(v) for k, v in raw.items()}

    def mget_features(self, entities: Iterable[str], group: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for e in entities:
            out[e] = self.get_features(e, group)
        return out


def build_stores(settings: Settings) -> tuple[PostgresStore, RedisStore]:
    return PostgresStore(settings.db_url), RedisStore(settings.redis_url)
