from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from .config import Settings

try:
    from feast import FeatureStore as _FeastFeatureStore
    _HAVE_FEAST = True
except Exception:  # pragma: no cover - optional in tests
    _FeastFeatureStore = None  # type: ignore
    _HAVE_FEAST = False


FEATURE_GROUP_FIELDS: dict[str, list[str]] = {
    "user_velocity": [
        "tx_count_1h", "tx_count_24h", "tx_count_7d", "tx_sum_24h",
        "distinct_cards_24h", "distinct_devices_24h", "distinct_ips_24h",
    ],
    "payment_history": ["success_fail_ratio_30d", "avg_ticket_30d", "first_payment_age_days"],
    "device": ["fingerprint_hash", "new_to_user", "known_bad", "emulator", "rooted"],
    "geolocation": ["ip_country", "distance_from_billing_km", "vpn_proxy", "geo_velocity_kmh"],
}


class FeatureStoreClient:
    def __init__(self, settings: Settings, repo_path: str = "feature_repo") -> None:
        self.settings = settings
        self.repo_path = repo_path
        self._store: Any = None
        self._lock = threading.Lock()

    @property
    def store(self) -> Any:
        if not _HAVE_FEAST:
            raise RuntimeError("feast not available")
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = _FeastFeatureStore(repo_path=self.repo_path)
        return self._store

    def get_online_features(
        self, entity_rows: list[dict[str, Any]], features: list[str]
    ) -> dict[str, list[Any]]:
        if not _HAVE_FEAST:
            return {f: [] for f in features}
        resp = self.store.get_online_features(features=features, entity_rows=entity_rows)
        return {k: list(v) for k, v in resp.to_dict().items()}

    def get_features(self, entity_id: str, group: str) -> dict[str, Any]:
        if not _HAVE_FEAST:
            return {}
        fields = FEATURE_GROUP_FIELDS.get(group, [])
        if not fields:
            return {}
        entity_key = "user_id" if group in {"user_velocity", "payment_history", "geolocation"} else "device_id"
        resp = self.store.get_online_features(
            features=[f"{group}:{f}" for f in fields],
            entity_rows=[{entity_key: entity_id}],
        )
        out: dict[str, Any] = {}
        for f in fields:
            key = f"{group}:{f}"
            vals = resp.get(key) or resp.get(f, [])
            if vals:
                out[f] = vals[0]
        return out

    def ping(self) -> bool:
        try:
            return bool(_HAVE_FEAST)
        except Exception:
            return False


class InMemoryFeatureStore(FeatureStoreClient):
    """Test-friendly feature store that returns canned features keyed by entity."""

    def __init__(self, settings: Settings | None = None, repo_path: str = "feature_repo",
                 fixtures: Mapping[tuple[str, str], Mapping[str, Any]] | None = None) -> None:
        if settings is None:
            settings = Settings()
        super().__init__(settings, repo_path)
        self._fixtures: dict[tuple[str, str], dict[str, Any]] = {
            (k[0], k[1]): dict(v) for k, v in (fixtures or {}).items()
        }
        self._writes: list[tuple[str, str, dict[str, Any]]] = []

    def seed(self, group: str, entity_id: str, values: Mapping[str, Any]) -> None:
        self._fixtures[(group, entity_id)] = dict(values)

    def get_features(self, entity_id: str, group: str) -> dict[str, Any]:
        return dict(self._fixtures.get((group, entity_id), {}))

    def get_online_features(
        self, entity_rows: list[dict[str, Any]], features: list[str]
    ) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {f: [] for f in features}
        for row in entity_rows:
            for f in features:
                group, field = f.split(":", 1) if ":" in f else f.split("__", 1)
                if group not in FEATURE_GROUP_FIELDS:
                    group, field = f.rsplit("_", 1)[0], f
                entity_key = row.get("user_id") or row.get("device_id") or row.get("tx_id") or ""
                fx = self._fixtures.get((group, entity_key), {})
                out[f].append(fx.get(field, 0))
        return out

    def write(self, group: str, entity_id: str, values: Mapping[str, Any]) -> None:
        self._fixtures[(group, entity_id)] = dict(values)
        self._writes.append((group, entity_id, dict(values)))

    @property
    def writes(self) -> list[tuple[str, str, dict[str, Any]]]:
        return list(self._writes)

    def ping(self) -> bool:
        return True


def build_feature_store(settings: Settings, repo_path: str = "feature_repo") -> FeatureStoreClient:
    return FeatureStoreClient(settings, repo_path=repo_path)
