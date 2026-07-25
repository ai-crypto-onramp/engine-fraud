from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .config import Settings

try:
    import mlflow
    _HAVE_MLFLOW = True
except Exception:  # pragma: no cover - optional in tests
    mlflow = None  # type: ignore
    _HAVE_MLFLOW = False

from .db import PostgresStore
from .scoring import ModelLoader


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class ModelRegistry:
    """Thin wrapper over MLflow + the model_versions table.

    For tests and environments without MLflow, falls back to in-memory
    registration seeded from `default_model_for_name`.
    """

    def __init__(self, settings: Settings, db: PostgresStore | None = None,
                 loader: ModelLoader | None = None) -> None:
        self.settings = settings
        self.db = db
        self.loader = loader or ModelLoader(settings.model_registry_url)
        self._lock = threading.Lock()
        self._versions: dict[str, dict[str, dict[str, Any]]] = {}
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        with self._lock:
            self._versions.setdefault("chargeback-xgb", {})
            self._versions["chargeback-xgb"]["v3.2.0-stub"] = {
                "name": "chargeback-xgb",
                "version": "v3.2.0-stub",
                "stage": "champion",
                "metrics": {"auc": 0.86, "pr_auc": 0.81, "calibration": 0.04},
                "traffic_split": {"champion": 0.9, "challenger": 0.1},
                "trained_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            self._versions.setdefault("velocity-isoforest", {})
            self._versions["velocity-isoforest"]["v1.4.0-stub"] = {
                "name": "velocity-isoforest",
                "version": "v1.4.0-stub",
                "stage": "champion",
                "metrics": {"auc": 0.78, "pr_auc": 0.69, "calibration": 0.07},
                "traffic_split": {"champion": 1.0},
                "trained_at": _now_iso(),
                "updated_at": _now_iso(),
            }
        self.loader.register_stage("chargeback-xgb", "champion", "v3.2.0-stub")
        self.loader.register_stage("velocity-isoforest", "champion", "v1.4.0-stub")

    def register(
        self, name: str, version: str, stage: str, metrics: Mapping[str, Any],
        traffic_split: Mapping[str, float], trained_at: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "name": name,
            "version": version,
            "stage": stage,
            "metrics": dict(metrics),
            "traffic_split": dict(traffic_split),
            "trained_at": trained_at or _now_iso(),
            "updated_at": _now_iso(),
        }
        with self._lock:
            self._versions.setdefault(name, {})
            self._versions[name][version] = record
        self.loader.register_stage(name, stage, version)
        if self.db is not None:
            try:
                self.db.insert_model_version(
                    name, version, stage, dict(metrics), dict(traffic_split), record["trained_at"]
                )
            except Exception:
                pass
        if _HAVE_MLFLOW and self.settings.model_registry_url:
            try:
                mlflow.set_tracking_uri(self.settings.model_registry_url)
                client = mlflow.tracking.MlflowClient()
                try:
                    client.create_registered_model(name)
                except Exception:
                    pass
                mv = client.create_model_version(
                    name=name, source=f"models:/{name}/{version}", version=version,
                )
                client.transition_model_version_stage(name, mv.version, stage)
            except Exception:
                pass
        return record

    def transition(self, name: str, version: str, to_stage: str) -> dict[str, Any]:
        with self._lock:
            rec = self._versions.get(name, {}).get(version)
            if rec is None:
                raise KeyError(f"model {name}@{version} not registered")
            rec["stage"] = to_stage
            rec["updated_at"] = _now_iso()
            out = dict(rec)
        self.loader.register_stage(name, to_stage, version)
        self.loader.invalidate(name)
        if self.db is not None:
            try:
                self.db.insert_model_version(
                    name, version, to_stage, dict(out["metrics"]),
                    dict(out["traffic_split"]), out["trained_at"],
                )
            except Exception:
                pass
        return out

    def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            versions = {n: dict(v) for n, v in self._versions.items()}
        for name, by_ver in versions.items():
            champion = None
            challenger = None
            split: dict[str, float] = {}
            updated = _now_iso()
            for ver, rec in by_ver.items():
                if rec["stage"] == "champion":
                    champion = ver
                    split = dict(rec.get("traffic_split", {}))
                    updated = rec.get("updated_at", updated)
                elif rec["stage"] == "challenger":
                    challenger = ver
                    if not split:
                        split = dict(rec.get("traffic_split", {}))
                    updated = max(updated, rec.get("updated_at", updated))
            if not split and champion is not None:
                split = {"champion": 1.0}
            out.append({
                "name": name,
                "champion": champion,
                "challenger": challenger,
                "traffic_split": split,
                "updated_at": updated,
            })
        return out

    def resolve_version(self, name: str, variant: str) -> str | None:
        with self._lock:
            by_ver = self._versions.get(name, {})
        for ver, rec in by_ver.items():
            if rec["stage"] == variant:
                return ver
        return None

    def get_model(self, name: str, version: str) -> Any:
        return self.loader.get(name, version)

    def traffic_split_for(self, name: str) -> dict[str, float]:
        with self._lock:
            by_ver = self._versions.get(name, {})
        for rec in by_ver.values():
            if rec["stage"] == "champion" and rec.get("traffic_split"):
                return dict(rec["traffic_split"])
        return {"champion": 1.0}
