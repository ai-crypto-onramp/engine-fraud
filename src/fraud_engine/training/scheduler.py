from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config import Settings
from ..db import PostgresStore
from ..registry import ModelRegistry
from .train_chargeback import train_chargeback, train_velocity


class PromotionGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._approvals: dict[tuple[str, str], bool] = {}

    def approve(self, name: str, version: str) -> bool:
        with self._lock:
            self._approvals[(name, version)] = True
        return True

    def is_approved(self, name: str, version: str) -> bool:
        with self._lock:
            return self._approvals.get((name, version), False)

    def reset(self) -> None:
        with self._lock:
            self._approvals.clear()


class RetrainScheduler:
    def __init__(
        self, settings: Settings, db: PostgresStore | None = None,
        registry: ModelRegistry | None = None, gate: PromotionGate | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.registry = registry
        self.gate = gate or PromotionGate()
        self._last_velocity_run: datetime | None = None
        self._last_chargeback_run: datetime | None = None

    def _watermark(self, last: datetime | None) -> str:
        if last is None:
            return (datetime.now(UTC) - timedelta(days=365)).isoformat()
        return last.isoformat()

    def run_velocity(self, dataset_path: str = "data/velocity_dataset.json") -> dict[str, Any] | None:
        if not self._due(self.settings.retrain_velocity_cron, self._last_velocity_run):
            return None
        result = train_velocity(dataset_path)
        self._last_velocity_run = datetime.now(UTC)
        return result

    def run_chargeback(self, dataset_path: str = "data/chargeback_dataset.json") -> dict[str, Any] | None:
        if not self._due(self.settings.retrain_chargeback_cron, self._last_chargeback_run):
            return None
        result = train_chargeback(dataset_path)
        self._last_chargeback_run = datetime.now(UTC)
        return result

    def promote(self, name: str, version: str, to_stage: str = "challenger") -> dict[str, Any] | None:
        if not self.gate.is_approved(name, version):
            return None
        if self.registry is None:
            return None
        return self.registry.transition(name, version, to_stage)

    def _due(self, _cron: str, last: datetime | None) -> bool:
        if last is None:
            return True
        return (datetime.now(UTC) - last) > timedelta(hours=23)

    def pull_labels(self, watermark: str) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        return self.db.fetch_labeled_since(watermark)
