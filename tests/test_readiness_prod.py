import pytest

from fraud_engine import app as app_module


def _flip_dev_mode(value: bool) -> None:
    app_module._DEV_MODE = value


def test_readiness_dev_mode_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_MODE", "1")
    _flip_dev_mode(True)
    try:
        results, failed, total = app_module.readiness_report()
        assert failed == 0
        assert total == len(app_module.READINESS_CHECKS)
        for name, _ in app_module.READINESS_CHECKS:
            assert results[name] == "ok"
    finally:
        _flip_dev_mode(False)


def test_readiness_prod_reports_down_for_unprobed_subsystems(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.delenv("KAFKA_BROKERS", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("MODEL_REGISTRY_URL", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    _flip_dev_mode(False)
    app_module._kafka_producer = None
    orig_fs = app_module._DEFAULT_FEATURE_STORE
    monkeypatch.setattr(app_module, "_DEFAULT_FEATURE_STORE", type("F", (), {"ping": staticmethod(lambda: False)})())
    try:
        results, failed, _ = app_module.readiness_report()
        assert failed > 0
        assert results["db"] == "down"
        assert results["mq"] == "down"
        assert results["features"] == "down"
        assert results["scoring"] == "down"
        assert results["audit"] == "down"
        assert results["kyt"] == "down"
        assert results["ledger"] == "down"
    finally:
        monkeypatch.setattr(app_module, "_DEFAULT_FEATURE_STORE", orig_fs)
        _flip_dev_mode(True)


def test_readiness_prod_db_ok_when_db_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_MODE", raising=False)
    _flip_dev_mode(False)

    class FakeDB:
        def ping(self) -> bool:
            return True

    monkeypatch.setattr(app_module, "_DEFAULT_DB", FakeDB())
    try:
        assert app_module.db_ready() is True
    finally:
        _flip_dev_mode(True)


def test_readiness_prod_mq_ok_when_producer_started(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_MODE", raising=False)
    _flip_dev_mode(False)
    monkeypatch.setattr(app_module, "_DEFAULT_SETTINGS",
                        type("S", (), {"kafka_brokers": ["localhost:9092"],
                                       "audit_topic": "audit.v1"})())
    orig_producer = app_module._kafka_producer
    app_module._kafka_producer = object()
    try:
        assert app_module.mq_ready() is True
        assert app_module.audit_ready() is True
    finally:
        app_module._kafka_producer = orig_producer
        _flip_dev_mode(True)


def test_readiness_prod_scoring_ok_when_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_MODE", raising=False)
    _flip_dev_mode(False)

    class RealModel:
        name = "chargeback-xgb"

    class FakeRegistry:
        def get_model(self, name, version):
            return RealModel()

    orig = app_module._DEFAULT_REGISTRY
    app_module._DEFAULT_REGISTRY = FakeRegistry()
    try:
        assert app_module.scoring_ready() is True
    finally:
        app_module._DEFAULT_REGISTRY = orig
        _flip_dev_mode(True)
