import pytest

from fraud_engine import app as app_module


def test_prod_guard_dev_mode_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_MODE", "1")
    import asyncio
    asyncio.run(app_module._enforce_prod_requirements())


def test_prod_guard_missing_envs_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("KAFKA_BROKERS", raising=False)
    monkeypatch.delenv("MODEL_REGISTRY_URL", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    import asyncio
    with pytest.raises(SystemExit):
        asyncio.run(app_module._enforce_prod_requirements())


def test_prod_guard_all_envs_set_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.setenv("DB_URL", "postgres://localhost/fraud")
    monkeypatch.setenv("REDIS_URL", "redis://localhost")
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    monkeypatch.setenv("MODEL_REGISTRY_URL", "http://localhost:8080")
    import asyncio
    asyncio.run(app_module._enforce_prod_requirements())
