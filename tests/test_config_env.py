import os

from fraud_engine.config import Settings, get_settings, require_runtime_settings


def test_settings_loads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("SCORE_THRESHOLD_HIGH", "0.88")
    monkeypatch.setenv("SCORE_THRESHOLD_MEDIUM", "0.55")
    monkeypatch.setenv("CHALLENGER_TRAFFIC_FRACTION", "0.25")
    monkeypatch.setenv("KAFKA_BROKERS", "a:9092,b:9092")
    monkeypatch.setenv("FORCE_VARIANT", "champion")
    s = Settings()
    assert s.port == 9090
    assert s.score_threshold_high == 0.88
    assert s.score_threshold_medium == 0.55
    assert s.challenger_traffic_fraction == 0.25
    assert s.kafka_brokers == ["a:9092", "b:9092"]
    assert s.kafka_brokers_str() == "a:9092,b:9092"
    assert s.force_variant == "champion"


def test_settings_defaults(monkeypatch) -> None:
    for k in ["DB_URL", "REDIS_URL", "MODEL_REGISTRY_URL", "KAFKA_BROKERS",
              "PORT", "SCORE_THRESHOLD_HIGH"]:
        monkeypatch.delenv(k, raising=False)
    s = get_settings()
    assert s.port == 8080
    assert s.db_url is None
    assert s.score_threshold_high == 0.75


def test_require_runtime_settings_raises(monkeypatch) -> None:
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("MODEL_REGISTRY_URL", raising=False)
    try:
        require_runtime_settings()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_settings_as_dict() -> None:
    s = Settings()
    d = s.as_dict()
    assert "port" in d and "db_url" in d and "kafka_brokers" in d


def test_require_runtime_settings_with_env(monkeypatch) -> None:
    monkeypatch.setenv("DB_URL", "postgresql://x")
    monkeypatch.setenv("REDIS_URL", "redis://x")
    monkeypatch.setenv("MODEL_REGISTRY_URL", "http://mlflow")
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")
    s = require_runtime_settings()
    assert s.db_url == "postgresql://x"
    assert s.redis_url == "redis://x"
