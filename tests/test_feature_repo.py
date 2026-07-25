import sys
from pathlib import Path


def test_feature_repo_imports() -> None:
    repo = Path(__file__).resolve().parents[1] / "feature_repo"
    sys.path.insert(0, str(repo))
    try:
        import features
        names = [v.name for v in features.feature_views]
        entities = [e.name for e in features.entities]
    finally:
        sys.path.pop(0)
    assert "user_velocity" in names
    assert "payment_history" in names
    assert "device" in names
    assert "geolocation" in names
    assert set(entities) == {"user_id", "device_id", "tx_id"}


def test_feature_store_yaml_exists() -> None:
    yaml = Path(__file__).resolve().parents[1] / "feature_repo" / "feature_store.yaml"
    assert yaml.exists()
    text = yaml.read_text()
    assert "fraud_engine" in text
    assert "redis" in text


def test_migrations_file_exists_and_has_tables() -> None:
    sql = Path(__file__).resolve().parents[1] / "migrations" / "001_init.sql"
    assert sql.exists()
    body = sql.read_text()
    for table in ["fraud_scores", "model_versions", "feature_values",
                  "chargeback_events", "drift_metrics"]:
        assert table in body
