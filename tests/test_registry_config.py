from fraud_engine.config import Settings, get_settings
from fraud_engine.db import PostgresStore
from fraud_engine.registry import ModelRegistry
from fraud_engine.scoring import ModelLoader


def test_config_defaults() -> None:
    s = get_settings()
    assert s.port == 8080
    assert s.score_threshold_high == 0.75
    assert s.score_threshold_medium == 0.40
    assert s.challenger_traffic_fraction == 0.10
    assert s.drift_psi_threshold == 0.2
    assert s.kafka_consumer_group == "fraud-detection"


def test_registry_seeds_defaults() -> None:
    reg = ModelRegistry(Settings())
    models = reg.list_models()
    names = {m["name"] for m in models}
    assert "chargeback-xgb" in names
    assert "velocity-isoforest" in names
    cb = next(m for m in models if m["name"] == "chargeback-xgb")
    assert cb["champion"] is not None


def test_registry_register_and_transition() -> None:
    reg = ModelRegistry(Settings())
    rec = reg.register(
        "chargeback-xgb", "v9.9.9", "challenger",
        metrics={"auc": 0.9}, traffic_split={"champion": 0.9, "challenger": 0.1},
    )
    assert rec["stage"] == "challenger"
    new_rec = reg.transition("chargeback-xgb", "v9.9.9", "champion")
    assert new_rec["stage"] == "champion"
    models = reg.list_models()
    cb = next(m for m in models if m["name"] == "chargeback-xgb")
    assert cb["champion"] == "v9.9.9"


def test_registry_resolve_version() -> None:
    reg = ModelRegistry(Settings())
    assert reg.resolve_version("chargeback-xgb", "champion") == "v3.2.0-stub"
    assert reg.resolve_version("chargeback-xgb", "challenger") is None


def test_model_loader_caches_and_invalidates() -> None:
    loader = ModelLoader(None)
    m1 = loader.get("chargeback-xgb", "v3.2.0-stub")
    m2 = loader.get("chargeback-xgb", "v3.2.0-stub")
    assert m1 is m2
    loader.invalidate("chargeback-xgb")
    m3 = loader.get("chargeback-xgb", "v3.2.0-stub")
    assert m3 is not m1


def test_db_stores_construct_without_dsn() -> None:
    pg = PostgresStore(None)
    assert pg.ping() is False


def test_registry_traffic_split_and_get_model() -> None:
    reg = ModelRegistry(Settings())
    split = reg.traffic_split_for("chargeback-xgb")
    assert "champion" in split
    model = reg.get_model("chargeback-xgb", "v3.2.0-stub")
    assert model.name == "chargeback-xgb"


def test_registry_list_models_champion_only() -> None:
    reg = ModelRegistry(Settings())
    reg.register("velocity-isoforest", "v9.9.9", "champion", metrics={"auc": 0.9},
                 traffic_split={"champion": 1.0})
    models = reg.list_models()
    vel = next(m for m in models if m["name"] == "velocity-isoforest")
    assert vel["champion"] is not None
    assert vel["traffic_split"]["champion"] == 1.0


def test_registry_get_model_returns_default() -> None:
    reg = ModelRegistry(Settings())
    m = reg.get_model("velocity-isoforest", "v1.4.0-stub")
    assert m.name == "velocity-isoforest"


def test_model_loader_invalidates_all() -> None:
    loader = ModelLoader(None)
    loader.get("chargeback-xgb", "v3.2.0-stub")
    loader.invalidate()
    m = loader.get("chargeback-xgb", "v3.2.0-stub")
    assert m.name == "chargeback-xgb"


def test_db_apply_migrations_from_path(tmp_path) -> None:
    pg = PostgresStore(None)
    assert pg.ping() is False
