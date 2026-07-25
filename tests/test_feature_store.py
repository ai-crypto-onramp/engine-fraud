from fraud_engine.feature_store import (
    FEATURE_GROUP_FIELDS,
    InMemoryFeatureStore,
)


def test_in_memory_store_write_and_read() -> None:
    store = InMemoryFeatureStore()
    store.write("user_velocity", "u1", {"tx_count_24h": 7})
    assert store.get_features("u1", "user_velocity")["tx_count_24h"] == 7
    assert store.get_features("u1", "device") == {}


def test_in_memory_store_get_online_features() -> None:
    store = InMemoryFeatureStore()
    store.write("user_velocity", "u1", {"tx_count_1h": 3, "tx_count_24h": 9})
    resp = store.get_online_features(
        entity_rows=[{"user_id": "u1"}],
        features=["user_velocity:tx_count_1h", "user_velocity:tx_count_24h"],
    )
    assert resp["user_velocity:tx_count_1h"] == [3]
    assert resp["user_velocity:tx_count_24h"] == [9]


def test_in_memory_store_records_writes() -> None:
    store = InMemoryFeatureStore()
    store.write("device", "fp1", {"known_bad": 1})
    assert store.writes == [("device", "fp1", {"known_bad": 1})]


def test_feature_group_fields_documented() -> None:
    for group in ["user_velocity", "payment_history", "device", "geolocation"]:
        assert group in FEATURE_GROUP_FIELDS
        assert FEATURE_GROUP_FIELDS[group]
