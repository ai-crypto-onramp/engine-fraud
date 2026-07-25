class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, str]] = {}

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.data.setdefault(key, {}).update(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.data.get(key, {}))

    def ping(self) -> bool:
        return True


def test_redis_store_set_and_get(monkeypatch) -> None:
    from fraud_engine import db as db_module
    fake = FakeRedis()
    monkeypatch.setattr(db_module, "_HAVE_REDIS", True)
    monkeypatch.setattr(db_module.redis.Redis, "from_url", classmethod(lambda cls, *a, **kw: fake))  # type: ignore[union-attr]
    store = db_module.RedisStore("redis://localhost:6379")
    assert store.ping() is True
    store.set_features("u1", "user_velocity", {"tx_count_24h": 5})
    out = store.get_features("u1", "user_velocity")
    assert out["tx_count_24h"] == 5
    batch = store.mget_features(["u1"], "user_velocity")
    assert batch["u1"]["tx_count_24h"] == 5
