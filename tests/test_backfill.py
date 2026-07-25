import json
from datetime import UTC, datetime, timezone
from pathlib import Path

from fraud_engine.feature_store import InMemoryFeatureStore
from fraud_engine.features.backfill import backfill, parse_events
from fraud_engine.features.engineering import StubIPLookupProvider


def _events_file(tmp_path: Path) -> str:
    events = [
        {
            "user_id": "u1", "tx_id": "t1", "ts": "2026-07-10T10:00:00Z",
            "amount_minor_units": 100, "card_id": "c1",
            "device_id": "fp_1", "ip": "1.1.1.1", "status": "success",
            "device": {"fingerprint": "fp_1", "emulator": False, "rooted": False},
            "billing": {"lat": 0.0, "lon": 0.0},
            "geo": {"lat": 1.0, "lon": 1.0},
        },
        {
            "user_id": "u1", "tx_id": "t2", "ts": "2026-07-11T10:00:00Z",
            "amount_minor_units": 200, "card_id": "c2",
            "device_id": "fp_2", "ip": "2.2.2.2", "status": "failed",
            "device": {"fingerprint": "fp_2", "emulator": False, "rooted": False},
            "billing": {"lat": 0.0, "lon": 0.0},
            "geo": {"lat": 2.0, "lon": 2.0},
        },
    ]
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events))
    return str(p)


def test_backfill_populates_feature_store(tmp_path: Path) -> None:
    events_path = _events_file(tmp_path)
    events = parse_events(events_path)
    store = InMemoryFeatureStore()
    offline_dir = tmp_path / "offline"
    backfill(
        events, since=datetime(2026, 1, 1, tzinfo=UTC),
        known_bad={"fp_bad_1"}, ip_lookup=StubIPLookupProvider({"1.1.1.1": {"country": "US"}}),
        store=store, offline_dir=str(offline_dir),
    )
    uv = store.get_features("u1", "user_velocity")
    assert uv["tx_count_24h"] >= 1
    df = store.get_features("fp_2", "device")
    assert df["new_to_user"] == 1
    geo = store.get_features("u1", "geolocation")
    assert "distance_from_billing_km" in geo
    assert (offline_dir / "user_velocity.jsonl").exists()


def test_backfill_cli_main(tmp_path: Path, monkeypatch) -> None:
    from fraud_engine.features import backfill as bf_module
    events_path = _events_file(tmp_path)
    offline_dir = tmp_path / "offline"
    monkeypatch.setattr(bf_module, "get_settings", lambda: __import__(
        "fraud_engine.config", fromlist=["Settings"]).Settings())
    rc = bf_module.main(["--since", "2026-01-01", "--events", events_path,
                         "--offline-dir", str(offline_dir)])
    assert rc == 0
    assert (offline_dir / "user_velocity.jsonl").exists()
