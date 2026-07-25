from datetime import UTC, datetime, timedelta, timezone

from fraud_engine.features.engineering import (
    StubIPLookupProvider,
    device_features,
    geo_velocity_kmh,
    geolocation_features,
    haversine_km,
    load_known_bad_devices,
    payment_history_features,
    user_velocity_features,
)


def _ts(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def test_haversine() -> None:
    d = haversine_km(0, 0, 0, 1)
    assert 100 < d < 120


def test_geo_velocity() -> None:
    v = geo_velocity_kmh(
        {"lat": 0.0, "lon": 0.0, "ts": "2026-07-13T10:00:00Z"},
        {"lat": 0.0, "lon": 1.0, "ts": "2026-07-13T11:00:00Z"},
    )
    assert v > 0


def test_user_velocity_counts_and_distinct() -> None:
    now = datetime.now(UTC)
    events = [
        {"ts": now - timedelta(hours=1), "amount_minor_units": 100, "card_id": "c1", "device_id": "d1", "ip": "1.1.1.1"},
        {"ts": now - timedelta(hours=2), "amount_minor_units": 200, "card_id": "c2", "device_id": "d2", "ip": "2.2.2.2"},
        {"ts": now - timedelta(days=3), "amount_minor_units": 300, "card_id": "c3", "device_id": "d3", "ip": "3.3.3.3"},
    ]
    feats = user_velocity_features(events, now)
    assert feats["tx_count_1h"] == 1
    assert feats["tx_count_24h"] == 2
    assert feats["tx_count_7d"] == 3
    assert feats["tx_sum_24h"] == 300
    assert feats["distinct_cards_24h"] == 2
    assert feats["distinct_devices_24h"] == 2
    assert feats["distinct_ips_24h"] == 2


def test_payment_history_features() -> None:
    now = datetime.now(UTC)
    events = [
        {"ts": now - timedelta(days=1), "status": "success", "amount_minor_units": 100},
        {"ts": now - timedelta(days=2), "status": "failed", "amount_minor_units": 0},
        {"ts": now - timedelta(days=5), "status": "authorized", "amount_minor_units": 200},
    ]
    feats = payment_history_features(events, now)
    assert feats["success_fail_ratio_30d"] == 2.0
    assert feats["avg_ticket_30d"] == 150.0
    assert feats["first_payment_age_days"] >= 5


def test_device_features_new_to_user_and_known_bad(tmp_path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("fp_bad_1\n# comment\nfp_bad_2\n")
    known = load_known_bad_devices(str(bad))
    assert known == {"fp_bad_1", "fp_bad_2"}
    df = device_features(
        fingerprint="fp_bad_1", seen_devices_for_user=["fp_old"],
        emulator=True, rooted=True, known_bad=known,
    )
    assert df["new_to_user"] == 1
    assert df["known_bad"] == 1
    assert df["emulator"] == 1
    assert df["rooted"] == 1
    df2 = device_features(
        fingerprint="fp_old", seen_devices_for_user=["fp_old"],
        emulator=False, rooted=False, known_bad=known,
    )
    assert df2["new_to_user"] == 0
    assert df2["known_bad"] == 0


def test_geolocation_features_with_provider() -> None:
    provider = StubIPLookupProvider({"1.2.3.4": {"country": "RU", "vpn_proxy": True}})
    g = geolocation_features(
        ip="1.2.3.4",
        billing={"lat": 0.0, "lon": 0.0},
        current={"lat": 1.0, "lon": 1.0, "ts": "2026-07-13T10:00:00Z"},
        prev_event={"lat": 0.0, "lon": 0.0, "ts": "2026-07-13T09:00:00Z"},
        ip_lookup=provider,
    )
    assert g["ip_country"] == "RU"
    assert g["vpn_proxy"] == 1
    assert g["distance_from_billing_km"] > 0
    assert g["geo_velocity_kmh"] >= 0


def test_geolocation_missing_prev_event() -> None:
    g = geolocation_features(
        ip="9.9.9.9",
        billing={"lat": 0.0, "lon": 0.0},
        current={"lat": 2.0, "lon": 2.0},
        prev_event=None,
    )
    assert g["geo_velocity_kmh"] == 0.0
    assert g["ip_country"] == ""
