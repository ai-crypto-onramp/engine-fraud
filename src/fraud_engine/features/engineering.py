from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any


def _parse_ts(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geo_velocity_kmh(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> float:
    distance = haversine_km(prev["lat"], prev["lon"], cur["lat"], cur["lon"])
    dt_h = (_parse_ts(cur["ts"]) - _parse_ts(prev["ts"])).total_seconds() / 3600.0
    if dt_h <= 0:
        return 0.0
    return distance / dt_h


def user_velocity_features(
    events: Sequence[Mapping[str, Any]], now: datetime
) -> dict[str, Any]:
    counts = {"1h": 0, "24h": 0, "7d": 0}
    sums = {"24h": 0.0}
    cards_24h: set[str] = set()
    devices_24h: set[str] = set()
    ips_24h: set[str] = set()
    for e in events:
        ts = _parse_ts(e["ts"])
        amount = float(e.get("amount_minor_units", 0) or 0)
        age = now - ts
        if age <= timedelta(hours=1):
            counts["1h"] += 1
        if age <= timedelta(hours=24):
            counts["24h"] += 1
            sums["24h"] += amount
            if e.get("card_id"):
                cards_24h.add(str(e["card_id"]))
            if e.get("device_id"):
                devices_24h.add(str(e["device_id"]))
            if e.get("ip"):
                ips_24h.add(str(e["ip"]))
        if age <= timedelta(days=7):
            counts["7d"] += 1
    return {
        "tx_count_1h": counts["1h"],
        "tx_count_24h": counts["24h"],
        "tx_count_7d": counts["7d"],
        "tx_sum_24h": sums["24h"],
        "distinct_cards_24h": len(cards_24h),
        "distinct_devices_24h": len(devices_24h),
        "distinct_ips_24h": len(ips_24h),
    }


def payment_history_features(events: Sequence[Mapping[str, Any]], now: datetime) -> dict[str, Any]:
    success = 0
    fail = 0
    total_amount = 0.0
    first_ts: datetime | None = None
    for e in events:
        ts = _parse_ts(e["ts"])
        if now - ts > timedelta(days=30):
            continue
        status = str(e.get("status", "")).lower()
        if status == "success" or status == "authorized":
            success += 1
            total_amount += float(e.get("amount_minor_units", 0) or 0)
        elif status in {"failed", "denied", "error"}:
            fail += 1
        if first_ts is None or ts < first_ts:
            first_ts = ts
    ratio = (success / fail) if fail > 0 else float(success)
    avg_ticket = (total_amount / success) if success > 0 else 0.0
    first_age = ((now - first_ts).days) if first_ts else -1
    return {
        "success_fail_ratio_30d": float(ratio),
        "avg_ticket_30d": float(avg_ticket),
        "first_payment_age_days": int(first_age),
    }


def device_features(
    fingerprint: str,
    seen_devices_for_user: Iterable[str],
    emulator: bool,
    rooted: bool,
    known_bad: Iterable[str],
) -> dict[str, Any]:
    known_bad_set = {str(x) for x in known_bad}
    return {
        "fingerprint_hash": fingerprint,
        "new_to_user": int(fingerprint not in {str(x) for x in seen_devices_for_user}),
        "known_bad": int(fingerprint in known_bad_set),
        "emulator": int(bool(emulator)),
        "rooted": int(bool(rooted)),
    }


def geolocation_features(
    ip: str,
    billing: Mapping[str, Any],
    current: Mapping[str, Any],
    prev_event: Mapping[str, Any] | None,
    ip_lookup: IPLookupProvider | None = None,
) -> dict[str, Any]:
    country = ""
    vpn_proxy = 0
    if ip_lookup is not None:
        info = ip_lookup.lookup(ip)
        country = info.get("country", "")
        vpn_proxy = int(bool(info.get("vpn_proxy", False)))
    distance = haversine_km(billing["lat"], billing["lon"], current["lat"], current["lon"])
    geo_v = 0.0
    if prev_event is not None:
        geo_v = geo_velocity_kmh(prev_event, {**current, "ts": current.get("ts", prev_event["ts"])})
    return {
        "ip_country": country,
        "distance_from_billing_km": float(distance),
        "vpn_proxy": vpn_proxy,
        "geo_velocity_kmh": float(geo_v),
    }


class IPLookupProvider:
    def lookup(self, ip: str) -> dict[str, Any]:
        raise NotImplementedError


class StubIPLookupProvider(IPLookupProvider):
    def __init__(self, mapping: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.mapping = dict(mapping or {})

    def lookup(self, ip: str) -> dict[str, Any]:
        return dict(self.mapping.get(ip, {"country": "US", "vpn_proxy": False}))


def load_known_bad_devices(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    out: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out
