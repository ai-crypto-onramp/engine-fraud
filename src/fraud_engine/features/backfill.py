from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fraud_engine.config import get_settings
from fraud_engine.feature_store import InMemoryFeatureStore
from fraud_engine.features.engineering import (
    StubIPLookupProvider,
    device_features,
    geolocation_features,
    load_known_bad_devices,
    payment_history_features,
    user_velocity_features,
)


def parse_events(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def group_events_by_user(events: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_user[str(e.get("user_id", ""))].append(dict(e))
    return by_user


def build_device_history(events: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = defaultdict(list)
    for e in events:
        uid = str(e.get("user_id", ""))
        dev = str(e.get("device_id") or e.get("device", {}).get("fingerprint", ""))
        if dev:
            seen[uid].append(dev)
    return seen


def backfill(
    events: Sequence[Mapping[str, Any]],
    *,
    since: datetime,
    known_bad: Iterable[str],
    ip_lookup: StubIPLookupProvider | None = None,
    store: InMemoryFeatureStore | None = None,
    offline_dir: str = "data/offline",
) -> InMemoryFeatureStore:
    if store is None:
        store = InMemoryFeatureStore()
    if ip_lookup is None:
        ip_lookup = StubIPLookupProvider()
    known_bad_set = set(known_bad)
    by_user = group_events_by_user(events)
    device_history = build_device_history(events)
    Path(offline_dir).mkdir(parents=True, exist_ok=True)
    offline_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for uid, user_events in by_user.items():
        relevant = [e for e in user_events if _parse_ts(e["ts"]) >= since]
        if not relevant:
            continue
        now = max(_parse_ts(e["ts"]) for e in relevant)
        uv = user_velocity_features(relevant, now)
        store.write("user_velocity", uid, uv)
        offline_records["user_velocity"].append({"user_id": uid, **uv})

        ph = payment_history_features(relevant, now)
        store.write("payment_history", uid, ph)
        offline_records["payment_history"].append({"user_id": uid, **ph})

        last = relevant[-1]
        dev_id = str(last.get("device_id") or last.get("device", {}).get("fingerprint", ""))
        if dev_id:
            seen_for_user = device_history.get(uid, [])
            dev_obj = last.get("device", {}) or {}
            df = device_features(
                fingerprint=dev_id,
                seen_devices_for_user=seen_for_user[:-1] if len(seen_for_user) > 1 else [],
                emulator=bool(dev_obj.get("emulator", False)),
                rooted=bool(dev_obj.get("rooted", False)),
                known_bad=known_bad_set,
            )
            store.write("device", dev_id, df)
            offline_records["device"].append({"device_id": dev_id, **df})

        billing = last.get("billing", {}) or {}
        cur = last.get("geo", {}) or {}
        if billing and cur:
            prev_geo = None
            if len(relevant) > 1:
                prev = relevant[-2]
                prev_geo_obj = prev.get("geo", {}) or {}
                if prev_geo_obj:
                    prev_geo = {
                        "lat": prev_geo_obj.get("lat", 0.0), "lon": prev_geo_obj.get("lon", 0.0),
                        "ts": prev.get("ts", cur.get("ts")),
                    }
            geo = geolocation_features(
                ip=str(last.get("ip", "")),
                billing=billing,
                current={**cur, "ts": last.get("ts", cur.get("ts"))},
                prev_event=prev_geo,
                ip_lookup=ip_lookup,
            )
            store.write("geolocation", uid, geo)
            offline_records["geolocation"].append({"user_id": uid, **geo})

    for group, rows in offline_records.items():
        path = Path(offline_dir) / f"{group}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return store


def _parse_ts(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.features.backfill")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD")
    parser.add_argument("--events", default="data/events.jsonl")
    parser.add_argument("--offline-dir", default="data/offline")
    args = parser.parse_args(argv)

    settings = get_settings()
    known_bad = load_known_bad_devices(settings.known_bad_devices_path)
    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
    if not os.path.exists(args.events):
        print(f"events file not found: {args.events}", file=sys.stderr)
        return 1
    events = parse_events(args.events)
    store = InMemoryFeatureStore(settings)
    backfill(
        events, since=since, known_bad=known_bad,
        store=store, offline_dir=args.offline_dir,
    )
    print(f"backfilled {len(events)} events since {args.since}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
