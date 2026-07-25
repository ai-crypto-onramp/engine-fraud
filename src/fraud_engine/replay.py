from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fraud_engine.scoring import default_model_for_name


def load_audit(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_feature_snapshot(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reconstruct(
    audit: Mapping[str, Any], feature_snapshot: Mapping[str, Any],
    model: Any | None = None,
) -> dict[str, Any]:
    if model is None:
        model = default_model_for_name(audit["model_version"].split("@")[0])
    vec = [float(feature_snapshot.get(n, 0) or 0) for n in model.feature_names]
    scores = model.predict_proba([vec])
    return {
        "tx_id": audit["tx_id"],
        "reconstructed_score": float(scores[0]),
        "original_score": audit["score"],
        "risk_band": audit["risk_band"],
        "model_version": audit["model_version"],
        "top_features": audit.get("top_features", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.replay")
    parser.add_argument("--audit", required=True, help="audit jsonl path")
    parser.add_argument("--snapshot", help="feature snapshot json path (single tx)")
    parser.add_argument("--tx-id", help="filter audit by tx_id")
    args = parser.parse_args(argv)

    audit_records = load_audit(args.audit)
    if args.tx_id is not None:
        audit_records = [r for r in audit_records if r.get("tx_id") == args.tx_id]
    if not audit_records:
        print("no audit records found", file=sys.stderr)
        return 1

    if args.snapshot:
        snapshot = load_feature_snapshot(args.snapshot)
        rec = reconstruct(audit_records[0], snapshot)
        print(json.dumps(rec, indent=2))
        return 0

    out: list[dict[str, Any]] = []
    for record in audit_records:
        snapshot_path = record.get("feature_snapshot_uri", "")
        if snapshot_path.startswith("feature_values:"):
            snapshot_path = f"data/snapshots/{snapshot_path.split(':',1)[1]}.json"
        if snapshot_path and Path(snapshot_path).exists():
            snap = load_feature_snapshot(snapshot_path)
        else:
            snap = {}
        out.append(reconstruct(record, snap))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
