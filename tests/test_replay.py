import json
from pathlib import Path

from fraud_engine.replay import load_audit, main, reconstruct
from fraud_engine.scoring import StubModel


def test_reconstruct_matches_audit(tmp_path: Path) -> None:
    audit = {
        "schema": "fraud.audit/v1", "tx_id": "t1", "user_id": "u1",
        "score": 0.5, "risk_band": "HIGH", "model_version": "chargeback-xgb@v3.2.0-stub",
        "variant": "champion", "top_features": [], "scored_at": "2026-07-13T10:00:00Z",
    }
    snapshot = {n: 0.0 for n in StubModel.feature_names}
    snapshot["known_bad"] = 1.0
    rec = reconstruct(audit, snapshot)
    assert rec["tx_id"] == "t1"
    assert 0.0 < rec["reconstructed_score"] < 1.0
    assert rec["risk_band"] == "HIGH"


def test_replay_cli(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    audit = {
        "schema": "fraud.audit/v1", "tx_id": "t1", "user_id": "u1",
        "score": 0.5, "risk_band": "HIGH", "model_version": "chargeback-xgb@v3.2.0-stub",
        "variant": "champion", "top_features": [], "scored_at": "2026-07-13T10:00:00Z",
        "feature_snapshot_uri": "",
    }
    audit_path.write_text(json.dumps(audit) + "\n")
    rc = main(["--audit", str(audit_path), "--tx-id", "t1"])
    assert rc == 0


def test_load_audit(tmp_path: Path) -> None:
    from fraud_engine.replay import load_audit
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"tx_id": "x"}) + "\n" + json.dumps({"tx_id": "y"}) + "\n")
    records = load_audit(str(p))
    assert len(records) == 2
    assert records[0]["tx_id"] == "x"


def test_replay_cli_no_records(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    rc = main(["--audit", str(p)])
    assert rc == 1


def test_replay_with_snapshot_arg(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps({"tx_count_24h": 5, "known_bad": 1}))
    audit = {
        "schema": "fraud.audit/v1", "tx_id": "t1", "user_id": "u1",
        "score": 0.5, "risk_band": "HIGH", "model_version": "chargeback-xgb@v3.2.0-stub",
        "variant": "champion", "top_features": [], "scored_at": "2026-07-13T10:00:00Z",
    }
    audit_path.write_text(json.dumps(audit) + "\n")
    rc = main(["--audit", str(audit_path), "--snapshot", str(snap_path)])
    assert rc == 0
