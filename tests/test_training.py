import json
from pathlib import Path

from fraud_engine.config import Settings
from fraud_engine.registry import ModelRegistry
from fraud_engine.training.scheduler import PromotionGate, RetrainScheduler
from fraud_engine.training.train_chargeback import (
    compute_metrics,
    train_chargeback,
    train_velocity,
)
from fraud_engine.training.train_velocity import main as velocity_main


def test_compute_metrics_handles_empty() -> None:
    m = compute_metrics([], [])
    assert m["auc"] == 0.5
    assert m["pr_auc"] == 0.5


def test_compute_metrics_returns_values() -> None:
    m = compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert 0.0 <= m["auc"] <= 1.0
    assert m["calibration"] >= 0.0


def test_train_chargeback(tmp_path: Path) -> None:
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps({"X": [[0.1] * 14] * 8, "y": [0, 1] * 4}))
    result = train_chargeback(str(dataset), offline_dir=str(tmp_path / "off"))
    assert result["name"] == "chargeback-xgb"
    assert result["stage"] == "challenger"
    assert "auc" in result["metrics"]
    assert Path(result["artifact"]).exists()


def test_promotion_gate_blocks_then_allows() -> None:
    gate = PromotionGate()
    assert not gate.is_approved("m", "v1")
    gate.approve("m", "v1")
    assert gate.is_approved("m", "v1")


def test_retrain_scheduler_promotion_requires_gate() -> None:
    reg = ModelRegistry(Settings())
    reg.register("chargeback-xgb", "v9.9.9", "staging", metrics={},
                 traffic_split={"champion": 0.9, "challenger": 0.1})
    sched = RetrainScheduler(Settings(), registry=reg)
    assert sched.promote("chargeback-xgb", "v9.9.9") is None
    sched.gate.approve("chargeback-xgb", "v9.9.9")
    rec = sched.promote("chargeback-xgb", "v9.9.9", to_stage="challenger")
    assert rec is not None and rec["stage"] == "challenger"


def test_train_velocity_function(tmp_path: Path) -> None:
    result = train_velocity(str(tmp_path / "ds.json"), offline_dir=str(tmp_path / "off"))
    assert result["name"] == "velocity-isoforest"
    assert result["stage"] == "challenger"


def test_train_velocity_cli(tmp_path: Path) -> None:
    dataset = tmp_path / "vds.json"
    dataset.write_text(json.dumps({"X": [[0.1] * 14] * 8, "y": [0, 1] * 4}))
    rc = velocity_main(["--dataset", str(dataset), "--version", "v9.9.9"])
    assert rc == 0


def test_retrain_scheduler_runs_when_due(tmp_path: Path) -> None:
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps({"X": [[0.1] * 14] * 8, "y": [0, 1] * 4}))
    sched = RetrainScheduler(Settings())
    res = sched.run_chargeback(str(dataset))
    assert res is not None and res["name"] == "chargeback-xgb"
    res2 = sched.run_chargeback(str(dataset))
    assert res2 is None


def test_promotion_gate_reset() -> None:
    gate = PromotionGate()
    gate.approve("m", "v1")
    assert gate.is_approved("m", "v1")
    gate.reset()
    assert not gate.is_approved("m", "v1")
