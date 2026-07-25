from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlflow
    _HAVE_MLFLOW = True
except Exception:  # pragma: no cover - optional in tests
    mlflow = None  # type: ignore
    _HAVE_MLFLOW = False

from fraud_engine.scoring import StubModel


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def compute_metrics(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    if not y_true or not y_score:
        return {"auc": 0.5, "pr_auc": 0.5, "calibration": 1.0}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        auc, pr_auc = 0.5, 0.5
    calibration = float(abs(float(np.mean(y_score)) - float(np.mean(y_true))))
    return {"auc": round(auc, 4), "pr_auc": round(pr_auc, 4), "calibration": round(calibration, 4)}


def train_chargeback(
    dataset_path: str, *, model_name: str = "chargeback-xgb",
    version: str = "v3.3.0-rc1", stage: str = "challenger",
    offline_dir: str = "data/offline",
) -> dict[str, Any]:
    Path(offline_dir).mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(dataset_path)
    model = StubModel()
    model.name = model_name
    model.version = version
    y_score = model.predict_proba(X)
    metrics = compute_metrics(y, y_score)
    snapshot_uri = f"{offline_dir}/snapshot_{model_name}_{version}.json"
    with open(snapshot_uri, "w", encoding="utf-8") as fh:
        json.dump({"features": model.feature_names, "X": X, "y": y}, fh)
    artifact = Path(offline_dir) / f"{model_name}_{version}.pkl"
    with open(artifact, "wb") as fh:
        pickle.dump(model, fh)
    if _HAVE_MLFLOW:
        try:
            mlflow.set_experiment(model_name)
            with mlflow.start_run(run_name=version):
                mlflow.log_params({"model": model_name, "version": version, "stage": stage})
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(artifact))
                mlflow.set_tag("feature_snapshot_uri", snapshot_uri)
        except Exception:
            pass
    return {
        "name": model_name, "version": version, "stage": stage,
        "metrics": metrics, "feature_snapshot_uri": snapshot_uri,
        "artifact": str(artifact), "trained_at": _now_iso(),
    }


def train_velocity(
    dataset_path: str, *, model_name: str = "velocity-isoforest",
    version: str = "v1.5.0-rc1", stage: str = "challenger",
    offline_dir: str = "data/offline",
) -> dict[str, Any]:
    return train_chargeback(
        dataset_path, model_name=model_name, version=version, stage=stage,
        offline_dir=offline_dir,
    )


def load_dataset(path: str) -> tuple[list[list[float]], list[int]]:
    if not os.path.exists(path):
        rng = np.random.default_rng(42)
        X = rng.random((64, 14)).tolist()
        y = (rng.random(64) > 0.7).astype(int).tolist()
        return X, y
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("X", []), data.get("y", [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.training.train_chargeback")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="data/chargeback_dataset.json")
    parser.add_argument("--version", default="v3.3.0-rc1")
    parser.add_argument("--stage", default="challenger")
    args = parser.parse_args(argv)
    result = train_chargeback(args.dataset, version=args.version, stage=args.stage)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
