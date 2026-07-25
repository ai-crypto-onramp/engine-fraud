import os

import pytest

from fraud_engine.scoring import (
    ModelLoader,
    RealModel,
    StubModel,
    _coerce_artifact,
    load_model_artifact,
)


def _train_dummy(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression

    est = LogisticRegression()
    X = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y = [0, 0, 1, 1]
    est.fit(X, y)
    artifact = {"predictor": est, "feature_names": ["a", "b"]}
    path = tmp_path / "m.joblib"
    joblib.dump(artifact, path)
    return path, est


def test_load_model_artifact_from_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _train_dummy(tmp_path)
    art = load_model_artifact("chargeback-xgb", "v1", model_path=str(path))
    assert "predictor" in art or hasattr(art, "predict_proba")


def test_load_model_artifact_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_artifact("chargeback-xgb", "v9", registry_url=None, model_path=None)


def test_real_model_predict_proba(tmp_path) -> None:
    _, est = _train_dummy(tmp_path)
    m = RealModel({"predictor": est, "feature_names": ["a", "b"]})
    scores = m.predict_proba([[0.5, 0.5], [2.5, 2.5]])
    assert len(scores) == 2
    for s in scores:
        assert 0.0 < s < 1.0


def test_real_model_predict_proba_1d_input(tmp_path) -> None:
    _, est = _train_dummy(tmp_path)
    m = RealModel({"predictor": est, "feature_names": ["a", "b"]})
    scores = m.predict_proba([0.5, 0.5])
    assert len(scores) == 1


def test_real_model_shap_contributions(tmp_path) -> None:
    _, est = _train_dummy(tmp_path)
    m = RealModel({"predictor": est, "feature_names": ["a", "b"]})
    contribs = m.shap_contributions([[0.5, 0.5], [2.5, 2.5]])
    assert len(contribs) == 2
    assert set(contribs[0].keys()) == {"a", "b"}


def test_real_model_shap_contributions_1d(tmp_path) -> None:
    _, est = _train_dummy(tmp_path)
    m = RealModel({"predictor": est, "feature_names": ["a", "b"]})
    contribs = m.shap_contributions([0.5, 0.5])
    assert len(contribs) == 1


def test_real_model_single_column_predict_proba() -> None:
    class SingleCol:
        def predict_proba(self, X):
            import numpy as np
            return np.asarray([[0.2], [0.8]])

    m = RealModel({"predictor": SingleCol(), "feature_names": ["a"]})
    scores = m.predict_proba([[1.0], [2.0]])
    assert scores == [0.2, 0.8]


def test_real_model_1d_raw_predict_proba() -> None:
    class Flat:
        def predict_proba(self, X):
            import numpy as np
            return np.asarray([0.3, 0.7])

    m = RealModel({"predictor": Flat(), "feature_names": ["a"]})
    scores = m.predict_proba([[1.0], [2.0]])
    assert scores == [0.3, 0.7]


def test_score_request_with_real_model(tmp_path) -> None:
    from fraud_engine.models.schemas import BehavioralFeatures, DeviceInfo, Money, ScoreRequest
    from fraud_engine.scoring import score_request

    _, est = _train_dummy(tmp_path)
    m = RealModel({"predictor": est, "feature_names": ["a", "b"]})
    req = ScoreRequest(
        user_id="u1", payment_id="p1", tx_id="t1",
        amount=Money(currency="USD", minor_units=100),
        device=DeviceInfo(fingerprint="fp1"),
        ip="1.2.3.4",
        behavioral_features=BehavioralFeatures(),
    )
    resp = score_request(
        req, features={"a": 1.0, "b": 2.0}, model=m,
        model_version="chargeback-xgb@v1",
        threshold_high=0.75, threshold_medium=0.40,
        challenger_fraction=0.0,
    )
    assert 0.0 <= resp.score <= 1.0
    assert resp.risk_band in {"LOW", "MEDIUM", "HIGH"}


def test_real_model_wraps_bare_estimator(tmp_path) -> None:
    _, est = _train_dummy(tmp_path)
    m = RealModel(est)
    assert len(m.feature_names) >= 2
    scores = m.predict_proba([[0.5, 0.5]])
    assert len(scores) == 1


def test_coerce_artifact_dict() -> None:
    predictor, names = _coerce_artifact({"predictor": "p", "feature_names": ["x"]})
    assert predictor == "p"
    assert names == ["x"]


def test_coerce_artifact_bare_falls_back_to_stub_names() -> None:
    class Bare:
        pass

    predictor, names = _coerce_artifact(Bare())
    assert isinstance(predictor, Bare)
    assert names == StubModel.feature_names


def test_model_loader_uses_model_path_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _train_dummy(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    loader = ModelLoader(registry_url=None)
    model = loader.get("chargeback-xgb", "v9.9.9")
    assert isinstance(model, RealModel)
    assert model.feature_names == ["a", "b"]


def test_model_loader_per_name_version_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _train_dummy(tmp_path)
    monkeypatch.setenv("MODEL_PATH_CHARGEBACK_XGB_V1_0_0", str(path))
    loader = ModelLoader(registry_url=None)
    model = loader.get("chargeback-xgb", "v1.0.0")
    assert isinstance(model, RealModel)


def test_model_loader_falls_back_to_stub_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_PATH_CHARGEBACK_XGB_V9_9_9", raising=False)
    loader = ModelLoader(registry_url=None)
    model = loader.get("chargeback-xgb", "v9.9.9")
    assert isinstance(model, StubModel)


def test_model_loader_cache_hits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _train_dummy(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    loader = ModelLoader(registry_url=None)
    m1 = loader.get("chargeback-xgb", "v1")
    m2 = loader.get("chargeback-xgb", "v1")
    assert m1 is m2


def test_model_loader_invalidate(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _train_dummy(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    loader = ModelLoader(registry_url=None)
    m1 = loader.get("chargeback-xgb", "v1")
    loader.invalidate("chargeback-xgb")
    m2 = loader.get("chargeback-xgb", "v1")
    assert m1 is not m2


def test_model_loader_load_artifact_failure_falls_back_to_stub(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a pickle")
    monkeypatch.setenv("MODEL_PATH", str(bad))
    loader = ModelLoader(registry_url=None)
    model = loader.get("chargeback-xgb", "v1")
    assert isinstance(model, StubModel)


def test_load_model_artifact_from_registry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import http.server
    import socketserver
    import threading

    path, _ = _train_dummy(tmp_path)
    payload = path.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as srv:
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            art = load_model_artifact(
                "chargeback-xgb", "v1", registry_url=f"http://127.0.0.1:{port}",
            )
            assert "predictor" in art or hasattr(art, "predict_proba")
        finally:
            srv.shutdown()
