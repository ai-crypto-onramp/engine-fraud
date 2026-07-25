from fraud_engine.models.schemas import BehavioralFeatures, DeviceInfo, Money, ScoreRequest
from fraud_engine.scoring import (
    StubModel,
    StubVelocityModel,
    deterministic_score,
    score_request,
    top_shap_features,
)


def _req(tx_id: str = "tx_1") -> ScoreRequest:
    return ScoreRequest(
        user_id="u1", payment_id="p1", tx_id=tx_id,
        amount=Money(currency="USD", minor_units=100),
        device=DeviceInfo(fingerprint="fp1", type="mobile_ios"),
        ip="1.2.3.4",
        behavioral_features=BehavioralFeatures(session_duration_ms=100, keystroke_entropy=0.5),
    )


def test_stub_model_predict_and_shap() -> None:
    m = StubModel()
    feats = {"tx_count_24h": 5, "known_bad": 1, "new_to_user": 1, "keystroke_entropy": 0.3}
    vec = [float(feats.get(n, 0) or 0) for n in m.feature_names]
    s = m.predict_proba([vec])[0]
    assert 0.0 < s < 1.0
    contribs = m.shap_contributions([vec])[0]
    assert "known_bad" in contribs


def test_score_request_returns_response_with_variant() -> None:
    req = _req("tx_2")
    feats = {"tx_count_24h": 5, "known_bad": 1}
    resp = score_request(
        req, features=feats, model=StubModel(),
        model_version="chargeback-xgb@v3.2.0-stub",
        threshold_high=0.75, threshold_medium=0.40,
        challenger_fraction=0.5, force_variant="champion",
    )
    assert resp.variant == "champion"
    assert resp.model_version.startswith("chargeback-xgb@")
    assert len(resp.top_features) <= 3
    for f in resp.top_features:
        assert isinstance(f.name, str)
        assert isinstance(f.shap, float)


def test_top_shap_features_sorted_by_abs() -> None:
    contribs = {"a": 0.1, "b": -0.5, "c": 0.3, "d": 0.0}
    out = top_shap_features(contribs, k=3)
    assert [f.name for f in out] == ["b", "c", "a"]


def test_risk_band_low_for_clean_request() -> None:
    req = _req("tx_3")
    feats = {"keystroke_entropy": 0.9, "tap_variance": 0.8}
    resp = score_request(
        req, features=feats, model=StubModel(),
        model_version="m@v1",
        threshold_high=0.75, threshold_medium=0.40,
        challenger_fraction=0.0,
    )
    assert resp.variant == "champion"
    assert resp.risk_band in {"LOW", "MEDIUM"}


def test_stub_velocity_model_name() -> None:
    m = StubVelocityModel()
    assert m.name == "velocity-isoforest"


def test_deterministic_score_is_stable() -> None:
    feats = {"a": 1, "b": 2}
    assert deterministic_score(feats) == deterministic_score(feats)
    assert 0.0 <= deterministic_score(feats) <= 1.0


def test_score_request_with_traffic_split_mapping() -> None:
    req = _req("tx_split")
    feats = {"tx_count_24h": 3}
    resp = score_request(
        req, features=feats, model=StubModel(),
        model_version="m@v1",
        threshold_high=0.75, threshold_medium=0.40,
        challenger_fraction=0.5,
        traffic_split={"champion": 0.8, "challenger": 0.2},
        force_variant="challenger",
    )
    assert resp.variant == "challenger"
