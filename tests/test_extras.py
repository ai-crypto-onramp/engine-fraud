from fraud_engine.observability.drift import ks_statistic, psi
from fraud_engine.scoring import StubModel, feature_vector, now_utc_iso


def test_feature_vector_pads_missing() -> None:
    vec = feature_vector({"tx_count_24h": 3}, StubModel.feature_names)
    assert "tx_count_24h" in StubModel.feature_names
    idx = StubModel.feature_names.index("tx_count_24h")
    assert vec[idx] == 3.0
    assert len(vec) == len(StubModel.feature_names)


def test_now_utc_iso_has_z_suffix() -> None:
    s = now_utc_iso()
    assert s.endswith("Z")


def test_psi_empty_inputs_return_zero() -> None:
    assert psi([], []) == 0.0


def test_ks_empty_inputs_return_zero() -> None:
    assert ks_statistic([], []) == 0.0


def test_psi_constant_distribution_zero() -> None:
    assert psi([0.5] * 20, [0.5] * 20) == 0.0


def test_ks_statistic_identical_zero() -> None:
    assert ks_statistic([0.1, 0.2, 0.3] * 10, [0.1, 0.2, 0.3] * 10) == 0.0


def test_psi_zero_range_returns_zero() -> None:
    assert psi([1.0] * 10, [1.0] * 10) == 0.0


def test_drift_report_skips_missing_baseline() -> None:
    from fraud_engine.observability.drift import drift_report
    report = drift_report({}, {"f1": [0.1, 0.2, 0.3]}, psi_threshold=0.2)
    assert report == []


def test_breached_features_empty() -> None:
    from fraud_engine.observability.drift import breached_features
    assert breached_features([]) == []
