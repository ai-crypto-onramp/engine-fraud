from fraud_engine.observability.drift import breached_features, drift_report, ks_statistic, psi


def test_psi_identical_distributions_zero() -> None:
    base = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    assert psi(base, base) == 0.0


def test_psi_shifted_distribution_positive() -> None:
    base = [0.1, 0.2, 0.3, 0.4, 0.5] * 50
    cur = [0.5, 0.6, 0.7, 0.8, 0.9] * 50
    assert psi(base, cur) > 0.0


def test_ks_statistic() -> None:
    base = list(range(100))
    cur = list(range(50, 150))
    k = ks_statistic(base, cur)
    assert 0.0 <= k <= 1.0


def test_drift_report_flags_breach() -> None:
    base = {"f1": [0.1, 0.2, 0.3, 0.4, 0.5] * 20}
    cur = {"f1": [0.6, 0.7, 0.8, 0.9, 1.0] * 20}
    report = drift_report(base, cur, psi_threshold=0.2)
    assert report[0]["feature"] == "f1"
    assert "psi" in report[0]
    assert "ks" in report[0]
    assert report[0]["breached"] in (True, False)


def test_breached_features_list() -> None:
    report = [
        {"feature": "a", "psi": 0.5, "ks": 0.1, "breached": True},
        {"feature": "b", "psi": 0.05, "ks": 0.0, "breached": False},
    ]
    assert breached_features(report) == ["a"]


def test_drift_monitor_emits_alert_and_persists() -> None:
    from fraud_engine.config import get_settings
    from fraud_engine.observability.monitoring import AlertSink, DriftMonitor
    sink = AlertSink()
    mon = DriftMonitor(get_settings(), db=None, alert_sink=sink)
    base = {"f1": [0.1, 0.2, 0.3, 0.4, 0.5] * 50}
    cur = {"f1": [0.5, 0.6, 0.7, 0.8, 0.9] * 50}
    report = mon.run("chargeback-xgb", base, cur)
    assert report
    assert sink.alerts
    assert sink.alerts[0]["schema"] == "fraud.alert.raised/v1"
    assert mon.breaches_for("chargeback-xgb")
