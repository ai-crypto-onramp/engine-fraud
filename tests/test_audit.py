from fraud_engine.audit import AuditEmitter, build_audit_payload
from fraud_engine.models.schemas import (
    BehavioralFeatures,
    DeviceInfo,
    Money,
    ScoreRequest,
    ScoreResponse,
    TopFeature,
)


def _req() -> ScoreRequest:
    return ScoreRequest(
        user_id="u", payment_id="p", tx_id="t",
        amount=Money(currency="USD", minor_units=1),
        device=DeviceInfo(fingerprint="fp"),
        ip="1.2.3.4",
        behavioral_features=BehavioralFeatures(),
    )


def _resp() -> ScoreResponse:
    return ScoreResponse(score=0.9, risk_band="HIGH", model_version="m@v1",
                         variant="challenger",
                         top_features=[TopFeature(name="known_bad", shap=0.3)],
                         scored_at="2026-07-13T10:00:00Z")


def test_audit_payload_has_required_fields() -> None:
    p = build_audit_payload(_req(), _resp(), feature_snapshot_uri="uri://x")
    for f in ["tx_id", "user_id", "score", "risk_band", "model_version",
              "variant", "top_features", "feature_snapshot_uri", "scored_at"]:
        assert f in p
    assert p["schema"] == "fraud.audit/v1"
    assert p["variant"] == "challenger"
    assert p["top_features"][0]["name"] == "known_bad"


def test_audit_emitter_persists_to_db() -> None:
    class FakeDB:
        def __init__(self) -> None:
            self.scores = []

        def insert_score(self, **kw) -> None:
            self.scores.append(kw)

    db = FakeDB()
    audit = AuditEmitter(db=db)
    audit.emit(_req(), _resp(), feature_snapshot_uri="uri")
    assert db.scores and db.scores[0]["tx_id"] == "t"
    assert db.scores[0]["variant"] == "challenger"


def test_audit_emitter_reset() -> None:
    audit = AuditEmitter()
    audit.emit(_req(), _resp())
    assert audit.emitted
    audit.reset()
    assert not audit.emitted
