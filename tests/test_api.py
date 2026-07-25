from httpx import ASGITransport, AsyncClient

from fraud_engine.app import app
from fraud_engine.audit import AuditEmitter
from fraud_engine.config import Settings
from fraud_engine.feature_store import InMemoryFeatureStore
from fraud_engine.models.schemas import (
    BehavioralFeatures,
    DeviceInfo,
    Money,
    ScoreRequest,
)
from fraud_engine.registry import ModelRegistry
from fraud_engine.scoring import StubModel


class FakeDB:
    def __init__(self) -> None:
        self.scores: list[dict] = []
        self.chargebacks: dict[tuple[str, str], dict] = {}

    def insert_score(self, **kw) -> None:
        self.scores.append(kw)

    def upsert_chargeback(self, tx_id, outcome, reason_code, source, reported_at) -> bool:
        key = (tx_id, reported_at)
        if key in self.chargebacks:
            return False
        self.chargebacks[key] = {"tx_id": tx_id, "outcome": outcome,
                                 "reason_code": reason_code, "source": source,
                                 "reported_at": reported_at}
        return True

    def insert_feature_values(self, *a, **kw) -> None: pass

    def insert_model_version(self, *a, **kw) -> None: pass

    def list_model_versions(self) -> list: return []

    def fetch_labeled_since(self, watermark) -> list: return []

    def insert_drift_metric(self, *a, **kw) -> None: pass


def _payload(tx_id: str = "tx_1") -> dict:
    return {
        "user_id": "usr_1",
        "payment_id": "pay_1",
        "tx_id": tx_id,
        "amount": {"currency": "USD", "minor_units": 25000},
        "device": {"fingerprint": "fp_1", "type": "mobile_ios",
                   "rooted": False, "emulator": False},
        "ip": "203.0.113.42",
        "behavioral_features": {"session_duration_ms": 18432,
                                "keystroke_entropy": 0.71, "tap_variance": 0.33},
    }


async def test_score_endpoint_returns_200() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/fraud/score", json=_payload("tx_a"))
    assert r.status_code == 200
    body = r.json()
    assert "score" in body and "risk_band" in body and "model_version" in body
    assert body["risk_band"] in {"HIGH", "MEDIUM", "LOW"}
    assert "top_features" in body and "scored_at" in body
    assert "variant" in body


async def test_score_endpoint_invalid_body_422() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/fraud/score", json={"user_id": "x"})
    assert r.status_code == 422


async def test_models_endpoint() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/v1/fraud/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    names = {m["name"] for m in body["models"]}
    assert "chargeback-xgb" in names


async def test_feedback_endpoint_valid_returns_204(monkeypatch) -> None:
    fake = FakeDB()
    monkeypatch.setattr("fraud_engine.app._DEFAULT_DB", fake)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/fraud/feedback", json={
            "tx_id": "tx_1", "outcome": "CHARGEBACK", "reason_code": "10.4",
            "reported_at": "2026-07-12T09:00:00Z",
        })
    assert r.status_code == 204
    assert ("tx_1", "2026-07-12T09:00:00Z") in fake.chargebacks


async def test_feedback_endpoint_invalid_outcome_422() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/fraud/feedback", json={
            "tx_id": "tx_1", "outcome": "nope",
            "reported_at": "2026-07-12T09:00:00Z",
        })
    assert r.status_code == 422


async def test_feedback_endpoint_idempotent(monkeypatch) -> None:
    fake = FakeDB()
    monkeypatch.setattr("fraud_engine.app._DEFAULT_DB", fake)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.post("/v1/fraud/feedback", json={
            "tx_id": "tx_2", "outcome": "FRAUD",
            "reported_at": "2026-07-12T10:00:00Z",
        })
        r2 = await client.post("/v1/fraud/feedback", json={
            "tx_id": "tx_2", "outcome": "FRAUD",
            "reported_at": "2026-07-12T10:00:00Z",
        })
    assert r1.status_code == 204 and r2.status_code == 204
    assert len(fake.chargebacks) == 1


def test_audit_emitter_emits_exactly_once() -> None:
    audit = AuditEmitter(db=None)
    req = ScoreRequest(
        user_id="u", payment_id="p", tx_id="t1",
        amount=Money(currency="USD", minor_units=1),
        device=DeviceInfo(fingerprint="fp"),
        ip="1.2.3.4",
        behavioral_features=BehavioralFeatures(),
    )
    from fraud_engine.models.schemas import ScoreResponse, TopFeature
    resp = ScoreResponse(score=0.5, risk_band="MEDIUM", model_version="m@v1",
                        variant="champion", top_features=[TopFeature(name="a", shap=0.1)],
                        scored_at="2026-07-13T10:00:00Z")
    audit.emit(req, resp)
    audit.emit(req, resp)
    assert len(audit.emitted) == 1
    assert audit.emitted[0]["schema"] == "fraud.audit/v1"
    assert audit.emitted[0]["tx_id"] == "t1"
    assert audit.emitted[0]["variant"] == "champion"
