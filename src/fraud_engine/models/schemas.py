from __future__ import annotations

from pydantic import BaseModel, Field


class Money(BaseModel):
    currency: str
    minor_units: int


class DeviceInfo(BaseModel):
    fingerprint: str
    type: str | None = None
    rooted: bool = False
    emulator: bool = False


class BehavioralFeatures(BaseModel):
    session_duration_ms: int = 0
    keystroke_entropy: float = 0.0
    tap_variance: float = 0.0


class ScoreRequest(BaseModel):
    user_id: str
    payment_id: str
    tx_id: str | None = None
    amount: Money
    device: DeviceInfo
    ip: str
    behavioral_features: BehavioralFeatures = Field(default_factory=BehavioralFeatures)


class TopFeature(BaseModel):
    name: str
    shap: float


class ScoreResponse(BaseModel):
    score: float
    risk_band: str
    model_version: str
    variant: str
    top_features: list[TopFeature]
    scored_at: str


class FeedbackRequest(BaseModel):
    tx_id: str
    outcome: str
    reason_code: str | None = None
    source: str = "API"
    reported_at: str


class ModelInfo(BaseModel):
    name: str
    champion: str | None = None
    challenger: str | None = None
    traffic_split: dict[str, float]
    updated_at: str
    drift_breaches: list[str] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthStatus(BaseModel):
    status: str
    healthy: str
    failed: str
    total: str
