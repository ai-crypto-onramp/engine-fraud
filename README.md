# Fraud Detection
![CI](https://github.com/ai-crypto-onramp/fraud-engine/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/ai-crypto-onramp/fraud-engine/branch/main/graph/badge.svg)](https://codecov.io/gh/ai-crypto-onramp/fraud-engine)

ML scoring on payment + behavioral signals (chargeback/velocity models); feeds the policy engine.

## Overview / Responsibilities

The Fraud Detection service is the ML-powered risk scoring layer of the crypto
on-ramp. It consumes payment and behavioral signals in real time, produces a
fraud risk score and risk band per transaction, and returns them to the
**Policy / Risk Engine** (the gatekeeper before MPC signing). It is a
"fast-follow" service in the launch sequence and hardens the platform against
chargeback, account-takeover, and velocity-based fraud.

Core responsibilities:

- Real-time fraud scoring on every payment + behavioral event.
- Maintaining chargeback risk and velocity anomaly models.
- Engineering and serving features (user, device, geolocation, payment history).
- Model versioning, A/B routing (champion/challenger), and retraining.
- Producing explainable scores (SHAP top features) for analysts and disputes.
- Closing the feedback loop by ingesting chargeback / fraud / clean outcomes.
- Emitting threshold-based alerts and an audit trail of every score.

## Language & Tech Stack

- **Language:** Python 3.11+
- **Web framework:** FastAPI (async REST + Kafka consumer workers)
- **ML:** scikit-learn, XGBoost; SHAP for explainability
- **Feature store:** Feast (online lookups via Redis)
- **Streaming:** Kafka consumer for `payment.*` and `chargeback.*` events
- **Model registry:** MLflow (model versions, stage transitions, A/B routing)
- **Storage:** PostgreSQL (scores, model versions, chargeback events)
- **Observability:** Prometheus metrics, OpenTelemetry traces, structured logs

## System Requirements

### Functional

- **Real-time scoring** on payment + behavioral signals, called synchronously on
  the transaction path and asynchronously via the event bus.
- **Chargeback risk model** estimating P(chargeback) for a given payment.
- **Velocity anomaly model** detecting unusual transaction frequency, amount, or
  breadth across cards/devices/IPs for a user or device cluster.
- **Device / fingerprint signals** ingestion (fingerprint hash, device type,
  rooted/emulator flags, known-bad device list).
- **Feature engineering pipeline** producing online + offline feature groups,
  backfilled from historical events and kept fresh from the stream.
- **Model versioning & A/B** — every model has a semantic version; traffic is
  split between champion and challenger via configurable routing.
- **Score explainability** — top contributing features per score via SHAP,
  returned in the API response and stored for audit.
- **Feedback loop** — chargeback, fraud-claim, and confirmed-clean outcomes are
  ingested as labels to retrain and evaluate models.
- **Threshold-based alerting** — scores above `SCORE_THRESHOLD_HIGH` trigger an
  immediate alert (to Policy Engine + notification); medium band routes to
  manual review queue.

### Non-Functional

| Requirement | Target |
|---|---|
| Score latency (p99) | < 200 ms |
| Feature lookup (p99) | < 50 ms |
| Model retrain cadence | daily (velocity) / weekly (chargeback) |
| Availability | 99.9% (scoring on transaction path) |
| Drift monitoring | PSI/KS drift checks on inputs + score distribution, daily |
| Throughput | ≥ 2,000 scores/sec sustained |
| Model rollback | < 5 min via MLflow stage transition |
| Score persistence | immutable, retained 7 years for compliance |

## Technical Specifications

### API Surface

Two surfaces:

1. **REST (synchronous)** — called inline by the Policy Engine on the
   transaction path. Must meet the p99 latency budget.
2. **Event bus (asynchronous)** — Kafka consumer ingests
   `payment.authorized`, `payment.captured`, `chargeback.received` and emits
   `fraud.scored` for downstream consumers.

### Endpoints

#### `POST /v1/fraud/score`

Synchronous score for a single transaction.

**Request:**
```json
{
  "user_id": "usr_7f3...",
  "payment_id": "pay_8a1...",
  "amount": { "currency": "USD", "minor_units": 25000 },
  "device": {
    "fingerprint": "fp_2c9...",
    "type": "mobile_ios",
    "rooted": false,
    "emulator": false
  },
  "ip": "203.0.113.42",
  "behavioral_features": {
    "session_duration_ms": 18432,
    "keystroke_entropy": 0.71,
    "tap_variance": 0.33
  }
}
```

**Response (200):**
```json
{
  "score": 0.82,
  "risk_band": "high",
  "model_version": "chargeback-xgb@v3.2.0",
  "top_features": [
    { "name": "velocity_card_24h", "shap": 0.21 },
    { "name": "chargeback_rate_30d", "shap": 0.18 },
    { "name": "device_new_to_user", "shap": 0.09 }
  ],
  "scored_at": "2026-07-06T12:00:00Z"
}
```

#### `GET /v1/fraud/models`

List registered models with current stage and A/B traffic split.

```json
{
  "models": [
    {
      "name": "chargeback-xgb",
      "champion": "v3.2.0",
      "challenger": "v3.3.0-rc1",
      "traffic_split": { "champion": 0.9, "challenger": 0.1 },
      "updated_at": "2026-07-05T03:00:00Z"
    },
    {
      "name": "velocity-isoforest",
      "champion": "v1.4.0",
      "challenger": null,
      "traffic_split": { "champion": 1.0 },
      "updated_at": "2026-06-28T03:00:00Z"
    }
  ]
}
```

#### `POST /v1/fraud/feedback`

Submit an outcome label for a previously-scored transaction; used by the
retraining pipeline.

**Request:**
```json
{
  "tx_id": "pay_8a1...",
  "outcome": "chargeback",
  "reason_code": "10.4",
  "reported_at": "2026-07-12T09:00:00Z"
}
```

`outcome` ∈ `{ chargeback, fraud, clean }`. **Response:** `204 No Content`.

### Data Model (PostgreSQL)

| Table | Purpose |
|---|---|
| `fraud_scores` | one row per score: `tx_id, user_id, score, risk_band, model_version, top_features(jsonb), scored_at` |
| `model_versions` | registered models: `name, version, stage (champion/challenger/archived), metrics(jsonb), trained_at` |
| `feature_values` | online-feature snapshot per score (for replay/debug): `tx_id, feature_group, payload(jsonb)` |
| `chargeback_events` | feedback labels: `tx_id, outcome, reason_code, reported_at, source` |

### Feature Groups (Feast)

| Feature group | Example features |
|---|---|
| `user_velocity` | tx count/sum last 1h, 24h, 7d; distinct cards/devices/IPs |
| `device` | fingerprint hash, new-to-user flag, known-bad flag, emulator/rooted |
| `geolocation` | IP country, distance from billing, VPN/proxy flag, geo-velocity |
| `payment_history` | success/fail ratio, avg ticket, first-payment age |
| `chargeback_history` | user chargeback rate 30d/90d, device chargeback rate |

### Integrations

| Direction | Channel | Counterparty | Payload |
|---|---|---|---|
| Consumes | Kafka `payment.authorized` / `payment.captured` | Payment Orchestration | payment + device + behavioral signals |
| Consumes | Kafka `chargeback.received` (webhook normalize by Rail Connectors) | Rail Connectors | chargeback reason, amount, tx ref |
| Emits | Kafka `fraud.scored` | Policy / Risk Engine | score, risk_band, model_version |
| Emits | Kafka `fraud.alert.raised` | Notification + ops | high-band alerts |
| Emits | Kafka `fraud.audit` | Audit / Event Log | every score (immutable) |
| Reads | REST (sync) | Feature Store (Feast/Redis) | online features |
| Reads/Writes | REST | MLflow Model Registry | model versions, stage transitions |

### ML Ops

- **Model registry (MLflow):** every trained model is logged with params,
  metrics (AUC, PR-AUC, calibration), and the feature snapshot URI. Stage
  transitions (`staging → champion → archived`) are audited.
- **A/B routing:** the service resolves the active split per model name from
  the registry and routes by stable hash of `tx_id`.
- **Champion / challenger:** challenger receives a configurable fraction of
  traffic; its offline + online metrics are compared before promotion.
- **Retraining:** scheduled jobs (daily for velocity, weekly for chargeback)
  pull labeled `chargeback_events` since the last run, retrain, and register a
  candidate. Promotion requires a human approval gate.
- **Drift monitoring:** daily PSI/KS tests on input features and score
  distribution per model; breaches open an alert and can trigger retraining.
- **Rollback:** reverting a champion is a single MLflow stage transition; the
  service hot-reloads the active version map.

## Dependencies

| Dependency | Purpose |
|---|---|
| PostgreSQL | persistent store for scores, model versions, feedback labels |
| Redis | online feature store backend (Feast) + low-latency caches |
| Feast | feature store: definitions, offline backfill, online serving |
| Kafka | event ingestion (`payment.*`, `chargeback.*`) and emission (`fraud.*`) |
| MLflow | model registry, experiment tracking, A/B routing metadata |
| Audit / Event Log | downstream consumer of `fraud.audit` for compliance |
| OpenTelemetry collector | traces/metrics export |

## Configuration

Environment variables (12-factor; loaded at startup):

| Variable | Required | Default | Description |
|---|---|---|---|
| `PORT` | no | `8080` | HTTP listen port |
| `DB_URL` | yes | — | PostgreSQL DSN (`postgresql://...`) |
| `REDIS_URL` | yes | — | Redis URL for online feature store |
| `KAFKA_BROKERS` | yes | — | comma-separated Kafka bootstrap brokers |
| `KAFKA_CONSUMER_GROUP` | no | `fraud-engine` | consumer group id |
| `MODEL_REGISTRY_URL` | yes | — | MLflow tracking URI |
| `FEATURE_STORE_URL` | no | `redis://redis:6379` | Feast online store URL |
| `SCORE_THRESHOLD_HIGH` | no | `0.75` | score ≥ → `high` band + immediate alert |
| `SCORE_THRESHOLD_MEDIUM` | no | `0.40` | score ≥ → `medium` band → manual review |
| `CHALLENGER_TRAFFIC_FRACTION` | no | `0.10` | default A/B split for challengers |
| `RETRAIN_VELOCITY_CRON` | no | `0 3 * * *` | daily velocity retrain schedule |
| `RETRAIN_CHARGEBACK_CRON` | no | `0 4 * * 1` | weekly chargeback retrain schedule |
| `DRIFT_PSI_THRESHOLD` | no | `0.2` | PSI above which a feature is flagged |
| `LOG_LEVEL` | no | `info` | structured log level |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | no | — | OpenTelemetry collector endpoint |

## Local Development

```bash
# Build the image
docker build -t fraud-engine:dev .

# Run with dependencies (Postgres, Redis, Kafka, MLflow)
docker compose up -d postgres redis kafka mlflow
docker compose run --rm fraud-engine

# Or run directly
uv sync
uv run uvicorn app.main:app --reload --port 8080

# Tests
uv run pytest

# Lint / typecheck
uv run ruff check .
uv run mypy app/

# Train a model locally (writes to local MLflow)
uv run python -m app.training.train_chargeback --config config/chargeback.yaml
uv run python -m app.training.train_velocity    --config config/velocity.yaml

# Backfill features into the offline + online store
uv run python -m app.features.backfill --since 2026-01-01
```

### Full dev loop with Make

```bash
make up            # docker compose up -d (Postgres, Redis, Kafka, MLflow, service)
make migrate       # apply migrations/001_init.sql against DB_URL
make test          # pytest with coverage
make lint          # ruff check src tests feature_repo
make typecheck     # mypy src/fraud_engine
make train         # run chargeback + velocity training entrypoints
make docker-build  # build the multi-stage Docker image
make clean         # remove build/coverage artifacts
```

### Source layout

```
src/fraud_engine/
  app.py             FastAPI app (healthz, readyz, /v1/fraud/* endpoints)
  config.py          12-factor settings (env vars)
  db.py              Postgres + Redis connection helpers and queries
  feature_store.py   Feast online store wrapper + InMemory test store
  registry.py        MLflow-backed model registry + model_versions table
  scoring.py         StubModel, SHAP top-features, score_request
  audit.py           fraud.audit emitter (exactly-once per score)
  replay.py          replay tool reconstructing scores from audit + snapshot
  models/schemas.py  Pydantic request/response models
  models/routing.py  stable-hash A/B routing + risk_band thresholds
  features/engineering.py  user_velocity / payment_history / device / geolocation
  features/backfill.py     backfill CLI (--since YYYY-MM-DD)
  consumers/kafka.py       aiokafka consumer for payment.*/chargeback.*
  observability/drift.py   PSI + KS helpers
  observability/monitoring.py  DriftMonitor + Prometheus metrics + alerts
  training/train_chargeback.py  chargeback training entrypoint
  training/train_velocity.py   velocity training entrypoint
  training/scheduler.py        daily/weekly retrain schedules + promotion gate
feature_repo/
  features.py          Feast entity + feature-view definitions
  feature_store.yaml   Feast repo config (Redis online store)
migrations/
  001_init.sql         fraud_scores, model_versions, feature_values,
                       chargeback_events, drift_metrics tables
```
