# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt
COPY pyproject.toml ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install .

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.11/site-packages:/app/src
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 fraud \
    && useradd --system --uid 1001 --gid fraud --create-home --home-dir /home/fraud fraud
COPY --from=builder /install /install
COPY --from=builder /build/src/ /app/src/
COPY migrations /app/migrations
COPY feature_repo /app/feature_repo
RUN chown -R fraud:fraud /app
USER fraud
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost:8080/healthz || exit 1
CMD ["uvicorn", "fraud_detection.app:app", "--host", "0.0.0.0", "--port", "8080"]