"""OpenTelemetry tracing + metrics wiring for the fraud-detection service.

Configures global tracer + meter providers from OTEL_EXPORTER_OTLP_ENDPOINT.
When the endpoint is unset, init is a no-op so tests never require a real
collector. Also installs FastAPI auto-instrumentation when available.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("fraud_engine.tracing")

_INITIALIZED = False


def init_tracing() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        _INITIALIZED = True
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.environ.get("OTEL_SERVICE_NAME", "fraud-detection")
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "ai-crypto-onramp",
            "deployment.environment": "dev",
        }
    )

    trace_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter, export_interval_millis=10000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    from opentelemetry import metrics
    metrics.set_meter_provider(meter_provider)

    _INITIALIZED = True


def instrument_app(app) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass
