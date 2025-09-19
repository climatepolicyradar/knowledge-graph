# Requires: opentelemetry-sdk, opentelemetry-exporter-otlp

import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Set resource attributes; add your own org/team identifiers
resource = Resource.create(
    {
        "service.name": os.getenv("OTEL_SERVICE_NAME", "knowledge-graph-flows"),
        "service.namespace": "knowledge-graph",
        "service.instance.id": "knowledge-graph-flows-1",
        "environment": "local",
    }
)

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    "https://otel.staging.climatepolicyradar.org"
)
os.environ["OTEL_EXPORT_OTLP_PROTOCOL"] = "http/protobuf"
os.environ["OTEL_SERVICE_NAME"] = "knowledge-graph-flows"
os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "service.namespace=knowledge-graph"

base_endpoint = os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.staging.climatepolicyradar.org"
)

# Tracing
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base_endpoint}/v1/traces"))
)
trace.set_tracer_provider(tracer_provider)

# Metrics
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=f"{base_endpoint}/v1/metrics")
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
