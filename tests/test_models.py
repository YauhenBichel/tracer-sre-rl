"""Tests for immutable data models."""

import pytest
from src.models import (
    MetricSample, LogEntry, LogLevel, TraceSpan, SpanStatus, Event,
    ServiceDefinition, TimelineEntry, ScenarioDefinition,
    GoldStandardRootCause, GoldStandardRemediation, GeneratedTelemetry,
)


def test_metric_sample_is_frozen():
    sample = MetricSample(timestamp=1.0, service="svc", metric_name="cpu", value=50.0)
    with pytest.raises(AttributeError):
        sample.value = 99.0


def test_service_definition_is_frozen():
    svc = ServiceDefinition(name="api", service_type="web-server", dependencies=("db",))
    with pytest.raises(AttributeError):
        svc.name = "changed"


def test_scenario_definition_is_frozen():
    scenario = ScenarioDefinition(
        id="test", name="test", description="", taxonomy_labels=(),
        services=(), timeline=(), gold_root_causes=(), gold_remediations=(),
    )
    with pytest.raises(AttributeError):
        scenario.name = "changed"


def test_trace_span_uses_enum_status():
    span = TraceSpan(
        trace_id="abc", span_id="def", parent_span_id=None,
        service="svc", operation="op", start_time=0.0,
        duration_ms=10.0, status=SpanStatus.OK,
    )
    assert span.status == SpanStatus.OK
    assert span.status != SpanStatus.ERROR


def test_generated_telemetry_is_frozen():
    t = GeneratedTelemetry()
    with pytest.raises(AttributeError):
        t.metrics = ()
