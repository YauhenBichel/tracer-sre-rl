"""Tests for event effect handlers."""

import pytest

from app.models import ServiceDefinition, TimelineEntry
from app.telemetry.effects.cascade import CascadeEffect
from app.telemetry.effects.connection_exhaustion import ConnectionExhaustionEffect
from app.telemetry.effects.disk_fill import DiskFillEffect
from app.telemetry.effects.error_spike import ErrorSpikeEffect
from app.telemetry.effects.latency_spike import LatencySpikeEffect
from app.telemetry.effects.memory_leak import MemoryLeakEffect
from app.telemetry.effects.registry import EventEffectRegistry, build_default_registry
from app.telemetry.effects.traffic_ramp import TrafficRampEffect


@pytest.fixture
def service():
    return ServiceDefinition(name="api", service_type="application", config={"max_connections": 100})


@pytest.fixture
def event():
    return TimelineEntry(time_offset_seconds=0, event_type="test", params={"duration_seconds": 100})


def test_traffic_ramp_increases_request_rate(service, event):
    effect = TrafficRampEffect()
    event = TimelineEntry(time_offset_seconds=0, event_type="traffic_ramp", params={"multiplier": 3.0})
    result = effect.apply(100.0, "request_rate", service, event, progress=1.0)

    assert result == pytest.approx(300.0)


def test_traffic_ramp_no_effect_on_unrelated(service, event):
    assert TrafficRampEffect().apply(100.0, "disk_usage_percent", service, event, 0.5) is None


def test_latency_spike_multiplies(service):
    event = TimelineEntry(time_offset_seconds=0, event_type="latency_spike", params={"factor": 10.0})
    result = LatencySpikeEffect().apply(20.0, "latency_p99_ms", service, event, progress=1.0)
    assert result == pytest.approx(200.0)


def test_error_spike_ramps_to_target(service):
    event = TimelineEntry(time_offset_seconds=0, event_type="error_spike", params={"target_rate": 0.5})
    result = ErrorSpikeEffect().apply(0.01, "error_rate", service, event, progress=1.0)
    assert result == pytest.approx(0.5)


def test_connection_exhaustion_approaches_max(service):
    event = TimelineEntry(time_offset_seconds=0, event_type="connection_exhaustion")
    result = ConnectionExhaustionEffect().apply(20.0, "active_connections", service, event, progress=1.0)
    assert result == pytest.approx(98.0)  # 100 * 0.98


def test_cascade_increases_latency(service):
    event = TimelineEntry(time_offset_seconds=0, event_type="cascade")
    result = CascadeEffect().apply(10.0, "latency_p99_ms", service, event, progress=1.0)
    assert result > 10.0


def test_disk_fill_reaches_target(service):
    event = TimelineEntry(time_offset_seconds=0, event_type="disk_fill", params={"target_percent": 95})
    result = DiskFillEffect().apply(30.0, "disk_usage_percent", service, event, progress=1.0)
    assert result == pytest.approx(95.0)


def test_memory_leak_grows_over_time(service):
    event = TimelineEntry(
        time_offset_seconds=0, event_type="memory_leak", params={"rate_per_minute": 5.0, "duration_seconds": 600}
    )
    result = MemoryLeakEffect().apply(30.0, "memory_percent", service, event, progress=1.0)
    assert result == pytest.approx(80.0)  # 30 + 5 * 10 minutes


def test_registry_returns_none_for_unknown():
    registry = EventEffectRegistry()
    assert registry.get("nonexistent") is None


def test_default_registry_has_all_handlers():
    registry = build_default_registry()
    expected = [
        "traffic_ramp",
        "resource_exhaustion",
        "latency_spike",
        "error_spike",
        "connection_exhaustion",
        "cascade",
        "symptom",
        "disk_fill",
        "memory_leak",
    ]

    assert len(expected) > 0
    for event_type in expected:
        assert registry.get(event_type) is not None, f"Missing handler for {event_type}"
