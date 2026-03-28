"""Tests for individual generators (metrics, logs, traces)."""

import random

import pytest

from src.generators.effects.registry import build_default_registry
from src.generators.logs import LogGenerator
from src.generators.metrics import MetricsGenerator
from src.generators.traces import TraceGenerator
from src.models import ServiceDefinition, TimelineEntry


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def services():
    return (
        ServiceDefinition(name="gateway", service_type="web-server", dependencies=("app",)),
        ServiceDefinition(name="app", service_type="application", dependencies=("db",)),
        ServiceDefinition(name="db", service_type="database", config={"max_connections": 100}),
    )


@pytest.fixture
def normal_events():
    return [TimelineEntry(time_offset_seconds=0, event_type="normal_traffic")]


@pytest.fixture
def error_events():
    return [
        TimelineEntry(
            time_offset_seconds=0,
            event_type="error_spike",
            service="app",
            params={"target_rate": 0.5, "duration_seconds": 100},
        )
    ]


class TestMetricsGenerator:
    def test_generates_samples_for_service(self, rng, services, normal_events):
        gen = MetricsGenerator(rng, build_default_registry())
        samples = gen.generate(services[0], timestamp=100, active_events=normal_events)

        assert len(samples) > 0
        assert all(s.service == "gateway" for s in samples)

    def test_all_values_non_negative(self, rng, services, normal_events):
        gen = MetricsGenerator(rng, build_default_registry())
        assert len(services) > 0
        for svc in services:
            samples = gen.generate(svc, timestamp=100, active_events=normal_events)
            assert all(s.value >= 0 for s in samples), f"Negative metric for {svc.name}"

    def test_percent_metrics_capped_at_100(self, rng, services, normal_events):
        gen = MetricsGenerator(rng, build_default_registry())
        assert len(services) > 0
        for svc in services:
            samples = gen.generate(svc, timestamp=100, active_events=normal_events)
            for s in samples:
                if "percent" in s.metric_name:
                    assert s.value <= 100, f"{s.metric_name} = {s.value}"


class TestLogGenerator:
    def test_generates_log_entry(self, rng, services, normal_events):
        gen = LogGenerator(rng)
        entry = gen.generate(services[0], timestamp=100, active_events=normal_events)

        assert entry is not None
        assert entry.service == "gateway"

    def test_error_events_increase_log_probability(self, services, error_events):
        normal = [TimelineEntry(time_offset_seconds=0, event_type="normal_traffic")]
        normal_count = sum(1 for s in range(100) if LogGenerator(random.Random(s)).should_generate(services[1], normal))
        error_count = sum(
            1 for s in range(100) if LogGenerator(random.Random(s)).should_generate(services[1], error_events)
        )

        assert error_count > normal_count

    def test_log_level_reflects_error_events(self, services, error_events):
        # Run multiple times — with error events, should produce ERROR level sometimes
        error_count = 0
        for seed in range(50):
            gen = LogGenerator(random.Random(seed))
            entry = gen.generate(services[1], timestamp=100, active_events=error_events)
            if entry and entry.level.value == "ERROR":
                error_count += 1
        assert error_count > 0, "Error events should sometimes produce ERROR logs"


class TestTraceGenerator:
    def test_generates_spans(self, rng, services, normal_events):
        gen = TraceGenerator(rng)
        spans = gen.generate(services, timestamp=100, active_events=normal_events)
        assert len(spans) > 0

    def test_spans_follow_dependency_chain(self, rng, services, normal_events):
        gen = TraceGenerator(rng)
        spans = gen.generate(services, timestamp=100, active_events=normal_events)
        service_names = {s.service for s in spans}

        assert "gateway" in service_names
        assert "app" in service_names
        assert "db" in service_names

    def test_all_spans_share_trace_id(self, rng, services, normal_events):
        gen = TraceGenerator(rng)
        spans = gen.generate(services, timestamp=100, active_events=normal_events)
        trace_ids = {s.trace_id for s in spans}

        assert len(trace_ids) == 1

    def test_error_events_cause_elevated_latency(self, services, error_events):
        normal_durations, error_durations = [], []
        normal_events = [TimelineEntry(time_offset_seconds=0, event_type="normal_traffic")]
        for seed in range(20):
            rng = random.Random(seed)
            normal_spans = TraceGenerator(rng).generate(services, 100, normal_events)
            error_spans = TraceGenerator(random.Random(seed)).generate(services, 100, error_events)
            normal_durations.extend(s.duration_ms for s in normal_spans if s.service == "app")
            error_durations.extend(s.duration_ms for s in error_spans if s.service == "app")

        assert len(normal_durations) > 0
        assert len(error_durations) > 0
        assert max(error_durations) > max(normal_durations)
