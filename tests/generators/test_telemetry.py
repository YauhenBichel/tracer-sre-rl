"""Tests for the synthetic telemetry generator."""

import pytest

from src.generators.loader import ScenarioLoader
from src.generators.telemetry import TelemetryGenerator


@pytest.fixture
def scenarios():
    return ScenarioLoader().load_all()


@pytest.fixture
def generator():
    return TelemetryGenerator(seed=42)


def test_load_scenarios(scenarios):
    assert len(scenarios) >= 4
    for s in scenarios:
        assert s.id
        assert s.name
        assert len(s.services) > 0
        assert len(s.timeline) > 0
        assert len(s.gold_root_causes) > 0
        assert len(s.gold_remediations) > 0


def test_generate_telemetry(generator, scenarios):
    assert len(scenarios) > 0
    for scenario in scenarios:
        telemetry = generator.generate(scenario)
        assert len(telemetry.metrics) > 0, f"No metrics for {scenario.name}"
        assert len(telemetry.logs) > 0, f"No logs for {scenario.name}"
        assert len(telemetry.traces) > 0, f"No traces for {scenario.name}"
        assert len(telemetry.events) > 0, f"No events for {scenario.name}"


def test_metrics_cover_all_services(generator, scenarios):
    assert len(scenarios) > 0
    for scenario in scenarios:
        telemetry = generator.generate(scenario)
        services_with_metrics = {m.service for m in telemetry.metrics}
        expected_services = {s.name for s in scenario.services}
        assert expected_services.issubset(services_with_metrics), (
            f"Missing metrics for: {expected_services - services_with_metrics} in {scenario.name}"
        )


def test_deterministic_with_seed(scenarios):
    scenario = scenarios[0]
    t1 = TelemetryGenerator(seed=123).generate(scenario)
    t2 = TelemetryGenerator(seed=123).generate(scenario)

    assert len(t1.metrics) == len(t2.metrics)
    for m1, m2 in zip(t1.metrics, t2.metrics, strict=True):
        assert m1.service == m2.service
        assert m1.metric_name == m2.metric_name
        assert m1.value == m2.value


def test_event_effects_visible_in_metrics(generator, scenarios):
    scenario = next(s for s in scenarios if "connection" in s.name.lower())
    telemetry = generator.generate(scenario)

    early = [
        m.value
        for m in telemetry.metrics
        if m.service == "postgres-primary" and m.timestamp < 200 and "connection" in m.metric_name
    ]
    late = [
        m.value
        for m in telemetry.metrics
        if m.service == "postgres-primary" and m.timestamp > 500 and "connection" in m.metric_name
    ]

    assert len(early) > 0, "Expected early connection metrics"
    assert len(late) > 0, "Expected late connection metrics"
    assert max(late) > max(early), "Connection count should increase during exhaustion"
