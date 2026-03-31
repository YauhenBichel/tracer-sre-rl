"""Tests for TelemetryFormatter."""

import pytest

from app.incidents.scenario_loader import ScenarioLoader
from app.rl_env.formatter import TelemetryFormatter
from app.telemetry.generator import TelemetryGenerator


@pytest.fixture
def formatter():
    scenario = ScenarioLoader().load("scenarios/db_connection_pool.yaml")
    telemetry = TelemetryGenerator(seed=42).generate(scenario)
    return TelemetryFormatter(telemetry, scenario)


def test_initial_observation_contains_alert(formatter):
    obs = formatter.initial_observation()
    assert "ALERT" in obs or "anomaly" in obs


def test_topology_lists_services(formatter):
    text = formatter.topology()

    assert "Service Topology" in text
    assert "api-gateway" in text
    assert "postgres-primary" in text


def test_metrics_returns_data(formatter):
    text = formatter.metrics("api-gateway", start_idx=0, end_idx=89)
    assert "Metrics for api-gateway" in text
    assert "avg=" in text


def test_metrics_no_data_for_invalid_service(formatter):
    text = formatter.metrics("nonexistent", start_idx=0, end_idx=89)
    assert "No metrics found" in text


def test_logs_returns_entries(formatter):
    text = formatter.logs("api-gateway", start_idx=0, end_idx=89)
    assert "Logs for" in text or "No logs" in text


def test_traces_returns_spans(formatter):
    text = formatter.traces("api-gateway")
    assert "Traces" in text or "No traces" in text


def test_alerts_returns_fired_alerts(formatter):
    text = formatter.alerts()
    assert "Alert" in text or "No alerts" in text
