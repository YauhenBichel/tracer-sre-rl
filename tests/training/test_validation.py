"""Tests for scenario validation and data quality checks."""

from __future__ import annotations

from app.incidents.scenario_loader import ScenarioLoader
from app.models import ScenarioDefinition
from app.training.validation import (
    check_data_contamination,
    validate_all,
    validate_scenario,
    validate_telemetry_quality,
)


def test_validate_builtin_scenarios():
    """All builtin scenarios must pass validation."""
    scenarios = ScenarioLoader().load_all()
    results = [(s.name, *validate_scenario(s)) for s in scenarios]

    assert all(valid for _, valid, _ in results), next(
        f"{name}: {reason}" for name, valid, reason in results if not valid
    )


def test_validate_telemetry_quality_builtins():
    """All builtins must produce ERROR logs (evidence of anomaly)."""
    scenarios = ScenarioLoader().load_all()
    results = [(s.name, *validate_telemetry_quality(s)) for s in scenarios]

    assert all(valid for _, valid, _ in results), next(
        f"{name}: {reason}" for name, valid, reason in results if not valid
    )


def test_rejects_empty_scenario():
    scenario = ScenarioDefinition(
        id="empty",
        name="Empty",
        description="",
        taxonomy_labels=(),
        services=(),
        timeline=(),
        gold_root_causes=(),
        gold_remediations=(),
    )
    valid, reason = validate_scenario(scenario)

    assert not valid
    assert "no services" in reason


def test_contamination_check_detects_overlap():
    scenarios = ScenarioLoader().load_all()
    overlap = check_data_contamination(scenarios, scenarios)
    assert len(overlap) == len(scenarios)


def test_contamination_check_no_overlap():
    scenarios = ScenarioLoader().load_all()
    overlap = check_data_contamination(scenarios[:3], scenarios[3:])
    assert len(overlap) == 0


def test_validate_all_returns_only_valid():
    scenarios = ScenarioLoader().load_all()
    valid = validate_all(scenarios)
    assert len(valid) == len(scenarios)  # all builtins should be valid
