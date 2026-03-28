"""Tests for the LLM agent tool adapter."""

from src.agent_adapter import TOOL_DEFINITIONS, SREToolAdapter
from src.generators.loader import ScenarioLoader


def _load_scenario():
    return ScenarioLoader().load_all()[0]


def test_adapter_reset_returns_initial_alert():
    adapter = SREToolAdapter(_load_scenario(), seed=42)
    result = adapter.reset()

    assert result.tool_name == "initial_alert"
    assert len(result.observation) > 0
    assert result.step == 0
    assert not result.done


def test_adapter_list_alerts():
    adapter = SREToolAdapter(_load_scenario(), seed=42)
    adapter.reset()
    result = adapter.call_tool("list_alerts")

    assert "Alert" in result.observation or "alert" in result.observation.lower()
    assert result.step == 1


def test_adapter_query_metrics_by_service_name():
    adapter = SREToolAdapter(_load_scenario(), seed=42)
    adapter.reset()
    service = adapter.service_names[0]
    result = adapter.call_tool("query_metrics", service=service)

    assert "Metrics" in result.observation or "metrics" in result.observation.lower()


def test_adapter_full_episode():
    adapter = SREToolAdapter(_load_scenario(), seed=42)
    adapter.reset()

    adapter.call_tool("list_alerts")
    adapter.call_tool("list_services")
    adapter.call_tool("query_metrics", service=adapter.service_names[0])
    adapter.call_tool("query_logs", service=adapter.service_names[0])

    diag = adapter.diagnosis_options[0]
    adapter.call_tool("diagnose", diagnosis=diag)

    rem = adapter.remediation_options[0]
    result = adapter.call_tool("remediate", action=rem)

    assert result.done
    assert result.reward > 0


def test_adapter_done_prevents_further_calls():
    adapter = SREToolAdapter(_load_scenario(), seed=42)
    adapter.reset()

    # Fast forward to done
    adapter.call_tool("list_alerts")
    adapter.call_tool("diagnose", diagnosis=adapter.diagnosis_options[0])
    adapter.call_tool("remediate", action=adapter.remediation_options[0])

    assert adapter.done
    result = adapter.call_tool("list_alerts")
    assert result.done
    assert "complete" in result.observation.lower()


def test_adapter_partial_service_match():
    adapter = SREToolAdapter(_load_scenario(), seed=42)
    adapter.reset()
    # Use partial name
    first_service = adapter.service_names[0]
    partial = first_service[:4]
    result = adapter.call_tool("query_metrics", service=partial)

    assert not result.done


def test_tool_definitions_have_required_fields():
    assert len(TOOL_DEFINITIONS) > 0
    for tool in TOOL_DEFINITIONS:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool


def test_adapter_exposes_options():
    adapter = SREToolAdapter(_load_scenario(), seed=42)

    assert len(adapter.diagnosis_options) > 0
    assert len(adapter.remediation_options) > 0
    assert len(adapter.service_names) > 0
