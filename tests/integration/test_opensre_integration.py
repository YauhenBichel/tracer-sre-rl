"""Tests for open-sre-agent integration.

Verifies that the RL environment correctly implements the interfaces
expected by opensre's execute_actions pipeline:
  - SimulatedAction matches InvestigationAction fields
  - parameter_extractor is not None (execute_actions rejects None)
  - function() returns dict with "source" and "available" keys
  - AgentState has all required fields including investigation_started_at
"""

from src.agent_adapter import SREToolAdapter
from src.generators.loader import ScenarioLoader
from src.integration.evidence_source import (
    SIMULATED_SOURCE_KEY,
    build_simulated_sources,
    create_simulated_actions,
)
from src.integration.state_adapter import (
    scenario_to_agent_state,
    score_agent_state,
)


def _load_scenario():
    return ScenarioLoader().load_all()[0]


# --- Evidence source tests ---


def test_simulated_actions_have_parameter_extractor():
    """execute_actions rejects actions with parameter_extractor=None."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    adapter.reset()
    for action in create_simulated_actions(adapter):
        assert action.parameter_extractor is not None, f"{action.name} has no parameter_extractor"


def test_simulated_actions_have_availability_check():
    """Actions should have availability_check for execute_actions filtering."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    adapter.reset()
    for action in create_simulated_actions(adapter):
        assert action.availability_check is not None, f"{action.name} has no availability_check"


def test_availability_check_passes_with_simulated_sources():
    """availability_check returns True when simulated sources are configured."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    adapter.reset()
    sources = build_simulated_sources(adapter)
    for action in create_simulated_actions(adapter):
        assert action.availability_check(sources), f"{action.name} availability check failed"


def test_parameter_extractor_returns_dict():
    """parameter_extractor must return a dict of kwargs for the function."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    adapter.reset()
    sources = build_simulated_sources(adapter)
    for action in create_simulated_actions(adapter):
        kwargs = action.parameter_extractor(sources)
        assert isinstance(kwargs, dict), f"{action.name} extractor returned {type(kwargs)}"


def test_function_returns_opensre_format():
    """function() must return dict with 'source' and 'available' keys."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    adapter.reset()
    sources = build_simulated_sources(adapter)

    for action in create_simulated_actions(adapter):
        kwargs = action.parameter_extractor(sources)
        result = action.function(**kwargs)
        assert isinstance(result, dict), f"{action.name} returned {type(result)}"
        assert "source" in result, f"{action.name} missing 'source' key"
        assert "available" in result, f"{action.name} missing 'available' key"
        assert result["available"] is True, f"{action.name} available is not True"


def test_simulated_actions_cover_key_tools():
    """Simulated actions cover the tools open-sre-agent needs for investigation."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    adapter.reset()
    names = {a.name for a in create_simulated_actions(adapter)}
    assert "get_alerts" in names
    assert "get_service_topology" in names
    assert "get_metrics" in names
    assert "get_error_logs" in names
    assert "get_traces" in names


def test_simulated_sources_has_connection_verified():
    """available_sources must have connection_verified for availability_check."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    sources = build_simulated_sources(adapter)
    assert sources[SIMULATED_SOURCE_KEY]["connection_verified"] is True


# --- State adapter tests ---


def test_agent_state_has_investigation_started_at():
    """AgentState must include investigation_started_at for timing calculations."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    state = scenario_to_agent_state(scenario, adapter)
    assert "investigation_started_at" in state
    assert isinstance(state["investigation_started_at"], float)


def test_agent_state_has_alert_json():
    """AgentState must include alert_json for downstream nodes."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    state = scenario_to_agent_state(scenario, adapter)
    assert "alert_json" in state
    assert isinstance(state["alert_json"], dict)
    assert "alert_name" in state["alert_json"]


def test_agent_state_has_resolved_integrations_with_endpoints():
    """resolved_integrations must have endpoint/api_key structure, not just available/type."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    state = scenario_to_agent_state(scenario, adapter)

    ri = state["resolved_integrations"]
    assert "grafana" in ri
    assert "endpoint" in ri["grafana"]
    assert "api_key" in ri["grafana"]
    assert "cloudwatch" in ri
    assert "log_group" in ri["cloudwatch"]
    assert "datadog" in ri
    assert "api_key" in ri["datadog"]


def test_agent_state_has_all_required_fields():
    """AgentState must have every field that opensre's pipeline reads."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)
    state = scenario_to_agent_state(scenario, adapter)

    required_fields = [
        "mode",
        "is_noise",
        "alert_name",
        "pipeline_name",
        "severity",
        "alert_source",
        "raw_alert",
        "alert_json",
        "planned_actions",
        "plan_rationale",
        "available_sources",
        "available_action_names",
        "resolved_integrations",
        "context",
        "evidence",
        "root_cause",
        "root_cause_category",
        "validated_claims",
        "non_validated_claims",
        "validity_score",
        "investigation_recommendations",
        "remediation_steps",
        "investigation_loop_count",
        "hypotheses",
        "executed_hypotheses",
        "investigation_started_at",
        "slack_context",
        "problem_md",
    ]
    for field in required_fields:
        assert field in state, f"Missing required field: {field}"


# --- Scoring tests ---


def test_score_correct_diagnosis():
    scenario = _load_scenario()
    gold_label = scenario.gold_root_causes[0].taxonomy_label
    gold_remediation = scenario.gold_remediations[0].action

    state = {
        "root_cause": gold_label.replace(".", " ").replace("_", " "),
        "root_cause_category": gold_label.split(".")[0],
        "remediation_steps": [gold_remediation],
        "executed_hypotheses": [
            {"action": "get_alerts", "service": "system"},
            {"action": "get_metrics", "service": scenario.services[0].name},
            {"action": "get_error_logs", "service": scenario.services[1].name},
        ],
    }
    result = score_agent_state(state, scenario)
    assert result["reward"] > 0.5


def test_score_wrong_diagnosis():
    scenario = _load_scenario()
    state = {
        "root_cause": "completely wrong diagnosis about unicorns",
        "root_cause_category": "unknown",
        "remediation_steps": ["restart everything"],
        "executed_hypotheses": [{"action": "get_alerts", "service": "system"}],
    }
    result = score_agent_state(state, scenario)
    assert result["reward"] < 0.5


def test_score_empty_diagnosis():
    scenario = _load_scenario()
    state = {
        "root_cause": "",
        "root_cause_category": "",
        "remediation_steps": [],
        "executed_hypotheses": [],
    }
    result = score_agent_state(state, scenario)
    assert result["reward"] == 0.0


# --- End-to-end test ---


def test_full_integration_flow():
    """Simulates the full opensre execute_actions pipeline against the RL env."""
    scenario = _load_scenario()
    adapter = SREToolAdapter(scenario, seed=42)

    # 1. Create state (what extract_alert node produces)
    state = scenario_to_agent_state(scenario, adapter)
    assert state["mode"] == "investigation"

    # 2. Create simulated actions + sources (what resolve_integrations produces)
    actions = create_simulated_actions(adapter)
    sources = build_simulated_sources(adapter)
    actions_by_name = {a.name: a for a in actions}

    # 3. Simulate execute_actions flow: check availability → extract params → call function
    for action_name in ["get_alerts", "get_service_topology", "get_metrics", "get_error_logs"]:
        action = actions_by_name[action_name]
        assert action.availability_check(sources)
        kwargs = action.parameter_extractor(sources)
        result = action.function(**kwargs)
        assert result["available"] is True

    # 4. Score with gold standard
    gold_label = scenario.gold_root_causes[0].taxonomy_label
    state["root_cause"] = gold_label.replace("_", " ")
    state["root_cause_category"] = gold_label.split(".")[0]
    state["remediation_steps"] = [scenario.gold_remediations[0].action]
    state["executed_hypotheses"] = [
        {"action": "get_alerts", "service": "system"},
        {"action": "get_metrics", "service": adapter.service_names[0]},
        {"action": "get_error_logs", "service": adapter.service_names[0]},
    ]

    result = score_agent_state(state, scenario)
    assert result["reward"] > 0.5
