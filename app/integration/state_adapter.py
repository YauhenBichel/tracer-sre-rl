"""AgentState adapter — bridges open-sre-agent state and RL environment.

Translates between:
  - open-sre-agent's AgentState (TypedDict from app.agent.state)
  - RL environment's ScenarioDefinition + observations + reward

Matches the exact field structure of AgentState including:
  investigation_started_at, alert_json, resolved_integrations, etc.
"""

from __future__ import annotations

import time
from typing import Any

from app.evaluation.reward_calculator import RewardCalculator
from app.integration.agent_adapter import SREToolAdapter
from app.integration.evidence_source import build_simulated_sources
from app.models import ScenarioDefinition


def scenario_to_agent_state(scenario: ScenarioDefinition, adapter: SREToolAdapter) -> dict[str, Any]:
    """Convert a scenario + initial observation into an open-sre-agent AgentState.

    Creates the same state shape that open-sre-agent's make_initial_state() produces,
    but sourced from the RL scenario instead of a real Slack/PagerDuty alert.

    See: app/agent/state.py — AgentState TypedDict and make_initial_state()
    """
    initial = adapter.reset()

    alert_events = [e for e in scenario.timeline if e.event_type == "alert"]
    first_alert = alert_events[0] if alert_events else None

    alert_name = first_alert.params.get("alert_name", scenario.name) if first_alert else scenario.name
    severity = first_alert.params.get("severity", "unknown") if first_alert else "unknown"
    pipeline_name = first_alert.service if first_alert else scenario.services[0].name

    # Build alert_json matching what extract_alert node expects
    alert_json = {
        "alert_name": alert_name,
        "pipeline_name": pipeline_name,
        "severity": severity,
        "source": "simulated_rl_env",
        "scenario_id": scenario.id,
        "description": scenario.description,
        "services": [s.name for s in scenario.services],
        "initial_observation": initial.observation,
    }

    # Build resolved_integrations matching what detect_sources expects
    # See: app/agent/nodes/resolve_integrations/node.py → _classify_integrations
    resolved_integrations = {
        "grafana": {
            "endpoint": "simulated://rl-env",
            "api_key": "simulated",
            "integration_id": "rl-env-grafana",
            "connection_verified": True,
        },
        "cloudwatch": {
            "log_group": "simulated://rl-env",
            "region": "us-east-1",
            "integration_id": "rl-env-cloudwatch",
            "connection_verified": True,
        },
        "datadog": {
            "api_key": "simulated",
            "app_key": "simulated",
            "site": "datadoghq.com",
            "integration_id": "rl-env-datadog",
            "connection_verified": True,
        },
        "_all": [],
    }

    # available_sources matching what parameter_extractors look up
    available_sources = build_simulated_sources(adapter)

    return {
        # Mode
        "mode": "investigation",
        "route": "",
        "is_noise": False,
        # Auth (empty for simulated)
        "org_id": "",
        "user_id": "",
        "user_email": "",
        "user_name": "",
        "organization_slug": "",
        # Chat mode (empty for investigation)
        "messages": [],
        # Alert input
        "alert_name": alert_name,
        "pipeline_name": pipeline_name,
        "severity": severity,
        # Empty string so opensre's detect_sources enables all integrations
        "alert_source": "",
        "raw_alert": alert_json,
        "alert_json": alert_json,
        # Investigation planning
        "planned_actions": [],
        "plan_rationale": "",
        "available_sources": available_sources,
        "available_action_names": [
            "query_grafana_alert_rules",
            "query_grafana_service_names",
            "query_grafana_metrics",
            "query_grafana_logs",
            "query_grafana_traces",
        ],
        "resolved_integrations": resolved_integrations,
        # Evidence
        "context": {},
        "evidence": {},
        # Analysis
        "root_cause": "",
        "root_cause_category": "",
        "validated_claims": [],
        "non_validated_claims": [],
        "validity_score": 0.0,
        "investigation_recommendations": [],
        "remediation_steps": [],
        "investigation_loop_count": 0,
        "hypotheses": [],
        "executed_hypotheses": [],
        "investigation_started_at": time.monotonic(),
        # Slack (empty for simulated)
        "slack_context": {},
        # LangGraph context
        "thread_id": "",
        "run_id": "",
        "_auth_token": "",  # nosec B105 — empty placeholder for opensre AgentState
        # Outputs
        "slack_message": "",
        "problem_md": "",
        "summary": "",
        "problem_report": {},
    }


def score_agent_state(state: dict[str, Any], scenario: ScenarioDefinition) -> dict[str, Any]:
    """Score a completed open-sre-agent investigation against the scenario's gold standard.

    Maps the agent's free-text root_cause and remediation_steps back to taxonomy
    labels, then computes reward via the same RewardCalculator used by the Gymnasium env.
    """
    calculator = RewardCalculator()

    diagnosis = _map_root_cause_to_label(
        state.get("root_cause", ""),
        state.get("root_cause_category", ""),
        scenario,
    )
    remediation = _map_remediation(state.get("remediation_steps", []), scenario)
    action_history = _build_action_history(state.get("executed_hypotheses", []))
    steps_taken = len(action_history) + 2  # +2 for diagnose + remediate

    if not diagnosis and not remediation:
        return {
            "reward": 0.0,
            "breakdown": None,
            "mapped_diagnosis": "",
            "mapped_remediation": "",
            "steps": steps_taken,
            "validity_score": state.get("validity_score", 0.0),
        }

    reward = calculator.calculate(
        scenario=scenario,
        diagnosis={"label": diagnosis} if diagnosis else None,
        remediation={"action": remediation} if remediation else None,
        steps_taken=steps_taken,
        max_steps=scenario.max_investigation_steps,
        action_history=action_history,
    )

    return {
        "reward": reward,
        "breakdown": calculator.last_breakdown,
        "mapped_diagnosis": diagnosis,
        "mapped_remediation": remediation,
        "steps": steps_taken,
        "validity_score": state.get("validity_score", 0.0),
    }


_SYNONYM_MAP = {
    "disk_full": ["disk usage", "disk full", "disk space", "no space", "out of disk"],
    "connection_pool": ["connection pool", "too many connections", "max connections", "connection exhaust"],
    "memory": ["memory", "oom", "out of memory", "heap"],
    "leak": ["leak", "growing steadily", "increasing over time"],
    "cpu_saturation": ["cpu", "cpu usage", "cpu saturation", "high load"],
    "dns": ["dns", "name resolution", "resolve"],
    "cascading_failure": ["cascading", "cascade", "downstream", "circuit breaker"],
    "upstream_timeout": ["timeout", "upstream", "slow dependency"],
    "partition": ["network partition", "unreachable", "connectivity"],
    "container_crash": ["crash", "restart", "crashloop", "oom kill"],
    "replication_lag": ["replication", "replica lag", "sync delay"],
    "iops_throttling": ["iops", "throttl", "disk i/o", "io wait"],
    "resource_exhaustion": ["resource exhaust", "exhaustion", "saturation"],
}


def _match_by_label_parts(text: str, scenario: ScenarioDefinition) -> str | None:
    """Match by checking if taxonomy label parts appear in the text."""
    for gold in scenario.gold_root_causes:
        if any(part.replace("_", " ") in text for part in gold.taxonomy_label.split(".")):
            return gold.taxonomy_label
    return None


def _match_by_synonyms(text: str, scenario: ScenarioDefinition) -> str | None:
    """Match using synonym expansion (e.g., 'disk usage' matches 'disk_full')."""
    for gold in scenario.gold_root_causes:
        leaf = gold.taxonomy_label.split(".")[-1]
        if any(syn in text for syn in _SYNONYM_MAP.get(leaf, [])):
            return gold.taxonomy_label
    return None


def _match_by_category(category: str, scenario: ScenarioDefinition) -> str | None:
    """Match the LLM's category string against taxonomy labels."""
    cat_lower = category.lower()
    for gold in scenario.gold_root_causes:
        if cat_lower in gold.taxonomy_label.lower():
            return gold.taxonomy_label
    for leaf, synonyms in _SYNONYM_MAP.items():
        if any(syn in cat_lower for syn in synonyms):
            for gold in scenario.gold_root_causes:
                if leaf in gold.taxonomy_label:
                    return gold.taxonomy_label
    return None


def _map_root_cause_to_label(root_cause: str, category: str, scenario: ScenarioDefinition) -> str:
    """Map the agent's free-text root cause to a taxonomy label."""
    if not root_cause:
        return ""

    text = (root_cause + " " + category).lower()

    return (
        _match_by_label_parts(text, scenario)
        or _match_by_synonyms(text, scenario)
        or (category and _match_by_category(category, scenario))
        or category
        or ""
    )


def _map_remediation(steps: list[str], scenario: ScenarioDefinition) -> str:
    """Map agent's remediation steps to a gold standard remediation action."""
    if not steps:
        return ""

    combined = " ".join(steps).lower()
    best_match = ""
    best_score = 0
    for gold in scenario.gold_remediations:
        keywords = gold.action.lower().split()
        matches = sum(1 for kw in keywords if kw in combined)
        if matches > best_score:
            best_score = matches
            best_match = gold.action

    return best_match or (steps[0] if steps else "")


def _build_action_history(executed_hypotheses: list[dict]) -> list[dict]:
    """Convert open-sre-agent's executed_hypotheses into RL action history format."""
    history: list[dict[str, str | int]] = []
    for hyp in executed_hypotheses:
        action_name = hyp.get("action", hyp.get("name", ""))
        target = hyp.get("service", hyp.get("target", "unknown"))

        rl_action = "QUERY_METRICS"
        if "log" in action_name.lower():
            rl_action = "QUERY_LOGS"
        elif "trace" in action_name.lower():
            rl_action = "QUERY_TRACES"
        elif "alert" in action_name.lower():
            rl_action = "LIST_ALERTS"
        elif "topology" in action_name.lower() or "service" in action_name.lower():
            rl_action = "LIST_SERVICES"

        history.append({"step": len(history) + 1, "action": rl_action, "target": target})

    return history
