"""Simulated evidence source for open-sre-agent integration.

Plugs into the open-sre-agent's InvestigationAction registry, replacing
real observability tool calls (Grafana, Datadog, CloudWatch) with synthetic
telemetry from the RL environment.

Matches the exact interface expected by:
  - app.agent.tools.tool_actions.investigation_registry.models.InvestigationAction
  - app.agent.nodes.investigate.execution.execute_actions._execute_with_retry()

Key compatibility requirements:
  - parameter_extractor(available_sources) → kwargs dict (MUST NOT be None)
  - availability_check(available_sources) → bool
  - function(**kwargs) → dict with "source", "available" fields
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from src.agent_adapter import SREToolAdapter

# Mirror EvidenceSource from app.agent.state
EvidenceSource = Literal[
    "storage",
    "batch",
    "tracer_web",
    "cloudwatch",
    "aws_sdk",
    "knowledge",
    "grafana",
    "datadog",
    "eks",
    "github",
    "sentry",
]

# Key used in available_sources to hold simulated env config
SIMULATED_SOURCE_KEY = "simulated_rl_env"


@dataclass
class SimulatedAction:
    """Mirrors open-sre-agent's InvestigationAction exactly.

    See: app/agent/tools/tool_actions/investigation_registry/models.py
    """

    name: str
    description: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    use_cases: list[str]
    requires: list[str]
    source: EvidenceSource
    function: Callable[..., dict[str, Any]]
    availability_check: Callable[[dict[str, dict]], bool] | None = None
    parameter_extractor: Callable[[dict[str, dict]], dict[str, Any]] | None = None


def create_simulated_actions(adapter: SREToolAdapter) -> list[SimulatedAction]:
    """Create simulated investigation actions backed by the RL environment.

    Each action has:
      - availability_check that verifies the simulated source is present
      - parameter_extractor that pulls service_name etc. from available_sources
      - function that returns data in the format opensre expects
    """

    def _sim_available(sources: dict) -> bool:
        return bool(sources.get(SIMULATED_SOURCE_KEY, {}).get("connection_verified"))

    def _extract_service(sources: dict) -> dict[str, Any]:
        sim = sources.get(SIMULATED_SOURCE_KEY, {})
        return {
            "service_name": sim.get("service_name", ""),
            "time_range_start": sim.get("time_range_start", 0),
            "time_range_end": sim.get("time_range_end", 89),
        }

    def _extract_service_only(sources: dict) -> dict[str, Any]:
        sim = sources.get(SIMULATED_SOURCE_KEY, {})
        return {"service_name": sim.get("service_name", "")}

    def _extract_empty(sources: dict) -> dict[str, Any]:
        return {}

    # Action names MUST match opensre's EVIDENCE_MAPPERS keys in post_process.py
    # so that summarize_execution_results correctly maps results into evidence.
    return [
        SimulatedAction(
            name="query_grafana_alert_rules",
            description="List all fired alerts with severity, service, and timestamp",
            inputs={},
            outputs={"rules": "Alert rules with severity and timestamps"},
            use_cases=["Initial incident triage", "Understanding alert scope"],
            requires=[],
            source="grafana",
            availability_check=_sim_available,
            parameter_extractor=_extract_empty,
            function=lambda **_: _format_grafana_alerts(adapter.call_tool("list_alerts")),
        ),
        SimulatedAction(
            name="query_grafana_service_names",
            description="Show service dependency graph and service names",
            inputs={},
            outputs={"service_names": "Service names and topology"},
            use_cases=["Understanding service architecture", "Tracing dependency chains"],
            requires=[],
            source="grafana",
            availability_check=_sim_available,
            parameter_extractor=_extract_empty,
            function=lambda **_: _format_grafana_services(adapter.call_tool("list_services")),
        ),
        SimulatedAction(
            name="query_grafana_metrics",
            description="Query time-series metrics for a specific service over a time range",
            inputs={
                "service_name": "Name of the service to query",
                "time_range_start": "Start of time range (index, each unit = 10s)",
                "time_range_end": "End of time range (index)",
            },
            outputs={"metrics": "Aggregated metric statistics (avg, min, max)"},
            use_cases=["Checking service health", "Identifying anomalous metrics"],
            requires=[],
            source="grafana",
            availability_check=_sim_available,
            parameter_extractor=_extract_service,
            function=lambda service_name="", time_range_start=0, time_range_end=89, **_: _format_grafana_metrics(
                adapter.call_tool(
                    "query_metrics",
                    service=service_name,
                    time_start=int(time_range_start),
                    time_end=int(time_range_end),
                ),
            ),
        ),
        SimulatedAction(
            name="query_grafana_logs",
            description="Query structured logs for a specific service",
            inputs={
                "service_name": "Name of the service to query",
                "time_range_start": "Start of time range (index)",
                "time_range_end": "End of time range (index)",
            },
            outputs={"logs": "Log entries with timestamp, level, and message"},
            use_cases=["Finding error messages", "Correlating log events with metrics"],
            requires=[],
            source="grafana",
            availability_check=_sim_available,
            parameter_extractor=_extract_service,
            function=lambda service_name="", time_range_start=0, time_range_end=89, **_: _format_grafana_logs(
                adapter.call_tool(
                    "query_logs", service=service_name, time_start=int(time_range_start), time_end=int(time_range_end)
                ),
            ),
        ),
        SimulatedAction(
            name="query_grafana_traces",
            description="Query distributed traces involving a specific service",
            inputs={"service_name": "Name of the service to query"},
            outputs={"traces": "Trace summaries with span counts and error status"},
            use_cases=["Tracing request flow", "Finding slow spans"],
            requires=[],
            source="grafana",
            availability_check=_sim_available,
            parameter_extractor=_extract_service_only,
            function=lambda service_name="", **_: _format_grafana_traces(
                adapter.call_tool("query_traces", service=service_name),
            ),
        ),
    ]


# Format functions match opensre's EVIDENCE_MAPPERS expectations
# (defined in opensre's post_process.py).


def _format_grafana_alerts(tool_result) -> dict[str, Any]:
    """Match _map_grafana_alert_rules: expects {rules: [...]}."""
    return {
        "source": "grafana_loki",
        "available": True,
        "rules": [{"name": "simulated_alert", "text": tool_result.observation}],
    }


def _format_grafana_services(tool_result) -> dict[str, Any]:
    """Match _map_grafana_service_names: expects {service_names: [...]}."""
    return {
        "source": "grafana_loki",
        "available": True,
        "service_names": [tool_result.observation],
    }


def _format_grafana_metrics(tool_result) -> dict[str, Any]:
    """Match _map_grafana_metrics: expects {metrics: [...]}."""
    return {
        "source": "grafana_loki",
        "available": True,
        "metrics": [{"name": "simulated", "data": tool_result.observation}],
    }


def _format_grafana_logs(tool_result) -> dict[str, Any]:
    """Match _map_grafana_logs: expects {logs: [...], error_logs: [...]}."""
    obs = tool_result.observation
    return {
        "source": "grafana_loki",
        "available": True,
        "logs": [obs],
        "error_logs": [line for line in obs.split("\n") if "ERROR" in line],
        "total_logs": 1,
        "service_name": "",
    }


def _format_grafana_traces(tool_result) -> dict[str, Any]:
    """Match _map_grafana_traces: expects {traces: [...]}."""
    return {
        "source": "grafana_loki",
        "available": True,
        "traces": [{"summary": tool_result.observation}],
    }


def build_simulated_sources(adapter: SREToolAdapter) -> dict[str, dict]:
    """Build available_sources dict matching what opensre parameter_extractors expect.

    This is passed to execute_actions() and then to each action's
    parameter_extractor(available_sources). The extractors look up
    sources[SIMULATED_SOURCE_KEY] for parameters.
    """
    return {
        SIMULATED_SOURCE_KEY: {
            "connection_verified": True,
            "service_name": adapter.service_names[0] if adapter.service_names else "",
            "time_range_start": 0,
            "time_range_end": 89,
            "services": adapter.service_names,
        },
    }
