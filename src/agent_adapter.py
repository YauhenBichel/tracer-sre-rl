"""Agent adapter — exposes the SRE RL environment as callable tools for LLM agents.

This bridges the Gymnasium RL environment with the open-sre-agent tool interface.
An LLM agent (via LangGraph, Claude tool_use, or any function-calling API) can
call these tools to investigate incidents, just as it would call real observability
tools (Grafana, Datadog, PagerDuty).

Usage:
    adapter = SREToolAdapter(scenario, seed=42)
    result = adapter.call_tool("query_metrics", service="postgres-primary",
                               time_start=40, time_end=89)

The adapter maintains episode state internally and tracks the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.environment.env import SREEnvironment
from src.environment.actions import ActionType
from src.models import ScenarioDefinition

# Tool definitions in the format expected by LLM function-calling APIs.
TOOL_DEFINITIONS = [
    {"name": "list_alerts", "description": "List all fired alerts with severity, service, and timestamp.", "parameters": {}},
    {"name": "list_services", "description": "Show the service topology including dependencies.", "parameters": {}},
    {"name": "query_metrics", "description": "Query time-series metrics for a service over a time range.",
     "parameters": {"service": {"type": "string"}, "time_start": {"type": "integer"}, "time_end": {"type": "integer"}}, "required": ["service"]},
    {"name": "query_logs", "description": "Query structured logs for a service over a time range.",
     "parameters": {"service": {"type": "string"}, "time_start": {"type": "integer"}, "time_end": {"type": "integer"}}, "required": ["service"]},
    {"name": "query_traces", "description": "Query distributed traces involving a specific service.",
     "parameters": {"service": {"type": "string"}}, "required": ["service"]},
    {"name": "diagnose", "description": "Submit a root cause diagnosis from the taxonomy.",
     "parameters": {"diagnosis": {"type": "string"}}, "required": ["diagnosis"]},
    {"name": "remediate", "description": "Propose a remediation action.",
     "parameters": {"action": {"type": "string"}}, "required": ["action"]},
]


@dataclass
class ToolResult:
    """Result from a tool call, mirroring real observability tool responses."""
    tool_name: str
    observation: str
    step: int
    done: bool = False
    reward: float = 0.0
    reward_breakdown: Any = None


class SREToolAdapter:
    """Wraps SREEnvironment as callable tools for LLM agents.

    Mirrors the tool interface of observability platforms (Grafana, Datadog,
    PagerDuty) that open-sre-agent integrates with.
    """

    def __init__(self, scenario: ScenarioDefinition, seed: int | None = None):
        self._env = SREEnvironment(scenario=scenario, seed=seed)
        self._service_index = {name: i for i, name in enumerate(self._env.service_names)}
        self._done = False
        self._last_reward = 0.0
        self._last_info: dict = {}

    def reset(self, seed: int | None = None) -> ToolResult:
        """Reset the episode and return the initial alert observation."""
        obs, info = self._env.reset(seed=seed)
        self._done = False
        self._last_reward = 0.0
        self._last_info = info
        return ToolResult(
            tool_name="initial_alert",
            observation=obs["text_observation"],
            step=0,
        )

    def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Call a tool by name with keyword arguments. Returns observation text."""
        if self._done:
            return ToolResult(
                tool_name=tool_name,
                observation="Episode is complete. Call reset() to start a new investigation.",
                step=self._env.step_count,
                done=True,
                reward=self._last_reward,
            )

        action = self._build_action(tool_name, **kwargs)
        obs, reward, terminated, truncated, info = self._env.step(action)

        self._done = terminated or truncated
        self._last_reward = reward
        self._last_info = info

        return ToolResult(
            tool_name=tool_name,
            observation=obs["text_observation"],
            step=self._env.step_count,
            done=self._done,
            reward=reward if self._done else 0.0,
            reward_breakdown=info.get("reward_breakdown"),
        )

    @property
    def available_tools(self) -> list[dict]:
        """Return tool definitions for LLM function-calling APIs."""
        return TOOL_DEFINITIONS

    @property
    def diagnosis_options(self) -> list[str]:
        """Return available diagnosis labels for the current scenario."""
        return self._env.diagnosis_options

    @property
    def remediation_options(self) -> list[str]:
        """Return available remediation actions for the current scenario."""
        return self._env.remediation_options

    @property
    def service_names(self) -> list[str]:
        return self._env.service_names

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reward(self) -> float:
        return self._last_reward

    def _build_action(self, tool_name: str, **kwargs) -> dict:
        """Convert a tool call into a Gymnasium action dict."""
        max_time = self._env.scenario.episode_duration_seconds // 10 - 1

        match tool_name:
            case "list_alerts":
                return {"action_type": ActionType.LIST_ALERTS, "target_service": 0,
                        "time_start": 0, "time_end": max_time,
                        "diagnosis_idx": 0, "remediation_idx": 0}

            case "list_services":
                return {"action_type": ActionType.LIST_SERVICES, "target_service": 0,
                        "time_start": 0, "time_end": max_time,
                        "diagnosis_idx": 0, "remediation_idx": 0}

            case "query_metrics" | "query_logs":
                svc_idx = self._resolve_service(kwargs.get("service", ""))
                action_type = ActionType.QUERY_METRICS if tool_name == "query_metrics" else ActionType.QUERY_LOGS
                return {"action_type": action_type, "target_service": svc_idx,
                        "time_start": kwargs.get("time_start", 0),
                        "time_end": kwargs.get("time_end", max_time),
                        "diagnosis_idx": 0, "remediation_idx": 0}

            case "query_traces":
                svc_idx = self._resolve_service(kwargs.get("service", ""))
                return {"action_type": ActionType.QUERY_TRACES, "target_service": svc_idx,
                        "time_start": 0, "time_end": max_time,
                        "diagnosis_idx": 0, "remediation_idx": 0}

            case "diagnose":
                label = kwargs.get("diagnosis", "")
                idx = self._resolve_diagnosis(label)
                return {"action_type": ActionType.DIAGNOSE, "target_service": 0,
                        "time_start": 0, "time_end": max_time,
                        "diagnosis_idx": idx, "remediation_idx": 0}

            case "remediate":
                action_text = kwargs.get("action", "")
                idx = self._resolve_remediation(action_text)
                return {"action_type": ActionType.REMEDIATE, "target_service": 0,
                        "time_start": 0, "time_end": max_time,
                        "diagnosis_idx": 0, "remediation_idx": idx}

            case _:
                raise ValueError(f"Unknown tool: {tool_name}. Available: {[t['name'] for t in TOOL_DEFINITIONS]}")

    def _resolve_service(self, service: str) -> int:
        """Resolve a service name to its index. Supports partial matching."""
        if service in self._service_index:
            return self._service_index[service]
        # Partial match
        for name, idx in self._service_index.items():
            if service.lower() in name.lower():
                return idx
        return 0

    def _resolve_diagnosis(self, label: str) -> int:
        """Resolve a diagnosis label to its index. Supports exact and partial matching."""
        options = self._env.diagnosis_options
        if label in options:
            return options.index(label)
        # Partial match
        label_lower = label.lower()
        for i, opt in enumerate(options):
            if label_lower in opt.lower():
                return i
        return 0

    def _resolve_remediation(self, action: str) -> int:
        """Resolve a remediation action to its index. Supports partial matching."""
        options = self._env.remediation_options
        if action in options:
            return options.index(action)
        action_lower = action.lower()
        for i, opt in enumerate(options):
            if action_lower in opt.lower():
                return i
        return 0
