"""Immutable value objects used across the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --- Enums ---

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"


# --- Telemetry data ---

@dataclass(frozen=True)
class MetricSample:
    timestamp: float
    service: str
    metric_name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LogEntry:
    timestamp: float
    service: str
    level: LogLevel
    message: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_time: float
    duration_ms: float
    status: SpanStatus
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    timestamp: float
    event_type: str
    service: str
    description: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedTelemetry:
    metrics: tuple[MetricSample, ...] = ()
    logs: tuple[LogEntry, ...] = ()
    traces: tuple[TraceSpan, ...] = ()
    events: tuple[Event, ...] = ()


# --- Scenario definition ---

@dataclass(frozen=True)
class ServiceDefinition:
    name: str
    service_type: str
    dependencies: tuple[str, ...] = ()
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineEntry:
    time_offset_seconds: int
    event_type: str
    service: str | None = None
    params: dict = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class GoldStandardRootCause:
    taxonomy_label: str
    relevance: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldStandardRemediation:
    action: str
    effectiveness: float


@dataclass(frozen=True)
class AgentAction:
    """An action the agent takes in the SRE environment."""
    action_type: int
    target_service: int = 0
    time_start: int = 0
    time_end: int = 0
    diagnosis_idx: int = 0
    remediation_idx: int = 0

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target_service": self.target_service,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "diagnosis_idx": self.diagnosis_idx,
            "remediation_idx": self.remediation_idx,
        }


@dataclass(frozen=True)
class ScenarioDefinition:
    id: str
    name: str
    description: str
    taxonomy_labels: tuple[str, ...]
    services: tuple[ServiceDefinition, ...]
    timeline: tuple[TimelineEntry, ...]
    gold_root_causes: tuple[GoldStandardRootCause, ...]
    gold_remediations: tuple[GoldStandardRemediation, ...]
    difficulty: float = 0.5
    max_investigation_steps: int = 20
    episode_duration_seconds: int = 900
