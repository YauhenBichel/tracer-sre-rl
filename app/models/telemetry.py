"""Telemetry data models — metrics, logs, traces, events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"


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
