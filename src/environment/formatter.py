"""TelemetryFormatter — renders telemetry data into human-readable text for agent observations."""

from __future__ import annotations

from collections import defaultdict

from src.constants import EVENT_ALERT, META_ALERT_NAME, META_SEVERITY, META_SEVERITY_UNKNOWN
from src.generators.telemetry import METRIC_INTERVAL_SECONDS
from src.models import GeneratedTelemetry, ScenarioDefinition, SpanStatus

MAX_LOG_RESULTS = 30
MAX_TRACE_RESULTS = 10


class TelemetryFormatter:
    def __init__(self, telemetry: GeneratedTelemetry, scenario: ScenarioDefinition):
        self._telemetry = telemetry
        self._scenario = scenario

    def initial_observation(self) -> str:
        alerts = [e for e in self._telemetry.events if e.event_type == EVENT_ALERT]
        services = ", ".join(s.name for s in self._scenario.services)
        if not alerts:
            return f"System anomaly detected.\nAvailable services: {services}\nBegin investigation."
        first = min(alerts, key=lambda a: a.timestamp)
        return (
            f"INCIDENT ALERT\n"
            f"Alert: {first.metadata.get(META_ALERT_NAME, first.description)}\n"
            f"Service: {first.service}\n"
            f"Severity: {first.metadata.get(META_SEVERITY, META_SEVERITY_UNKNOWN)}\n"
            f"Time: {first.timestamp:.0f}s\n"
            f"Description: {first.description}\n\n"
            f"Available services: {services}\n"
            f"You have {self._scenario.max_investigation_steps} investigation steps."
        )

    def topology(self) -> str:
        lines = ["=== Service Topology ==="]
        for svc in self._scenario.services:
            deps = ", ".join(svc.dependencies) if svc.dependencies else "none"
            lines.append(f"  {svc.name} ({svc.service_type}) -> deps: [{deps}]")
        return "\n".join(lines)

    def metrics(self, service: str, start_idx: int, end_idx: int) -> str:
        t_start, t_end = self._time_range(start_idx, end_idx)
        samples = [m for m in self._telemetry.metrics if m.service == service and t_start <= m.timestamp <= t_end]
        if not samples:
            return f"No metrics found for {service} in [{t_start}s, {t_end}s]"

        grouped: dict[str, list[float]] = defaultdict(list)
        for s in samples:
            grouped[s.metric_name].append(s.value)

        lines = [f"=== Metrics for {service} [{t_start}s - {t_end}s] ==="]
        for name, values in sorted(grouped.items()):
            lines.append(
                f"  {name}: avg={sum(values) / len(values):.2f}, min={min(values):.2f}, max={max(values):.2f} ({len(values)} samples)"
            )
        return "\n".join(lines)

    def logs(self, service: str, start_idx: int, end_idx: int) -> str:
        t_start, t_end = self._time_range(start_idx, end_idx)
        entries = sorted(
            (log for log in self._telemetry.logs if log.service == service and t_start <= log.timestamp <= t_end),
            key=lambda entry: entry.timestamp,
        )
        if not entries:
            return f"No logs found for {service} in [{t_start}s, {t_end}s]"

        lines = [f"=== Logs for {service} [{t_start}s - {t_end}s] ==="]
        lines.extend(
            f"  [{entry.timestamp:.0f}s] [{entry.level.value}] {entry.message}" for entry in entries[:MAX_LOG_RESULTS]
        )
        if len(entries) > MAX_LOG_RESULTS:
            lines.append(f"  ... and {len(entries) - MAX_LOG_RESULTS} more entries")
        return "\n".join(lines)

    def traces(self, service: str) -> str:
        spans = [s for s in self._telemetry.traces if s.service == service]
        if not spans:
            return f"No traces found for {service}"

        grouped: dict[str, list] = defaultdict(list)
        for span in spans:
            grouped[span.trace_id].append(span)

        lines = [f"=== Traces involving {service} ({len(spans)} spans) ==="]
        for trace_id, trace_spans in list(grouped.items())[:MAX_TRACE_RESULTS]:
            total = sum(s.duration_ms for s in trace_spans)
            has_error = any(s.status == SpanStatus.ERROR for s in trace_spans)
            lines.append(
                f"  Trace {trace_id[:8]}...: {len(trace_spans)} spans, total={total:.1f}ms, status={'ERROR' if has_error else 'OK'}"
            )
            for span in sorted(trace_spans, key=lambda s: s.start_time):
                status = "ERROR" if span.status == SpanStatus.ERROR else "ok"
                lines.append(f"    [{span.service}] {span.operation} {span.duration_ms:.1f}ms {status}")
        return "\n".join(lines)

    def alerts(self) -> str:
        alert_events = sorted(
            (e for e in self._telemetry.events if e.event_type == "alert"),
            key=lambda a: a.timestamp,
        )
        if not alert_events:
            return "No alerts fired."
        lines = ["=== Fired Alerts ==="]
        lines.extend(
            f"  [{a.timestamp:.0f}s] [{a.metadata.get(META_SEVERITY, '?')}] {a.service}: {a.metadata.get(META_ALERT_NAME, a.description)}"
            for a in alert_events
        )
        return "\n".join(lines)

    def _time_range(self, start_idx: int, end_idx: int) -> tuple[int, int]:
        return (
            start_idx * METRIC_INTERVAL_SECONDS,
            min((end_idx + 1) * METRIC_INTERVAL_SECONDS, self._scenario.episode_duration_seconds),
        )
