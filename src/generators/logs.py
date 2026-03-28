"""Log generator — produces event-correlated structured log entries.

Log messages are selected based on active timeline events, not randomly.
A connection_exhaustion event produces "Too many connections" logs,
not "Deadlock" or "Disk space" errors.
"""

from __future__ import annotations

import random

from src.config import log_templates
from src.constants import ERROR_EVENT_TYPES, WARN_EVENT_TYPES, EVENT_CASCADE, SERVICE_TYPE_APPLICATION
from src.models import LogEntry, LogLevel, ServiceDefinition, TimelineEntry

BASE_LOG_PROBABILITY = 0.2
EVENT_LOG_PROBABILITY_INCREMENT = 0.3
MAX_LOG_PROBABILITY = 0.9
ERROR_LEVEL_PROBABILITY = 0.7
WARN_LEVEL_PROBABILITY = 0.5
LOG_TIMESTAMP_JITTER_SECONDS = 9

# Maps event types → log templates that should appear during that event.
# This ensures logs are causally coherent with the scenario timeline.
EVENT_LOG_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "connection_exhaustion": {
        "ERROR": ["Too many connections: max={max}, active={active}"],
        "WARN": ["Connection count approaching limit: {active}/{max}"],
    },
    "error_spike": {
        "ERROR": [
            "Database connection failed: {error}",
            "Unhandled exception in {operation}: {error}",
            "Upstream connection failed: {upstream} - {error}",
        ],
        "WARN": ["Slow database query: {duration}ms for {query_type}"],
    },
    "latency_spike": {
        "WARN": [
            "Slow upstream response from {upstream}: {duration}ms",
            "Slow database query: {duration}ms for {query_type}",
            "Long-running query detected: {duration}ms",
        ],
    },
    "cascade": {
        "ERROR": [
            "Upstream connection failed: {upstream} - {error}",
            "Circuit breaker OPEN for {upstream}: {failure_count} failures in {window}s",
        ],
        "WARN": ["Slow upstream response from {upstream}: {duration}ms"],
    },
    "resource_exhaustion": {
        "ERROR": ["Out of memory, rejecting writes"],
        "WARN": ["Thread pool utilisation high: {active}/{max} threads"],
    },
    "memory_leak": {
        "WARN": ["Memory usage: {used_mb}MB / {max_mb}MB"],
    },
    "disk_fill": {
        "ERROR": ["Disk space critically low: {free_mb}MB remaining"],
        "WARN": ["Disk space critically low: {free_mb}MB remaining"],
    },
    "traffic_ramp": {
        "WARN": ["Connection pool nearing capacity: {active}/{max} active"],
    },
}


class LogGenerator:

    def __init__(self, rng: random.Random):
        self._rng = rng

    def should_generate(self, service: ServiceDefinition, active_events: list[TimelineEntry]) -> bool:
        prob = BASE_LOG_PROBABILITY
        for e in active_events:
            if e.service == service.name or e.event_type == EVENT_CASCADE:
                prob = min(MAX_LOG_PROBABILITY, prob + EVENT_LOG_PROBABILITY_INCREMENT)
        return self._rng.random() < prob

    def generate(self, service: ServiceDefinition, timestamp: int,
                 active_events: list[TimelineEntry]) -> LogEntry | None:
        level = self._pick_level(service, active_events)
        template = self._pick_template(service, level, active_events)
        if template is None:
            return None

        message = self._render(template, service, active_events)
        return LogEntry(
            timestamp=float(timestamp) + self._rng.uniform(0, LOG_TIMESTAMP_JITTER_SECONDS),
            service=service.name,
            level=level,
            message=message,
            attributes={"service_type": service.service_type, "hostname": f"{service.name}-{self._rng.randint(1, 3)}"},
        )

    def _pick_level(self, service: ServiceDefinition, active: list[TimelineEntry]) -> LogLevel:
        def has(types: set[str]) -> bool:
            return any(e.event_type in types and (e.service == service.name or e.event_type == EVENT_CASCADE) for e in active)
        if has(ERROR_EVENT_TYPES) and self._rng.random() < ERROR_LEVEL_PROBABILITY:
            return LogLevel.ERROR
        if has(WARN_EVENT_TYPES) and self._rng.random() < WARN_LEVEL_PROBABILITY:
            return LogLevel.WARN
        return LogLevel.INFO

    def _pick_template(self, service: ServiceDefinition, level: LogLevel,
                       active_events: list[TimelineEntry]) -> str | None:
        """Select a log template that is relevant to the active events.

        If error/warn events are active, use event-specific templates.
        Falls back to generic templates from baselines.yaml only for INFO
        or when no event-specific template exists.
        """
        if level in (LogLevel.ERROR, LogLevel.WARN):
            # Collect templates from all active events affecting this service
            event_templates: list[str] = []
            for e in active_events:
                if e.service != service.name and e.event_type != EVENT_CASCADE:
                    continue
                templates_for_event = EVENT_LOG_TEMPLATES.get(e.event_type, {})
                event_templates.extend(templates_for_event.get(level.value, []))

            if event_templates:
                return self._rng.choice(event_templates)

        # Fallback to generic templates from config
        all_templates = log_templates().get(service.service_type, log_templates().get(SERVICE_TYPE_APPLICATION, {}))
        options = all_templates.get(level.value, all_templates.get("INFO", []))
        return self._rng.choice(options) if options else None

    def _render(self, template: str, service: ServiceDefinition,
                active_events: list[TimelineEntry] | None = None) -> str:
        """Render a log template with values correlated to active events."""
        generators = self._build_value_generators(service, active_events)
        try:
            values = {k: v() for k, v in generators.items() if f"{{{k}}}" in template}
            return template.format_map(values)
        except (KeyError, IndexError):
            return template

    def _build_value_generators(self, service: ServiceDefinition,
                                active_events: list[TimelineEntry] | None) -> dict:
        """Build placeholder value generators correlated to active events.

        During error events, values reflect incident state (high latency,
        5xx status codes, connections near max). During normal operation,
        values reflect healthy baselines.
        """
        rng = self._rng
        has_errors = active_events and any(
            e.event_type in ERROR_EVENT_TYPES and (e.service == service.name or e.event_type == EVENT_CASCADE)
            for e in active_events
        )
        max_conn = service.config.get("max_connections", 100)
        upstream = rng.choice(list(service.dependencies)) if service.dependencies else "unknown"

        return {
            # HTTP request fields
            "method": lambda: rng.choice(["GET", "POST", "PUT", "DELETE"]),
            "path": lambda: rng.choice(["/api/users", "/api/orders", "/api/health"]),
            "status": lambda: rng.choice([500, 503, 502, 200]) if has_errors else rng.choice([200, 200, 200, 201]),
            "duration": lambda: rng.randint(1000, 15000) if has_errors else rng.randint(1, 200),
            "upstream": lambda: upstream,
            # Connection fields — high during incidents
            "active": lambda: rng.randint(int(max_conn * 0.85), max_conn) if has_errors else rng.randint(10, int(max_conn * 0.5)),
            "idle": lambda: rng.randint(0, 5) if has_errors else rng.randint(5, 20),
            "max": lambda: max_conn,
            # Error fields
            "error": lambda: rng.choice(["connection refused", "connection timeout", "connection reset by peer"]),
            "failure_count": lambda: rng.randint(5, 50),
            "window": lambda: rng.choice([30, 60, 120]),
            # Application fields
            "operation": lambda: rng.choice(["getUserById", "createOrder", "updateInventory"]),
            "user_id": lambda: f"usr_{rng.randint(1000, 9999)}",
            "query_type": lambda: rng.choice(["SELECT", "INSERT", "UPDATE"]),
            "client_ip": lambda: f"10.0.{rng.randint(1, 255)}.{rng.randint(1, 255)}",
            # Database fields
            "lag_ms": lambda: rng.randint(50, 500) if has_errors else rng.randint(0, 10),
            "tx1": lambda: f"tx_{rng.randint(100, 999)}",
            "tx2": lambda: f"tx_{rng.randint(100, 999)}",
            "free_mb": lambda: rng.randint(5, 50) if has_errors else rng.randint(200, 500),
            # Queue fields
            "topic": lambda: rng.choice(["orders", "events", "notifications"]),
            "partition": lambda: rng.randint(0, 11),
            "group": lambda: rng.choice(["order-processor", "analytics", "notifier"]),
            "lag": lambda: rng.randint(1000, 10000) if has_errors else rng.randint(0, 100),
            # Resource fields
            "used_mb": lambda: rng.randint(3000, 4000) if has_errors else rng.randint(500, 2000),
            "max_mb": lambda: 4096,
            "rate": lambda: rng.randint(500, 2000) if has_errors else rng.randint(0, 50),
            "percent": lambda: rng.randint(85, 99) if has_errors else rng.randint(30, 60),
        }
