"""Trace generator — produces distributed trace spans across service topology."""

from __future__ import annotations

import random
import uuid

from app.config import base_latency_ms
from app.constants import ERROR_EVENT_TYPES, EVENT_CASCADE, SERVICE_TYPE_WEB_SERVER
from app.models import ServiceDefinition, SpanStatus, TimelineEntry, TraceSpan

MAX_TRACE_DEPTH = 5
DEFAULT_BASE_LATENCY_MS = 10
LATENCY_NOISE_FACTOR = 0.2  # gaussian std dev as fraction of base latency
ERROR_LATENCY_MIN_MULTIPLIER = 3
ERROR_LATENCY_MAX_MULTIPLIER = 20
MIN_SPAN_DURATION_MS = 0.1
# Base error probability when an error event is active but has no explicit target_rate
DEFAULT_ERROR_STATUS_PROBABILITY = 0.3


class TraceGenerator:
    def __init__(self, rng: random.Random):
        self._rng = rng

    def generate(
        self, services: tuple[ServiceDefinition, ...], timestamp: int, active_events: list[TimelineEntry]
    ) -> list[TraceSpan]:
        trace_id = uuid.uuid4().hex[:32]
        entry = next((s for s in services if s.service_type == SERVICE_TYPE_WEB_SERVER), services[0])
        service_map = {s.name: s for s in services}
        return self._build_spans(entry, service_map, trace_id, None, timestamp, active_events, depth=0)

    def _build_spans(
        self,
        service: ServiceDefinition,
        service_map: dict[str, ServiceDefinition],
        trace_id: str,
        parent_id: str | None,
        timestamp: int,
        active: list[TimelineEntry],
        depth: int,
    ) -> list[TraceSpan]:
        if depth > MAX_TRACE_DEPTH:
            return []

        # Derive error probability from the actual error_rate in active events,
        # so traces are correlated with metrics (not independent coin flips)
        error_prob = self._error_probability(service, active)
        is_affected = error_prob > 0

        base = base_latency_ms().get(service.service_type, DEFAULT_BASE_LATENCY_MS)
        duration = base + self._rng.gauss(0, base * LATENCY_NOISE_FACTOR)
        status = SpanStatus.OK
        if is_affected:
            duration *= self._rng.uniform(ERROR_LATENCY_MIN_MULTIPLIER, ERROR_LATENCY_MAX_MULTIPLIER)
            if self._rng.random() < error_prob:
                status = SpanStatus.ERROR

        span_id = uuid.uuid4().hex[:16]
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            service=service.name,
            operation=f"{service.service_type}.handle_request",
            start_time=float(timestamp) + self._rng.uniform(0, 1),
            duration_ms=round(max(MIN_SPAN_DURATION_MS, duration), 2),
            status=status,
            attributes={"service_type": service.service_type},
        )
        result = [span]
        for dep_name in service.dependencies:
            if dep := service_map.get(dep_name):
                result.extend(self._build_spans(dep, service_map, trace_id, span_id, timestamp, active, depth + 1))
        return result

    @staticmethod
    def _error_probability(service: ServiceDefinition, active: list[TimelineEntry]) -> float:
        """Derive error probability from active events affecting this service.

        Uses the event's target_rate parameter when available (e.g., error_spike
        with target_rate=0.35), so trace errors correlate with metric error rates.
        For events without an explicit rate (e.g., cascade), uses a default.
        """
        max_prob = 0.0
        for e in active:
            if e.event_type not in ERROR_EVENT_TYPES:
                continue
            if e.service != service.name and e.event_type != EVENT_CASCADE:
                continue
            # Use the event's target error rate if specified, otherwise default
            prob = e.params.get("target_rate", DEFAULT_ERROR_STATUS_PROBABILITY)
            max_prob = max(max_prob, prob)
        return max_prob
