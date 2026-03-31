from __future__ import annotations

from src.constants import (
    METRIC_CPU_PERCENT,
    METRIC_ENQUEUE_RATE,
    METRIC_LATENCY,
    METRIC_MEMORY_PERCENT,
    METRIC_QUERY_RATE,
    METRIC_REQUEST_RATE,
)

from .base import EventEffectHandler

RATE_METRICS = {METRIC_REQUEST_RATE, METRIC_QUERY_RATE, METRIC_ENQUEUE_RATE}
RESOURCE_METRICS = {METRIC_CPU_PERCENT, METRIC_MEMORY_PERCENT}

# How much CPU/memory scales relative to traffic increase
CPU_MEMORY_SCALING_FACTOR = 0.3
# How much latency scales relative to traffic increase
LATENCY_SCALING_FACTOR = 0.5
DEFAULT_TRAFFIC_MULTIPLIER = 2.0


class TrafficRampEffect(EventEffectHandler):
    """Applies traffic ramp effects to metrics during active events."""

    def apply(self, base_value, metric_name, service, event, progress):
        multiplier = event.params.get("multiplier", DEFAULT_TRAFFIC_MULTIPLIER)
        if metric_name in RATE_METRICS:
            return base_value * (1.0 + (multiplier - 1.0) * progress)
        if metric_name in RESOURCE_METRICS:
            return base_value * (1.0 + (multiplier - 1.0) * CPU_MEMORY_SCALING_FACTOR * progress)
        if METRIC_LATENCY in metric_name:
            return base_value * (1.0 + (multiplier - 1.0) * LATENCY_SCALING_FACTOR * progress)
        return None
