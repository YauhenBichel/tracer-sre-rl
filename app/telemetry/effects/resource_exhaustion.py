from __future__ import annotations

from app.constants import METRIC_CPU_PERCENT, METRIC_MEMORY_PERCENT

from .base import EventEffectHandler

DEFAULT_RESOURCE_METRICS = {METRIC_CPU_PERCENT, METRIC_MEMORY_PERCENT}
DEFAULT_RESOURCE_CEILING_PERCENT = 95


class ResourceExhaustionEffect(EventEffectHandler):
    """Applies resource exhaustion effects to metrics during active events."""

    def apply(self, base_value, metric_name, service, event, progress):
        target = event.params.get("metric", "")
        if metric_name == target or (not target and metric_name in DEFAULT_RESOURCE_METRICS):
            ceiling = event.params.get("ceiling", DEFAULT_RESOURCE_CEILING_PERCENT)
            return base_value + (ceiling - base_value) * progress
        return None
