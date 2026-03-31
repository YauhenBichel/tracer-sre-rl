from __future__ import annotations

from app.constants import METRIC_LATENCY

from .base import EventEffectHandler

DEFAULT_LATENCY_SPIKE_FACTOR = 10.0


class LatencySpikeEffect(EventEffectHandler):
    """Applies latency spike effects to metrics during active events."""

    def apply(self, base_value, metric_name, service, event, progress):
        if METRIC_LATENCY in metric_name:
            return base_value * (1.0 + (event.params.get("factor", DEFAULT_LATENCY_SPIKE_FACTOR) - 1.0) * progress)
        return None
