from __future__ import annotations

from src.constants import METRIC_ERROR_RATE, METRIC_LATENCY

from .base import EventEffectHandler

LATENCY_MULTIPLIER = 5.0
ERROR_RATE_FLOOR = 0.1


class CascadeEffect(EventEffectHandler):
    def apply(self, base_value, metric_name, service, event, progress):
        if METRIC_LATENCY in metric_name:
            return base_value * (1.0 + LATENCY_MULTIPLIER * progress)
        if METRIC_ERROR_RATE in metric_name:
            return max(base_value, ERROR_RATE_FLOOR * progress)
        return None
