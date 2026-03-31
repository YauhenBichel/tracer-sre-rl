from __future__ import annotations

from app.constants import METRIC_MEMORY_PERCENT

from .base import EventEffectHandler

SECONDS_PER_MINUTE = 60.0
DEFAULT_LEAK_RATE_PER_MINUTE = 2.0
DEFAULT_LEAK_DURATION_SECONDS = 300


class MemoryLeakEffect(EventEffectHandler):
    """Applies memory leak effects to metrics during active events."""

    def apply(self, base_value, metric_name, service, event, progress):
        if metric_name == METRIC_MEMORY_PERCENT:
            rate = event.params.get("rate_per_minute", DEFAULT_LEAK_RATE_PER_MINUTE)
            duration = event.params.get("duration_seconds", DEFAULT_LEAK_DURATION_SECONDS)
            return base_value + rate * (progress * duration) / SECONDS_PER_MINUTE
        return None
