from __future__ import annotations

from .base import EventEffectHandler

# Default target multiplier when no explicit target_value is provided
DEFAULT_SYMPTOM_MULTIPLIER = 3


class SymptomEffect(EventEffectHandler):
    def apply(self, base_value, metric_name, service, event, progress):
        if metric_name == event.params.get("metric", ""):
            target = event.params.get("target_value", base_value * DEFAULT_SYMPTOM_MULTIPLIER)
            return base_value + (target - base_value) * progress
        return None
