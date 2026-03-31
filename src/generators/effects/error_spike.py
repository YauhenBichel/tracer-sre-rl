from __future__ import annotations

from src.constants import METRIC_ERROR_RATE

from .base import EventEffectHandler

DEFAULT_TARGET_ERROR_RATE = 0.3


class ErrorSpikeEffect(EventEffectHandler):
    """Applies error spike effects to metrics during active events."""

    def apply(self, base_value, metric_name, service, event, progress):
        if METRIC_ERROR_RATE in metric_name:
            return base_value + (event.params.get("target_rate", DEFAULT_TARGET_ERROR_RATE) - base_value) * progress
        return None
