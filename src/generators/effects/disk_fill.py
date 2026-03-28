from __future__ import annotations

from src.constants import METRIC_DISK_USAGE_PERCENT
from .base import EventEffectHandler

DEFAULT_DISK_FILL_TARGET_PERCENT = 98


class DiskFillEffect(EventEffectHandler):

    def apply(self, base_value, metric_name, service, event, progress):
        if metric_name == METRIC_DISK_USAGE_PERCENT:
            return base_value + (event.params.get("target_percent", DEFAULT_DISK_FILL_TARGET_PERCENT) - base_value) * progress
        return None
