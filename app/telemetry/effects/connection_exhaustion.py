from __future__ import annotations

from app.constants import METRIC_CONNECTION

from .base import EventEffectHandler

MAX_CONN_UTILISATION = 0.98
DEFAULT_MAX_CONNECTIONS = 100


class ConnectionExhaustionEffect(EventEffectHandler):
    """Applies connection exhaustion effects to metrics during active events."""

    def apply(self, base_value, metric_name, service, event, progress):
        if METRIC_CONNECTION in metric_name:
            max_conn = service.config.get("max_connections", DEFAULT_MAX_CONNECTIONS)
            return base_value + (max_conn * MAX_CONN_UTILISATION - base_value) * progress
        return None
