"""Action types for the SRE environment."""

from __future__ import annotations

from enum import IntEnum


class ActionType(IntEnum):
    QUERY_METRICS = 0
    QUERY_LOGS = 1
    QUERY_TRACES = 2
    LIST_SERVICES = 3
    LIST_ALERTS = 4
    DIAGNOSE = 5
    REMEDIATE = 6


QUERY_ACTIONS = frozenset({ActionType.QUERY_METRICS, ActionType.QUERY_LOGS, ActionType.QUERY_TRACES})
