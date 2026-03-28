"""Scores safety compliance — penalises poor investigation behaviour.

Two penalties:
- Hasty diagnosis: diagnosing before querying enough telemetry sources.
- Tunnel vision: only looking at one service before concluding.
"""

from __future__ import annotations

from src.config import reward_config
from src.environment.actions import ActionType, QUERY_ACTIONS

_QUERY_ACTION_NAMES = frozenset(a.name for a in QUERY_ACTIONS)
_DIAGNOSE_ACTION_NAME = ActionType.DIAGNOSE.name


class SafetyScorer:

    def __init__(self):
        cfg = reward_config()["safety"]
        self._min_queries: int = cfg["min_queries_before_diagnosis"]
        self._min_services: int = cfg.get("min_services_queried", 2)
        self._hasty_penalty: float = cfg["hasty_diagnosis_penalty"]
        self._tunnel_penalty: float = cfg["tunnel_vision_penalty"]

    def score(self, action_history: list[dict]) -> float:
        queries_count, services_queried = self._count_queries_before_diagnosis(action_history)

        result = 1.0
        if queries_count < self._min_queries:
            result -= self._hasty_penalty
        # Tunnel vision: diagnosing without querying enough distinct services
        if len(services_queried) < self._min_services:
            result -= self._tunnel_penalty

        return max(0.0, result)

    @staticmethod
    def _count_queries_before_diagnosis(history: list[dict]) -> tuple[int, set[str]]:
        """Count query actions and unique services queried before the first DIAGNOSE."""
        count = 0
        services = set()
        for action in history:
            if action["action"] == _DIAGNOSE_ACTION_NAME:
                break
            if action["action"] in _QUERY_ACTION_NAMES:
                count += 1
                services.add(action["target"])
        return count, services
