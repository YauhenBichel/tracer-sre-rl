"""Scores remediation quality against gold-standard actions.

Exact match returns the gold-standard effectiveness score.
Generic actions (restart, scale) receive small partial credit
since they sometimes work but don't address root cause.
"""

from __future__ import annotations

from src.config import reward_config
from src.models import ScenarioDefinition


class RemediationScorer:
    def __init__(self):
        self._partial_credit: dict[str, float] = reward_config()["remediation"]["partial_credit"]

    def score(self, proposed: str, scenario: ScenarioDefinition) -> float:
        for gold in scenario.gold_remediations:
            if proposed == gold.action:
                return gold.effectiveness

        proposed_lower = proposed.lower()
        for keyword, credit in self._partial_credit.items():
            if keyword in proposed_lower:
                return credit

        return 0.0
