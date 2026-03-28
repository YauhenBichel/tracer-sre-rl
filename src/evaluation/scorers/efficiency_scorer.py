"""Scores investigation efficiency — fewer steps yields a higher score.

An agent that diagnoses correctly in ≤5 steps (list alerts, 2 queries,
diagnose, remediate) gets a perfect score. Score declines linearly
to zero at max_steps.

The efficiency score is scaled by the diagnosis score in RewardCalculator
so that being fast but wrong provides no efficiency benefit.
"""

from __future__ import annotations

from src.config import reward_config


class EfficiencyScorer:
    def __init__(self):
        self._ideal_steps: int = reward_config()["efficiency"]["ideal_steps"]

    def score(self, steps_taken: int, max_steps: int) -> float:
        if max_steps == 0 or steps_taken <= self._ideal_steps:
            return 1.0
        remaining = max_steps - self._ideal_steps
        if remaining <= 0:
            return 0.0
        return max(0.0, 1.0 - (steps_taken - self._ideal_steps) / remaining)
