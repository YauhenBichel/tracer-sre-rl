"""RewardCalculator — composes individual scorers into a multi-dimensional reward."""

from __future__ import annotations

from src.config import reward_config
from src.models import ScenarioDefinition
from src.evaluation.reward_breakdown import RewardBreakdown, RewardWeights
from src.evaluation.scorers import (
    DiagnosisScorer, EfficiencyScorer, RemediationScorer, SafetyScorer,
)


class RewardCalculator:

    def __init__(self):
        self._diagnosis = DiagnosisScorer()
        self._efficiency = EfficiencyScorer()
        self._remediation = RemediationScorer()
        self._safety = SafetyScorer()
        w = reward_config()["weights"]
        self._weights = RewardWeights(
            diagnosis=w["diagnosis"],
            efficiency=w["efficiency"],
            remediation=w["remediation"],
            safety=w["safety"],
        )
        self.last_breakdown: RewardBreakdown | None = None

    def calculate(
        self,
        scenario: ScenarioDefinition,
        diagnosis: dict | None,
        remediation: dict | None,
        steps_taken: int,
        max_steps: int,
        action_history: list[dict],
    ) -> float:
        diagnosis_score = self._diagnosis.score(label=diagnosis["label"], scenario=scenario) if diagnosis else 0.0
        raw_efficiency = self._efficiency.score(steps_taken=steps_taken, max_steps=max_steps)
        # Efficiency is only valuable when the diagnosis is correct:
        # being fast but wrong should not be rewarded.
        efficiency_score = raw_efficiency * diagnosis_score

        self.last_breakdown = RewardBreakdown(
            diagnosis_score=diagnosis_score,
            efficiency_score=efficiency_score,
            remediation_score=self._remediation.score(proposed=remediation["action"], scenario=scenario) if remediation else 0.0,
            safety_score=self._safety.score(action_history=action_history),
            weights=self._weights,
        )
        return self.last_breakdown.total
