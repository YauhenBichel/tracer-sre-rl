"""Immutable breakdown of reward components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    diagnosis: float
    efficiency: float
    remediation: float
    safety: float


@dataclass(frozen=True)
class RewardBreakdown:
    diagnosis_score: float
    efficiency_score: float
    remediation_score: float
    safety_score: float
    weights: RewardWeights

    @property
    def total(self) -> float:
        return (
            self.weights.diagnosis * self.diagnosis_score
            + self.weights.efficiency * self.efficiency_score
            + self.weights.remediation * self.remediation_score
            + self.weights.safety * self.safety_score
        )
