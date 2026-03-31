"""Agent action model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentAction:
    """An action the agent takes in the SRE environment."""

    action_type: int
    target_service: int = 0
    time_start: int = 0
    time_end: int = 0
    diagnosis_idx: int = 0
    remediation_idx: int = 0

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target_service": self.target_service,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "diagnosis_idx": self.diagnosis_idx,
            "remediation_idx": self.remediation_idx,
        }
