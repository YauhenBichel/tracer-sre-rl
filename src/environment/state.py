"""Episode state for the SRE environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_SERVICES = 10


@dataclass
class EpisodeState:
    """Mutable state for a single episode. Reset at the start of each episode."""

    obs_text: str = ""
    step_count: int = 0
    services_queried: np.ndarray = field(default_factory=lambda: np.zeros(MAX_SERVICES, dtype=np.int8))
    alerts_seen: int = 0
    diagnosed: bool = False
    remediated: bool = False
    diagnosis_label: str | None = None
    remediation_action: str | None = None
    action_history: list[dict] = field(default_factory=list)

    def record_action(self, step: int, action_name: str, target: str):
        self.action_history.append({"step": step, "action": action_name, "target": target})


@dataclass(frozen=True)
class StepResult:
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)
