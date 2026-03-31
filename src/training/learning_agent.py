"""Tabular Q-learning agent that improves over episodes.

Demonstrates that the reward signal actually drives learning:
the agent starts random, then learns which investigation actions
lead to higher rewards for different alert patterns.

State features (discretised):
  - alert_service_type: web-server, application, database, cache, queue
  - num_services_queried: 0, 1, 2, 3+
  - has_seen_alerts: bool
  - has_seen_topology: bool

Actions: LIST_ALERTS, LIST_SERVICES, QUERY_METRICS(svc_i), QUERY_LOGS(svc_i), DIAGNOSE, REMEDIATE

This is NOT a production RL algorithm — it's a proof that the environment
and reward signal can drive measurable improvement.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

import numpy as np

from src.environment.actions import ActionType
from src.environment.env import SREEnvironment

logger = logging.getLogger(__name__)

# Discretised action space (simplified from full 7-action x service space)
ACTIONS = [
    ("LIST_ALERTS", 0),
    ("LIST_SERVICES", 0),
    ("QUERY_METRICS", 0),
    ("QUERY_METRICS", 1),
    ("QUERY_METRICS", 2),
    ("QUERY_LOGS", 0),
    ("QUERY_LOGS", 1),
    ("DIAGNOSE", 0),
    ("REMEDIATE", 0),
]

ACTION_TYPE_MAP = {
    "LIST_ALERTS": ActionType.LIST_ALERTS,
    "LIST_SERVICES": ActionType.LIST_SERVICES,
    "QUERY_METRICS": ActionType.QUERY_METRICS,
    "QUERY_LOGS": ActionType.QUERY_LOGS,
    "DIAGNOSE": ActionType.DIAGNOSE,
    "REMEDIATE": ActionType.REMEDIATE,
}


def _extract_state(obs: dict, env: SREEnvironment) -> tuple:
    """Extract discretised state features from observation."""
    text = obs.get("text_observation", "").lower()
    queried = int(np.sum(obs.get("services_queried", np.zeros(1))))

    has_alerts = "alert" in text or obs.get("alerts_seen", 0) > 0
    has_topology = "topology" in text or "deps" in text

    # Detect dominant signal in observation
    signal = "none"
    if "error" in text or "500" in text or "fail" in text:
        signal = "error"
    elif "latency" in text or "slow" in text:
        signal = "latency"
    elif "connection" in text or "pool" in text:
        signal = "connection"
    elif "disk" in text or "memory" in text:
        signal = "resource"

    return (
        min(queried, 3),
        has_alerts,
        has_topology,
        env.diagnosed,
        signal,
    )


class QLearningAgent:
    """Tabular Q-learning agent for SRE investigation.

    Learns a Q-table mapping (state, action) → expected reward.
    Uses epsilon-greedy exploration with decay.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.q_table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(ACTIONS)))
        self._rng = random.Random(seed)
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._prev_state: tuple | None = None
        self._prev_action_idx: int | None = None

    def act(self, obs: dict, env: SREEnvironment) -> dict:
        """Select an action using epsilon-greedy policy."""
        state = _extract_state(obs, env)

        # Epsilon-greedy
        if self._rng.random() < self.epsilon:
            action_idx = self._rng.randint(0, len(ACTIONS) - 1)
        else:
            action_idx = int(np.argmax(self.q_table[state]))

        # Force valid transitions
        if env.diagnosed and not env.remediated:
            action_idx = 8  # REMEDIATE
        elif env.step_count >= env.scenario.max_investigation_steps - 2 and not env.diagnosed:
            action_idx = 7  # DIAGNOSE

        self._prev_state = state
        self._prev_action_idx = action_idx

        action_name, target = ACTIONS[action_idx]
        max_time = env.scenario.episode_duration_seconds // 10 - 1
        target = min(target, env.num_services - 1)

        return {
            "action_type": ACTION_TYPE_MAP[action_name],
            "target_service": target,
            "time_start": max_time // 2,
            "time_end": max_time,
            "diagnosis_idx": self._rng.randint(0, max(0, len(env.diagnosis_options) - 1)),
            "remediation_idx": self._rng.randint(0, max(0, len(env.remediation_options) - 1)),
        }

    def update(self, obs: dict, env: SREEnvironment, reward: float, done: bool) -> None:
        """Update Q-table from (state, action, reward, next_state)."""
        if self._prev_state is None:
            return

        next_state = _extract_state(obs, env)
        best_next = np.max(self.q_table[next_state]) if not done else 0.0

        old_q = self.q_table[self._prev_state][self._prev_action_idx]
        self.q_table[self._prev_state][self._prev_action_idx] = old_q + self.lr * (
            reward + self.discount * best_next - old_q
        )

    def end_episode(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._prev_state = None
        self._prev_action_idx = None
