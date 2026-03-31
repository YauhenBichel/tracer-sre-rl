"""Episode runner — runs episodes and collects trajectories for RL training.

Collects full trajectories (observation, action, reward) that can be:
  1. Exported as JSONL for LLM fine-tuning (GRPO/DPO on opensre's LLM)
  2. Used directly with Gymnasium-compatible RL libraries (Stable-Baselines3, etc.)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.incidents.scenario_loader import ScenarioLoader
from app.models import ScenarioDefinition
from app.rl_env.actions import ActionType
from app.rl_env.env import SREEnvironment

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """A single step in an episode trajectory."""

    observation: str
    action_type: str
    target_service: str
    reward: float = 0.0
    done: bool = False


@dataclass
class EpisodeResult:
    """Collected trajectory from a single episode."""

    scenario_id: str
    scenario_name: str
    seed: int
    steps: int
    reward: float
    diagnosis_score: float = 0.0
    efficiency_score: float = 0.0
    remediation_score: float = 0.0
    safety_score: float = 0.0
    trajectory: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class TrainingStats:
    """Aggregate statistics across episodes."""

    episodes_run: int = 0
    total_reward: float = 0.0
    rewards_by_scenario: dict[str, list[float]] = field(default_factory=dict)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.episodes_run)

    def record(self, result: EpisodeResult) -> None:
        self.episodes_run += 1
        self.total_reward += result.reward
        self.rewards_by_scenario.setdefault(result.scenario_id, []).append(result.reward)


class EpisodeRunner:
    """Runs episodes and collects trajectories for RL training.

    Supports two data sources:
      - Builtin YAML scenarios (hand-authored, always available)
      - Real incident data from VOID/Aiops-Dataset (via IncidentReplaySource)

    And difficulty-based curriculum:
      - Early training: only easy scenarios (max_difficulty low)
      - Later training: all scenarios including hard ones
    """

    def __init__(
        self,
        scenarios_dir: str | None = None,
        max_difficulty: float | None = None,
        scenarios: list[ScenarioDefinition] | None = None,
        incident_db_path: str | None = None,
    ):
        loader = ScenarioLoader(scenarios_dir) if scenarios_dir else ScenarioLoader()
        if scenarios is not None:
            self._scenarios = scenarios + loader.load_all()
        elif incident_db_path is not None:
            from app.incidents.repository.sqlite_repository import (
                SqliteIncidentRepository,
            )
            from app.incidents.scenario_generator import ScenarioGenerator

            repo = SqliteIncidentRepository(incident_db_path)
            try:
                incidents = repo.load_all()
            finally:
                repo.close()
            generated = ScenarioGenerator().batch_generate(incidents)
            self._scenarios = generated + loader.load_all()
        else:
            self._scenarios = loader.load_all(max_difficulty=max_difficulty)

        if max_difficulty is not None:
            self._scenarios = [s for s in self._scenarios if s.difficulty <= max_difficulty]

        if not self._scenarios:
            raise ValueError("No scenarios found")

        # Validate scenarios — reject unplayable, check telemetry quality
        from app.training.validation import check_data_contamination, validate_all

        self._scenarios = validate_all(self._scenarios)
        if not self._scenarios:
            raise ValueError("No valid scenarios after validation")

        # Train/eval split — reserve 20% for evaluation (never used during training)
        self._rng_split = random.Random(0)  # fixed seed for reproducible split
        self._rng_split.shuffle(self._scenarios)
        split_idx = max(1, int(len(self._scenarios) * 0.8))
        self._train_scenarios = self._scenarios[:split_idx]
        self._eval_scenarios = self._scenarios[split_idx:]

        # Data contamination check
        contaminated = check_data_contamination(self._train_scenarios, self._eval_scenarios)
        if contaminated:
            logger.warning(
                "Data contamination: %d scenario IDs in both train and eval",
                len(contaminated),
            )

        self._rng = random.Random()
        self._stats = TrainingStats()
        logger.info(
            "EpisodeRunner initialised with %d scenarios (max_difficulty=%s)",
            len(self._scenarios),
            max_difficulty,
        )

    @property
    def stats(self) -> TrainingStats:
        return self._stats

    @property
    def scenarios(self) -> list[ScenarioDefinition]:
        return self._scenarios

    @property
    def train_scenarios(self) -> list[ScenarioDefinition]:
        return self._train_scenarios

    @property
    def eval_scenarios(self) -> list[ScenarioDefinition]:
        return self._eval_scenarios

    def sample_scenario(self) -> ScenarioDefinition:
        """Sample a random scenario from the training set (not eval)."""
        return self._rng.choice(self._train_scenarios)

    def evaluate(self, num_episodes: int = 20, agent_fn: Callable | None = None) -> float:
        """Run episodes on the held-out eval set and return average reward.

        Use this to measure generalisation: train on train_scenarios,
        evaluate on eval_scenarios the agent has never seen.
        """
        rewards = []
        for i in range(num_episodes):
            scenario = self._eval_scenarios[i % len(self._eval_scenarios)]
            try:
                result = self.run_episode(scenario=scenario, seed=i + 10000, agent_fn=agent_fn)
                rewards.append(result.reward)
            except Exception as e:
                logger.warning("Eval episode %d failed: %s", i, e)
        return sum(rewards) / max(1, len(rewards))

    def run_episode(
        self,
        scenario: ScenarioDefinition | None = None,
        seed: int | None = None,
        agent_fn: Callable[..., Any] | None = None,
    ) -> EpisodeResult:
        """Run a single episode and collect the full trajectory.

        The trajectory contains every (observation, action, reward) step,
        which can be exported for LLM fine-tuning or RL policy updates.
        """
        if scenario is None:
            scenario = self.sample_scenario()
        if seed is None:
            seed = self._rng.randint(0, 2**31)
        if agent_fn is None:
            agent_fn = _random_agent

        env = SREEnvironment(scenario=scenario, seed=seed)
        obs, info = env.reset()
        trajectory: list[TrajectoryStep] = []

        step = 0
        reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent_fn(obs, env)
            action_type = ActionType(action["action_type"]).name
            target_idx = min(action["target_service"], env.num_services - 1)
            target_service = env.service_names[target_idx]

            prev_obs_text = obs["text_observation"]
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            trajectory.append(
                TrajectoryStep(
                    observation=prev_obs_text,
                    action_type=action_type,
                    target_service=target_service,
                    reward=reward if (terminated or truncated) else 0.0,
                    done=terminated or truncated,
                )
            )

        bd = info.get("reward_breakdown")
        result = EpisodeResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            seed=seed,
            steps=step,
            reward=reward,
            diagnosis_score=bd.diagnosis_score if bd else 0.0,
            efficiency_score=bd.efficiency_score if bd else 0.0,
            remediation_score=bd.remediation_score if bd else 0.0,
            safety_score=bd.safety_score if bd else 0.0,
            trajectory=trajectory,
        )
        self._stats.record(result)
        env.close()
        return result

    def run_batch(self, num_episodes: int, agent_fn: Callable[..., Any] | None = None) -> list[EpisodeResult]:
        """Run multiple episodes and return all results with trajectories."""
        results = []
        failed = 0
        progress_interval = max(1, num_episodes // 10)
        for i in range(num_episodes):
            try:
                result = self.run_episode(agent_fn=agent_fn)
                results.append(result)
            except Exception as e:
                failed += 1
                logger.warning("[episode=%d] failed: %s", i + 1, e)
            if (i + 1) % progress_interval == 0 or i == num_episodes - 1:
                pct = (i + 1) / num_episodes * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(
                    f"  {bar} {i + 1}/{num_episodes} episodes, avg reward: {self._stats.avg_reward:.3f}",
                    end="\r",
                )
        print()  # newline after progress bar
        if failed:
            print(f"  ⚠ {failed}/{num_episodes} episodes failed (see logs)")
        return results


def _random_agent(_obs: dict, env: SREEnvironment) -> dict:
    """Baseline random agent for testing the training loop."""
    rng = random.Random()
    max_time = env.scenario.episode_duration_seconds // 10 - 1

    if not env.diagnosed:
        if env.step_count < 3 or rng.random() > 0.3:
            action_type = rng.choice(
                [
                    ActionType.QUERY_METRICS,
                    ActionType.QUERY_LOGS,
                    ActionType.LIST_SERVICES,
                    ActionType.LIST_ALERTS,
                ]
            )
        else:
            action_type = ActionType.DIAGNOSE
    else:
        action_type = ActionType.REMEDIATE

    return {
        "action_type": action_type,
        "target_service": rng.randint(0, env.num_services - 1),
        "time_start": rng.randint(0, max_time // 2),
        "time_end": rng.randint(max_time // 2, max_time),
        "diagnosis_idx": rng.randint(0, max(0, len(env.diagnosis_options) - 1)),
        "remediation_idx": rng.randint(0, max(0, len(env.remediation_options) - 1)),
    }
