"""SREEnvironment — Gymnasium RL environment for AI SRE agent training."""

from __future__ import annotations

import logging
import random

import gymnasium as gym
from gymnasium import spaces

from src.config import distractor_diagnoses, distractor_remediations
from src.constants import EVENT_ALERT, TAXONOMY_SEPARATOR
from src.environment.actions import QUERY_ACTIONS, ActionType
from src.environment.formatter import TelemetryFormatter
from src.environment.state import MAX_SERVICES, EpisodeState, StepResult
from src.evaluation.reward_calculator import RewardCalculator
from src.generators.perturbation import perturb_scenario
from src.generators.telemetry import METRIC_INTERVAL_SECONDS, TelemetryGenerator
from src.models import GeneratedTelemetry, ScenarioDefinition

logger = logging.getLogger(__name__)

MAX_OBS_LENGTH = 4096
MAX_DIAGNOSIS_OPTIONS = 15
MAX_ALERTS_OBSERVED = 20  # observation space bound for alerts_seen
TRUNCATION_REWARD_FACTOR = 0.5


class SREEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}  # type: ignore[assignment]  # noqa: RUF012

    def __init__(self, scenario: ScenarioDefinition, seed: int | None = None, render_mode: str | None = None):
        super().__init__()

        if not scenario.services:
            raise ValueError(f"Scenario '{scenario.name}' has no services defined")
        if not scenario.gold_root_causes:
            raise ValueError(f"Scenario '{scenario.name}' has no gold standard root causes")

        self.scenario = scenario
        self.render_mode = render_mode
        self._seed = seed
        self.num_services = len(scenario.services)
        self.service_names = [s.name for s in scenario.services]
        self._reward_calculator = RewardCalculator()

        self.diagnosis_options = self._build_diagnosis_options()
        self.remediation_options = [r.action for r in scenario.gold_remediations] + distractor_remediations()

        self.observation_space = spaces.Dict(
            {
                "text_observation": spaces.Text(max_length=MAX_OBS_LENGTH, min_length=0),
                "step_count": spaces.Discrete(scenario.max_investigation_steps + 1),
                "alerts_seen": spaces.Discrete(MAX_ALERTS_OBSERVED),
                "services_queried": spaces.MultiBinary(MAX_SERVICES),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(len(ActionType)),
                "target_service": spaces.Discrete(MAX_SERVICES),
                "time_start": spaces.Discrete(scenario.episode_duration_seconds // METRIC_INTERVAL_SECONDS),
                "time_end": spaces.Discrete(scenario.episode_duration_seconds // METRIC_INTERVAL_SECONDS),
                "diagnosis_idx": spaces.Discrete(max(1, len(self.diagnosis_options))),
                "remediation_idx": spaces.Discrete(max(1, len(self.remediation_options))),
            }
        )

        self._state = EpisodeState()
        self._formatter: TelemetryFormatter | None = None
        self._telemetry: GeneratedTelemetry | None = None

        logger.info("SREEnvironment initialised for scenario '%s' with %d services", scenario.name, self.num_services)
        self._start_episode(seed)

    # --- Gymnasium API ---

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._start_episode(seed if seed is not None else self._seed)
        logger.debug("Episode reset for scenario '%s'", self.scenario.name)
        return self._observation(), {"scenario": self.scenario.name}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        self._state.step_count += 1
        action_type = ActionType(action["action_type"])
        target_idx = min(action["target_service"], self.num_services - 1)
        service = self.service_names[target_idx]

        self._state.record_action(self._state.step_count, action_type.name, service)
        logger.debug("Step %d: %s -> %s", self._state.step_count, action_type.name, service)

        self._execute(action_type, action, target_idx, service)

        result = self._evaluate()
        if result.terminated:
            logger.info("Episode completed at step %d with reward %.4f", self._state.step_count, result.reward)
        elif result.truncated:
            logger.info("Episode truncated at step %d", self._state.step_count)

        return self._observation(), result.reward, result.terminated, result.truncated, result.info

    def render(self) -> str | None:  # type: ignore[override]
        if self.render_mode == "human":
            logger.info("Step %d:\n%s", self._state.step_count, self._state.obs_text)
        return self._state.obs_text if self.render_mode == "ansi" else None

    # --- Properties for external access ---

    @property
    def diagnosed(self) -> bool:
        return self._state.diagnosed

    @property
    def step_count(self) -> int:
        return self._state.step_count

    # --- Private ---

    def _start_episode(self, seed: int | None):
        rng = random.Random(seed)
        perturbed = perturb_scenario(self.scenario, rng)
        telemetry = TelemetryGenerator(seed=seed).generate(perturbed)
        self._telemetry = telemetry
        self._formatter = TelemetryFormatter(telemetry, perturbed)
        self._state = EpisodeState(obs_text=self._formatter.initial_observation())

    def _execute(self, action_type: ActionType, action: dict, target_idx: int, service: str):
        s = self._state
        default_end = self.scenario.episode_duration_seconds // METRIC_INTERVAL_SECONDS - 1
        t_start = action.get("time_start", 0)
        t_end = action.get("time_end", default_end)

        if action_type in QUERY_ACTIONS:
            s.services_queried[target_idx] = 1

        assert self._formatter is not None
        assert self._telemetry is not None

        match action_type:
            case ActionType.QUERY_METRICS:
                s.obs_text = self._formatter.metrics(service, t_start, t_end)
            case ActionType.QUERY_LOGS:
                s.obs_text = self._formatter.logs(service, t_start, t_end)
            case ActionType.QUERY_TRACES:
                s.obs_text = self._formatter.traces(service)
            case ActionType.LIST_SERVICES:
                s.obs_text = self._formatter.topology()
            case ActionType.LIST_ALERTS:
                s.obs_text = self._formatter.alerts()
                s.alerts_seen = sum(1 for e in self._telemetry.events if e.event_type == EVENT_ALERT)
            case ActionType.DIAGNOSE:
                idx = min(action.get("diagnosis_idx", 0), len(self.diagnosis_options) - 1)
                s.diagnosed = True
                s.diagnosis_label = self.diagnosis_options[idx]
                s.obs_text = f"Diagnosis submitted: {s.diagnosis_label}"
                if not s.remediated:
                    s.obs_text += "\nNow propose a remediation action."
            case ActionType.REMEDIATE:
                idx = min(action.get("remediation_idx", 0), len(self.remediation_options) - 1)
                s.remediated = True
                s.remediation_action = self.remediation_options[idx]
                s.obs_text = f"Remediation proposed: {s.remediation_action}"

    def _evaluate(self) -> StepResult:
        s = self._state
        if s.diagnosed and s.remediated:
            reward = self._reward_calculator.calculate(
                self.scenario,
                {"label": s.diagnosis_label},
                {"action": s.remediation_action},
                s.step_count,
                self.scenario.max_investigation_steps,
                s.action_history,
            )
            return StepResult(
                reward=reward, terminated=True, info={"reward_breakdown": self._reward_calculator.last_breakdown}
            )

        if s.step_count >= self.scenario.max_investigation_steps:
            reward = 0.0
            if s.diagnosed:
                reward = (
                    self._reward_calculator.calculate(
                        self.scenario,
                        {"label": s.diagnosis_label},
                        None,
                        s.step_count,
                        self.scenario.max_investigation_steps,
                        s.action_history,
                    )
                    * TRUNCATION_REWARD_FACTOR
                )
            return StepResult(reward=reward, truncated=True, info={"reason": "max_steps_exceeded"})

        return StepResult()

    def _observation(self) -> dict:
        s = self._state
        return {
            "text_observation": s.obs_text[:MAX_OBS_LENGTH],
            "step_count": s.step_count,
            "alerts_seen": min(s.alerts_seen, MAX_ALERTS_OBSERVED - 1),
            "services_queried": s.services_queried.copy(),
        }

    def _build_diagnosis_options(self) -> list[str]:
        options = set()
        for rc in self.scenario.gold_root_causes:
            options.add(rc.taxonomy_label)
            parts = rc.taxonomy_label.split(TAXONOMY_SEPARATOR)
            for i in range(1, len(parts)):
                options.add(".".join(parts[:i]))
        for d in distractor_diagnoses():
            if len(options) >= MAX_DIAGNOSIS_OPTIONS:
                break
            options.add(d)
        return sorted(options)
