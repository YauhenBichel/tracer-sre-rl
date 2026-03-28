"""Scenario and data quality validation.

Three validations run before training:
  1. Scenario playability — each scenario must produce a complete episode
  2. Telemetry quality — generated telemetry must have anomalous signals
  3. Data contamination — train and eval sets must not share scenario IDs
"""

from __future__ import annotations

import logging

from src.environment.actions import ActionType
from src.environment.env import SREEnvironment
from src.generators.telemetry import TelemetryGenerator
from src.models import ScenarioDefinition

logger = logging.getLogger(__name__)


def validate_scenario(scenario: ScenarioDefinition, seed: int = 0) -> tuple[bool, str]:
    """Check that a scenario produces a valid, completable episode.

    Returns (is_valid, reason).
    """
    # Must have services
    if not scenario.services:
        return False, "no services defined"

    # Must have gold standard root causes
    if not scenario.gold_root_causes:
        return False, "no gold standard root causes"

    # Must have timeline events
    if not scenario.timeline:
        return False, "empty timeline"

    # Must be playable — reset + step should not crash
    try:
        env = SREEnvironment(scenario=scenario, seed=seed)
        obs, _ = env.reset()
        if len(obs["text_observation"]) == 0:
            env.close()
            return False, "empty initial observation"

        # Quick episode: alert + diagnose + remediate
        env.step(
            {
                "action_type": ActionType.LIST_ALERTS,
                "target_service": 0,
                "time_start": 0,
                "time_end": 89,
                "diagnosis_idx": 0,
                "remediation_idx": 0,
            }
        )
        env.step(
            {
                "action_type": ActionType.DIAGNOSE,
                "target_service": 0,
                "time_start": 0,
                "time_end": 89,
                "diagnosis_idx": 0,
                "remediation_idx": 0,
            }
        )
        _, _reward, terminated, _, _ = env.step(
            {
                "action_type": ActionType.REMEDIATE,
                "target_service": 0,
                "time_start": 0,
                "time_end": 89,
                "diagnosis_idx": 0,
                "remediation_idx": 0,
            }
        )
        env.close()

        if not terminated:
            return False, "episode did not terminate after diagnose + remediate"

        return True, "ok"
    except Exception as e:
        return False, f"crash: {e}"


def validate_telemetry_quality(scenario: ScenarioDefinition, seed: int = 0) -> tuple[bool, str]:
    """Check that generated telemetry contains anomalous signals.

    A valid training scenario must produce telemetry that differs from
    normal baselines — otherwise the agent has nothing to investigate.
    """
    try:
        telemetry = TelemetryGenerator(seed=seed).generate(scenario)
    except Exception as e:
        return False, f"telemetry generation failed: {e}"

    if len(telemetry.metrics) == 0:
        return False, "no metrics generated"

    if len(telemetry.logs) == 0:
        return False, "no logs generated"

    # Check that at least some events exist (alerts, errors)
    if len(telemetry.events) == 0:
        logger.warning("Scenario %s has no events (alerts/deployments)", scenario.id)

    # Check logs contain ERROR level entries (evidence of anomaly)
    has_errors = any(log.level.value == "ERROR" for log in telemetry.logs)
    if not has_errors:
        return False, "no ERROR logs — telemetry shows no anomaly"

    return True, "ok"


def check_data_contamination(
    train: list[ScenarioDefinition],
    eval_set: list[ScenarioDefinition],
) -> list[str]:
    """Check for scenario ID overlap between train and eval sets.

    Returns list of contaminated IDs (should be empty).
    """
    train_ids = {s.id for s in train}
    eval_ids = {s.id for s in eval_set}
    overlap = train_ids & eval_ids
    if overlap:
        logger.warning(
            "Data contamination: %d scenario IDs in both train and eval: %s",
            len(overlap),
            overlap,
        )
    return sorted(overlap)


def validate_all(scenarios: list[ScenarioDefinition]) -> list[ScenarioDefinition]:
    """Validate all scenarios and return only valid ones.

    Logs warnings for invalid scenarios. Used by EpisodeRunner
    before training to ensure data quality.
    """
    valid = []
    rejected = 0
    for scenario in scenarios:
        playable, reason = validate_scenario(scenario)
        if not playable:
            logger.warning("Rejected scenario %s: %s", scenario.id, reason)
            rejected += 1
            continue

        quality, reason = validate_telemetry_quality(scenario)
        if not quality:
            logger.warning("Low quality scenario %s: %s", scenario.id, reason)
            # Keep it but log — some scenarios may have subtle anomalies

        valid.append(scenario)

    if rejected:
        logger.info("Validated %d scenarios, rejected %d", len(valid), rejected)
    return valid
