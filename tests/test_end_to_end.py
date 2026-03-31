"""End-to-end test — full pipeline from scenario to reward.

Validates the complete flow:
  scenario YAML → telemetry generation → environment → agent actions → reward
"""

from __future__ import annotations

from app.incidents.scenario_loader import ScenarioLoader
from app.rl_env.actions import ActionType
from app.rl_env.env import SREEnvironment


def test_full_episode_produces_nonzero_reward():
    """Run a full investigation episode and verify reward is computed."""
    scenario = ScenarioLoader().load_all()[0]
    env = SREEnvironment(scenario=scenario, seed=42)
    obs, info = env.reset()

    assert len(obs["text_observation"]) > 0
    assert info["scenario"] == scenario.name

    max_time = scenario.episode_duration_seconds // 10 - 1

    # Investigate: list alerts, query 2 services
    env.step(
        {
            "action_type": ActionType.LIST_ALERTS,
            "target_service": 0,
            "time_start": 0,
            "time_end": max_time,
            "diagnosis_idx": 0,
            "remediation_idx": 0,
        }
    )
    env.step(
        {
            "action_type": ActionType.QUERY_METRICS,
            "target_service": 0,
            "time_start": max_time // 2,
            "time_end": max_time,
            "diagnosis_idx": 0,
            "remediation_idx": 0,
        }
    )
    env.step(
        {
            "action_type": ActionType.QUERY_LOGS,
            "target_service": 1,
            "time_start": max_time // 2,
            "time_end": max_time,
            "diagnosis_idx": 0,
            "remediation_idx": 0,
        }
    )

    # Diagnose with correct label
    gold_label = scenario.gold_root_causes[0].taxonomy_label
    diag_idx = env.diagnosis_options.index(gold_label)
    _obs, reward, terminated, _truncated, info = env.step(
        {
            "action_type": ActionType.DIAGNOSE,
            "target_service": 0,
            "time_start": 0,
            "time_end": max_time,
            "diagnosis_idx": diag_idx,
            "remediation_idx": 0,
        }
    )
    assert not terminated  # not done until remediation

    # Remediate
    _obs, reward, terminated, _truncated, info = env.step(
        {
            "action_type": ActionType.REMEDIATE,
            "target_service": 0,
            "time_start": 0,
            "time_end": max_time,
            "diagnosis_idx": 0,
            "remediation_idx": 0,
        }
    )

    assert terminated
    assert reward > 0.5
    assert "reward_breakdown" in info
    bd = info["reward_breakdown"]
    assert bd.diagnosis_score > 0.9
    assert bd.safety_score == 1.0

    env.close()


def test_different_seeds_produce_different_telemetry():
    """Perturbation ensures each seed produces different episodes."""
    scenario = ScenarioLoader().load_all()[0]

    env1 = SREEnvironment(scenario=scenario, seed=1)
    _obs1, _ = env1.reset()
    env1.step(
        {
            "action_type": ActionType.QUERY_METRICS,
            "target_service": 0,
            "time_start": 40,
            "time_end": 89,
            "diagnosis_idx": 0,
            "remediation_idx": 0,
        }
    )
    text1 = env1._state.obs_text
    env1.close()

    env2 = SREEnvironment(scenario=scenario, seed=2)
    _obs2, _ = env2.reset()
    env2.step(
        {
            "action_type": ActionType.QUERY_METRICS,
            "target_service": 0,
            "time_start": 40,
            "time_end": 89,
            "diagnosis_idx": 0,
            "remediation_idx": 0,
        }
    )
    text2 = env2._state.obs_text
    env2.close()

    assert text1 != text2


def test_all_scenarios_run_to_completion():
    """Every scenario YAML can be loaded and run without errors."""
    scenarios = ScenarioLoader().load_all()
    assert len(scenarios) > 0
    for scenario in scenarios:
        env = SREEnvironment(scenario=scenario, seed=42)
        _obs, _info = env.reset()

        # Quick episode: list alerts + diagnose + remediate
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
                "action_type": ActionType.QUERY_METRICS,
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
        _, reward, terminated, _, _ = env.step(
            {
                "action_type": ActionType.REMEDIATE,
                "target_service": 0,
                "time_start": 0,
                "time_end": 89,
                "diagnosis_idx": 0,
                "remediation_idx": 0,
            }
        )

        assert terminated, f"Scenario {scenario.name} did not terminate"
        assert reward >= 0, f"Scenario {scenario.name} produced negative reward"
        env.close()
