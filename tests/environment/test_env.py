"""Tests for the Gymnasium RL environment."""

import pytest

from src.environment.actions import ActionType
from src.environment.env import SREEnvironment
from src.generators.loader import ScenarioLoader


@pytest.fixture
def scenarios():
    return ScenarioLoader().load_all()


@pytest.fixture
def env(scenarios):
    env = SREEnvironment(scenario=scenarios[0], seed=42)
    yield env
    env.close()


def _action(action_type, target=0, time_start=0, time_end=89, diagnosis_idx=0, remediation_idx=0):
    return {
        "action_type": action_type,
        "target_service": target,
        "time_start": time_start,
        "time_end": time_end,
        "diagnosis_idx": diagnosis_idx,
        "remediation_idx": remediation_idx,
    }


def test_env_reset(env):
    obs, info = env.reset()

    assert "text_observation" in obs
    assert obs["step_count"] == 0
    assert "scenario" in info


def test_env_query_metrics(env):
    env.reset()
    obs, reward, terminated, _, _ = env.step(_action(ActionType.QUERY_METRICS))

    assert "Metrics for" in obs["text_observation"] or "No metrics" in obs["text_observation"]
    assert not terminated
    assert reward == 0.0


def test_env_query_logs(env):
    env.reset()
    obs, _, _, _, _ = env.step(_action(ActionType.QUERY_LOGS))
    assert obs["step_count"] == 1


def test_env_list_services(env):
    env.reset()
    obs, _, _, _, _ = env.step(_action(ActionType.LIST_SERVICES))
    assert "Service Topology" in obs["text_observation"]


def test_env_list_alerts(env):
    env.reset()
    obs, _, _, _, _ = env.step(_action(ActionType.LIST_ALERTS))
    assert "Alert" in obs["text_observation"] or "No alerts" in obs["text_observation"]


def test_env_full_episode(env):
    env.reset()
    env.step(_action(ActionType.LIST_ALERTS))
    env.step(_action(ActionType.QUERY_METRICS, target=0, time_start=40))
    env.step(_action(ActionType.QUERY_METRICS, target=1, time_start=40))
    _obs, reward, terminated, _, _ = env.step(_action(ActionType.DIAGNOSE))

    assert not terminated

    _obs, reward, terminated, _, _ = env.step(_action(ActionType.REMEDIATE))
    assert terminated
    assert reward > 0


def test_env_truncation(scenarios):
    scenario = scenarios[0]
    env = SREEnvironment(scenario=scenario, seed=42)
    env.reset()
    for i in range(scenario.max_investigation_steps):
        _, _, _, truncated, _ = env.step(_action(ActionType.QUERY_METRICS, target=i % len(scenario.services)))
        if truncated:
            break

    assert truncated
    env.close()


def test_all_scenarios_work(scenarios):
    assert len(scenarios) > 0
    for scenario in scenarios:
        env = SREEnvironment(scenario=scenario, seed=42)
        obs, info = env.reset()
        assert obs["text_observation"]
        assert info["scenario"] == scenario.name
        obs, _, _, _, _ = env.step(_action(ActionType.LIST_SERVICES))
        assert obs["step_count"] == 1
        env.close()
