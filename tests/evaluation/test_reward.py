"""Tests for the reward calculation system."""

import pytest

from src.evaluation.reward_calculator import RewardCalculator
from src.generators.loader import ScenarioLoader


@pytest.fixture
def calculator():
    return RewardCalculator()


@pytest.fixture
def scenarios():
    return ScenarioLoader().load_all()


def test_perfect_diagnosis(calculator, scenarios):
    scenario = next(s for s in scenarios if "connection" in s.name.lower())
    label = scenario.gold_root_causes[0].taxonomy_label
    score = calculator._diagnosis.score(label=label, scenario=scenario)

    assert score >= 0.9


def test_partial_diagnosis(calculator, scenarios):
    scenario = next(s for s in scenarios if "connection" in s.name.lower())
    score = calculator._diagnosis.score(label="infrastructure.database.replication_lag", scenario=scenario)
    assert 0.3 < score < 0.9


def test_wrong_diagnosis(calculator, scenarios):
    scenario = next(s for s in scenarios if "connection" in s.name.lower())
    score = calculator._diagnosis.score(label="external.cloud_provider", scenario=scenario)
    assert score < 0.3


def test_efficiency_rewards_fast(calculator):
    assert calculator._efficiency.score(steps_taken=3, max_steps=20) > calculator._efficiency.score(
        steps_taken=15, max_steps=20
    )


def test_efficiency_perfect_at_min(calculator):
    assert calculator._efficiency.score(steps_taken=2, max_steps=20) == 1.0


def test_remediation_exact_match(calculator, scenarios):
    scenario = next(s for s in scenarios if "connection" in s.name.lower())
    best = max(scenario.gold_remediations, key=lambda r: r.effectiveness)
    assert calculator._remediation.score(proposed=best.action, scenario=scenario) == best.effectiveness


def test_remediation_restart_low(calculator, scenarios):
    scenario = scenarios[0]
    assert calculator._remediation.score(proposed="Restart all services", scenario=scenario) <= 0.2


def test_safety_penalises_hasty(calculator):
    assert calculator._safety.score(action_history=[{"step": 1, "action": "DIAGNOSE", "target": "a"}]) < 1.0


def test_safety_rewards_thorough(calculator):
    history = [
        {"step": 1, "action": "LIST_ALERTS", "target": "system"},
        {"step": 2, "action": "QUERY_METRICS", "target": "a"},
        {"step": 3, "action": "QUERY_METRICS", "target": "b"},
        {"step": 4, "action": "QUERY_LOGS", "target": "a"},
        {"step": 5, "action": "DIAGNOSE", "target": "a"},
    ]
    assert calculator._safety.score(action_history=history) == 1.0


def test_full_reward(calculator, scenarios):
    scenario = next(s for s in scenarios if "connection" in s.name.lower())
    label = scenario.gold_root_causes[0].taxonomy_label
    best_rem = max(scenario.gold_remediations, key=lambda r: r.effectiveness)
    reward = calculator.calculate(
        scenario=scenario,
        diagnosis={"label": label},
        remediation={"action": best_rem.action},
        steps_taken=5,
        max_steps=20,
        action_history=[
            {"step": 1, "action": "LIST_ALERTS", "target": "system"},
            {"step": 2, "action": "QUERY_METRICS", "target": "a"},
            {"step": 3, "action": "QUERY_METRICS", "target": "b"},
            {"step": 4, "action": "QUERY_LOGS", "target": "a"},
            {"step": 5, "action": "DIAGNOSE", "target": "a"},
        ],
    )

    assert reward > 0.7
    assert calculator.last_breakdown.diagnosis_score >= 0.9
