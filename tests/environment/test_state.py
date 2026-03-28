"""Tests for EpisodeState and StepResult."""

from src.environment.state import EpisodeState, StepResult


def test_episode_state_initial_values():
    state = EpisodeState()

    assert state.step_count == 0
    assert not state.diagnosed
    assert not state.remediated
    assert state.diagnosis_label is None
    assert state.action_history == []


def test_record_action():
    state = EpisodeState()
    state.record_action(1, "QUERY_METRICS", "api-gateway")
    state.record_action(2, "DIAGNOSE", "api-gateway")

    assert len(state.action_history) == 2
    assert state.action_history[0]["action"] == "QUERY_METRICS"
    assert state.action_history[1]["step"] == 2


def test_step_result_defaults():
    result = StepResult()

    assert result.reward == 0.0
    assert not result.terminated
    assert not result.truncated
    assert result.info == {}


def test_step_result_is_frozen():
    import pytest

    result = StepResult(reward=0.5, terminated=True)
    with pytest.raises(AttributeError):
        result.reward = 0.9
