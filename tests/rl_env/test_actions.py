"""Tests for action types."""

from __future__ import annotations

from app.rl_env.actions import QUERY_ACTIONS, ActionType


def test_action_type_has_seven_actions():
    assert len(ActionType) == 7


def test_query_actions_are_subset():
    result = all(action in ActionType for action in QUERY_ACTIONS)

    assert result


def test_action_type_values_are_unique():
    values = [a.value for a in ActionType]

    assert len(values) == len(set(values))
