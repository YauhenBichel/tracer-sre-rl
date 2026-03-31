"""Tests for crawler quality scoring."""

from app.incidents.quality import quality_score


def test_all_fields_present():
    item = {"name": "incident", "updates": [1, 2], "severity": "high"}
    assert quality_score(item, ["name", "updates", "severity"]) == 1.0


def test_no_fields_present():
    assert quality_score({}, ["name", "updates"]) == 0.0


def test_partial_fields():
    item = {"name": "incident"}
    assert quality_score(item, ["name", "updates"]) == 0.5


def test_pipe_or_syntax():
    item = {"most-recent-update": {"text": "fixed"}}
    assert quality_score(item, ["updates|most-recent-update"]) == 1.0


def test_pipe_or_neither_present():
    assert quality_score({}, ["updates|most-recent-update"]) == 0.0


def test_empty_field_specs():
    assert quality_score({"a": 1}, []) == 0.0
