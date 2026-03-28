"""Tests for incident replay — converting real incidents to RL scenarios."""

from __future__ import annotations

from src.crawler.models import NormalisedIncident
from src.generators.incident_replay import IncidentReplaySource


def _make_incident(**overrides) -> NormalisedIncident:
    defaults = {
        "id": "void-123",
        "source": "void",
        "source_url": "https://example.com",
        "title": "Database connection pool exhaustion during traffic spike",
        "summary": "Traffic spike caused DB connections to max out",
        "taxonomy_labels": ["infrastructure.database.connection_pool"],
        "quality_score": 0.7,
        "impact": {"severity": "critical"},
    }
    defaults.update(overrides)
    return NormalisedIncident(**defaults)


def test_load_from_incidents():
    source = IncidentReplaySource()
    incidents = [_make_incident(id=f"void-{i}") for i in range(3)]
    count = source.load_from_incidents(incidents)
    assert count == 3
    assert len(source.scenarios) == 3


def test_filters_by_quality():
    source = IncidentReplaySource()
    incidents = [
        _make_incident(id="good", quality_score=0.8),
        _make_incident(id="bad", quality_score=0.1),
    ]
    count = source.load_from_incidents(incidents, min_quality=0.5)
    assert count == 1


def test_filters_by_taxonomy():
    source = IncidentReplaySource()
    incidents = [
        _make_incident(id="labeled", taxonomy_labels=["infrastructure.database.connection_pool"]),
        _make_incident(id="unlabeled", taxonomy_labels=[]),
    ]
    count = source.load_from_incidents(incidents)
    assert count == 1


def test_generated_scenarios_are_playable():
    """Scenarios from real incidents should work in the RL environment."""
    source = IncidentReplaySource()
    source.load_from_incidents([_make_incident()])

    scenario = source.scenarios[0]
    assert scenario.id.startswith("generated-")
    assert len(scenario.services) >= 2
    assert len(scenario.timeline) >= 2
    assert len(scenario.gold_root_causes) >= 1

    # Actually run it in the environment
    from src.environment.env import SREEnvironment
    env = SREEnvironment(scenario=scenario, seed=42)
    obs, info = env.reset()
    assert len(obs["text_observation"]) > 0
    env.close()


def test_load_builtin_scenarios():
    source = IncidentReplaySource()
    count = source.load_builtin_scenarios()
    assert count >= 5


def test_get_all_sorted_by_difficulty():
    source = IncidentReplaySource()
    source.load_builtin_scenarios()
    source.load_from_incidents([
        _make_incident(id="easy", impact={"severity": "minor"}),
        _make_incident(id="hard", impact={"severity": "critical"}),
    ])
    all_scenarios = source.get_all()
    difficulties = [s.difficulty for s in all_scenarios]
    assert difficulties == sorted(difficulties)
