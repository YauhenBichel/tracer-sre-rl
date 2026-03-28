"""Tests for the incident-to-scenario generator."""

import tempfile

import yaml

from src.crawler.models import NormalisedIncident
from src.crawler.scenario_generator import ScenarioGenerator


def _make_incident(**overrides) -> NormalisedIncident:
    defaults = {
        "id": "test-001",
        "source": "test",
        "source_url": "https://example.com/incident/1",
        "title": "Database outage due to connection pool exhaustion",
        "summary": "Traffic spike caused DB connections to max out",
        "taxonomy_labels": ["infrastructure.database.connection_pool"],
        "quality_score": 0.6,
        "impact": {"severity": "critical"},
    }
    defaults.update(overrides)
    return NormalisedIncident(**defaults)


def test_generate_produces_valid_scenario():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident())

    assert scenario["id"] == "generated-test-001"
    assert "name" in scenario
    assert "services" in scenario
    assert "timeline" in scenario
    assert "gold_standard" in scenario
    assert len(scenario["services"]) >= 2
    assert len(scenario["timeline"]) >= 2


def test_generate_includes_taxonomy_labels():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident())

    assert "infrastructure.database.connection_pool" in scenario["taxonomy_labels"]


def test_generate_picks_database_topology_for_db_incident():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident(taxonomy_labels=["infrastructure.database.connection_pool"]))

    service_types = {s["service_type"] for s in scenario["services"]}

    assert "database" in service_types


def test_generate_picks_application_topology_for_app_incident():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident(taxonomy_labels=["application.memory.leak"]))

    service_types = {s["service_type"] for s in scenario["services"]}

    assert "application" in service_types


def test_generate_uses_generic_timeline_for_unknown_taxonomy():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident(taxonomy_labels=["operational.deployment.bad_deploy"]))

    event_types = {e["event_type"] for e in scenario["timeline"]}

    assert "normal_traffic" in event_types


def test_generate_handles_no_taxonomy_labels():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident(taxonomy_labels=[]))

    assert scenario["taxonomy_labels"] == []
    assert len(scenario["timeline"]) >= 2


def test_difficulty_from_severity():
    gen = ScenarioGenerator()
    critical = gen.generate(_make_incident(impact={"severity": "critical"}))
    minor = gen.generate(_make_incident(impact={"severity": "minor"}))

    assert critical["difficulty"] > minor["difficulty"]


def test_gold_standard_has_root_cause():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident())

    gold = scenario["gold_standard"]

    assert len(gold["root_causes"]) >= 1
    assert gold["root_causes"][0]["taxonomy_label"] == "infrastructure.database.connection_pool"


def test_gold_standard_includes_remediations():
    gen = ScenarioGenerator()
    incident = _make_incident(remediation=["Increase max_connections", "Add connection pooler"])
    scenario = gen.generate(incident)

    remediations = scenario["gold_standard"]["remediations"]

    assert len(remediations) == 2
    assert remediations[0]["action"] == "Increase max_connections"


def test_gold_standard_fallback_remediation():
    gen = ScenarioGenerator()
    scenario = gen.generate(_make_incident(remediation=[]))

    remediations = scenario["gold_standard"]["remediations"]

    assert len(remediations) >= 1


def test_generate_yaml_produces_valid_yaml():
    gen = ScenarioGenerator()
    yaml_str = gen.generate_yaml(_make_incident())
    parsed = yaml.safe_load(yaml_str)

    assert parsed["id"] == "generated-test-001"
    assert isinstance(parsed["services"], list)


def test_save_writes_file():
    gen = ScenarioGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = gen.save(_make_incident(), tmpdir)
        assert path.exists()
        content = yaml.safe_load(path.read_text())

    assert content["id"] == "generated-test-001"


def test_generate_yaml_header_contains_source():
    gen = ScenarioGenerator()
    yaml_str = gen.generate_yaml(_make_incident())

    assert "Source: https://example.com/incident/1" in yaml_str
    assert "Quality score:" in yaml_str


def test_generate_definition_returns_scenario():
    gen = ScenarioGenerator()
    scenario = gen.generate_definition(_make_incident())

    assert scenario.id == "generated-test-001"
    assert len(scenario.services) >= 2
    assert len(scenario.timeline) >= 2
    assert len(scenario.gold_root_causes) >= 1


def test_generate_definition_is_playable():
    """Generated ScenarioDefinition should work in the RL environment."""
    from src.environment.env import SREEnvironment

    gen = ScenarioGenerator()
    scenario = gen.generate_definition(_make_incident())
    env = SREEnvironment(scenario=scenario, seed=42)
    obs, _info = env.reset()

    assert len(obs["text_observation"]) > 0
    env.close()


def test_batch_generate_filters_by_quality():
    gen = ScenarioGenerator()
    incidents = [
        _make_incident(id="good", quality_score=0.8),
        _make_incident(id="bad", quality_score=0.1),
    ]
    results = gen.batch_generate(incidents, min_quality=0.5)

    assert len(results) == 1


def test_batch_generate_filters_by_taxonomy():
    gen = ScenarioGenerator()
    incidents = [
        _make_incident(id="labeled", taxonomy_labels=["infrastructure.database.connection_pool"]),
        _make_incident(id="unlabeled", taxonomy_labels=[]),
    ]
    results = gen.batch_generate(incidents)

    assert len(results) == 1
