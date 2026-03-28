"""Tests for scenario loader."""

import pytest

from src.generators.loader import ScenarioLoader


@pytest.fixture
def loader():
    return ScenarioLoader()


def test_load_all_finds_scenarios(loader):
    scenarios = loader.load_all()
    assert len(scenarios) >= 5


def test_load_single_scenario(loader):
    loader.load_all()
    path = "scenarios/db_connection_pool.yaml"
    scenario = loader.load(path)

    assert scenario.id == "scenario-db-conn-pool-001"
    assert len(scenario.services) == 5


def test_scenario_fields_are_tuples(loader):
    scenario = loader.load("scenarios/memory_leak.yaml")

    assert isinstance(scenario.services, tuple)
    assert isinstance(scenario.timeline, tuple)
    assert isinstance(scenario.gold_root_causes, tuple)
    assert isinstance(scenario.gold_remediations, tuple)
    assert isinstance(scenario.taxonomy_labels, tuple)


def test_service_dependencies_are_tuples(loader):
    scenario = loader.load("scenarios/cascading_failure.yaml")

    assert len(scenario.services) > 0
    for svc in scenario.services:
        assert isinstance(svc.dependencies, tuple)


def test_invalid_scenario_raises(loader, tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("id: test\nname: test\n")
    with pytest.raises(ValueError, match="missing required fields"):
        loader.load(str(bad_file))


def test_empty_directory(tmp_path):
    loader = ScenarioLoader(str(tmp_path))
    assert loader.load_all() == []
