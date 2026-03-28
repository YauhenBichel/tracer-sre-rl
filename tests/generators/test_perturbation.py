"""Tests for scenario perturbation."""

import random

from src.generators.loader import ScenarioLoader
from src.generators.perturbation import perturb_scenario


def _load_scenario():
    loader = ScenarioLoader()
    return loader.load_all()[0]


def test_perturbation_changes_timing():
    scenario = _load_scenario()
    rng = random.Random(42)
    perturbed = perturb_scenario(scenario, rng)

    # At least some timeline entries should have different offsets
    original_offsets = [e.time_offset_seconds for e in scenario.timeline]
    perturbed_offsets = [e.time_offset_seconds for e in perturbed.timeline]

    assert original_offsets != perturbed_offsets


def test_perturbation_preserves_services():
    scenario = _load_scenario()
    rng = random.Random(42)
    perturbed = perturb_scenario(scenario, rng)

    assert perturbed.services == scenario.services


def test_perturbation_preserves_gold_standard():
    scenario = _load_scenario()
    rng = random.Random(42)
    perturbed = perturb_scenario(scenario, rng)

    assert perturbed.gold_root_causes == scenario.gold_root_causes
    assert perturbed.gold_remediations == scenario.gold_remediations


def test_different_seeds_produce_different_perturbations():
    scenario = _load_scenario()
    p1 = perturb_scenario(scenario, random.Random(1))
    p2 = perturb_scenario(scenario, random.Random(2))

    offsets1 = [e.time_offset_seconds for e in p1.timeline]
    offsets2 = [e.time_offset_seconds for e in p2.timeline]

    assert offsets1 != offsets2


def test_perturbation_keeps_events_within_episode():
    scenario = _load_scenario()
    rng = random.Random(42)
    perturbed = perturb_scenario(scenario, rng)

    assert len(perturbed.timeline) > 0
    for entry in perturbed.timeline:
        assert 0 <= entry.time_offset_seconds < perturbed.episode_duration_seconds


def test_perturbation_preserves_event_types():
    scenario = _load_scenario()
    rng = random.Random(42)
    perturbed = perturb_scenario(scenario, rng)

    original_types = [e.event_type for e in scenario.timeline]
    perturbed_types = [e.event_type for e in perturbed.timeline]

    assert original_types == perturbed_types
