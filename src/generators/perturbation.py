"""Scenario perturbation — creates randomised variations of base scenarios.

This is critical for generalisation: the agent must learn to investigate
failure patterns, not memorise specific scenarios. Each seed produces a
different perturbation of timing, effect magnitudes, and noise profiles.
"""

from __future__ import annotations

import random
from dataclasses import replace

from src.models import ScenarioDefinition, TimelineEntry

# How much to jitter event timing (fraction of original offset)
TIMING_JITTER_FRACTION = 0.15
# How much to scale effect magnitudes (e.g., error_rate, latency factor)
MAGNITUDE_JITTER_FRACTION = 0.2
# Min/max multiplier for episode duration variation
DURATION_SCALE_MIN = 0.8
DURATION_SCALE_MAX = 1.2


def perturb_scenario(scenario: ScenarioDefinition, rng: random.Random) -> ScenarioDefinition:
    """Create a randomised variation of a scenario.

    Perturbs:
    - Event timing offsets (±15% jitter)
    - Effect magnitudes in params (±20% jitter on numeric values)
    - Episode duration (±20%)

    Preserves:
    - Service topology (same services and dependencies)
    - Event types and order
    - Gold standard labels and remediations
    - Taxonomy labels
    """
    duration_scale = rng.uniform(DURATION_SCALE_MIN, DURATION_SCALE_MAX)
    new_duration = max(300, int(scenario.episode_duration_seconds * duration_scale))

    new_timeline = tuple(_perturb_event(entry, rng, new_duration) for entry in scenario.timeline)

    return replace(
        scenario,
        timeline=new_timeline,
        episode_duration_seconds=new_duration,
    )


def _perturb_event(entry: TimelineEntry, rng: random.Random, max_duration: int) -> TimelineEntry:
    """Perturb a single timeline event's timing and numeric params."""
    # Jitter the timing offset
    if entry.time_offset_seconds > 0:
        jitter = int(entry.time_offset_seconds * TIMING_JITTER_FRACTION)
        new_offset = entry.time_offset_seconds + rng.randint(-jitter, jitter)
        new_offset = max(0, min(new_offset, max_duration - 10))
    else:
        new_offset = 0

    # Jitter numeric params (effect magnitudes)
    new_params = {}
    for key, value in entry.params.items():
        if isinstance(value, (int, float)) and key not in ("alert_name", "severity"):
            scale = 1.0 + rng.uniform(-MAGNITUDE_JITTER_FRACTION, MAGNITUDE_JITTER_FRACTION)
            new_value = value * scale
            new_params[key] = int(new_value) if isinstance(value, int) else round(new_value, 3)
        else:
            new_params[key] = value

    return replace(entry, time_offset_seconds=new_offset, params=new_params)
