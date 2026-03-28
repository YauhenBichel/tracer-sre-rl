"""Scenario loader — reads YAML scenario definitions into immutable ScenarioDefinition objects."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import yaml

from src.models import (
    GoldStandardRemediation, GoldStandardRootCause,
    ScenarioDefinition, ServiceDefinition, TimelineEntry,
)

logger = logging.getLogger(__name__)

SCENARIOS_DIR = str(Path(__file__).parent.parent.parent / "scenarios")
REQUIRED_FIELDS = ("id", "name", "services", "timeline")


class ScenarioLoader:

    def __init__(self, scenarios_dir: str = SCENARIOS_DIR):
        self._dir = scenarios_dir

    def load(self, path: str) -> ScenarioDefinition:
        logger.info("Loading scenario from %s", path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse(data, path)

    def load_all(self, *, max_difficulty: float | None = None) -> list[ScenarioDefinition]:
        """Load all scenarios, optionally filtered by maximum difficulty.

        Scenarios are returned sorted by difficulty (easiest first) for
        curriculum learning — train on easy scenarios first, then harder ones.
        """
        paths = sorted(glob.glob(os.path.join(self._dir, "*.yaml")))
        if not paths:
            logger.warning("No scenarios found in %s", self._dir)
        scenarios = [self.load(p) for p in paths]
        if max_difficulty is not None:
            scenarios = [s for s in scenarios if s.difficulty <= max_difficulty]
        return sorted(scenarios, key=lambda s: s.difficulty)

    @staticmethod
    def _parse(data: dict, source: str) -> ScenarioDefinition:
        missing = [k for k in REQUIRED_FIELDS if k not in data]
        if missing:
            raise ValueError(f"Scenario {source} missing required fields: {missing}")

        gold = data.get("gold_standard", {})

        return ScenarioDefinition(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            taxonomy_labels=tuple(data.get("taxonomy_labels", [])),
            services=tuple(
                ServiceDefinition(
                    name=s["name"],
                    service_type=s["service_type"],
                    dependencies=tuple(s.get("dependencies", [])),
                    config=s.get("config", {}),
                ) for s in data["services"]
            ),
            timeline=tuple(
                TimelineEntry(
                    time_offset_seconds=t["time_offset_seconds"],
                    event_type=t["event_type"],
                    service=t.get("service"),
                    params=t.get("params", {}),
                    description=t.get("description", ""),
                ) for t in data["timeline"]
            ),
            gold_root_causes=tuple(
                GoldStandardRootCause(
                    taxonomy_label=rc["taxonomy_label"],
                    relevance=rc["relevance"],
                    evidence=tuple(rc.get("evidence", [])),
                ) for rc in gold.get("root_causes", [])
            ),
            gold_remediations=tuple(
                GoldStandardRemediation(
                    action=r["action"],
                    effectiveness=r["effectiveness"],
                ) for r in gold.get("remediations", [])
            ),
            difficulty=data.get("difficulty", 0.5),
            max_investigation_steps=data.get("max_investigation_steps", 20),
            episode_duration_seconds=data.get("episode_duration_seconds", 900),
        )
