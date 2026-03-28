"""Scenario loader — reads YAML scenario definitions into immutable ScenarioDefinition objects."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import yaml

from src.models import ScenarioDefinition

logger = logging.getLogger(__name__)

SCENARIOS_DIR = str(Path(__file__).parent.parent.parent / "scenarios")
REQUIRED_FIELDS = ("id", "name", "services", "timeline")


class ScenarioLoader:
    def __init__(self, scenarios_dir: str = SCENARIOS_DIR):
        self._dir = scenarios_dir

    def load(self, path: str) -> ScenarioDefinition | None:
        """Load a single scenario YAML. Returns None if file is missing or invalid."""
        logger.info("Loading scenario from %s", path)
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("Scenario file not found: %s", path)
            return None
        except yaml.YAMLError as e:
            logger.error("Invalid YAML in %s: %s", path, e)
            return None
        return self._parse(data, path)

    def load_all(self, *, max_difficulty: float | None = None) -> list[ScenarioDefinition]:
        """Load all scenarios, optionally filtered by maximum difficulty.

        Scenarios are returned sorted by difficulty (easiest first) for
        curriculum learning — train on easy scenarios first, then harder ones.
        """
        paths = sorted(glob.glob(os.path.join(self._dir, "*.yaml")))
        if not paths:
            logger.warning("No scenarios found in %s", self._dir)
        scenarios = [s for p in paths if (s := self.load(p)) is not None]
        if max_difficulty is not None:
            scenarios = [s for s in scenarios if s.difficulty <= max_difficulty]
        return sorted(scenarios, key=lambda s: s.difficulty)

    @staticmethod
    def _parse(data: dict, source: str) -> ScenarioDefinition:
        missing = [k for k in REQUIRED_FIELDS if k not in data]
        if missing:
            raise ValueError(f"Scenario {source} missing required fields: {missing}")

        return ScenarioDefinition.from_dict(data)
