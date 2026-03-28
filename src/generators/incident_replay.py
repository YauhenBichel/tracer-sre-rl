"""Incident replay — generates scenarios from real incident databases.

Instead of hand-authoring scenario YAML files, this module converts real
incidents from VOID, Aiops-Dataset, and other sources into playable
scenarios for the RL environment.

The key insight: we use real incident data to define WHAT happens (timeline,
root cause, affected services), and synthetic telemetry generation for HOW
it looks (metrics, logs, traces). This gives us real-world failure patterns
at synthetic-telemetry speed.

Data flow:
  VOID/Aiops-Dataset → NormalisedIncident → ReplayableScenario → SREEnvironment
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.crawler.models import NormalisedIncident
from src.crawler.scenario_generator import ScenarioGenerator
from src.generators.loader import ScenarioLoader
from src.models import ScenarioDefinition

logger = logging.getLogger(__name__)


class IncidentReplaySource:
    """Loads real incidents and converts them into RL-ready scenarios.

    Supports multiple data sources:
      - VOID database (via crawler)
      - Aiops-Dataset (via CSV loader)
      - SQLite database (previously crawled incidents)
      - Any list of NormalisedIncident objects
    """

    def __init__(self):
        self._generator = ScenarioGenerator()
        self._scenarios: list[ScenarioDefinition] = []

    @property
    def scenarios(self) -> list[ScenarioDefinition]:
        return self._scenarios

    def load_from_incidents(self, incidents: list[NormalisedIncident],
                            min_quality: float = 0.3) -> int:
        """Convert NormalisedIncidents into ScenarioDefinitions.

        Filters by quality score and taxonomy label availability.
        Returns number of scenarios generated.
        """
        count = 0
        for incident in incidents:
            if incident.quality_score < min_quality:
                continue
            if not incident.taxonomy_labels:
                continue
            try:
                scenario_dict = self._generator.generate(incident)
                scenario = _dict_to_scenario(scenario_dict)
                self._scenarios.append(scenario)
                count += 1
            except Exception as e:
                logger.warning("Failed to generate scenario from %s: %s", incident.id, e)
        logger.info("Generated %d scenarios from %d incidents", count, len(incidents))
        return count

    def load_from_db(self, db_path: str, min_quality: float = 0.3) -> int:
        """Load incidents from SQLite database (previously crawled)."""
        from src.crawler.repository.sqlite_repository import SqliteIncidentRepository
        repo = SqliteIncidentRepository(db_path)
        try:
            incidents = repo.load_all()
        finally:
            repo.close()
        return self.load_from_incidents(incidents, min_quality)

    def load_from_aiops_dataset(self, groundtruth_path: str) -> int:
        """Load labeled fault scenarios from the Aiops-Dataset."""
        from src.crawler.crawlers.aiops_dataset_loader import load_aiops_groundtruth
        incidents = load_aiops_groundtruth(groundtruth_path)
        return self.load_from_incidents(incidents, min_quality=0.0)

    def load_builtin_scenarios(self, scenarios_dir: str | None = None) -> int:
        """Load the hand-authored YAML scenarios as a baseline."""
        loader = ScenarioLoader(scenarios_dir) if scenarios_dir else ScenarioLoader()
        builtin = loader.load_all()
        self._scenarios.extend(builtin)
        return len(builtin)

    def get_all(self) -> list[ScenarioDefinition]:
        """Return all loaded scenarios (real + builtin), sorted by difficulty."""
        return sorted(self._scenarios, key=lambda s: s.difficulty)


def _dict_to_scenario(data: dict) -> ScenarioDefinition:
    """Convert a scenario dict (from ScenarioGenerator) to ScenarioDefinition."""
    from src.models import (
        GoldStandardRemediation, GoldStandardRootCause,
        ServiceDefinition, TimelineEntry,
    )

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
            ) for s in data.get("services", [])
        ),
        timeline=tuple(
            TimelineEntry(
                time_offset_seconds=t["time_offset_seconds"],
                event_type=t["event_type"],
                service=t.get("service"),
                params=t.get("params", {}),
                description=t.get("description", ""),
            ) for t in data.get("timeline", [])
        ),
        gold_root_causes=tuple(
            GoldStandardRootCause(
                taxonomy_label=rc["taxonomy_label"],
                relevance=rc.get("relevance", 0.8),
                evidence=tuple(rc.get("evidence", [])),
            ) for rc in gold.get("root_causes", [])
        ),
        gold_remediations=tuple(
            GoldStandardRemediation(
                action=r["action"],
                effectiveness=r.get("effectiveness", 0.7),
            ) for r in gold.get("remediations", [])
        ),
        difficulty=data.get("difficulty", 0.5),
        max_investigation_steps=data.get("max_investigation_steps", 20),
        episode_duration_seconds=data.get("episode_duration_seconds", 900),
    )
