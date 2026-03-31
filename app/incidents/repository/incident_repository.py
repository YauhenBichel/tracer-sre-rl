"""Incident repository interface."""

from __future__ import annotations

from typing import Protocol

from app.incidents.models import NormalisedIncident


class IncidentRepository(Protocol):
    def save(self, incident: NormalisedIncident) -> None: ...
    def count(self) -> int: ...
    def count_by_source(self) -> dict[str, int]: ...
    def close(self) -> None: ...
