"""Base class for event effect handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import ServiceDefinition, TimelineEntry


class EventEffectHandler(ABC):
    @abstractmethod
    def apply(
        self, base_value: float, metric_name: str, service: ServiceDefinition, event: TimelineEntry, progress: float
    ) -> float | None: ...
