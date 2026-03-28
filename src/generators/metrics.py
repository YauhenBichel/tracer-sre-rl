"""Metrics generator — produces time-series metric samples for services."""

from __future__ import annotations

import logging
import random

from src.config import metric_baselines
from src.constants import EVENT_CASCADE, METRIC_PERCENT, SERVICE_TYPE_APPLICATION
from src.generators.effects.registry import EventEffectRegistry
from src.models import MetricSample, ServiceDefinition, TimelineEntry

logger = logging.getLogger(__name__)

DEFAULT_EVENT_DURATION_SECONDS = 300
METRIC_NOISE_FACTOR = 0.05  # gaussian std dev as fraction of baseline range
PERCENT_METRIC_CAP = 100


class MetricsGenerator:
    def __init__(self, rng: random.Random, effects: EventEffectRegistry):
        self._rng = rng
        self._effects = effects

    def generate(
        self, service: ServiceDefinition, timestamp: int, active_events: list[TimelineEntry]
    ) -> list[MetricSample]:
        baselines = metric_baselines().get(service.service_type, metric_baselines().get(SERVICE_TYPE_APPLICATION, {}))
        if not baselines:
            logger.warning("No baselines for service type '%s', skipping metrics", service.service_type)
            return []

        samples = []
        for name, (low, high) in baselines.items():
            value = (low + high) / 2 + self._rng.gauss(0, (high - low) * METRIC_NOISE_FACTOR)
            value = self._apply_effects(value, name, service, active_events, timestamp)
            if METRIC_PERCENT in name:
                value = min(value, PERCENT_METRIC_CAP)
            samples.append(
                MetricSample(
                    timestamp=float(timestamp),
                    service=service.name,
                    metric_name=name,
                    value=round(max(0, value), 3),
                    labels={"service_type": service.service_type},
                )
            )
        return samples

    def _apply_effects(
        self, value: float, metric: str, service: ServiceDefinition, events: list[TimelineEntry], timestamp: int
    ) -> float:
        for event in events:
            if not _event_affects_service(event, service):
                continue
            duration = event.params.get("duration_seconds", DEFAULT_EVENT_DURATION_SECONDS)
            progress = min(1.0, (timestamp - event.time_offset_seconds) / max(1, duration))
            handler = self._effects.get(event.event_type)
            if handler:
                result = handler.apply(value, metric, service, event, progress)
                if result is not None:
                    value = result
        return value


def _event_affects_service(event: TimelineEntry, service: ServiceDefinition) -> bool:
    if event.service is None or event.service == service.name:
        return True
    return event.event_type == EVENT_CASCADE and event.service != service.name
