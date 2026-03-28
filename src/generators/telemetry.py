"""TelemetryGenerator — composes MetricsGenerator, LogGenerator, TraceGenerator."""

from __future__ import annotations

import random

from src.constants import SYSTEM_EVENT_TYPES
from src.generators.effects.registry import build_default_registry
from src.generators.logs import LogGenerator
from src.generators.metrics import MetricsGenerator
from src.generators.traces import TraceGenerator
from src.models import Event, GeneratedTelemetry, ScenarioDefinition, TimelineEntry

METRIC_INTERVAL_SECONDS = 10
TRACE_SAMPLE_RATE = 0.3


class TelemetryGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        effects = build_default_registry()
        self._metrics = MetricsGenerator(self._rng, effects)
        self._logs = LogGenerator(self._rng)
        self._traces = TraceGenerator(self._rng)

    def generate(self, scenario: ScenarioDefinition) -> GeneratedTelemetry:
        metrics, logs, traces = [], [], []

        for ts in range(0, scenario.episode_duration_seconds, METRIC_INTERVAL_SECONDS):
            active = self._active_events(scenario.timeline, ts)

            for service in scenario.services:
                metrics.extend(self._metrics.generate(service, ts, active))
                if self._logs.should_generate(service, active):
                    entry = self._logs.generate(service, ts, active)
                    if entry:
                        logs.append(entry)

            if self._rng.random() < TRACE_SAMPLE_RATE:
                traces.extend(self._traces.generate(scenario.services, ts, active))

        return GeneratedTelemetry(
            metrics=tuple(metrics),
            logs=tuple(logs),
            traces=tuple(traces),
            events=self._system_events(scenario.timeline),
        )

    @staticmethod
    def _active_events(timeline: tuple[TimelineEntry, ...], ts: int) -> list[TimelineEntry]:
        return [
            e
            for e in timeline
            if e.time_offset_seconds <= ts <= e.time_offset_seconds + e.params.get("duration_seconds", float("inf"))
        ]

    @staticmethod
    def _system_events(timeline: tuple[TimelineEntry, ...]) -> tuple[Event, ...]:
        return tuple(
            Event(
                timestamp=float(e.time_offset_seconds),
                event_type=e.event_type,
                service=e.service or "system",
                description=e.description,
                metadata=e.params,
            )
            for e in timeline
            if e.event_type in SYSTEM_EVENT_TYPES
        )
