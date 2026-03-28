"""Event effect registry — maps event types to handlers."""

from __future__ import annotations

from .base import EventEffectHandler
from .traffic_ramp import TrafficRampEffect
from .resource_exhaustion import ResourceExhaustionEffect
from .latency_spike import LatencySpikeEffect
from .error_spike import ErrorSpikeEffect
from .connection_exhaustion import ConnectionExhaustionEffect
from .cascade import CascadeEffect
from .symptom import SymptomEffect
from .disk_fill import DiskFillEffect
from .memory_leak import MemoryLeakEffect


class EventEffectRegistry:

    def __init__(self):
        self._handlers: dict[str, EventEffectHandler] = {}

    def register(self, event_type: str, handler: EventEffectHandler) -> "EventEffectRegistry":
        self._handlers[event_type] = handler
        return self

    def get(self, event_type: str) -> EventEffectHandler | None:
        return self._handlers.get(event_type)


def build_default_registry() -> EventEffectRegistry:
    return (
        EventEffectRegistry()
        .register("traffic_ramp", TrafficRampEffect())
        .register("resource_exhaustion", ResourceExhaustionEffect())
        .register("latency_spike", LatencySpikeEffect())
        .register("error_spike", ErrorSpikeEffect())
        .register("connection_exhaustion", ConnectionExhaustionEffect())
        .register("cascade", CascadeEffect())
        .register("symptom", SymptomEffect())
        .register("disk_fill", DiskFillEffect())
        .register("memory_leak", MemoryLeakEffect())
    )
