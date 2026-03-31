"""Converts crawled NormalisedIncidents into YAML scenario templates.

This bridges Pillar 1 (Crawling & Indexing) with Pillar 5 (Test Case Generation).
It maps incident metadata to the scenario YAML structure used by the telemetry
generator and RL environment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from app.incidents.models import NormalisedIncident
from app.models import ScenarioDefinition
from app.models.taxonomy import build_default_taxonomy

logger = logging.getLogger(__name__)

# Default topology templates per failure category
_SERVICE_TOPOLOGIES: dict[str, list[dict[str, Any]]] = {
    "infrastructure.database": [
        {"name": "api-gateway", "service_type": "web-server", "dependencies": ["app-service"]},
        {"name": "app-service", "service_type": "application", "dependencies": ["database-primary"]},
        {"name": "database-primary", "service_type": "database", "config": {"max_connections": 100}},
    ],
    "infrastructure.network": [
        {"name": "api-gateway", "service_type": "web-server", "dependencies": ["app-service"]},
        {"name": "app-service", "service_type": "application", "dependencies": ["backend-service"]},
        {"name": "backend-service", "service_type": "application", "dependencies": ["database-primary"]},
        {"name": "database-primary", "service_type": "database"},
    ],
    "infrastructure.storage": [
        {"name": "api-gateway", "service_type": "web-server", "dependencies": ["app-service"]},
        {"name": "app-service", "service_type": "application", "dependencies": ["database-primary"]},
        {"name": "database-primary", "service_type": "database", "config": {"max_connections": 200}},
    ],
    "application": [
        {"name": "api-gateway", "service_type": "web-server", "dependencies": ["app-service"]},
        {"name": "app-service", "service_type": "application", "dependencies": ["database-primary", "cache"]},
        {"name": "database-primary", "service_type": "database"},
        {"name": "cache", "service_type": "cache"},
    ],
    "operational": [
        {"name": "api-gateway", "service_type": "web-server", "dependencies": ["app-service"]},
        {"name": "app-service", "service_type": "application", "dependencies": ["database-primary"]},
        {"name": "database-primary", "service_type": "database"},
    ],
}

DEFAULT_TOPOLOGY: list[dict[str, Any]] = _SERVICE_TOPOLOGIES["operational"]

# Maps taxonomy leaf nodes to timeline event templates
_EVENT_TEMPLATES = {
    "infrastructure.database.connection_pool": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "traffic_ramp",
            "params": {"multiplier": 3.0, "duration_seconds": 600},
            "description": "Traffic spike begins",
        },
        {
            "time_offset_seconds": 420,
            "event_type": "connection_exhaustion",
            "service": "database-primary",
            "params": {"duration_seconds": 480},
            "description": "Connection pool saturating",
        },
        {
            "time_offset_seconds": 500,
            "event_type": "error_spike",
            "service": "app-service",
            "params": {"target_rate": 0.3, "duration_seconds": 400},
            "description": "Application errors from connection failures",
        },
        {
            "time_offset_seconds": 540,
            "event_type": "alert",
            "service": "database-primary",
            "params": {"alert_name": "Connection Pool Exhaustion", "severity": "critical"},
            "description": "ALERT: Database connection pool exhausted",
        },
    ],
    "infrastructure.network.dns": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "latency_spike",
            "service": "app-service",
            "params": {"factor": 5.0, "duration_seconds": 600},
            "description": "DNS resolution failures causing timeouts",
        },
        {
            "time_offset_seconds": 360,
            "event_type": "error_spike",
            "service": "app-service",
            "params": {"target_rate": 0.4, "duration_seconds": 540},
            "description": "Failed DNS lookups",
        },
        {
            "time_offset_seconds": 400,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "High Error Rate", "severity": "critical"},
            "description": "ALERT: Service error rate above threshold",
        },
    ],
    "infrastructure.storage.disk_full": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 60,
            "event_type": "disk_fill",
            "service": "database-primary",
            "params": {"target_percent": 98, "duration_seconds": 400},
            "description": "Disk usage growing",
        },
        {
            "time_offset_seconds": 420,
            "event_type": "latency_spike",
            "service": "database-primary",
            "params": {"factor": 10.0, "duration_seconds": 480},
            "description": "Disk I/O saturated",
        },
        {
            "time_offset_seconds": 480,
            "event_type": "error_spike",
            "service": "database-primary",
            "params": {"target_rate": 0.9, "duration_seconds": 420},
            "description": "Write failures from full disk",
        },
        {
            "time_offset_seconds": 520,
            "event_type": "alert",
            "service": "database-primary",
            "params": {"alert_name": "Disk Full", "severity": "critical"},
            "description": "ALERT: Disk full",
        },
    ],
    "application.memory.leak": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 120,
            "event_type": "memory_leak",
            "service": "app-service",
            "params": {"rate_per_minute": 3.0, "duration_seconds": 700},
            "description": "Memory growing steadily",
        },
        {
            "time_offset_seconds": 500,
            "event_type": "resource_exhaustion",
            "service": "app-service",
            "params": {"metric": "memory_percent", "ceiling": 95, "duration_seconds": 400},
            "description": "OOM approaching",
        },
        {
            "time_offset_seconds": 540,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "Memory Usage Critical", "severity": "warning"},
            "description": "ALERT: High memory usage",
        },
    ],
    "application.dependency.cascading_failure": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "latency_spike",
            "service": "app-service",
            "params": {"factor": 15.0, "duration_seconds": 600},
            "description": "Upstream dependency slow",
        },
        {
            "time_offset_seconds": 400,
            "event_type": "cascade",
            "service": "api-gateway",
            "params": {"duration_seconds": 500},
            "description": "Cascade propagating to API gateway",
        },
        {
            "time_offset_seconds": 450,
            "event_type": "alert",
            "service": "api-gateway",
            "params": {"alert_name": "Cascading Failure Detected", "severity": "critical"},
            "description": "ALERT: Multiple services degraded",
        },
    ],
    "infrastructure.compute.cpu_saturation": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 200,
            "event_type": "resource_exhaustion",
            "service": "app-service",
            "params": {"metric": "cpu_percent", "ceiling": 95, "duration_seconds": 600},
            "description": "CPU usage climbing",
        },
        {
            "time_offset_seconds": 400,
            "event_type": "latency_spike",
            "service": "app-service",
            "params": {"factor": 8.0, "duration_seconds": 500},
            "description": "Latency increasing from CPU saturation",
        },
        {
            "time_offset_seconds": 450,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "CPU Saturation", "severity": "critical"},
            "description": "ALERT: CPU above 90%",
        },
    ],
    "infrastructure.compute.container_crash": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "error_spike",
            "service": "app-service",
            "params": {"target_rate": 0.8, "duration_seconds": 600},
            "description": "Container crashing and restarting",
        },
        {
            "time_offset_seconds": 320,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "Container CrashLoopBackOff", "severity": "critical"},
            "description": "ALERT: Container restarting",
        },
    ],
    "infrastructure.compute.instance_failure": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "error_spike",
            "service": "app-service",
            "params": {"target_rate": 1.0, "duration_seconds": 600},
            "description": "Node unreachable",
        },
        {
            "time_offset_seconds": 310,
            "event_type": "cascade",
            "service": "api-gateway",
            "params": {"duration_seconds": 590},
            "description": "Services on failed node unavailable",
        },
        {
            "time_offset_seconds": 330,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "Node Down", "severity": "critical"},
            "description": "ALERT: Node not responding",
        },
    ],
    "infrastructure.network.partition": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "latency_spike",
            "service": "app-service",
            "params": {"factor": 20.0, "duration_seconds": 600},
            "description": "Network partition causing timeouts",
        },
        {
            "time_offset_seconds": 350,
            "event_type": "error_spike",
            "service": "app-service",
            "params": {"target_rate": 0.5, "duration_seconds": 550},
            "description": "Connection failures across partition",
        },
        {
            "time_offset_seconds": 380,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "Network Connectivity Lost", "severity": "critical"},
            "description": "ALERT: Services unreachable",
        },
    ],
    "infrastructure.storage.iops_throttling": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 200,
            "event_type": "latency_spike",
            "service": "database-primary",
            "params": {"factor": 15.0, "duration_seconds": 700},
            "description": "Disk I/O throttled",
        },
        {
            "time_offset_seconds": 400,
            "event_type": "error_spike",
            "service": "database-primary",
            "params": {"target_rate": 0.2, "duration_seconds": 500},
            "description": "Query timeouts from slow I/O",
        },
        {
            "time_offset_seconds": 420,
            "event_type": "alert",
            "service": "database-primary",
            "params": {"alert_name": "Disk I/O Throttling", "severity": "warning"},
            "description": "ALERT: IOPS throttled",
        },
    ],
    "application.dependency.upstream_timeout": [
        {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
        {
            "time_offset_seconds": 300,
            "event_type": "latency_spike",
            "service": "app-service",
            "params": {"factor": 30.0, "duration_seconds": 600},
            "description": "Upstream dependency timing out",
        },
        {
            "time_offset_seconds": 400,
            "event_type": "error_spike",
            "service": "app-service",
            "params": {"target_rate": 0.4, "duration_seconds": 500},
            "description": "Timeout errors propagating",
        },
        {
            "time_offset_seconds": 430,
            "event_type": "alert",
            "service": "app-service",
            "params": {"alert_name": "Upstream Timeout", "severity": "critical"},
            "description": "ALERT: Upstream service unresponsive",
        },
    ],
}

# Fallback timeline for taxonomy labels without specific templates
_GENERIC_TIMELINE = [
    {"time_offset_seconds": 0, "event_type": "normal_traffic", "description": "System operating normally"},
    {
        "time_offset_seconds": 300,
        "event_type": "error_spike",
        "service": "app-service",
        "params": {"target_rate": 0.3, "duration_seconds": 600},
        "description": "Service degradation begins",
    },
    {
        "time_offset_seconds": 400,
        "event_type": "latency_spike",
        "service": "app-service",
        "params": {"factor": 5.0, "duration_seconds": 500},
        "description": "Latency increasing",
    },
    {
        "time_offset_seconds": 500,
        "event_type": "alert",
        "service": "app-service",
        "params": {"alert_name": "Service Degraded", "severity": "critical"},
        "description": "ALERT: Service degraded",
    },
]

# Difficulty estimates by severity
_SEVERITY_DIFFICULTY = {
    "critical": 0.6,
    "major": 0.5,
    "high": 0.5,
    "medium": 0.4,
    "minor": 0.3,
    "low": 0.2,
}
DEFAULT_DIFFICULTY = 0.5

DEFAULT_MAX_STEPS = 20
DEFAULT_EPISODE_DURATION = 900
DEFAULT_REMEDIATION_EFFECTIVENESS = 0.7
DEFAULT_ROOT_CAUSE_RELEVANCE = 0.8


class ScenarioGenerator:
    """Converts NormalisedIncidents into YAML scenario templates."""

    def __init__(self):
        self._taxonomy = build_default_taxonomy()

    def generate(self, incident: NormalisedIncident) -> dict:
        """Convert a single incident into a scenario dict (YAML-serialisable)."""
        taxonomy_label = self._best_taxonomy_label(incident)
        services = self._pick_topology(taxonomy_label)
        primary_service = self._find_primary_service(services, taxonomy_label)
        timeline = self._build_timeline(taxonomy_label, primary_service)

        return {
            "id": f"generated-{incident.id}",
            "name": self._build_name(incident),
            "description": incident.summary or incident.title,
            "taxonomy_labels": incident.taxonomy_labels or ([taxonomy_label] if taxonomy_label else []),
            "difficulty": self._estimate_difficulty(incident),
            "services": services,
            "timeline": timeline,
            "gold_standard": self._build_gold_standard(incident, taxonomy_label),
            "max_investigation_steps": DEFAULT_MAX_STEPS,
            "episode_duration_seconds": DEFAULT_EPISODE_DURATION,
        }

    def generate_yaml(self, incident: NormalisedIncident) -> str:
        """Convert a single incident into a YAML string."""
        scenario = self.generate(incident)
        header = (
            f"# Auto-generated scenario from incident: {incident.source}/{incident.id}\n"
            f"# Source: {incident.source_url}\n"
            f"# Quality score: {incident.quality_score:.2f}\n\n"
        )
        return header + yaml.dump(scenario, default_flow_style=False, sort_keys=False)

    def save(self, incident: NormalisedIncident, output_dir: str) -> Path:
        """Generate a scenario YAML and write it to output_dir."""
        scenario_yaml = self.generate_yaml(incident)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", incident.id)
        path = Path(output_dir) / f"generated_{safe_id}.yaml"
        path.write_text(scenario_yaml)
        logger.info("Generated scenario: %s", path)
        return path

    def _best_taxonomy_label(self, incident: NormalisedIncident) -> str:
        """Pick the most specific valid taxonomy label from the incident."""
        for label in incident.taxonomy_labels:
            if self._taxonomy.find(label) is not None:
                return label
        # Fallback: try partial matching
        for label in incident.taxonomy_labels:
            parts = label.split(".")
            for depth in range(len(parts), 0, -1):
                candidate = ".".join(parts[:depth])
                if self._taxonomy.find(candidate) is not None:
                    return candidate
        return ""

    def _pick_topology(self, taxonomy_label: str) -> list[dict[str, Any]]:
        """Select a service topology template based on the taxonomy category."""
        if not taxonomy_label:
            return list(DEFAULT_TOPOLOGY)
        # Try exact prefix match, then walk up the taxonomy path
        parts = taxonomy_label.split(".")
        for depth in range(len(parts), 0, -1):
            prefix = ".".join(parts[:depth])
            if prefix in _SERVICE_TOPOLOGIES:
                return list(_SERVICE_TOPOLOGIES[prefix])
        return list(DEFAULT_TOPOLOGY)

    @staticmethod
    def _find_primary_service(services: list[dict[str, Any]], taxonomy_label: str) -> str:
        """Determine which service is the primary failure target."""
        if "database" in taxonomy_label:
            for s in services:
                if s["service_type"] == "database":
                    return str(s["name"])
        if "network" in taxonomy_label or "application" in taxonomy_label:
            for s in services:
                if s["service_type"] == "application":
                    return str(s["name"])
        return str(services[-1]["name"]) if services else "app-service"

    def _build_timeline(self, taxonomy_label: str, primary_service: str) -> list[dict]:
        """Build event timeline from templates, substituting the primary service."""
        template = _EVENT_TEMPLATES.get(taxonomy_label, _GENERIC_TIMELINE)
        timeline = []
        for event in template:
            entry = dict(event)
            # Replace template service names with the actual primary service
            if "service" in entry and entry["service"] in ("app-service", "database-primary"):
                entry["service"] = primary_service
            timeline.append(entry)
        return timeline

    @staticmethod
    def _estimate_difficulty(incident: NormalisedIncident) -> float:
        """Estimate scenario difficulty from incident metadata."""
        severity = incident.impact.get("severity", "medium")
        if isinstance(severity, str):
            return _SEVERITY_DIFFICULTY.get(severity.lower(), DEFAULT_DIFFICULTY)
        return DEFAULT_DIFFICULTY

    @staticmethod
    def _build_name(incident: NormalisedIncident) -> str:
        """Build a clean scenario name from incident title."""
        title = incident.title[:80].strip()
        if not title:
            return f"Incident {incident.id}"
        return title

    @staticmethod
    def _build_gold_standard(incident: NormalisedIncident, taxonomy_label: str) -> dict:
        """Build gold standard section from incident data."""
        root_causes = []
        if taxonomy_label:
            root_causes.append(
                {
                    "taxonomy_label": taxonomy_label,
                    "relevance": DEFAULT_ROOT_CAUSE_RELEVANCE,
                    "evidence": [incident.summary] if incident.summary else [],
                }
            )

        remediations = [
            {"action": action, "effectiveness": DEFAULT_REMEDIATION_EFFECTIVENESS} for action in incident.remediation
        ]
        # Always include a generic remediation so the scenario is playable
        if not remediations:
            remediations.append(
                {
                    "action": "Investigate and remediate root cause",
                    "effectiveness": DEFAULT_REMEDIATION_EFFECTIVENESS,
                }
            )

        return {
            "root_causes": root_causes,
            "remediations": remediations,
        }

    def generate_definition(self, incident: NormalisedIncident) -> ScenarioDefinition:
        """Convert a single incident directly to an immutable ScenarioDefinition."""
        return ScenarioDefinition.from_dict(self.generate(incident))

    def batch_generate(self, incidents: list[NormalisedIncident], min_quality: float = 0.3) -> list[ScenarioDefinition]:
        """Convert a list of incidents to ScenarioDefinitions, filtering by quality."""
        results = []
        for incident in incidents:
            if incident.quality_score < min_quality or not incident.taxonomy_labels:
                continue
            try:
                results.append(self.generate_definition(incident))
            except Exception as e:
                logger.warning("Failed to generate scenario from %s: %s", incident.id, e)
        logger.info("Generated %d scenarios from %d incidents", len(results), len(incidents))
        return results
