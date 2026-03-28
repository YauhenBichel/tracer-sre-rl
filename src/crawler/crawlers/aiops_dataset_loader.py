"""Aiops-Dataset groundtruth loader.

Reads groundtruth-all.csv from the Aiops-Dataset and converts labeled
fault scenarios into NormalisedIncident objects for scenario generation.

CSV format (actual columns from the dataset):
  timestamp,level,cmdb_id,failure_type

  - timestamp: Unix epoch seconds
  - level: "node", "pod", or "service"
  - cmdb_id: instance name (e.g., "node-6", "recommendationservice-0")
  - failure_type: human-readable fault description

Download from: https://mega.nz/file/SA1VCRoJ#wLSzQdE1p0M4-l5mGbRhUqqEU7t34XJGzbJLXWowEiM
Extract groundtruth/groundtruth-all.csv into data/
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.crawler.models import NormalisedIncident
from src.utils.datetime_utils import DateTimeUtils

logger = logging.getLogger(__name__)

SOURCE_NAME = "aiops_dataset"

# Maps Aiops-Dataset failure_type strings to taxonomy labels.
_FAULT_TYPE_MAPPING = {
    "CPU Load": "infrastructure.compute.cpu_saturation",
    "CPU Failure": "infrastructure.compute.cpu_saturation",
    "CPU Spiking": "infrastructure.compute.cpu_saturation",
    "Memory Load": "application.memory.leak",
    "Memory Consumption": "application.memory.leak",
    "Network Latency": "infrastructure.network.partition",
    "Network Packet Loss": "infrastructure.network.partition",
    "Packet Corruption": "infrastructure.network.partition",
    "Packet Duplication": "infrastructure.network.partition",
    "Process Termination": "infrastructure.compute.container_crash",
    "Read I/O Load": "infrastructure.storage.iops_throttling",
    "Write I/O Load": "infrastructure.storage.iops_throttling",
    "Disk Read": "infrastructure.storage.iops_throttling",
    "Disk Write": "infrastructure.storage.iops_throttling",
    "Disk Space": "infrastructure.storage.disk_full",
}


def load_aiops_groundtruth(groundtruth_path: str) -> list[NormalisedIncident]:
    """Load fault incidents from the Aiops-Dataset groundtruth CSV.

    Args:
        groundtruth_path: Path to groundtruth-all.csv.

    Returns:
        List of NormalisedIncident objects ready for ScenarioGenerator.batch_generate().
    """
    path = Path(groundtruth_path)
    if not path.exists():
        logger.warning("Aiops-Dataset groundtruth not found: %s", path)
        return []

    results = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                results.append(_normalise_row(row))
            except Exception as e:
                logger.warning("Failed to normalise row: %s", e)

    logger.info("Aiops-Dataset: loaded %d incidents from %s", len(results), path)
    return results


def _normalise_row(row: dict) -> NormalisedIncident:
    """Convert a single groundtruth CSV row to NormalisedIncident."""
    timestamp = row.get("timestamp", "")
    level = row.get("level", "")
    cmdb_id = row.get("cmdb_id", "")
    failure_type = row.get("failure_type", "unknown")

    return NormalisedIncident(
        id=f"{SOURCE_NAME}-{timestamp}-{cmdb_id}",
        source=SOURCE_NAME,
        source_url="https://github.com/bbyldebb/Aiops-Dataset",
        title=f"{failure_type} on {cmdb_id}",
        summary=f"Level: {level}, Instance: {cmdb_id}, Fault: {failure_type}",
        root_causes=[cmdb_id],
        affected_services=[cmdb_id] if cmdb_id else [],
        impact={"level": level, "fault_type": failure_type},
        taxonomy_labels=_classify_fault_type(failure_type),
        ingested_at=DateTimeUtils.now_iso(),
        quality_score=0.7,
    )


def _classify_fault_type(failure_type: str) -> list[str]:
    """Map Aiops-Dataset failure_type string to taxonomy labels."""
    for keyword, label in _FAULT_TYPE_MAPPING.items():
        if keyword.lower() in failure_type.lower():
            return [label]
    return ["operational.deployment.bad_deploy"]
