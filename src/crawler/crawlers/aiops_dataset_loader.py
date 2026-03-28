"""Aiops-Dataset loader — imports labeled fault scenarios from bbyldebb/Aiops-Dataset.

The dataset contains log/metric/trace data from a 46-instance e-commerce
microservice system with ground-truth fault labels. This loader reads
the groundtruth CSV files and normalises them into NormalisedIncident format,
which can then be fed into the ScenarioGenerator.

Dataset: https://github.com/bbyldebb/Aiops-Dataset
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.crawler.models import NormalisedIncident
from src.utils.datetime_utils import DateTimeUtils

logger = logging.getLogger(__name__)

SOURCE_NAME = "aiops_dataset"


def load_aiops_groundtruth(groundtruth_path: str) -> list[NormalisedIncident]:
    """Load fault incidents from the Aiops-Dataset groundtruth CSV.

    Expected CSV columns: timestamp, duration, root_cause_instance,
    root_cause_type, failure_type (exact columns may vary).

    Args:
        groundtruth_path: Path to groundtruth-all.csv or individual date CSV.

    Returns:
        List of normalised incidents ready for scenario generation.
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
    """Convert a groundtruth CSV row to NormalisedIncident."""
    timestamp = row.get("timestamp", row.get("start_time", ""))
    root_cause = row.get("root_cause_instance", row.get("root_cause", ""))
    fault_type = row.get("failure_type", row.get("root_cause_type", "unknown"))
    duration = row.get("duration", row.get("duration_min", ""))

    title = f"{fault_type} on {root_cause}"
    taxonomy_labels = _classify_fault_type(fault_type)

    return NormalisedIncident(
        id=f"{SOURCE_NAME}-{timestamp}-{root_cause}",
        source=SOURCE_NAME,
        source_url="https://github.com/bbyldebb/Aiops-Dataset",
        title=title,
        summary=f"Fault type: {fault_type}, Root cause instance: {root_cause}, Duration: {duration}",
        root_causes=[root_cause],
        affected_services=[root_cause] if root_cause else [],
        impact={"duration": duration, "fault_type": fault_type},
        taxonomy_labels=taxonomy_labels,
        ingested_at=DateTimeUtils.now_iso(),
        quality_score=0.7,
    )


def _classify_fault_type(fault_type: str) -> list[str]:
    """Map Aiops-Dataset fault types to taxonomy labels."""
    ft = fault_type.lower()
    mapping = {
        "cpu": "infrastructure.compute.cpu_saturation",
        "memory": "application.memory.leak",
        "disk": "infrastructure.storage.disk_full",
        "network": "infrastructure.network.partition",
        "pod": "infrastructure.compute.container_crash",
        "node": "infrastructure.compute.instance_failure",
        "latency": "application.dependency.upstream_timeout",
        "error": "application.dependency.cascading_failure",
    }
    for keyword, label in mapping.items():
        if keyword in ft:
            return [label]
    return ["operational.deployment.bad_deploy"]
