"""Tests for Aiops-Dataset groundtruth loader."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from src.crawler.crawlers.aiops_dataset_loader import load_aiops_groundtruth


def _write_csv(rows: list[dict]) -> str:
    """Write test CSV and return path."""
    path = Path(tempfile.mktemp(suffix=".csv"))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_load_groundtruth():
    path = _write_csv([
        {"timestamp": "1651338400", "level": "node", "cmdb_id": "node-6",
         "failure_type": "Node CPU Failure"},
        {"timestamp": "1651339370", "level": "pod", "cmdb_id": "frontend-0",
         "failure_type": "Kubernetes Container Memory Load"},
    ])
    incidents = load_aiops_groundtruth(path)
    assert len(incidents) == 2
    assert incidents[0].source == "aiops_dataset"
    assert "CPU" in incidents[0].title


def test_taxonomy_classification():
    path = _write_csv([
        {"timestamp": "1", "level": "node", "cmdb_id": "n1", "failure_type": "Node CPU Failure"},
        {"timestamp": "2", "level": "pod", "cmdb_id": "s1", "failure_type": "Kubernetes Container Memory Load"},
        {"timestamp": "3", "level": "pod", "cmdb_id": "s2", "failure_type": "Kubernetes Container Network Latency"},
        {"timestamp": "4", "level": "node", "cmdb_id": "n2", "failure_type": "Node Disk Space Consumption"},
        {"timestamp": "5", "level": "pod", "cmdb_id": "s3", "failure_type": "Kubernetes Container Read I/O Load"},
        {"timestamp": "6", "level": "pod", "cmdb_id": "s4", "failure_type": "Kubernetes Container Process Termination"},
    ])
    incidents = load_aiops_groundtruth(path)

    assert "infrastructure.compute.cpu_saturation" in incidents[0].taxonomy_labels
    assert "application.memory.leak" in incidents[1].taxonomy_labels
    assert "infrastructure.network.partition" in incidents[2].taxonomy_labels
    assert "infrastructure.storage.disk_full" in incidents[3].taxonomy_labels
    assert "infrastructure.storage.iops_throttling" in incidents[4].taxonomy_labels
    assert "infrastructure.compute.container_crash" in incidents[5].taxonomy_labels


def test_missing_file_returns_empty():
    incidents = load_aiops_groundtruth("/nonexistent/path.csv")
    assert incidents == []


def test_incidents_are_playable():
    """Generated scenarios from Aiops-Dataset should work in the RL environment."""
    from src.crawler.scenario_generator import ScenarioGenerator
    from src.environment.env import SREEnvironment

    path = _write_csv([
        {"timestamp": "1651338400", "level": "node", "cmdb_id": "node-6",
         "failure_type": "Node CPU Spiking"},
    ])
    incidents = load_aiops_groundtruth(path)
    scenarios = ScenarioGenerator().batch_generate(incidents, min_quality=0.0)

    assert len(scenarios) == 1
    env = SREEnvironment(scenario=scenarios[0], seed=42)
    obs, _ = env.reset()
    assert len(obs["text_observation"]) > 0
    env.close()


def test_real_groundtruth_file():
    """Test against the actual downloaded groundtruth if present."""
    path = Path("data/groundtruth-all.csv")
    if not path.exists():
        return  # skip if not downloaded

    incidents = load_aiops_groundtruth(str(path))
    assert len(incidents) == 241
    assert all(i.source == "aiops_dataset" for i in incidents)
    assert all(len(i.taxonomy_labels) > 0 for i in incidents)
