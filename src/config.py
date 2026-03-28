"""Load configuration from YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"


@lru_cache
def _load(filename: str) -> dict:
    with open(CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)


def metric_baselines() -> dict[str, dict[str, tuple[float, float]]]:
    raw = _load("baselines.yaml")["metric_baselines"]
    return {svc: {m: tuple(v) for m, v in metrics.items()} for svc, metrics in raw.items()}


def base_latency_ms() -> dict[str, float]:
    return _load("baselines.yaml")["base_latency_ms"]


def log_templates() -> dict[str, dict[str, list[str]]]:
    return _load("baselines.yaml")["log_templates"]


def distractor_diagnoses() -> list[str]:
    return _load("environment.yaml")["distractor_diagnoses"]


def distractor_remediations() -> list[str]:
    return _load("environment.yaml")["distractor_remediations"]


def crawler_config(name: str) -> dict:
    return _load("crawlers.yaml")[name]


def taxonomy_config() -> dict:
    return _load("taxonomy.yaml")


def reward_config() -> dict:
    return _load("reward.yaml")
