"""Load configuration from YAML files."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"


@lru_cache
def _load(filename: str) -> dict:
    path = CONFIG_DIR / filename
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error("Config file not found: %s", path)
        return {}
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in %s: %s", path, e)
        return {}


def metric_baselines() -> dict[str, dict[str, tuple[float, float]]]:
    raw = _load("baselines.yaml")["metric_baselines"]
    return {svc: {m: tuple(v) for m, v in metrics.items()} for svc, metrics in raw.items()}


def base_latency_ms() -> dict[str, float]:
    result: dict[str, float] = _load("baselines.yaml")["base_latency_ms"]
    return result


def log_templates() -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = _load("baselines.yaml")["log_templates"]
    return result


def distractor_diagnoses() -> list[str]:
    result: list[str] = _load("distractors.yaml")["distractor_diagnoses"]
    return result


def distractor_remediations() -> list[str]:
    result: list[str] = _load("distractors.yaml")["distractor_remediations"]
    return result


def crawler_config(name: str) -> dict:
    result: dict = _load("crawlers.yaml")[name]
    return result


def taxonomy_config() -> dict:
    return _load("taxonomy.yaml")


def reward_config() -> dict:
    return _load("reward.yaml")
