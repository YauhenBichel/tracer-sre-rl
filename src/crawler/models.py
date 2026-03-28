"""Data models for incident crawling."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

from src.config import crawler_config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


@dataclass
class NormalisedIncident:
    id: str
    source: str
    source_url: str
    title: str
    summary: str
    timeline: list[dict] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    affected_services: list[str] = field(default_factory=list)
    impact: dict = field(default_factory=dict)
    remediation: list[str] = field(default_factory=list)
    taxonomy_labels: list[str] = field(default_factory=list)
    ingested_at: str = ""
    quality_score: float = 0.0


class IncidentCrawler(ABC):
    """Base class for all incident crawlers.

    Subclasses must define `source_name` and implement `crawl()`.
    Shared HTTP fetching is provided via `_fetch()`.
    """

    source_name: str

    def __init__(self):
        self._cfg = crawler_config(self.source_name)

    @abstractmethod
    def crawl(self) -> list[NormalisedIncident]: ...

    def _fetch(self, url: str, **kwargs) -> requests.Response:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp
