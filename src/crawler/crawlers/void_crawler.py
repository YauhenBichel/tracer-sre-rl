"""VOID (Verica Open Incident Database) crawler.

The VOID contains ~10,000 incidents from ~590 organisations.
This crawler fetches the public incident list and normalises
each entry into the NormalisedIncident format.

See: https://www.thevoid.community/
"""

from __future__ import annotations

import logging

import requests

from src.crawler.models import IncidentCrawler, NormalisedIncident
from src.utils.datetime_utils import DateTimeUtils
from .quality import quality_score

logger = logging.getLogger(__name__)

DEFAULT_QUALITY_SCORE = 0.5


class VOIDIncidentCrawler(IncidentCrawler):
    """Crawls the Verica Open Incident Database (VOID)."""

    source_name = "void"

    def crawl(self) -> list[NormalisedIncident]:
        logger.info("Crawling VOID incidents...")
        try:
            resp = self._fetch(self._cfg["url"])
        except requests.RequestException as e:
            logger.error("VOID fetch failed: %s", e)
            return []

        results = []
        for item in resp.json():
            try:
                results.append(self._normalise(item))
            except Exception as e:
                logger.warning("VOID normalise failed: %s", e)

        logger.info("VOID: %d incidents", len(results))
        return results

    def _normalise(self, item: dict) -> NormalisedIncident:
        incident_id = item.get("id", item.get("slug", ""))
        title = item.get("title", item.get("name", ""))
        url = item.get("url", item.get("link", ""))
        company = item.get("company", item.get("organization", ""))
        categories = item.get("categories", item.get("tags", []))

        return NormalisedIncident(
            id=f"{self.source_name}-{incident_id}",
            source=self.source_name,
            source_url=url,
            title=title,
            summary=item.get("summary", item.get("description", title)),
            affected_services=[company] if company else [],
            taxonomy_labels=self._classify(categories, title),
            ingested_at=DateTimeUtils.now_iso(),
            quality_score=quality_score(item, self._cfg.get("quality_fields", [])),
        )

    def _classify(self, categories: list, title: str) -> list[str]:
        """Map VOID categories to taxonomy labels."""
        labels = []
        text = " ".join(str(c) for c in categories) + " " + title
        text_lower = text.lower()

        mapping = {
            "infrastructure.network.dns": ["dns"],
            "infrastructure.database.connection_pool": ["database", "connection"],
            "infrastructure.compute.oom": ["oom", "memory"],
            "infrastructure.storage.disk_full": ["disk", "storage"],
            "operational.deployment.bad_deploy": ["deploy", "release", "rollout"],
            "operational.configuration.resource_limits": ["config", "misconfigur"],
            "operational.capacity.traffic_spike": ["traffic", "capacity", "scale", "load"],
            "application.dependency.cascading_failure": ["cascade", "cascading", "downstream"],
            "infrastructure.network.partition": ["network", "partition", "connectivity"],
            "application.memory.leak": ["memory leak", "leak"],
        }
        for label, keywords in mapping.items():
            if any(kw in text_lower for kw in keywords):
                labels.append(label)

        return labels
