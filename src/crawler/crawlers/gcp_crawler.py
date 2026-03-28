"""GCP incident crawler."""

from __future__ import annotations

import logging

import requests

from src.crawler.models import IncidentCrawler, NormalisedIncident
from src.utils.datetime_utils import DateTimeUtils

from .quality import quality_score

logger = logging.getLogger(__name__)

DEFAULT_SEVERITY = "medium"
DEFAULT_TITLE = "Unknown"


class GCPIncidentCrawler(IncidentCrawler):
    source_name = "gcp"

    def crawl(self) -> list[NormalisedIncident]:
        logger.info("Crawling GCP incidents...")
        try:
            resp = self._fetch(self._cfg["url"])
        except requests.RequestException as e:
            logger.error("GCP fetch failed: %s", e)
            return []

        results = []
        for item in resp.json():
            try:
                results.append(self._normalise(item))
            except Exception as e:
                logger.warning("GCP normalise failed: %s", e)
        logger.info("GCP: %d incidents", len(results))
        return results

    def _normalise(self, item: dict) -> NormalisedIncident:
        updates = item.get("most-recent-update", item.get("updates", []))
        if isinstance(updates, dict):
            updates = [updates]
        timeline = [
            {"timestamp": u.get("when", ""), "event": u.get("text", "")}
            for u in (updates if isinstance(updates, list) else [])
        ]
        number = item.get("number", item.get("id", ""))

        return NormalisedIncident(
            id=f"{self.source_name}-{number}",
            source=self.source_name,
            source_url=f"{self._cfg['base_incident_url']}{number}",
            title=item.get("external_desc", item.get("service_name", DEFAULT_TITLE)),
            summary=item.get("external_desc", ""),
            timeline=timeline,
            affected_services=[item["service_name"]] if "service_name" in item else [],
            impact={"severity": item.get("severity", DEFAULT_SEVERITY)},
            ingested_at=DateTimeUtils.now_iso(),
            quality_score=quality_score(item, self._cfg["quality_fields"]),
        )
