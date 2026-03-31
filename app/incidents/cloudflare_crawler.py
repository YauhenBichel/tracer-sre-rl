"""Cloudflare incident crawler via Atlassian Statuspage API."""

from __future__ import annotations

import logging

import requests

from app.incidents.models import IncidentCrawler, NormalisedIncident
from app.utils.datetime_utils import DateTimeUtils

from .quality import quality_score

logger = logging.getLogger(__name__)

DEFAULT_IMPACT = "minor"


class CloudflareIncidentCrawler(IncidentCrawler):
    source_name = "cloudflare"

    def crawl(self) -> list[NormalisedIncident]:
        logger.info("Crawling Cloudflare incidents...")
        results = []
        max_pages = self._cfg.get("max_pages", 5)

        try:
            for page in range(1, max_pages + 1):
                resp = self._fetch(self._cfg["url"], params={"page": page})
                incidents = resp.json().get("incidents", [])
                if not incidents:
                    break
                for item in incidents:
                    try:
                        results.append(self._normalise(item))
                    except Exception as e:
                        logger.warning("Cloudflare normalise failed: %s", e)
        except requests.RequestException as e:
            logger.error("Cloudflare fetch failed: %s", e)
        logger.info("Cloudflare: %d incidents", len(results))
        return results

    def _normalise(self, item: dict) -> NormalisedIncident:
        timeline = [
            {"timestamp": u.get("created_at", ""), "event": u.get("body", ""), "status": u.get("status", "")}
            for u in item.get("incident_updates", [])
        ]
        return NormalisedIncident(
            id=f"{self.source_name}-{item['id']}",
            source=self.source_name,
            source_url=item.get("shortlink", ""),
            title=item.get("name", ""),
            summary=item.get("name", ""),
            timeline=timeline,
            affected_services=[c["name"] for c in item.get("components", [])],
            impact={"severity": item.get("impact", DEFAULT_IMPACT), "duration_minutes": self._duration(item)},
            ingested_at=DateTimeUtils.now_iso(),
            quality_score=quality_score(item, self._cfg["quality_fields"]),
        )

    @staticmethod
    def _duration(item: dict) -> int | None:
        try:
            if resolved := item.get("resolved_at"):
                return DateTimeUtils.duration_minutes(item["created_at"], resolved)
        except (ValueError, KeyError):
            pass
        return None
