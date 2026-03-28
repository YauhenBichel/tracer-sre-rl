"""GitHub post-mortem crawler — parses curated post-mortem collections."""

from __future__ import annotations

import hashlib
import logging
import re

import requests

from src.crawler.models import IncidentCrawler, NormalisedIncident
from src.utils.datetime_utils import DateTimeUtils

logger = logging.getLogger(__name__)

LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
SECTION_PREFIX = "##"
MAX_TITLE_LENGTH = 200
DEFAULT_QUALITY_SCORE = 0.4


class GitHubPostmortemCrawler(IncidentCrawler):
    source_name = "github_postmortems"

    def crawl(self) -> list[NormalisedIncident]:
        logger.info("Crawling GitHub post-mortems...")
        try:
            resp = self._fetch(self._cfg["url"])
        except requests.RequestException as e:
            logger.error("GitHub PM fetch failed: %s", e)
            return []

        skip_urls = self._cfg.get("skip_urls", [])
        results = []
        category = ""

        for line in resp.text.split("\n"):
            if line.startswith(SECTION_PREFIX):
                category = line.strip("# ").strip()
            match = LINK_RE.search(line)
            if match:
                desc, url = match.group(1), match.group(2)
                if any(skip in url for skip in skip_urls):
                    continue
                results.append(
                    NormalisedIncident(
                        id=f"{self.source_name}-{hashlib.sha256(url.encode()).hexdigest()[:12]}",
                        source=self.source_name,
                        source_url=url,
                        title=desc[:MAX_TITLE_LENGTH],
                        summary=f"[{category}] {desc}",
                        taxonomy_labels=self._classify(f"{desc} {category}"),
                        ingested_at=DateTimeUtils.now_iso(),
                        quality_score=DEFAULT_QUALITY_SCORE,
                    )
                )
        logger.info("GitHub PM: %d incidents", len(results))
        return results

    def _classify(self, text: str) -> list[str]:
        patterns = self._cfg.get("category_patterns", {})
        return [label for label, pattern in patterns.items() if re.search(pattern, text)]
