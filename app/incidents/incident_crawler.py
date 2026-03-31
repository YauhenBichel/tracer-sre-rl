"""Run all incident crawlers and store results."""

from __future__ import annotations

import logging

from .crawlers import (
    CloudflareIncidentCrawler,
    GCPIncidentCrawler,
    GitHubPostmortemCrawler,
    IncidentCrawler,
)
from .repository import IncidentRepository, SqliteIncidentRepository

logger = logging.getLogger(__name__)

DEFAULT_CRAWLERS: list[type[IncidentCrawler]] = [
    GCPIncidentCrawler,
    CloudflareIncidentCrawler,
    GitHubPostmortemCrawler,
]


def run_crawler(repository: IncidentRepository | None = None, db_path: str = "incidents.db") -> dict[str, int]:
    repo = repository or SqliteIncidentRepository(db_path)
    try:
        for crawler_cls in DEFAULT_CRAWLERS:
            try:
                crawler = crawler_cls()
                for incident in crawler.crawl():
                    repo.save(incident)
            except Exception as e:
                logger.error("%s crawler failed: %s", crawler_cls.source_name, e)

        counts = repo.count_by_source()
        logger.info("Total incidents: %d", repo.count())
        return counts
    finally:
        if repository is None:
            repo.close()
