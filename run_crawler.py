#!/usr/bin/env python3
"""Run the incident data crawler and optionally generate scenarios.

Crawls public incident data from GCP, Cloudflare, and GitHub post-mortems,
storing normalised results in a local SQLite database.

Usage:
    python run_crawler.py [--db-path incidents.db]
    python run_crawler.py --generate-scenarios [--output-dir scenarios/generated]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.crawler.incident_crawler import run_crawler
from src.crawler.repository.sqlite_repository import SqliteIncidentRepository
from src.crawler.scenario_generator import ScenarioGenerator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MIN_QUALITY_FOR_SCENARIO = 0.3


def main():
    parser = argparse.ArgumentParser(description="Incident Data Crawler")
    parser.add_argument(
        "--db-path", default="incidents.db",
        help="Path to SQLite database (default: incidents.db)",
    )
    parser.add_argument(
        "--generate-scenarios", action="store_true",
        help="Generate scenario YAML files from crawled incidents",
    )
    parser.add_argument(
        "--output-dir", default="scenarios/generated",
        help="Output directory for generated scenarios (default: scenarios/generated)",
    )
    args = parser.parse_args()

    print("Starting incident data crawler...")
    counts = run_crawler(db_path=args.db_path)

    print("\nCrawl complete. Incidents by source:")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
    print(f"  Total: {sum(counts.values())}")

    if args.generate_scenarios:
        _generate_scenarios(args.db_path, args.output_dir)


def _generate_scenarios(db_path: str, output_dir: str):
    """Load incidents from DB and generate scenario YAML files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    repo = SqliteIncidentRepository(db_path)
    try:
        incidents = repo.load_all()
    finally:
        repo.close()

    # Filter to incidents with taxonomy labels and minimum quality
    eligible = [
        inc for inc in incidents
        if inc.taxonomy_labels and inc.quality_score >= MIN_QUALITY_FOR_SCENARIO
    ]

    if not eligible:
        print("\nNo eligible incidents for scenario generation (need taxonomy_labels and quality >= "
              f"{MIN_QUALITY_FOR_SCENARIO}).")
        return

    generator = ScenarioGenerator()
    generated = 0
    for incident in eligible:
        try:
            generator.save(incident, str(output))
            generated += 1
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to generate scenario for %s: %s", incident.id, e)

    print(f"\nGenerated {generated} scenario(s) in {output}/")


if __name__ == "__main__":
    main()
