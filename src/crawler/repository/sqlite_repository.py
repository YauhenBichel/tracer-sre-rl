"""SQLite implementation of IncidentRepository.

See incident_repository.py for the interface contract.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from src.crawler.models import NormalisedIncident


class SqliteIncidentRepository:

    def __init__(self, db_path: str = "incidents.db"):
        self._conn = sqlite3.connect(db_path)
        self._ensure_schema()

    def save(self, incident: NormalisedIncident) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO incidents (id, source, title, data, ingested_at, quality_score) VALUES (?, ?, ?, ?, ?, ?)",
            (incident.id, incident.source, incident.title, json.dumps(asdict(incident)), incident.ingested_at, incident.quality_score),
        )
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    def count_by_source(self) -> dict[str, int]:
        return dict(self._conn.execute("SELECT source, COUNT(*) FROM incidents GROUP BY source").fetchall())

    def load_all(self) -> list[NormalisedIncident]:
        """Load all incidents from the database."""
        rows = self._conn.execute("SELECT data FROM incidents ORDER BY ingested_at DESC").fetchall()
        return [NormalisedIncident(**json.loads(row[0])) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                data JSON NOT NULL,
                ingested_at TEXT NOT NULL,
                quality_score REAL DEFAULT 0.0
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON incidents(source)")
        self._conn.commit()
