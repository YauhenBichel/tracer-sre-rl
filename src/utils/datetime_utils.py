"""Reusable datetime utilities."""

from __future__ import annotations

from datetime import datetime, timezone


class DateTimeUtils:

    @staticmethod
    def now_iso() -> str:
        """Current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def parse_iso(value: str) -> datetime:
        """Parse an ISO 8601 string, handling trailing 'Z' as UTC."""
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def duration_minutes(start_iso: str, end_iso: str) -> int:
        """Calculate duration in minutes between two ISO 8601 timestamps."""
        start = DateTimeUtils.parse_iso(start_iso)
        end = DateTimeUtils.parse_iso(end_iso)
        return int((end - start).total_seconds() / 60)
