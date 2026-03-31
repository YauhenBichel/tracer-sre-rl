"""Tests for DateTimeUtils."""

from app.utils.datetime_utils import DateTimeUtils


def test_now_iso_returns_string():
    result = DateTimeUtils.now_iso()
    assert isinstance(result, str)
    assert "T" in result


def test_parse_iso_with_z_suffix():
    dt = DateTimeUtils.parse_iso("2024-01-15T14:30:00Z")
    assert dt.year == 2024
    assert dt.hour == 14


def test_parse_iso_with_offset():
    dt = DateTimeUtils.parse_iso("2024-01-15T14:30:00+00:00")
    assert dt.year == 2024


def test_duration_minutes():
    start = "2024-01-15T14:00:00Z"
    end = "2024-01-15T14:45:00Z"
    assert DateTimeUtils.duration_minutes(start, end) == 45


def test_duration_minutes_cross_hour():
    start = "2024-01-15T14:30:00Z"
    end = "2024-01-15T16:00:00Z"
    assert DateTimeUtils.duration_minutes(start, end) == 90
