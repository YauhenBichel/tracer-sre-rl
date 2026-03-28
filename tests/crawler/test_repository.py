"""Tests for incident repository."""

import pytest
from src.crawler.models import NormalisedIncident
from src.crawler.repository.sqlite_repository import SqliteIncidentRepository


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "test.db")
    repository = SqliteIncidentRepository(db_path)
    yield repository
    repository.close()


@pytest.fixture
def sample_incident():
    return NormalisedIncident(
        id="test-001",
        source="test",
        source_url="https://example.com",
        title="Test incident",
        summary="A test incident",
        ingested_at="2024-01-01T00:00:00",
        quality_score=0.8,
    )


def test_save_and_count(repo, sample_incident):
    assert repo.count() == 0
    repo.save(sample_incident)
    assert repo.count() == 1


def test_upsert_replaces(repo, sample_incident):
    repo.save(sample_incident)
    repo.save(sample_incident)
    assert repo.count() == 1


def test_count_by_source(repo):
    repo.save(NormalisedIncident(id="a", source="gcp", source_url="", title="A", summary="", ingested_at=""))
    repo.save(NormalisedIncident(id="b", source="gcp", source_url="", title="B", summary="", ingested_at=""))
    repo.save(NormalisedIncident(id="c", source="cloudflare", source_url="", title="C", summary="", ingested_at=""))
    counts = repo.count_by_source()
    assert counts["gcp"] == 2
    assert counts["cloudflare"] == 1


def test_empty_repo(repo):
    assert repo.count() == 0
    assert repo.count_by_source() == {}
