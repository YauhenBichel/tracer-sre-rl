"""Tests for the failure taxonomy."""

from app.evaluation.scorers.diagnosis_scorer import DiagnosisScorer
from app.models.taxonomy import build_default_taxonomy


def test_taxonomy_builds():
    root = build_default_taxonomy()
    assert root.name == "root"
    assert len(root.children) > 0


def test_taxonomy_categories():
    root = build_default_taxonomy()
    assert {"infrastructure", "application", "operational", "external"} == set(root.children.keys())


def test_taxonomy_find():
    root = build_default_taxonomy()
    node = root.find("infrastructure.database.connection_pool")

    assert node is not None
    assert node.name == "connection_pool"


def test_taxonomy_find_missing():
    assert build_default_taxonomy().find("nonexistent.path") is None


def test_taxonomy_leaves():
    leaves = build_default_taxonomy().get_all_leaves()
    assert len(leaves) > 20
    assert all(leaf.is_leaf for leaf in leaves)


def test_similarity():
    scorer = DiagnosisScorer()

    assert (
        scorer._similarity("infrastructure.database.connection_pool", "infrastructure.database.connection_pool") == 1.0
    )
    assert (
        0.5
        < scorer._similarity("infrastructure.database.connection_pool", "infrastructure.database.replication_lag")
        < 1.0
    )
    assert 0.2 < scorer._similarity("infrastructure.database.connection_pool", "infrastructure.network.dns") < 0.6
    assert scorer._similarity("infrastructure.database.connection_pool", "application.memory.leak") < 0.3
