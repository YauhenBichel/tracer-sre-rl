"""Tests for training metrics and accuracy reporting."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from app.incidents.scenario_loader import ScenarioLoader
from app.training.episode_runner import EpisodeRunner
from app.training.metrics import print_accuracy_matrix, print_data_summary


def test_print_data_summary_shows_taxonomy():
    scenarios = ScenarioLoader().load_all()
    captured = StringIO()

    with patch("sys.stdout", captured):
        print_data_summary(scenarios)

    assert "Taxonomy" in captured.getvalue()


def test_print_accuracy_matrix_shows_overall():
    runner = EpisodeRunner()
    results = runner.run_batch(3)
    captured = StringIO()

    with patch("sys.stdout", captured):
        print_accuracy_matrix(results, runner.scenarios)

    assert "OVERALL" in captured.getvalue()


def test_print_accuracy_matrix_empty_results():
    scenarios = ScenarioLoader().load_all()
    captured = StringIO()

    with patch("sys.stdout", captured):
        print_accuracy_matrix([], scenarios)

    assert "No results" in captured.getvalue()
