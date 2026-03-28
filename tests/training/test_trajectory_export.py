"""Tests for trajectory export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.training.episode_runner import EpisodeRunner
from src.training.trajectory_export import export_jsonl, export_preference_pairs


def test_export_jsonl_produces_valid_json():
    runner = EpisodeRunner()
    results = runner.run_batch(3)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    count = export_jsonl(results, path)
    assert count == 3

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert "prompt" in data
        assert "actions" in data
        assert "reward" in data
        assert isinstance(data["reward"], float)
        assert len(data["actions"]) > 0


def test_export_jsonl_contains_trajectory():
    runner = EpisodeRunner()
    results = runner.run_batch(1)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    export_jsonl(results, path)
    data = json.loads(Path(path).read_text().strip())

    assert len(data["observations"]) == len(data["actions"])
    for action in data["actions"]:
        assert "tool" in action
        assert "target" in action


def test_export_preference_pairs():
    runner = EpisodeRunner()
    # Run enough episodes to get reward variance
    results = runner.run_batch(20)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name

    count = export_preference_pairs(results, path, reward_threshold=0.01)

    if count > 0:
        line = Path(path).read_text().strip().split("\n")[0]
        pair = json.loads(line)
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"]["reward"] > pair["rejected"]["reward"]
        assert pair["reward_gap"] >= 0.01
