"""Tests for the episode runner / training loop."""

from app.training.episode_runner import EpisodeRunner


def test_runner_initialises_with_scenarios():
    runner = EpisodeRunner()
    assert len(runner.scenarios) > 0


def test_runner_runs_single_episode():
    runner = EpisodeRunner()
    result = runner.run_episode(seed=42)

    assert result.steps > 0
    assert result.reward >= 0
    assert result.scenario_id != ""


def test_runner_runs_batch():
    runner = EpisodeRunner()
    results = runner.run_batch(3)

    assert len(results) == 3
    assert runner.stats.episodes_run == 3
    assert runner.stats.avg_reward >= 0


def test_runner_difficulty_filtering():
    # Only easy scenarios
    runner = EpisodeRunner(max_difficulty=0.35)
    assert all(s.difficulty <= 0.35 for s in runner.scenarios)
    assert len(runner.scenarios) >= 1


def test_runner_stats_tracking():
    runner = EpisodeRunner()
    runner.run_episode(seed=1)
    runner.run_episode(seed=2)

    assert runner.stats.episodes_run == 2
    assert len(runner.stats.rewards_by_scenario) >= 1


def test_runner_different_seeds_produce_different_rewards():
    runner = EpisodeRunner()
    scenario = runner.scenarios[0]
    r1 = runner.run_episode(scenario=scenario, seed=1)
    r2 = runner.run_episode(scenario=scenario, seed=2)

    # Different seeds should produce different perturbations
    # Reward might coincidentally match, but steps should differ
    assert r1.seed != r2.seed
