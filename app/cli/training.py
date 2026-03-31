#!/usr/bin/env python3
"""Run the RL training loop.

Reads data sources from config/training.yaml by default. Edit that file
to change what scenarios are used — no code changes needed.

Usage:
    python run_training.py                    # Uses config/training.yaml
    python run_training.py --episodes 500     # Override episode count
    python run_training.py --config my.yaml   # Use a different config
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from app.cli.reward_check import heuristic_agent_actions, oracle_agent_actions
from app.incidents.aiops_dataset_loader import load_aiops_groundtruth
from app.incidents.repository.sqlite_repository import SqliteIncidentRepository
from app.incidents.scenario_generator import ScenarioGenerator
from app.incidents.scenario_loader import ScenarioLoader
from app.models import ScenarioDefinition
from app.rl_env.actions import ActionType
from app.training import EpisodeRunner, export_jsonl, export_preference_pairs
from app.training.metrics import print_accuracy_matrix, print_data_summary

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_CONFIG = "config/training.yaml"


def _load_scenarios_from_config(config: dict) -> list[ScenarioDefinition]:
    """Load scenarios from all sources listed in config."""
    sources = config.get("sources", {})
    scenarios: list[ScenarioDefinition] = []

    # Builtin YAML scenarios
    builtin_dir = sources.get("builtin_scenarios")
    if builtin_dir and Path(builtin_dir).exists():
        loader = ScenarioLoader(builtin_dir)
        builtin = loader.load_all()
        scenarios.extend(builtin)
        print(f"  Builtin scenarios: {len(builtin)} from {builtin_dir}")

    # Aiops-Dataset groundtruth CSV
    aiops_path = sources.get("aiops_groundtruth")
    if aiops_path and Path(aiops_path).exists():
        incidents = load_aiops_groundtruth(aiops_path)
        generated = ScenarioGenerator().batch_generate(incidents, min_quality=0.0)
        scenarios.extend(generated)
        print(f"  Aiops-Dataset: {len(generated)} scenarios from {aiops_path}")

    # Crawled incident database
    db_path = sources.get("incident_db")
    if db_path and Path(db_path).exists():
        repo = SqliteIncidentRepository(db_path)
        try:
            incidents = repo.load_all()
        finally:
            repo.close()
        generated = ScenarioGenerator().batch_generate(incidents)
        scenarios.extend(generated)
        print(f"  Crawled incidents: {len(generated)} scenarios from {db_path}")

    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="SRE RL Training Loop")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"Training config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count from config")
    parser.add_argument("--export", default=None, help="Export trajectories as JSONL")
    parser.add_argument("--export-pairs", default=None, help="Export preference pairs for DPO")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Create it from the template: cp config/training.yaml config/my_training.yaml")
        return
    except yaml.YAMLError as e:
        print(f"Invalid YAML in {args.config}: {e}")
        return

    episodes = args.episodes or config.get("episodes", 100)
    max_difficulty = config.get("max_difficulty")

    # Step 1: Load data
    print(f"\n{'=' * 60}")
    print("Step 1/4: Loading data sources")
    print(f"{'=' * 60}")
    scenarios = _load_scenarios_from_config(config)

    if max_difficulty is not None:
        before = len(scenarios)
        scenarios = [s for s in scenarios if s.difficulty <= max_difficulty]
        print(f"  Difficulty filter (≤{max_difficulty}): {before} → {len(scenarios)} scenarios")

    if not scenarios:
        print("  No scenarios found. Check config/training.yaml sources.")
        return

    # Step 2: Validate
    print(f"\n{'=' * 60}")
    print("Step 2/4: Validating scenarios")
    print(f"{'=' * 60}")
    agent_fn = _get_agent_fn(config.get("agent", {}))
    runner = EpisodeRunner(scenarios=scenarios)
    print(f"  Valid scenarios: {len(runner.scenarios)}")
    print(f"  Train set: {len(runner.train_scenarios)} scenarios")
    print(f"  Eval set:  {len(runner.eval_scenarios)} scenarios (held out)")
    print_data_summary(runner.scenarios)

    # Step 3: Train
    print(f"\n{'=' * 60}")
    print(f"Step 3/4: Training ({episodes} episodes)")
    print(f"{'=' * 60}")
    results = runner.run_batch(episodes, agent_fn=agent_fn)
    stats = runner.stats

    # Step 4: Evaluate
    eval_episodes = config.get("eval_episodes", 20)
    print(f"\n{'=' * 60}")
    print(f"Step 4/4: Evaluating on held-out set ({eval_episodes} episodes)")
    print(f"{'=' * 60}")
    eval_reward = runner.evaluate(num_episodes=eval_episodes, agent_fn=agent_fn)

    _print_summary(stats, eval_reward, results, runner.scenarios)
    _export_results(args, config, results)


def _print_summary(stats, eval_reward: float, results, scenarios) -> None:
    """Print training results summary."""
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"  Train episodes:   {stats.episodes_run}")
    print(f"  Train avg reward: {stats.avg_reward:.3f}")
    print(f"  Eval avg reward:  {eval_reward:.3f}")
    gap = abs(stats.avg_reward - eval_reward) / max(stats.avg_reward, 0.001)
    if gap < 0.1:
        print(f"  Generalisation:   good (train-eval gap: {gap:.1%})")
    elif gap < 0.3:
        print(f"  Generalisation:   moderate (train-eval gap: {gap:.1%})")
    else:
        print(f"  Generalisation:   poor (train-eval gap: {gap:.1%} — may be overfitting)")

    print(f"\n{'=' * 60}")
    print("Accuracy by Taxonomy Label")
    print(f"{'=' * 60}")
    print_accuracy_matrix(results, scenarios)


def _export_results(args, config: dict, results) -> None:
    """Export trajectories and preference pairs if configured."""
    export_path = args.export or config.get("export_jsonl")
    if export_path:
        count = export_jsonl(results, export_path)
        print(f"\nExported {count} trajectories to {export_path}")

    pairs_path = args.export_pairs or config.get("export_pairs")
    if pairs_path:
        count = export_preference_pairs(results, pairs_path)
        print(f"Exported {count} preference pairs to {pairs_path}")


def _get_agent_fn(agent_config: dict):
    """Return an agent function based on config."""
    agent_type = agent_config.get("type", "random") if isinstance(agent_config, dict) else "random"

    if agent_type == "random":
        return None  # EpisodeRunner uses _random_agent by default

    if agent_type == "heuristic":

        def _heuristic(_obs, env):
            if not hasattr(env, "heuristic_iter"):
                env.heuristic_iter = iter(heuristic_agent_actions(env))
            try:
                action = next(env.heuristic_iter)
                return action.to_dict()
            except StopIteration:
                return {
                    "action_type": ActionType.REMEDIATE,
                    "target_service": 0,
                    "time_start": 0,
                    "time_end": 0,
                    "diagnosis_idx": 0,
                    "remediation_idx": 0,
                }

        return _heuristic

    if agent_type == "oracle":

        def _oracle(_obs, env):
            if not hasattr(env, "oracle_iter"):
                env.oracle_iter = iter(oracle_agent_actions(env))
            try:
                action = next(env.oracle_iter)
                return action.to_dict()
            except StopIteration:
                return {
                    "action_type": ActionType.REMEDIATE,
                    "target_service": 0,
                    "time_start": 0,
                    "time_end": 0,
                    "diagnosis_idx": 0,
                    "remediation_idx": 0,
                }

        return _oracle

    print(f"  Agent type: {agent_type} (using random)")
    return None


if __name__ == "__main__":
    main()
