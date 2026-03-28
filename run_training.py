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

from src.crawler.scenario_generator import ScenarioGenerator
from src.generators.loader import ScenarioLoader
from src.models import ScenarioDefinition
from src.training import EpisodeRunner, export_jsonl, export_preference_pairs

logging.basicConfig(
    level=logging.INFO,
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
        from src.crawler.crawlers.aiops_dataset_loader import load_aiops_groundtruth
        incidents = load_aiops_groundtruth(aiops_path)
        generated = ScenarioGenerator().batch_generate(incidents, min_quality=0.0)
        scenarios.extend(generated)
        print(f"  Aiops-Dataset: {len(generated)} scenarios from {aiops_path}")

    # Crawled incident database
    db_path = sources.get("incident_db")
    if db_path and Path(db_path).exists():
        from src.crawler.repository.sqlite_repository import SqliteIncidentRepository
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
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Training config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count from config")
    parser.add_argument("--export", default=None,
                        help="Export trajectories as JSONL")
    parser.add_argument("--export-pairs", default=None,
                        help="Export preference pairs for DPO")
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

    # Load scenarios from all configured sources
    print(f"Loading data sources from {args.config}:")
    scenarios = _load_scenarios_from_config(config)

    if max_difficulty is not None:
        scenarios = [s for s in scenarios if s.difficulty <= max_difficulty]

    if not scenarios:
        print("No scenarios found. Check config/training.yaml sources.")
        return

    # Select agent from config
    agent_fn = _get_agent_fn(config.get("agent", {}))

    runner = EpisodeRunner(scenarios=scenarios)
    print(f"\nTraining on {len(runner.scenarios)} scenarios, {episodes} episodes...")

    results = runner.run_batch(episodes, agent_fn=agent_fn)
    stats = runner.stats

    print(f"\nTraining complete.")
    print(f"  Episodes:    {stats.episodes_run}")
    print(f"  Avg reward:  {stats.avg_reward:.3f}")
    print(f"\nPer-scenario breakdown:")
    for scenario_id, rewards in sorted(stats.rewards_by_scenario.items()):
        avg = sum(rewards) / len(rewards)
        print(f"  {scenario_id}: avg={avg:.3f} ({len(rewards)} episodes)")

    # Export (from CLI args or config)
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
        from demo import heuristic_agent_actions
        from src.environment.actions import ActionType
        from src.generators.telemetry import METRIC_INTERVAL_SECONDS

        def _heuristic(obs, env):
            if not hasattr(env, "_heuristic_iter"):
                env._heuristic_iter = iter(heuristic_agent_actions(env))
            try:
                action = next(env._heuristic_iter)
                return action.to_dict()
            except StopIteration:
                return {"action_type": ActionType.REMEDIATE, "target_service": 0,
                        "time_start": 0, "time_end": 0, "diagnosis_idx": 0, "remediation_idx": 0}
        return _heuristic

    if agent_type == "oracle":
        from demo import oracle_agent_actions
        from src.environment.actions import ActionType

        def _oracle(obs, env):
            if not hasattr(env, "_oracle_iter"):
                env._oracle_iter = iter(oracle_agent_actions(env))
            try:
                action = next(env._oracle_iter)
                return action.to_dict()
            except StopIteration:
                return {"action_type": ActionType.REMEDIATE, "target_service": 0,
                        "time_start": 0, "time_end": 0, "diagnosis_idx": 0, "remediation_idx": 0}
        return _oracle

    print(f"  Agent type: {agent_type} (using random)")
    return None


if __name__ == "__main__":
    main()
