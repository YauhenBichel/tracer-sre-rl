#!/usr/bin/env python3
"""Run the RL training loop — episodes across scenarios with stats collection.

Usage:
    # Train on builtin scenarios only
    python run_training.py --episodes 100

    # Train on real incident data from crawled database
    python run_training.py --episodes 1000 --incident-db data/incidents.db

    # Curriculum: start with easy scenarios only
    python run_training.py --episodes 500 --max-difficulty 0.4

    # Export trajectories for LLM fine-tuning
    python run_training.py --episodes 1000 --export training_data/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import logging

from src.training import EpisodeRunner, export_jsonl, export_preference_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SRE RL Training Loop")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    parser.add_argument("--max-difficulty", type=float, default=None,
                        help="Maximum scenario difficulty for curriculum learning")
    parser.add_argument("--incident-db", default=None,
                        help="Path to SQLite database of crawled incidents (adds real-data scenarios)")
    parser.add_argument("--export", default=None,
                        help="Export trajectories as JSONL for LLM fine-tuning")
    parser.add_argument("--export-pairs", default=None,
                        help="Export preference pairs for DPO training")
    args = parser.parse_args()

    runner = EpisodeRunner(
        max_difficulty=args.max_difficulty,
        incident_db_path=args.incident_db,
    )
    print(f"Training on {len(runner.scenarios)} scenarios, {args.episodes} episodes...")

    results = runner.run_batch(args.episodes)
    stats = runner.stats

    print(f"\nTraining complete.")
    print(f"  Episodes:    {stats.episodes_run}")
    print(f"  Avg reward:  {stats.avg_reward:.3f}")
    print(f"\nPer-scenario breakdown:")
    for scenario_id, rewards in sorted(stats.rewards_by_scenario.items()):
        avg = sum(rewards) / len(rewards)
        print(f"  {scenario_id}: avg={avg:.3f} ({len(rewards)} episodes)")

    if args.export:
        count = export_jsonl(results, args.export)
        print(f"\nExported {count} trajectories to {args.export}")

    if args.export_pairs:
        count = export_preference_pairs(results, args.export_pairs)
        print(f"Exported {count} preference pairs to {args.export_pairs}")


if __name__ == "__main__":
    main()
