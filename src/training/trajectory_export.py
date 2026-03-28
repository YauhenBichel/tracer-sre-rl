"""Trajectory export — converts collected episodes into training data.

Supports two export formats:

1. JSONL for LLM fine-tuning (GRPO/DPO):
   Each line is a training example: prompt (observations) + completion (actions) + reward.
   Use with opensre's LLM to fine-tune investigation strategy via reinforcement learning.

2. Pairs for preference learning (DPO/RLHF):
   Pairs high-reward and low-reward trajectories on the same scenario
   for Direct Preference Optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.training.episode_runner import EpisodeResult

logger = logging.getLogger(__name__)


def export_jsonl(results: list[EpisodeResult], output_path: str) -> int:
    """Export episodes as JSONL training data for LLM fine-tuning.

    Each line contains:
      - prompt: the scenario context + observations the agent saw
      - actions: the sequence of tool calls the agent made
      - reward: the final episode reward (used as GRPO signal)
      - metadata: scenario_id, seed, component scores

    Returns number of examples written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(path, "w") as f:
        for result in results:
            if not result.trajectory:
                continue

            # Build the prompt from observations
            observations = [step.observation for step in result.trajectory]
            actions = [
                {"tool": step.action_type, "target": step.target_service}
                for step in result.trajectory
            ]

            example = {
                "prompt": observations[0] if observations else "",
                "observations": observations,
                "actions": actions,
                "reward": result.reward,
                "scenario_id": result.scenario_id,
                "scenario_name": result.scenario_name,
                "seed": result.seed,
                "scores": {
                    "diagnosis": result.diagnosis_score,
                    "efficiency": result.efficiency_score,
                    "remediation": result.remediation_score,
                    "safety": result.safety_score,
                },
            }
            f.write(json.dumps(example) + "\n")
            count += 1

    logger.info("Exported %d episodes to %s", count, path)
    return count


def export_preference_pairs(
    results: list[EpisodeResult],
    output_path: str,
    reward_threshold: float = 0.2,
) -> int:
    """Export preference pairs for DPO training.

    Groups episodes by scenario_id, then pairs high-reward (chosen)
    with low-reward (rejected) trajectories. The reward gap must exceed
    reward_threshold to form a meaningful pair.

    Format per line: {"chosen": {...}, "rejected": {...}, "scenario_id": "..."}

    Returns number of pairs written.
    """
    # Group by scenario
    by_scenario: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_scenario.setdefault(result.scenario_id, []).append(result)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(path, "w") as f:
        for scenario_id, episodes in by_scenario.items():
            if len(episodes) < 2:
                continue

            sorted_eps = sorted(episodes, key=lambda e: e.reward, reverse=True)
            best = sorted_eps[0]
            worst = sorted_eps[-1]

            if best.reward - worst.reward < reward_threshold:
                continue

            pair = {
                "scenario_id": scenario_id,
                "chosen": _episode_to_dict(best),
                "rejected": _episode_to_dict(worst),
                "reward_gap": round(best.reward - worst.reward, 4),
            }
            f.write(json.dumps(pair) + "\n")
            count += 1

    logger.info("Exported %d preference pairs to %s", count, path)
    return count


def _episode_to_dict(result: EpisodeResult) -> dict:
    """Convert an episode result to a serialisable dict."""
    return {
        "reward": result.reward,
        "steps": result.steps,
        "seed": result.seed,
        "actions": [
            {"tool": step.action_type, "target": step.target_service}
            for step in result.trajectory
        ],
        "observations": [step.observation for step in result.trajectory],
    }
