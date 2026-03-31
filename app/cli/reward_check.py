#!/usr/bin/env python3
"""Verify the reward function scores investigation quality correctly.

Runs three test agents against failure scenarios to prove the reward
signal discriminates between bad, mediocre, and good investigation:
  - random:    picks random actions (should score lowest)
  - heuristic: investigates systematically but guesses diagnosis (middle)
  - oracle:    investigates + knows the correct answer (should score highest)
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

from app.incidents.scenario_loader import ScenarioLoader
from app.models import AgentAction
from app.rl_env.actions import ActionType
from app.rl_env.env import SREEnvironment
from app.telemetry.generator import METRIC_INTERVAL_SECONDS

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_LOG_QUERIES = 3
# Number of characters to show per observation in console output
OBSERVATION_PRINT_LIMIT = 500
DEFAULT_SEED = 42


# --- Shared helpers ---


def log_step(step: int, action: AgentAction, obs: dict, reward: float, env: SREEnvironment):
    name = ActionType(action.action_type).name
    target = env.service_names[min(action.target_service, env.num_services - 1)]
    print(f"--- Step {step}: {name} -> {target} ---")
    print(obs["text_observation"][:OBSERVATION_PRINT_LIMIT])
    if reward:
        print(f"\nReward: {reward:.4f}")
    print()


def log_episode_result(agent_name: str, scenario_name: str, step: int, reward: float, info: dict):
    print(f"\n[{agent_name}] {scenario_name} — Steps: {step}, Reward: {reward:.4f}")
    if bd := info.get("reward_breakdown"):
        w = bd.weights
        print(f"  Diagnosis:   {bd.diagnosis_score:.3f} (w={w.diagnosis})")
        print(f"  Efficiency:  {bd.efficiency_score:.3f} (w={w.efficiency})")
        print(f"  Remediation: {bd.remediation_score:.3f} (w={w.remediation})")
        print(f"  Safety:      {bd.safety_score:.3f} (w={w.safety})")


def run_episode(env: SREEnvironment, actions_iter, agent_name: str, verbose: bool = True):
    """Run a full episode given an action iterator. Returns (steps, reward, info)."""
    obs, info = env.reset()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"[{agent_name}] Scenario: {info['scenario']}")
        print(f"{'=' * 60}")
        print(f"{obs['text_observation']}\n")

    step = 0
    reward = 0.0
    terminated = truncated = False

    for action in actions_iter:
        obs, reward, terminated, truncated, info = env.step(action.to_dict())
        step += 1
        if verbose:
            log_step(step, action, obs, reward, env)
        if terminated or truncated:
            break

    log_episode_result(agent_name, info.get("scenario", env.scenario.name), step, reward, info)
    return step, reward, info


# --- Random agent ---


def random_agent_actions(env: SREEnvironment, rng: random.Random):
    """Yield random actions until the episode would end."""
    max_time = env.scenario.episode_duration_seconds // METRIC_INTERVAL_SECONDS - 1

    # Take a few random query actions, then diagnose and remediate
    query_types = [
        ActionType.QUERY_METRICS,
        ActionType.QUERY_LOGS,
        ActionType.QUERY_TRACES,
        ActionType.LIST_SERVICES,
        ActionType.LIST_ALERTS,
    ]

    num_queries = rng.randint(1, min(8, env.scenario.max_investigation_steps - 2))
    for _ in range(num_queries):
        yield AgentAction(
            action_type=rng.choice(query_types),
            target_service=rng.randint(0, env.num_services - 1),
            time_start=rng.randint(0, max_time // 2),
            time_end=rng.randint(max_time // 2, max_time),
        )

    # Random diagnosis
    yield AgentAction(
        action_type=ActionType.DIAGNOSE,
        diagnosis_idx=rng.randint(0, max(0, len(env.diagnosis_options) - 1)),
    )
    # Random remediation
    yield AgentAction(
        action_type=ActionType.REMEDIATE,
        remediation_idx=rng.randint(0, max(0, len(env.remediation_options) - 1)),
    )


# --- Heuristic agent (no gold labels) ---


def heuristic_agent_actions(env: SREEnvironment):
    """Systematic investigation, then pick first diagnosis/remediation (no cheating)."""
    max_time = env.scenario.episode_duration_seconds // METRIC_INTERVAL_SECONDS - 1
    half_time = max_time // 2

    yield AgentAction(action_type=ActionType.LIST_ALERTS)
    yield AgentAction(action_type=ActionType.LIST_SERVICES)

    for svc in range(env.num_services):
        yield AgentAction(
            action_type=ActionType.QUERY_METRICS,
            target_service=svc,
            time_start=half_time,
            time_end=max_time,
        )

    for svc in range(min(MAX_LOG_QUERIES, env.num_services)):
        yield AgentAction(
            action_type=ActionType.QUERY_LOGS,
            target_service=svc,
            time_start=half_time,
            time_end=max_time,
        )

    # Pick first option — no access to gold labels
    yield AgentAction(action_type=ActionType.DIAGNOSE, diagnosis_idx=0)
    yield AgentAction(action_type=ActionType.REMEDIATE, remediation_idx=0)


# --- Oracle agent (uses gold labels) ---


def oracle_agent_actions(env: SREEnvironment):
    """Systematic investigation + gold-standard diagnosis/remediation."""
    max_time = env.scenario.episode_duration_seconds // METRIC_INTERVAL_SECONDS - 1
    half_time = max_time // 2

    yield AgentAction(action_type=ActionType.LIST_ALERTS)
    yield AgentAction(action_type=ActionType.LIST_SERVICES)

    for svc in range(env.num_services):
        yield AgentAction(
            action_type=ActionType.QUERY_METRICS,
            target_service=svc,
            time_start=half_time,
            time_end=max_time,
        )

    for svc in range(min(MAX_LOG_QUERIES, env.num_services)):
        yield AgentAction(
            action_type=ActionType.QUERY_LOGS,
            target_service=svc,
            time_start=half_time,
            time_end=max_time,
        )

    # Use gold labels to pick correct diagnosis
    gold_labels = {g.taxonomy_label for g in env.scenario.gold_root_causes}
    diag_idx = next((i for i, opt in enumerate(env.diagnosis_options) if opt in gold_labels), 0)
    yield AgentAction(action_type=ActionType.DIAGNOSE, diagnosis_idx=diag_idx)

    # Use gold labels to pick best remediation
    best_idx, best_eff = 0, 0.0
    for i, opt in enumerate(env.remediation_options):
        for g in env.scenario.gold_remediations:
            if opt == g.action and g.effectiveness > best_eff:
                best_idx, best_eff = i, g.effectiveness
    yield AgentAction(action_type=ActionType.REMEDIATE, remediation_idx=best_idx)


# --- Runner ---

AGENTS = {
    "random": lambda env, seed: random_agent_actions(env, random.Random(seed)),
    "heuristic": lambda env, _seed: heuristic_agent_actions(env),
    "oracle": lambda env, _seed: oracle_agent_actions(env),
}


def main():
    parser = argparse.ArgumentParser(description="SRE RL Environment Demo")
    parser.add_argument("--scenario", "-s", help="Scenario YAML file path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--agent", choices=list(AGENTS.keys()), default=None, help="Agent type (default: run all three for comparison)"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show final results, not step-by-step output")
    args = parser.parse_args()

    loader = ScenarioLoader()
    loaded = loader.load(args.scenario) if args.scenario else None
    scenarios = [loaded] if loaded else loader.load_all()
    if not scenarios:
        logger.error("No scenarios found.")
        sys.exit(1)

    agent_names = [args.agent] if args.agent else list(AGENTS.keys())
    verbose = not args.quiet

    # Collect results for summary table
    results: list[dict] = []

    for scenario in scenarios:
        for agent_name in agent_names:
            env = SREEnvironment(scenario=scenario, seed=args.seed)
            action_iter = AGENTS[agent_name](env, args.seed)
            steps, reward, info = run_episode(env, action_iter, agent_name, verbose=verbose)
            bd = info.get("reward_breakdown")
            results.append(
                {
                    "scenario": scenario.name,
                    "agent": agent_name,
                    "reward": reward,
                    "steps": steps,
                    "diagnosis": bd.diagnosis_score if bd else 0.0,
                    "efficiency": bd.efficiency_score if bd else 0.0,
                    "remediation": bd.remediation_score if bd else 0.0,
                    "safety": bd.safety_score if bd else 0.0,
                }
            )
            env.close()

    # Print summary table
    if len(results) > 1:
        print(f"\n{'=' * 90}")
        print("BASELINE COMPARISON")
        print(f"{'=' * 90}")
        print(f"{'Scenario':<40} {'Agent':<12} {'Reward':>7} {'Diag':>6} {'Eff':>6} {'Rem':>6} {'Safe':>6}")
        print("-" * 90)
        for r in results:
            print(
                f"{r['scenario'][:39]:<40} {r['agent']:<12} {r['reward']:>7.3f} "
                f"{r['diagnosis']:>6.3f} {r['efficiency']:>6.3f} {r['remediation']:>6.3f} {r['safety']:>6.3f}"
            )

        # Print per-agent averages
        print("-" * 90)
        for agent_name in agent_names:
            agent_results = [r for r in results if r["agent"] == agent_name]
            avg = {
                k: sum(r[k] for r in agent_results) / len(agent_results)
                for k in ("reward", "diagnosis", "efficiency", "remediation", "safety")
            }
            print(
                f"{'AVERAGE':<40} {agent_name:<12} {avg['reward']:>7.3f} "
                f"{avg['diagnosis']:>6.3f} {avg['efficiency']:>6.3f} {avg['remediation']:>6.3f} {avg['safety']:>6.3f}"
            )


if __name__ == "__main__":
    main()
