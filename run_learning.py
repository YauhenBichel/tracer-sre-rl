#!/usr/bin/env python3
"""Demonstrate that the reward signal drives learning.

Runs a Q-learning agent over many episodes and shows the reward
improving over time with live progress.

Usage:
    python run_learning.py                    # 500 episodes
    python run_learning.py --episodes 1000    # more training
"""

from __future__ import annotations

import argparse
import logging

from src.environment.env import SREEnvironment
from src.generators.loader import ScenarioLoader
from src.training.learning_agent import QLearningAgent

logging.basicConfig(level=logging.WARNING)

WINDOW = 50


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning agent training demo")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    scenarios = ScenarioLoader().load_all()
    agent = QLearningAgent()
    rewards: list[float] = []

    print(f"Training Q-learning agent on {len(scenarios)} scenarios, {args.episodes} episodes...")
    print(f"{'Episode':>8}  {'Reward':>8}  {'Avg(50)':>8}  {'Epsilon':>8}  Progress")
    print("-" * 65)

    for ep in range(args.episodes):
        scenario = scenarios[ep % len(scenarios)]
        env = SREEnvironment(scenario=scenario, seed=ep)
        obs, _ = env.reset()

        episode_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            agent.update(obs, env, reward, terminated or truncated)
            episode_reward = reward if (terminated or truncated) else episode_reward

        agent.end_episode()
        rewards.append(episode_reward)
        env.close()

        # Live progress every WINDOW episodes
        if (ep + 1) % WINDOW == 0 or ep == args.episodes - 1:
            avg = sum(rewards[max(0, len(rewards) - WINDOW):]) / min(WINDOW, len(rewards))
            pct = (ep + 1) / args.episodes
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            print(f"{ep + 1:>8}  {episode_reward:>8.3f}  {avg:>8.3f}  {agent.epsilon:>8.3f}  {bar} {pct:.0%}")

    # Summary
    first = sum(rewards[:WINDOW]) / min(WINDOW, len(rewards))
    last = sum(rewards[-WINDOW:]) / min(WINDOW, len(rewards))
    improvement = ((last - first) / max(first, 0.001)) * 100

    print(f"\n{'='*65}")
    print(f"  First {WINDOW} episodes avg:  {first:.3f}")
    print(f"  Last {WINDOW} episodes avg:   {last:.3f}")
    print(f"  Improvement:            {improvement:+.1f}%")
    print(f"  Q-table states learned: {len(agent.q_table)}")

    if last > first:
        print(f"\n  The agent learned. Reward improved over training.")
    else:
        print(f"\n  No improvement detected. May need more episodes or tuning.")


if __name__ == "__main__":
    main()
