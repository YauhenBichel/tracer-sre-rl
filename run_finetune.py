#!/usr/bin/env python3
"""Fine-tune an LLM on collected investigation trajectories.

Reads trajectory JSONL from make export, formats it for GRPO or DPO,
and runs the training loop. This is the step that makes opensre's
LLM actually improve from the RL environment's reward signal.

Requires: pip install trl transformers torch

Usage:
    # Step 1: Collect trajectories
    make export

    # Step 2: Fine-tune (requires GPU)
    python run_finetune.py --data training_data/trajectories.jsonl
    python run_finetune.py --data training_data/trajectories.jsonl --method dpo --pairs training_data/pairs.jsonl

    # Step 3: Deploy to opensre
    # Update opensre's LLM config to point to the fine-tuned model
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_trajectories(path: str) -> list[dict]:
    """Load JSONL trajectory file."""
    data = []
    with open(path) as f:
        data.extend(json.loads(line) for line in f)
    logger.info("Loaded %d trajectories from %s", len(data), path)
    return data


def _format_for_grpo(trajectories: list[dict]) -> list[dict]:
    """Format trajectories for Group Relative Policy Optimisation.

    GRPO needs: prompt, completion, reward.
    - prompt: the initial alert observation
    - completion: the sequence of tool calls the agent made
    - reward: the episode reward (used to weight the update)
    """
    examples = []
    for traj in trajectories:
        prompt = traj.get("prompt", "")
        if not prompt:
            continue

        # Format actions as a tool-call sequence
        actions = traj.get("actions", [])
        completion = "\n".join(
            f"Action: {a['tool']}(target={a['target']})" for a in actions
        )

        examples.append({
            "prompt": prompt,
            "completion": completion,
            "reward": traj.get("reward", 0.0),
        })

    logger.info("Formatted %d examples for GRPO", len(examples))
    return examples


def _format_for_dpo(pairs_path: str) -> list[dict]:
    """Format preference pairs for Direct Preference Optimisation.

    DPO needs: prompt, chosen, rejected.
    """
    examples = []
    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line)
            chosen = pair.get("chosen", {})
            rejected = pair.get("rejected", {})

            chosen_actions = "\n".join(
                f"Action: {a['tool']}(target={a['target']})" for a in chosen.get("actions", [])
            )
            rejected_actions = "\n".join(
                f"Action: {a['tool']}(target={a['target']})" for a in rejected.get("actions", [])
            )

            if chosen.get("observations"):
                examples.append({
                    "prompt": chosen["observations"][0],
                    "chosen": chosen_actions,
                    "rejected": rejected_actions,
                })

    logger.info("Formatted %d pairs for DPO", len(examples))
    return examples


def run_grpo(data_path: str, model_name: str, output_dir: str, epochs: int) -> None:
    """Run GRPO fine-tuning on trajectory data."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415
    except ImportError:
        print("GRPO requires: pip install trl transformers torch")
        print("Then run on a machine with GPU.")
        print()
        _show_dry_run(data_path, "grpo")
        return

    trajectories = _load_trajectories(data_path)
    examples = _format_for_grpo(trajectories)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        logging_steps=10,
    )

    trainer = GRPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=examples,
    )

    print(f"Training for {epochs} epochs on {len(examples)} examples...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


def run_dpo(pairs_path: str, model_name: str, output_dir: str, epochs: int) -> None:
    """Run DPO fine-tuning on preference pairs."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from trl import DPOConfig, DPOTrainer  # noqa: PLC0415
    except ImportError:
        print("DPO requires: pip install trl transformers torch")
        print("Then run on a machine with GPU.")
        print()
        _show_dry_run(pairs_path, "dpo")
        return

    examples = _format_for_dpo(pairs_path)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        logging_steps=10,
    )

    trainer = DPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=examples,
    )

    print(f"Training for {epochs} epochs on {len(examples)} pairs...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


def _show_dry_run(data_path: str, method: str) -> None:
    """Show what would happen without actually running training."""
    print(f"{'='*60}")
    print(f"Dry run: {method.upper()} fine-tuning")
    print(f"{'='*60}")

    data = _load_trajectories(data_path)
    if method == "grpo":
        examples = _format_for_grpo(data)
    else:
        examples = data

    print(f"  Training examples: {len(examples)}")
    if examples:
        rewards = [e.get("reward", 0) for e in (data if method == "grpo" else examples)]
        print(f"  Reward range: {min(rewards):.3f} - {max(rewards):.3f}")
        print(f"  Avg reward: {sum(rewards)/len(rewards):.3f}")

    print("\n  Example prompt (first 200 chars):")
    if examples:
        print(f"    {examples[0].get('prompt', '')[:200]}...")

    print("\n  To run for real:")
    print("    pip install trl transformers torch")
    print(f"    python run_finetune.py --data {data_path} --method {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLM on investigation trajectories")
    parser.add_argument("--data", default="training_data/trajectories.jsonl",
                        help="Path to trajectory JSONL")
    parser.add_argument("--pairs", default="training_data/pairs.jsonl",
                        help="Path to preference pairs JSONL (for DPO)")
    parser.add_argument("--method", choices=["grpo", "dpo"], default="grpo",
                        help="Fine-tuning method (default: grpo)")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                        help="Base model to fine-tune")
    parser.add_argument("--output", default="models/sre-agent",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"No trajectory data found at {args.data}")
        print("Run first: make export")
        return

    if args.method == "grpo":
        run_grpo(args.data, args.model, args.output, args.epochs)
    else:
        if not Path(args.pairs).exists():
            print(f"No preference pairs found at {args.pairs}")
            print("Run first: python run_training.py --export-pairs training_data/pairs.jsonl")
            return
        run_dpo(args.pairs, args.model, args.output, args.epochs)


if __name__ == "__main__":
    main()
