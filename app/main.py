"""Main entry point for the SRE RL training environment.

Usage:
    python -m app.main train          # Run training episodes
    python -m app.main check-reward   # Verify reward function
    python -m app.main crawl          # Fetch real incidents
    python -m app.main export         # Save trajectories for fine-tuning
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("SRE RL Training Environment")
        print()
        print("Commands:")
        print("  train         Run training episodes (reads config/training.yaml)")
        print("  check-reward  Verify the reward function scores agents correctly")
        print("  crawl         Fetch real incidents from public APIs")
        print("  export        Save trajectories for LLM fine-tuning")
        print("  finetune      Run LLM fine-tuning on saved trajectories")
        print()
        print("Usage: python -m app.main <command>")
        print("Or use make: make train, make check-reward, make crawl")
        return

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip command from argv for sub-parsers

    if command == "train":
        from app.cli.training import main as train_main

        train_main()
    elif command == "check-reward":
        from app.cli.reward_check import main as check_main

        check_main()
    elif command == "crawl":
        from app.cli.crawler import main as crawl_main

        crawl_main()
    elif command == "export":
        sys.argv = [sys.argv[0], "--export", "training_data/trajectories.jsonl"]
        from app.cli.training import main as train_main

        train_main()
    elif command == "finetune":
        from app.cli.finetune import main as finetune_main

        finetune_main()
    else:
        print(f"Unknown command: {command}")
        print("Run 'python -m app.main' to see available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()
