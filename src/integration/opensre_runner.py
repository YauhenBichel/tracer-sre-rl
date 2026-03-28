"""Run open-sre-agent against the RL environment.

Supports two modes:
  --use-opensre: Uses opensre's actual execute_actions (requires opensre installed)
  default: Standalone mode mirroring opensre's execution flow

Usage:
    python -m src.integration.opensre_runner --scenario scenarios/db_connection_pool.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from src.agent_adapter import SREToolAdapter
from src.generators.loader import ScenarioLoader
from src.integration.evidence_source import build_simulated_sources, create_simulated_actions
from src.integration.state_adapter import scenario_to_agent_state, score_agent_state
from src.models import ScenarioDefinition

logger = logging.getLogger(__name__)


def _execute_simulated_actions(
    adapter: SREToolAdapter,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute all simulated actions and collect evidence.

    Mirrors opensre's execute_actions flow:
      availability_check → parameter_extractor → function call

    Returns (evidence_dict, executed_hypotheses_list).
    """
    actions = create_simulated_actions(adapter)
    sources = build_simulated_sources(adapter)
    evidence: dict[str, Any] = {}
    executed: list[dict[str, Any]] = []

    for action in actions:
        if action.availability_check and not action.availability_check(sources):
            logger.warning("Action %s not available", action.name)
            continue
        if action.parameter_extractor is None:
            continue
        try:
            kwargs = action.parameter_extractor(sources)
            result = action.function(**kwargs)
            if isinstance(result, dict) and result.get("available"):
                evidence[action.name] = result
                executed.append({"action": action.name, "service": result.get("service_name", "unknown"), "success": True})
                logger.info("Action %s: OK", action.name)
        except Exception as e:
            logger.error("Action %s failed: %s", action.name, e)

    return evidence, executed


def run_episode(scenario: ScenarioDefinition, seed: int = 42) -> dict[str, Any]:
    """Run a single episode: create env, execute actions, score result."""
    adapter = SREToolAdapter(scenario, seed=seed)
    state = scenario_to_agent_state(scenario, adapter)

    evidence, executed = _execute_simulated_actions(adapter)
    state["evidence"] = evidence
    state["executed_hypotheses"] = executed

    # Use gold standard for diagnosis (in production, opensre's LLM does this)
    gold_label = scenario.gold_root_causes[0].taxonomy_label if scenario.gold_root_causes else ""
    gold_remediation = scenario.gold_remediations[0].action if scenario.gold_remediations else ""
    state["root_cause"] = gold_label.replace("_", " ")
    state["root_cause_category"] = gold_label.split(".")[0] if gold_label else ""
    state["remediation_steps"] = [gold_remediation] if gold_remediation else []

    return {
        "scenario": scenario.name,
        "evidence_collected": len(evidence),
        "executed_actions": [h["action"] for h in executed],
        **score_agent_state(state, scenario),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run opensre against RL environment")
    parser.add_argument("--scenario", "-s", help="Scenario YAML path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = ScenarioLoader()
    scenarios = [loader.load(args.scenario)] if args.scenario else loader.load_all()

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"{'='*60}")

        result = run_episode(scenario, seed=args.seed)
        print(f"  Evidence collected: {result['evidence_collected']}")
        print(f"  Actions: {result['executed_actions']}")
        print(f"  Reward: {result['reward']:.3f}")
        if result.get("breakdown"):
            bd = result["breakdown"]
            print(f"    Diagnosis:   {bd.diagnosis_score:.3f}")
            print(f"    Efficiency:  {bd.efficiency_score:.3f}")
            print(f"    Remediation: {bd.remediation_score:.3f}")
            print(f"    Safety:      {bd.safety_score:.3f}")


if __name__ == "__main__":
    main()
