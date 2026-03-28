"""Run open-sre-agent's investigation pipeline against the RL environment.

This is the learning infrastructure: opensre investigates a simulated incident,
the reward is computed, and the full trajectory is saved for policy improvement.

Two modes:
  Standalone: mirrors opensre's flow without requiring opensre installed
  Full pipeline: runs opensre's actual LangGraph graph (requires opensre + LLM)

Usage:
    python -m src.integration.opensre_runner                          # standalone
    python -m src.integration.opensre_runner -s scenarios/disk_full.yaml
    PYTHONPATH="../opensre:." python -m src.integration.opensre_runner --use-opensre  # full
"""

from __future__ import annotations

import argparse
import logging
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
    """Execute simulated actions mirroring opensre's execute_actions flow."""
    actions = create_simulated_actions(adapter)
    sources = build_simulated_sources(adapter)
    evidence: dict[str, Any] = {}
    executed: list[dict[str, Any]] = []

    for action in actions:
        try:
            if action.availability_check and not action.availability_check(sources):
                continue
            if action.parameter_extractor is None:
                continue
            kwargs = action.parameter_extractor(sources)
            result = action.function(**kwargs)
            if isinstance(result, dict) and result.get("available"):
                evidence[action.name] = result
                executed.append(
                    {"action": action.name, "service": result.get("service_name", "unknown"), "success": True}
                )
        except Exception as e:
            logger.warning("Action %s failed: %s", action.name, e)

    return evidence, executed


def run_standalone(scenario: ScenarioDefinition, seed: int = 42) -> dict[str, Any]:
    """Run investigation without opensre — uses gold standard for diagnosis.

    This mode doesn't require an LLM. It demonstrates that the environment
    produces evidence, the reward scores it, and trajectories can be collected.
    """
    adapter = SREToolAdapter(scenario, seed=seed)
    state = scenario_to_agent_state(scenario, adapter)
    evidence, executed = _execute_simulated_actions(adapter)
    state["evidence"] = evidence
    state["executed_hypotheses"] = executed

    # Use gold standard (standalone mode has no LLM to reason)
    if scenario.gold_root_causes:
        gold = scenario.gold_root_causes[0]
        state["root_cause"] = gold.taxonomy_label.replace("_", " ")
        state["root_cause_category"] = gold.taxonomy_label.split(".")[0]
    if scenario.gold_remediations:
        state["remediation_steps"] = [scenario.gold_remediations[0].action]

    return {
        "scenario": scenario.name,
        "mode": "standalone",
        "evidence_collected": len(evidence),
        "actions": [h["action"] for h in executed],
        **score_agent_state(state, scenario),
    }


def run_with_opensre(scenario: ScenarioDefinition, seed: int = 42) -> dict[str, Any]:
    """Run opensre's full LangGraph pipeline against the simulated environment.

    This is the real learning loop: opensre's LLM investigates, diagnoses,
    and proposes remediation. The reward measures how well it did.

    Requires: opensre installed, LLM provider configured (see opensre onboard).
    """
    try:
        from app.agent.graph_pipeline import build_graph
        from app.agent.tools.tool_actions.investigation_registry import get_available_actions
        from app.agent.tools.tool_actions.investigation_registry.models import InvestigationAction
    except ImportError:
        logger.error("opensre not installed. Install with: pip install -e ../opensre")
        return {"error": "opensre not installed"}

    adapter = SREToolAdapter(scenario, seed=seed)
    state = scenario_to_agent_state(scenario, adapter)

    # Override alert_source so opensre's detect_sources enables all integrations
    state["alert_source"] = ""

    # Build simulated actions as real InvestigationAction objects
    sim_actions = create_simulated_actions(adapter)
    build_simulated_sources(adapter)

    real_actions = [
        InvestigationAction(
            name=sa.name,
            description=sa.description,
            inputs=sa.inputs,
            outputs=sa.outputs,
            use_cases=sa.use_cases,
            requires=sa.requires,
            source=sa.source,
            function=sa.function,
            availability_check=sa.availability_check,
            parameter_extractor=sa.parameter_extractor,
        )
        for sa in sim_actions
    ]

    # Inject simulated actions into opensre's action registry
    get_available_actions()
    try:
        import app.agent.tools.tool_actions.investigation_registry.actions as registry

        registry._simulated_actions = real_actions

        # Monkey-patch get_available_actions to return our simulated ones
        original_fn = registry.get_available_actions
        registry.get_available_actions = lambda: real_actions

        # Run the full LangGraph pipeline
        logger.info("Running opensre pipeline for: %s", scenario.name)
        graph = build_graph()
        final_state = graph.invoke(state)

        # Score the result
        result = score_agent_state(final_state, scenario)
        return {
            "scenario": scenario.name,
            "mode": "opensre",
            "root_cause": final_state.get("root_cause", ""),
            "root_cause_category": final_state.get("root_cause_category", ""),
            "validity_score": final_state.get("validity_score", 0.0),
            "remediation_steps": final_state.get("remediation_steps", []),
            "investigation_loops": final_state.get("investigation_loop_count", 0),
            **result,
        }
    except Exception as e:
        logger.error("opensre pipeline failed: %s", e)
        return {"error": str(e), "scenario": scenario.name}
    finally:
        # Restore original actions
        registry.get_available_actions = original_fn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run opensre against RL environment")
    parser.add_argument("--scenario", "-s", help="Scenario YAML path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-opensre", action="store_true", help="Use opensre's full LangGraph pipeline (requires LLM)"
    )
    args = parser.parse_args()

    loader = ScenarioLoader()
    if args.scenario:
        loaded = loader.load(args.scenario)
        scenarios = [loaded] if loaded else []
    else:
        scenarios = loader.load_all()

    run_fn = run_with_opensre if args.use_opensre else run_standalone

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario.name}")
        print(f"{'=' * 60}")

        try:
            result = run_fn(scenario, seed=args.seed)
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue
            print(f"  Mode: {result['mode']}")
            print(f"  Evidence: {result.get('evidence_collected', '?')}")
            print(f"  Reward: {result['reward']:.3f}")
            if result.get("breakdown"):
                bd = result["breakdown"]
                print(f"    Diagnosis:   {bd.diagnosis_score:.3f}")
                print(f"    Efficiency:  {bd.efficiency_score:.3f}")
                print(f"    Remediation: {bd.remediation_score:.3f}")
                print(f"    Safety:      {bd.safety_score:.3f}")
            if result.get("root_cause"):
                print(f"  Root cause: {result['root_cause'][:80]}")
        except Exception as e:
            logger.error("Failed on %s: %s", scenario.name, e)


if __name__ == "__main__":
    main()
