"""Run open-sre-agent's LangGraph pipeline against simulated scenarios.

Two modes:
  Standalone: mirrors opensre's flow without opensre installed
  Full pipeline: builds a custom LangGraph graph with simulated evidence sources

Usage:
    python -m src.integration.opensre_runner
    python -m src.integration.opensre_runner --use-opensre
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
    """Run without opensre — uses gold standard for diagnosis."""
    adapter = SREToolAdapter(scenario, seed=seed)
    state = scenario_to_agent_state(scenario, adapter)
    evidence, executed = _execute_simulated_actions(adapter)
    state["evidence"] = evidence
    state["executed_hypotheses"] = executed

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


def run_with_opensre(scenario: ScenarioDefinition, seed: int = 42) -> dict[str, Any]:  # noqa: PLR0915
    """Build a custom LangGraph graph with simulated evidence sources.

    Replaces opensre's investigate and plan_actions nodes with versions
    that use our simulated actions. No monkey-patching — a clean graph.
    """
    try:
        from app.agent.nodes import (
            node_diagnose_root_cause,
            node_extract_alert,
            node_publish_findings,
            node_resolve_integrations,
        )
        from app.agent.nodes.auth import inject_auth_node
        from app.agent.nodes.investigate.execution import execute_actions
        from app.agent.nodes.investigate.models import InvestigateInput, InvestigateOutput
        from app.agent.nodes.investigate.processing import summarize_execution_results
        from app.agent.nodes.plan_actions.build_prompt import plan_actions_with_llm, select_actions
        from app.agent.nodes.plan_actions.detect_sources import detect_sources
        from app.agent.nodes.plan_actions.node import InvestigationPlan
        from app.agent.output import get_tracker
        from app.agent.routing import (
            route_after_extract,
            route_by_mode,
            route_investigation_loop,
        )
        from app.agent.state import AgentState
        from app.agent.tools.clients import get_llm
        from app.agent.tools.tool_actions.investigation_registry.models import InvestigationAction
        from langgraph.graph import END, StateGraph
    except ImportError as e:
        logger.error("opensre not installed: %s", e)
        return {"error": f"opensre not installed: {e}"}

    adapter = SREToolAdapter(scenario, seed=seed)
    state = scenario_to_agent_state(scenario, adapter)

    sim_actions = create_simulated_actions(adapter)
    sim_sources = build_simulated_sources(adapter)
    state["alert_source"] = ""
    state["available_sources"] = sim_sources

    # Build real InvestigationAction objects from our SimulatedActions
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

    # --- Custom nodes that use simulated actions directly ---

    def sim_plan_actions(state: dict) -> dict:
        """Plan actions using simulated action registry."""
        input_data = InvestigateInput.from_state(state)
        tracker = get_tracker()
        tracker.start("plan_actions", "Planning evidence gathering")

        # Detect sources + inject simulated
        sources = detect_sources(
            input_data.raw_alert,
            input_data.context,
            resolved_integrations=state.get("resolved_integrations"),
        )
        sources.update(sim_sources)

        # Select from our simulated actions
        available, names = select_actions(
            actions=real_actions,
            available_sources=sources,
            executed_hypotheses=input_data.executed_hypotheses,
        )

        if not names:
            tracker.complete("plan_actions", fields_updated=["planned_actions"], message="No actions available")
            return {
                "planned_actions": [],
                "plan_rationale": "",
                "available_sources": sources,
                "available_action_names": [],
            }

        llm = get_llm()
        plan = plan_actions_with_llm(
            llm=llm,
            plan_model=InvestigationPlan,
            problem_md=input_data.problem_md,
            executed_hypotheses=input_data.executed_hypotheses,
            available_actions=available,
            available_sources=sources,
            memory_context="",
        )

        planned = plan.actions if plan else []
        rationale = plan.rationale if plan else ""
        tracker.complete("plan_actions", fields_updated=["planned_actions"], message=f"Planned: {planned}")
        return {
            "planned_actions": planned,
            "plan_rationale": rationale,
            "available_sources": sources,
            "available_action_names": names,
        }

    def sim_investigate(state: dict) -> dict:
        """Execute planned actions using simulated action registry."""
        input_data = InvestigateInput.from_state(state)
        tracker = get_tracker()
        tracker.start("investigate", "Executing planned actions")

        planned = state.get("planned_actions", [])
        sources = state.get("available_sources", {})
        rationale = state.get("plan_rationale", "")

        if not planned:
            tracker.complete("investigate", fields_updated=["evidence"], message="No actions")
            return {"evidence": input_data.evidence}

        actions_by_name = {a.name: a for a in real_actions}
        available = {name: actions_by_name[name] for name in planned if name in actions_by_name}

        results = execute_actions(planned, available, sources)
        evidence, executed_hyps, summary = summarize_execution_results(
            execution_results=results,
            current_evidence=input_data.evidence,
            executed_hypotheses=input_data.executed_hypotheses,
            investigation_loop_count=input_data.investigation_loop_count,
            rationale=rationale,
        )

        tracker.complete("investigate", fields_updated=["evidence"], message=summary)
        output = InvestigateOutput(evidence=evidence, executed_hypotheses=executed_hyps)
        return output.to_dict()  # type: ignore[no-any-return]

    # --- Build custom graph ---

    try:
        graph = StateGraph(AgentState)
        graph.add_node("inject_auth", inject_auth_node)
        graph.add_node("extract_alert", node_extract_alert)
        graph.add_node("resolve_integrations", node_resolve_integrations)
        graph.add_node("plan_actions", sim_plan_actions)  # type: ignore[type-var]
        graph.add_node("investigate", sim_investigate)  # type: ignore[type-var]
        graph.add_node("diagnose", node_diagnose_root_cause)
        graph.add_node("publish", node_publish_findings)

        graph.set_entry_point("inject_auth")
        graph.add_conditional_edges("inject_auth", route_by_mode, {"chat": END, "investigation": "extract_alert"})
        graph.add_conditional_edges(
            "extract_alert", route_after_extract, {"end": END, "investigate": "resolve_integrations"}
        )
        graph.add_edge("resolve_integrations", "plan_actions")
        graph.add_edge("plan_actions", "investigate")
        graph.add_edge("investigate", "diagnose")
        graph.add_conditional_edges(
            "diagnose", route_investigation_loop, {"investigate": "plan_actions", "publish": "publish"}
        )
        graph.add_edge("publish", END)

        compiled = graph.compile()

        logger.info("Running custom opensre graph for: %s", scenario.name)
        final_state = compiled.invoke(state)  # type: ignore[call-overload]

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
        logger.error("opensre pipeline failed: %s", e, exc_info=True)
        return {"error": str(e), "scenario": scenario.name}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run opensre against RL environment")
    parser.add_argument("--scenario", "-s", help="Scenario YAML path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-opensre", action="store_true", help="Use opensre's LangGraph pipeline (requires opensre + LLM)"
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
            if result.get("root_cause"):
                print(f"  Root cause: {result['root_cause'][:100]}")
            if result.get("breakdown"):
                bd = result["breakdown"]
                print(f"    Diagnosis:   {bd.diagnosis_score:.3f}")
                print(f"    Efficiency:  {bd.efficiency_score:.3f}")
                print(f"    Remediation: {bd.remediation_score:.3f}")
                print(f"    Safety:      {bd.safety_score:.3f}")
        except Exception as e:
            logger.error("Failed on %s: %s", scenario.name, e)


if __name__ == "__main__":
    main()
