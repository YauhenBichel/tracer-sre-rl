# opensre Integration: Custom LangGraph

This document describes how tracer-sre-rl plugs into opensre's LangGraph pipeline.

## The Problem

opensre investigates real incidents by calling real tools (Grafana, Datadog, CloudWatch). During RL training, we need to replace those real tool calls with simulated telemetry — without modifying opensre's code.

## The Solution

We build a **custom LangGraph graph** that reuses opensre's nodes (extract_alert, diagnose, publish) but replaces two nodes with our simulated versions:

```
                    opensre nodes (unchanged)     our nodes (simulated)
                    ─────────────────────────     ─────────────────────

inject_auth ──→ extract_alert ──→ resolve_integrations
                                        │
                                        ▼
                                  sim_plan_actions        ← OURS
                                  (uses opensre's LLM to plan,
                                   but offers our simulated
                                   actions instead of real ones)
                                        │
                                        ▼
                                  sim_investigate          ← OURS
                                  (executes planned actions
                                   against synthetic telemetry)
                                        │
                                        ▼
                                  diagnose                 ← opensre
                                  (LLM analyzes evidence,
                                   proposes root cause)
                                        │
                                  ┌─────┴─────┐
                                  ▼           ▼
                            plan_actions    publish        ← opensre
                            (loop back)    (final report)
```

## What Each Node Does

### sim_plan_actions (replaces opensre's plan_actions)

1. Calls opensre's `detect_sources()` to find available integrations
2. Adds our `simulated_rl_env` source with `connection_verified: True`
3. Runs `select_actions()` against our `SimulatedAction` list (not real Grafana/Datadog)
4. Calls opensre's LLM via `plan_actions_with_llm()` to decide which actions to execute

The LLM sees our action descriptions and chooses which to call — the same decision-making as production, but with simulated tools.

### sim_investigate (replaces opensre's investigate)

1. Reads planned actions from state
2. Looks up each action in our simulated registry
3. Calls opensre's `execute_actions()` — the exact same execution function used in production
4. Calls opensre's `summarize_execution_results()` to process evidence

The evidence flows through opensre's standard processing pipeline. Action names match opensre's `EVIDENCE_MAPPERS` (`query_grafana_alert_rules`, `query_grafana_metrics`, `query_grafana_logs`, `query_grafana_traces`, `query_grafana_service_names`) so evidence is correctly classified.

### Unchanged opensre nodes

- **inject_auth** — sets up authentication context
- **extract_alert** — parses the alert from state
- **resolve_integrations** — finds available integrations
- **diagnose** — LLM analyzes collected evidence, proposes root cause
- **publish** — generates final investigation report

## Verified Result

opensre's LLM investigated a simulated "Disk Full" scenario:

```
● Planning  5.5s  Planned: ['query_grafana_alert_rules', 'query_grafana_metrics']
● Gathering evidence  1ms  grafana:1 alert rules, grafana:1 metric series
● Diagnosing  8.1s  confidence 62%

● Planning  5.3s  Planned: ['query_grafana_service_names', 'query_grafana_logs']
● Gathering evidence  1ms  grafana:1 services, grafana:1 logs (13 errors)
● Diagnosing  13.2s  confidence 100%

Root cause: "high disk usage on postgres-primary caused cascading failures"
Category: resource_exhaustion
Validity: 100%
Investigation loops: 3
```

The LLM autonomously decided what to query, gathered synthetic evidence, and correctly identified the root cause.

## How to Run

```bash
# Standalone (no opensre needed, uses correct answers for diagnosis)
python -m app.integration.opensre_runner

# With opensre's LLM (requires opensre installed + API key)
PYTHONPATH="../opensre:." python -m app.main check-reward
PYTHONPATH="../opensre:." python -m app.integration.opensre_runner --use-opensre
```

## Code Location

- `app/integration/opensre_runner.py` — custom graph builder + `run_with_opensre()`
- `app/integration/evidence_source.py` — `SimulatedAction` matching opensre's `InvestigationAction`
- `app/integration/state_adapter.py` — `AgentState` conversion + reward scoring with synonym mapping
- `app/integration/agent_adapter.py` — `SREToolAdapter` wrapping Gymnasium env as callable tools
