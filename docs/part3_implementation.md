# Part 3: MVP Code Implementation

## Scope Choice: Why I Built All Four

The assessment suggests four MVP scopes:
1. A crawler that indexes incident reports and classifies against a taxonomy
2. A synthetic telemetry generator for distributed system observability data
3. A minimal RL environment with observation space, action space, and reward function
4. An evaluation harness that scores agent responses against known resolutions

I built all four — not to show breadth, but because **none of them solves the core problem in isolation.**

- A crawler without a telemetry generator produces incidents that can't be replayed
- A telemetry generator without a reward function produces data nobody can learn from
- An RL environment without real incident data trains on made-up scenarios
- An evaluation harness without an environment has nothing to evaluate

The novel contribution is the **pipeline that connects them**: real incidents from VOID → auto-generated scenarios → synthetic telemetry → RL environment → multi-dimensional reward → trajectory export for LLM fine-tuning → opensre integration. Each component exists to serve this pipeline.

## What I Targeted: The Reward Signal

The highest-risk component is the reward signal. The Part 1 research establishes why: coding agents have tractable rewards (tests pass or fail), but distributed systems have ambiguous, delayed, context-dependent correctness. If the reward signal doesn't work, the entire architecture fails regardless of how good the simulation is.

The reward had to satisfy four properties simultaneously:
1. **Handle multiple valid root causes** — an incident can have several defensible diagnoses
2. **Provide gradient for partial progress** — "right service, wrong component" must score higher than "completely wrong"
3. **Not reward speed over correctness** — a fast wrong answer must score lower than a slow right one
4. **Discriminate meaningfully between agent quality levels** — random, heuristic, and expert agents must get clearly different scores

All four are validated in the baseline comparison (random: 0.321, heuristic: 0.464, oracle: 0.875).

## What I Built

### Component 1: Incident Data Pipeline (Pillar 1 + 5)

**What:** Three API crawlers + one CSV loader that fetch real incident data, plus a scenario generator that converts them into playable RL episodes.

**Why:** Hand-authoring YAML scenarios doesn't scale (5 scenarios = 29% taxonomy coverage). Real incidents provide ground-truth failure patterns. The Aiops-Dataset groundtruth CSV (241 labeled faults from a 46-instance microservice system) is included in the repo at `data/groundtruth-all.csv` (16 KB). The crawlers fetch 261 more incidents from public APIs.

**How the training data is generated:**
1. `run_crawler.py` runs 3 crawlers (GCP, Cloudflare, GitHub) → 261 incidents → `data/incidents.db` (SQLite)
2. `load_aiops_groundtruth("data/groundtruth-all.csv")` reads 241 labeled faults from the included CSV
3. `ScenarioGenerator.batch_generate(incidents)` converts each incident to a `ScenarioDefinition` by picking a topology template, building an event timeline, and setting gold-standard root causes
4. `EpisodeRunner` combines generated + builtin scenarios → 246+ total scenarios for training

**Key files:**
- `src/crawler/models.py` — `NormalisedIncident` dataclass + `IncidentCrawler` ABC
- `src/crawler/crawlers/` — GCP, Cloudflare, GitHub crawlers + Aiops-Dataset CSV loader
- `src/crawler/scenario_generator.py` — converts incidents to playable scenarios (`batch_generate()`, `generate_definition()`)

### Component 2: Synthetic Telemetry Generator (Pillar 3)

**What:** Generates correlated metrics, logs, and traces for any scenario YAML.

**Why:** RL training needs 100K+ episodes. Real infrastructure costs $5/episode. Synthetic telemetry costs $0.001/episode — 5000x cheaper. The telemetry must be realistic enough that investigation skills transfer to real Grafana/Datadog dashboards.

**How:** The `TelemetryGenerator` orchestrates three sub-generators:
- `MetricsGenerator` — time-series metrics with 9 effect handlers (error spike, latency spike, connection exhaustion, cascade, etc.) that respond to scenario timeline events
- `LogGenerator` — event-correlated log messages. A connection exhaustion event produces "Too many connections: max=100, active=97", not random deadlock errors. Log severity, status codes, and durations are correlated with active events.
- `TraceGenerator` — distributed traces across the service topology. Error probability derived from the event's `target_rate` parameter, matching metrics.

Scenario perturbation (`perturb_scenario()`) jitters timing ±15% and magnitudes ±20% per seed, so the same scenario produces different telemetry every episode — preventing memorisation.

**Key files:**
- `src/generators/telemetry.py` — orchestrator
- `src/generators/metrics.py`, `logs.py`, `traces.py` — sub-generators
- `src/generators/effects/` — 9 effect handlers (registry pattern)
- `src/generators/perturbation.py` — per-seed variation

### Component 3: RL Environment (Pillar 3 + 4)

**What:** Gymnasium-compatible environment with tool-based observation/action space.

**Why:** The agent must learn *investigation strategy* — what to query, in what order — not just pattern matching. The tool interface (query metrics, query logs, query traces, list services, list alerts, diagnose, remediate) mirrors how real SREs use Grafana and Datadog, and matches opensre's `InvestigationAction` interface.

**How:** `SREEnvironment(gym.Env)` implements `reset()` and `step()`. Observations are natural language text (metrics tables, log entries, trace summaries) — designed for LLM agents. The agent doesn't see all telemetry at once; it must choose what to query, mimicking real SRE investigation where information retrieval is itself a skill.

**Key files:**
- `src/environment/env.py` — Gymnasium environment
- `src/environment/actions.py` — 7 action types
- `src/environment/formatter.py` — telemetry → text rendering
- `src/environment/state.py` — episode state tracking

### Component 4: Evaluation Harness (Pillar 4)

**What:** Multi-dimensional reward function with hierarchical partial credit.

**Why:** This is the hard part. A binary reward (right/wrong) can't capture the richness of SRE investigation. The VOID database shows only 25% of incidents have a single definitive root cause. The reward must handle ambiguity, partial correctness, and the interaction between investigation thoroughness and speed.

**How:** Four scorers compose the reward:
- `DiagnosisScorer` — hierarchical taxonomy matching. Exact match: 1.0. Right subcategory, wrong leaf: 0.7. Right category: 0.4. Wrong: 0.0. Supports multiple gold-standard root causes with relevance weights.
- `EfficiencyScorer` — linear decay from `ideal_steps` (5) to `max_steps`. Multiplied by diagnosis score so fast+wrong = 0.
- `RemediationScorer` — exact match against gold-standard remediations with effectiveness weights. Partial credit for generic actions (restart: 0.1, scale: 0.2).
- `SafetyScorer` — penalises hasty diagnosis (<2 queries: -0.3) and tunnel vision (<2 services queried: -0.2).

All thresholds are configurable in `config/reward.yaml`.

**Key files:**
- `src/evaluation/reward_calculator.py` — composes 4 scorers
- `src/evaluation/scorers/` — individual scorer implementations
- `src/evaluation/reward_breakdown.py` — immutable breakdown with weights
- `config/reward.yaml` — all thresholds and weights

### Component 5: opensre Integration (not in suggested scopes)

**What:** Bridge between the RL environment and Tracer's open-sre-agent LangGraph pipeline.

**Why:** The assessment says "your architecture should plug into the Tracer open-sre-agent ecosystem." Without integration, the RL environment is a standalone prototype. With integration, it's a training system for the production agent.

**How:** Three integration modules:
- `SimulatedAction` — drop-in replacement for opensre's `InvestigationAction`. Has `parameter_extractor`, `availability_check`, and `function` matching the exact interface that `execute_actions()` expects. Verified against opensre's actual code.
- `scenario_to_agent_state()` — creates an opensre `AgentState` from a scenario YAML, including all fields the pipeline reads (`investigation_started_at`, `alert_json`, `resolved_integrations` with endpoint/api_key structure).
- `score_agent_state()` — maps opensre's free-text `root_cause` and `remediation_steps` back to taxonomy labels for reward computation.

Tested end-to-end: opensre's `execute_actions()` successfully runs all 5 simulated actions. 5/5 succeed.

**Key files:**
- `src/integration/evidence_source.py` — `SimulatedAction` + `build_simulated_sources()`
- `src/integration/state_adapter.py` — `AgentState` conversion + scoring
- `src/agent_adapter.py` — `SREToolAdapter` for LLM function-calling APIs
- `src/tool_definitions.py` — tool definitions for Claude tool_use / OpenAI functions

### Component 6: Training Loop

**What:** Episode runner with trajectory collection, JSONL export, and curriculum support.

**Why:** The RL loop doesn't close without a training data pipeline. The environment generates episodes, the reward scores them, but something must collect trajectories and export them in a format that LLM fine-tuning frameworks (trl, DeepSpeed) can consume.

**How:** `EpisodeRunner` samples scenarios (with optional difficulty filtering for curriculum learning), runs episodes, collects full trajectories (observation, action, reward per step), and tracks per-scenario statistics. `export_jsonl()` and `export_preference_pairs()` output training data for GRPO and DPO respectively.

**Key files:**
- `src/training/episode_runner.py` — episode runner with trajectory collection
- `src/training/trajectory_export.py` — JSONL and preference pair export

## How I Used AI-Assisted Development

This project was built entirely using **Claude Code** as the primary development tool. The workflow:

1. **Architecture first.** Drafted the 5-pillar architecture document, then implemented each pillar. Claude Code generated initial scaffolding; I refined for correctness and realism.

2. **Test-driven iteration.** After each component, ran `pytest` to verify. Claude Code helped write test scaffolds; I validated that the tests exercise meaningful behaviour (not just "does it not crash").

3. **Critical self-review.** Asked Claude Code to independently audit the codebase against the assessment requirements. It identified 6 breaking issues in the opensre integration (missing `parameter_extractor`, wrong `resolved_integrations` structure, etc.) — all fixed.

4. **Where I intervened manually:**
   - Reward function tuning — initial weights produced degenerate behaviour (fast wrong answers scored too high). The `efficiency × diagnosis` coupling was a manual design decision.
   - Log template fidelity — Claude Code generated generic templates, but a connection pool scenario was showing "Deadlock detected" errors. I restructured the log generator to be event-aware.
   - opensre integration — required reading the actual opensre source code (`execute_actions.py`, `state.py`, `routing.py`) to understand the exact interface contracts.

## Compute Estimates

Measured on Apple M-series laptop, single core:

| Metric | Measured Value | Reasoning |
|--------|---------------|-----------|
| **Telemetry generation** | 38ms per episode | 5 scenarios, 3-8 services each, 90 time steps, 9 effect handlers. Scales linearly with services × time steps. |
| **Full episode (random agent)** | 29ms per episode | Includes generation + 5-15 agent steps + reward computation. |
| **Memory per episode** | 2.5MB | Telemetry held in memory (tuples of frozen dataclasses). No accumulation across episodes. |
| **Throughput** | 120,000 episodes/hour | Single CPU core, random agent. With LLM agent (~200ms/step × 10 steps), drops to ~1,800 eps/hr per GPU. |
| **Test suite** | 155 tests in 3.1 seconds | Full coverage of all components. |

### At Scale (100K training episodes)

| Resource | Layer 1 (Synthetic, 95K eps) | Layer 2 (AIOpsLab, 5K eps) | Total |
|----------|-----|-----|-------|
| **Time** | ~48 hours (1 GPU for LLM inference) | ~500 hours (real microservices) | ~3 weeks |
| **CPU** | 48 CPU-minutes (telemetry generation only) | ~80 CPU-hours (microservice orchestration) | ~80 CPU-hours |
| **GPU** | 1× A100 for LLM inference | 1× A100 for LLM inference | 1× A100 |
| **Memory** | 2.5MB peak (per episode, independent) | 8GB peak (running microservices) | 8GB peak |
| **Storage** | ~100MB (trajectory JSONL) | ~5GB (telemetry recordings) | ~5GB |
| **Cloud cost** | ~$150 (A100 spot, 48hrs) | ~$500 (8-core instances, 500hrs) | **~$650** |

### Reasoning

- **Layer 1 GPU time:** 95K episodes × 10 steps/episode × 200ms/step LLM inference = 190K seconds = 53 hours. One A100 at ~$2/hr spot = ~$106.
- **Layer 1 CPU time:** 95K episodes × 38ms = 60 minutes. Negligible vs GPU cost.
- **Layer 2 time:** AIOpsLab deploys DeathStarBench (~30 services), injects faults, waits for propagation (~5-10 min/episode). 5K episodes × 6 min = 500 hours.
- **Storage:** JSONL trajectory = ~1KB/episode × 100K = 100MB. Layer 2 telemetry recordings are larger (~1MB/episode).

## Setup Instructions

### Quick Start (Local)

```bash
# Clone and install
git clone <repo-url>
cd tracer-sre-rl
pip install -r requirements.txt

# Run tests (155 tests, ~3 seconds)
python -m pytest tests/ -v

# Run baseline comparison (3 agents × 5 scenarios)
python demo.py --quiet

# Run training loop with trajectory export
python run_training.py --episodes 100 --export training_data/trajectories.jsonl

# Run opensre integration
python -m src.integration.opensre_runner
```

### With opensre

```bash
# Install opensre in a venv
git clone https://github.com/Tracer-Cloud/open-sre-agent ../opensre
cd ../opensre && python3 -m venv .venv && source .venv/bin/activate
make install && pip install gymnasium numpy pyyaml requests

# Run integration against opensre's actual execute_actions
cd ../tracer-sre-rl
PYTHONPATH="../opensre:." python -m src.integration.opensre_runner --use-opensre
```

### Docker

```bash
docker compose up tests       # Run test suite
docker compose up sre-rl-env  # Run baseline comparison
docker compose up training    # Run training loop
docker compose up crawler     # Crawl incident data
```

## Limitations and Honest Gaps

**What works well:**
- Reward signal discriminates clearly (2.7× gap between oracle and random)
- opensre integration verified against actual `execute_actions` pipeline
- Event-correlated telemetry (logs match active events, not random errors)
- 149 passing tests, all README steps verified to run

**What doesn't work yet:**
- No actual RL policy update — trajectories are collected and exported, but GRPO/PPO training requires a GPU cluster and trl/DeepSpeed (outside MVP scope)
- Log metric values are approximate, not numerically identical to metric time-series (log says "active=97", metric says "avg=72")
- 5 hand-authored scenarios cover only 29% of the taxonomy — the VOID pipeline can generate more, but hasn't been run at scale yet
- The VOID crawler assumes a public API that may require a data partnership to access
- No empirical validation of sim-to-real transfer — whether synthetic-trained agents work on real Grafana dashboards is the biggest open question
