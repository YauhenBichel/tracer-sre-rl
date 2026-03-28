# SRE Reinforcement Learning Environment

A high-fidelity training environment for AI SRE agents to learn incident investigation, diagnosis, and remediation of distributed system failures.

Built for the [Tracer open-sre-agent](https://github.com/Tracer-Cloud/open-sre-agent) ecosystem. Plugs into opensre's LangGraph pipeline as a simulated evidence source, replacing real Grafana/Datadog/CloudWatch during training.

## MVP Focus: The Reward Signal

The highest-risk component in RL for SRE is the **reward signal**. For coding agents, the reward is tractable — tests pass or fail. For distributed systems, "correct" is ambiguous (multiple valid root causes), delayed (remediation effects unfold over time), and context-dependent. Without a usable reward signal, no amount of environment fidelity matters.

This MVP targets the reward signal and the surrounding infrastructure needed to validate it:
- **Multi-dimensional reward** (diagnosis accuracy × efficiency + remediation quality + safety) that provides gradient signal even for partially correct answers
- **Hierarchical taxonomy matching** with partial credit (right service layer but wrong component = 0.7, not 0.0)
- **Real-data-driven scenarios** from VOID (~10K incidents) and Aiops-Dataset, not hand-tuned synthetic data
- **Baseline validation** proving the reward discriminates: oracle (0.875) >> heuristic (0.464) >> random (0.321)
- **opensre integration** verified against the actual `execute_actions` pipeline — the same tool interface the production agent uses

## Architecture

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                     Data Plane                               │
                    │                                                              │
                    │  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
                    │  │ GCP Crawler  │   │  Cloudflare  │   │  GitHub PM       │  │
                    │  │             │   │  Crawler     │   │  Crawler         │  │
                    │  └──────┬──────┘   └──────┬───────┘   └────────┬─────────┘  │
                    │         └────────────┬─────┴──────────────────┬─┘            │
                    │                ┌─────▼─────┐          ┌──────▼──────┐       │
                    │                │ SQLite DB │◄────────►│ Scenario    │       │
                    │                │ Incidents │          │ Generator   │       │
                    │                └───────────┘          └──────┬──────┘       │
                    └─────────────────────────────────────────────┼───────────────┘
                                                                  │
                    ┌─────────────────────────────────────────────┼───────────────┐
                    │                  Simulation Plane            │               │
                    │                                              │               │
                    │  ┌──────────────┐   ┌────────────────────┐  │               │
                    │  │ Scenario     │◄──┘  config/            │  │               │
                    │  │ YAML         │      baselines.yaml     │  │               │
                    │  └──────┬───────┘      taxonomy.yaml      │  │               │
                    │         │              reward.yaml         │  │               │
                    │  ┌──────▼───────┐   └────────────────────┘  │               │
                    │  │ Telemetry    │                            │               │
                    │  │ Generator    │──► Metrics, Logs, Traces   │               │
                    │  └──────┬───────┘                            │               │
                    │         │                                    │               │
                    └─────────┼────────────────────────────────────┘               │
                              │                                                    │
                    ┌─────────┼────────────────────────────────────────────────────┘
                    │         │            Training Plane
                    │  ┌──────▼───────┐
                    │  │ SRE Env      │◄──────► SRE Agent
                    │  │ (Gymnasium)  │         (actions / observations)
                    │  └──────┬───────┘
                    │         │
                    │  ┌──────▼───────┐
                    │  │ Reward       │──► diagnosis, efficiency,
                    │  │ Engine       │    remediation, safety
                    │  └──────────────┘
                    └──────────────────────────────────────────────────────────────
```

**Core components:**

| Component | Description |
|-----------|-------------|
| `src/generators/` | Generates synthetic metrics, logs, traces, and events for failure scenarios |
| `src/environment/` | Gymnasium-compatible RL environment with tool-based observation/action spaces |
| `src/evaluation/` | Multi-dimensional reward function (diagnosis accuracy, efficiency, remediation, safety) |
| `src/taxonomy.py` | Hierarchical failure classification taxonomy (~30 leaf failure types) |
| `src/crawler/` | Incident data crawler for GCP, Cloudflare, and GitHub post-mortems |
| `src/crawler/scenario_generator.py` | Converts crawled incidents into YAML scenario templates |
| `src/integration/` | Bridge to [open-sre-agent](https://github.com/Tracer-Cloud/open-sre-agent) — simulated evidence sources + AgentState adapter |
| `src/training/` | RL training loop with curriculum support and episode collection |
| `scenarios/` | YAML scenario definitions for 5 distinct failure classes |

## Quick Start

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (152 tests)
python -m pytest tests/ -v

# Run demo (3 baseline agents on all 5 scenarios)
python demo.py

# Run a specific scenario with a specific agent
python demo.py --scenario scenarios/db_connection_pool.yaml --agent oracle

# Run the incident data crawler
python run_crawler.py

# Generate scenarios from crawled incidents
python run_crawler.py --generate-scenarios

# Run the RL training loop (100 episodes with random agent baseline)
python run_training.py --episodes 100
```

### open-sre-agent Integration

The RL environment plugs into [Tracer's open-sre-agent](https://github.com/Tracer-Cloud/open-sre-agent) as a simulated evidence source, replacing real Grafana/Datadog/CloudWatch tool calls during training.

```bash
# 1. Clone and install open-sre-agent
git clone https://github.com/Tracer-Cloud/open-sre-agent ../opensre
cd ../opensre
python3 -m venv .venv && source .venv/bin/activate
make install

# 2. Install tracer-sre-rl deps in the same venv
pip install gymnasium numpy pyyaml requests

# 3. Run the integration (standalone mode — no LLM needed)
cd ../tracer-sre-rl
python -m src.integration.opensre_runner

# 4. Run with opensre's actual execute_actions pipeline
PYTHONPATH="../opensre:." python -m src.integration.opensre_runner --use-opensre
```

The integration provides:
- **`SimulatedAction`** — drop-in replacement for opensre's `InvestigationAction`, with matching `parameter_extractor`, `availability_check`, and `function` interfaces
- **`scenario_to_agent_state()`** — creates an opensre `AgentState` from a scenario YAML (replaces Slack/PagerDuty alert ingestion)
- **`score_agent_state()`** — scores the agent's `root_cause` and `remediation_steps` against the scenario's gold standard

### Docker

```bash
# Run demo
docker compose up sre-rl-env

# Run tests
docker compose up tests

# Run crawler (stores results in ./data/incidents.db)
docker compose up crawler
```

## Scenarios

| Scenario | Failure Class | Difficulty | Services |
|----------|--------------|------------|----------|
| DB Connection Pool Exhaustion | `infrastructure.database.connection_pool` | 0.4 | 5 |
| Gradual Memory Leak → OOM | `application.memory.leak` | 0.5 | 6 |
| Cascading Failure from Slow Payment | `application.dependency.cascading_failure` | 0.6 | 8 |
| Intermittent DNS Failure | `infrastructure.network.dns` | 0.7 | 8 |
| Disk Full from Debug Logs | `infrastructure.storage.disk_full` | 0.3 | 3 |

## RL Environment API

```python
from src.generators.loader import ScenarioLoader
from src.environment.env import SREEnvironment
from src.environment.actions import ActionType

loader = ScenarioLoader()
scenario = loader.load("scenarios/db_connection_pool.yaml")
env = SREEnvironment(scenario=scenario, seed=42)

obs, info = env.reset()

# Agent actions: query_metrics, query_logs, query_traces,
#                list_services, list_alerts, diagnose, remediate
action = {
    "action_type": ActionType.QUERY_METRICS,
    "target_service": 0,       # Service index
    "time_start": 40,          # Start of time range (10s intervals)
    "time_end": 89,            # End of time range
    "diagnosis_idx": 0,        # For DIAGNOSE action
    "remediation_idx": 0,      # For REMEDIATE action
}

obs, reward, terminated, truncated, info = env.step(action)
```

### Observation Space

- `text_observation`: Natural language output (metrics tables, log entries, trace summaries, alert lists)
- `step_count`: Current investigation step
- `alerts_seen`: Number of alerts viewed
- `services_queried`: Binary vector of queried services

### Reward Function

```
R = 0.40 × R_diagnosis + 0.20 × R_efficiency + 0.25 × R_remediation + 0.15 × R_safety
```

- **Diagnosis (0.40)**: Hierarchical taxonomy matching with partial credit
- **Efficiency (0.20)**: Fewer steps = higher score (perfect at ≤5 steps, linear decay to max_steps). Multiplied by diagnosis score so fast+wrong = 0.
- **Remediation (0.25)**: Matched against gold-standard fixes with effectiveness weights
- **Safety (0.15)**: Penalises hasty diagnosis (<2 queries) and tunnel vision (<2 services queried)

### Baseline Agent Results

| Agent | Avg Reward | Diagnosis | Efficiency | Remediation | Safety |
|-------|-----------|-----------|------------|-------------|--------|
| Random | 0.321 | 0.244 | 0.244 | 0.100 | 1.000 |
| Heuristic (no gold labels) | 0.464 | 0.160 | 0.077 | 0.940 | 1.000 |
| Oracle (gold labels) | 0.875 | 0.960 | 0.480 | 0.980 | 1.000 |

## Compute Estimates

### Layer 1: Synthetic Telemetry (this MVP)

Measured on Apple M-series laptop, single core:

| Resource | Per Episode (measured) | At Scale (100K episodes) |
|----------|-----------|---------------------------|
| CPU | ~38ms telemetry generation, ~29ms full episode with random agent | ~48 CPU-minutes for 100K episodes |
| Memory | ~2.5MB per episode (episodes are independent, no accumulation) | ~2.5MB peak |
| Storage | ~1KB per trajectory (JSONL export) | ~100MB for 100K episodes |
| GPU | None for environment; 1× A100 for LLM agent inference | 1× A100 for inference |

**Throughput**: ~120,000 episodes/hour on a single CPU core (telemetry generation + random agent, excluding LLM inference). With LLM inference (~200ms/step × 10 steps), throughput drops to ~1,800 episodes/hour per GPU.

### Layer 2: Docker Compose (future)

| Resource | Per Episode | At Scale (100 episodes/day) |
|----------|-----------|---------------------------|
| CPU | 2-4 cores × 5-10 min | ~40-60 CPU-hours/day |
| Memory | 4-8GB (running services) | 8GB peak |
| Storage | ~1GB (Docker images + data) | ~1GB |

### Layer 3: Kubernetes + Chaos (future)

| Resource | Per Episode | At Scale (10 episodes/day) |
|----------|-----------|---------------------------|
| CPU | 8-16 cores × 15-30 min | ~40-80 CPU-hours/day |
| Memory | 16-32GB (K8s cluster) | 32GB peak |
| Storage | ~10GB (cluster state + telemetry) | ~10GB |
| Cost | ~$2-5 per episode (cloud) | ~$20-50/day |

### Full Training Run Estimate

To train an agent on 100K episodes (covering all 5 scenarios with parameter variations):

| Layer | Episodes | Time (1 GPU + 8 CPU) | Cost (cloud) |
|-------|----------|---------------------|------|
| L1 (synthetic) | 95,000 | ~48 hours | ~$150 (A100 spot) |
| L2 (Docker) | 4,500 | ~38 hours | ~$100 |
| L3 (K8s) | 500 | ~250 hours | ~$1,250 |
| **Total** | **100,000** | **~14 days** | **~$1,500** |

## Project Structure

```
tracer-sre-rl/
├── docs/
│   ├── part1_research.md          # Part 1: Research & Constraints
│   ├── part2_architecture.md      # Part 2: Architecture Design
│   └── part4_reflection.md        # Part 4: Results & Reflection
├── src/
│   ├── models.py                  # Immutable data models (metrics, logs, traces, scenarios)
│   ├── taxonomy.py                # Hierarchical failure taxonomy (~30 leaf types)
│   ├── config.py                  # YAML config loader with LRU caching
│   ├── constants.py               # Shared constants (event types, metric names)
│   ├── crawler/                   # Incident data crawling
│   │   ├── models.py              # NormalisedIncident + IncidentCrawler ABC
│   │   ├── incident_crawler.py    # Orchestrator for all crawlers
│   │   ├── scenario_generator.py  # Converts incidents → scenario YAML
│   │   ├── crawlers/              # Crawler implementations
│   │   │   ├── gcp_crawler.py     # Google Cloud incident feed
│   │   │   ├── cloudflare_crawler.py  # Cloudflare Statuspage API
│   │   │   ├── github_crawler.py  # GitHub post-mortem collections
│   │   │   └── quality.py         # Quality scoring for incidents
│   │   └── repository/            # Storage layer
│   │       ├── incident_repository.py  # Protocol (interface)
│   │       └── sqlite_repository.py    # SQLite implementation
│   ├── generators/                # Synthetic telemetry engine
│   │   ├── telemetry.py           # Orchestrates metrics + logs + traces
│   │   ├── metrics.py             # Time-series metric generation
│   │   ├── logs.py                # Structured log generation
│   │   ├── traces.py              # Distributed trace generation
│   │   ├── loader.py              # YAML scenario loader & validator
│   │   └── effects/               # Failure effect handlers (9 types)
│   │       ├── base.py            # EventEffectHandler ABC
│   │       ├── registry.py        # Effect type → handler mapping
│   │       ├── error_spike.py
│   │       ├── latency_spike.py
│   │       ├── traffic_ramp.py
│   │       ├── cascade.py
│   │       ├── connection_exhaustion.py
│   │       ├── resource_exhaustion.py
│   │       ├── memory_leak.py
│   │       ├── disk_fill.py
│   │       └── symptom.py
│   ├── environment/               # Gymnasium RL environment
│   │   ├── env.py                 # SREEnvironment (reset/step/render)
│   │   ├── actions.py             # ActionType enum (7 actions)
│   │   ├── state.py               # EpisodeState + StepResult
│   │   └── formatter.py           # Telemetry → text rendering
│   ├── evaluation/                # Reward calculation
│   │   ├── reward_calculator.py   # Composes 4 scorers
│   │   ├── reward_breakdown.py    # RewardBreakdown + RewardWeights
│   │   └── scorers/               # Individual reward components
│   │       ├── diagnosis_scorer.py    # Hierarchical taxonomy matching
│   │       ├── efficiency_scorer.py   # Step-count scoring
│   │       ├── remediation_scorer.py  # Gold-standard matching
│   │       └── safety_scorer.py       # Investigation quality
│   └── utils/
│       └── datetime_utils.py      # ISO 8601 helpers
├── scenarios/                     # YAML scenario definitions
│   ├── cascading_failure.yaml
│   ├── db_connection_pool.yaml
│   ├── disk_full.yaml
│   ├── dns_failure.yaml
│   └── memory_leak.yaml
├── config/                        # Configuration files
│   ├── taxonomy.yaml              # Failure taxonomy definition
│   ├── baselines.yaml             # Metric baselines & log templates
│   ├── reward.yaml                # Reward weights & thresholds
│   ├── crawlers.yaml              # Crawler endpoints & patterns
│   └── environment.yaml           # Distractor diagnoses & remediations
├── tests/                         # Test suite (152 tests)
├── demo.py                        # Demo runner with heuristic agents
├── run_crawler.py                 # Crawler + scenario generator CLI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Development Notes

- Built with Claude Code as part of the development workflow
- Python 3.11+ required
- No external API keys needed for the core environment (crawler needs network access)
- Deterministic: same seed produces identical telemetry for reproducible RL training
