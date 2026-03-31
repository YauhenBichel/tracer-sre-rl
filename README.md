# SRE Reinforcement Learning Environment

A training environment for AI SRE agents to learn incident investigation, diagnosis, and remediation. Built for the [Tracer open-sre-agent](https://github.com/Tracer-Cloud/open-sre-agent) ecosystem.

## Get Started

```bash
git clone <repo-url> && cd tracer-sre-rl
make quickstart     # install + test + demo — everything in one command
```

Or step by step: `make install` → `make test` → `make check-reward`

## What You Can Do

| Command | What it does | Time |
|---------|-------------|------|
| `make check-reward` | Verify the reward function scores agents correctly | 2s |
| `make train` | Run training episodes and show accuracy per failure type | 10s |
| `make export` | Save training trajectories to a file for LLM fine-tuning | 10s |
| `make finetune` | Run LLM fine-tuning on saved trajectories (preview without GPU) | 2s |
| `make crawl` | Fetch real incidents from GCP, Cloudflare, and GitHub | 30s |
| `make test` | Run all tests | 6s |
| `make help` | Show all commands | instant |

## Configure Training

Edit `config/training.yaml` — no code changes needed:

| Setting | What it controls | Default |
|---------|-----------------|---------|
| `sources.builtin_scenarios` | Hand-authored failure scenarios | `scenarios/` |
| `sources.aiops_groundtruth` | Real faults from Aiops-Dataset (241 incidents) | `data/groundtruth-all.csv` |
| `sources.incident_db` | Crawled incidents (uncomment after `make crawl`) | disabled |
| `episodes` | Number of training episodes | 100 |
| `agent.type` | Test agent: `random`, `heuristic`, or `oracle` | `random` |

Then: `make train`

## Training Output

Every `make train` run shows:

```
Step 1/4: Loading data sources
  Builtin scenarios: 5
  Aiops-Dataset: 241 scenarios

Step 2/4: Validating scenarios
  Valid: 246, Train: 200, Eval: 51
  Taxonomy: 15 leaves covered
  Taxonomy distribution:
    infrastructure.network.partition           88 ████████████████████
    infrastructure.compute.cpu_saturation      41 ████████████████████
    application.memory.leak                    30 ██████████████████
    ...

Step 3/4: Training (100 episodes)
  ████████████████████ 100/100 episodes, avg reward: 0.331

Step 4/4: Evaluating on held-out set
  Train avg reward: 0.331
  Eval avg reward:  0.325
  Generalisation:   good (train-eval gap: 2.3%)

Accuracy by Taxonomy Label
  infrastructure.network.partition         31    0.324   0.307   0.246
  infrastructure.compute.cpu_saturation    15    0.336   0.251   0.184
  application.memory.leak                  15    0.282   0.228   0.200
  ...
```

## opensre End-to-End (Verified)

opensre's LLM investigated a simulated "Disk Full" scenario through the full LangGraph pipeline:

```
  ● Planning  5.5s  Planned: ['query_grafana_alert_rules', 'query_grafana_metrics']
  ● Gathering evidence  1ms  grafana:1 alert rules, grafana:1 metric series
  ● Diagnosing  8.1s  confidence 62%          ← needs more evidence
  ● Planning  5.3s  Planned: ['query_grafana_service_names', 'query_grafana_logs']
  ● Gathering evidence  1ms  grafana:1 services, grafana:1 logs (13 errors)
  ● Diagnosing  13.2s  confidence 100%        ← found the root cause

  Root cause: "high disk usage on postgres-primary caused cascading failures"
  Category: resource_exhaustion
  Validity: 100%
  Investigation loops: 3
```

The LLM autonomously decided what to query, gathered synthetic evidence, and correctly identified the root cause — all through opensre's production LangGraph pipeline with simulated evidence sources.

## Full Learning Pipeline

```bash
make export         # 1. Export trajectories as JSONL
make finetune       # 2. Fine-tune LLM (dry run, or real with GPU + trl)
```

## Train opensre

opensre investigates simulated incidents, gets scored, trajectories saved for LLM fine-tuning:

```bash
# 1. Install opensre (one time)
git clone https://github.com/Tracer-Cloud/open-sre-agent ../opensre
cd ../opensre && python3 -m venv .venv && source .venv/bin/activate
make install && pip install gymnasium numpy pyyaml requests
opensre onboard

# 2. Run opensre against simulated scenarios
cd ../tracer-sre-rl
PYTHONPATH="../opensre:." python -m src.integration.opensre_runner

# 3. With opensre's full LLM pipeline
PYTHONPATH="../opensre:." python -m src.integration.opensre_runner --use-opensre
```

## How It Works

```
Real incidents (Aiops-Dataset, GCP, Cloudflare, GitHub)
  → ScenarioGenerator → topology + timeline + correct answers
    → TelemetryGenerator → correlated metrics, logs, traces
      → SREEnvironment → agent investigates (7 tool-based actions)
        → RewardCalculator → diagnosis × efficiency + remediation + safety
          → Trajectory export → JSONL for GRPO/DPO fine-tuning
```

Each episode perturbed per seed (±15% timing, ±20% magnitudes). 246 scenarios × unlimited seeds = unlimited training data.

## Baseline Results

| Agent | Reward | Diagnosis | Efficiency | Remediation | Safety |
|-------|--------|-----------|------------|-------------|--------|
| Random | 0.321 | 0.244 | 0.244 | 0.100 | 1.000 |
| Heuristic | 0.464 | 0.160 | 0.077 | 0.940 | 1.000 |
| Oracle | 0.875 | 0.960 | 0.480 | 0.980 | 1.000 |

2.7× gap between oracle and random = clear RL learning signal.

## Training Data

| Source | Size | In Repo | Command |
|--------|------|---------|---------|
| 5 builtin scenarios | 5 YAML files | Yes | `make train` |
| 241 Aiops-Dataset faults [1] | 16 KB CSV | Yes | `make train` |
| 261 crawled incidents | 288 KB SQLite | Generated | `make crawl` |

See [docs/data_sources.md](docs/data_sources.md) for large datasets (LogHub, AIOpsLab).

## Docker

```bash
docker compose run --rm check-reward   # Verify reward function
docker compose run --rm test           # Run 169 tests
docker compose run --rm train          # Run training episodes
docker compose run --rm export         # Save trajectories
docker compose run --rm crawl          # Fetch real incidents
```

## CI

GitHub Actions on every PR to `main`: tests on Python 3.11 + 3.12, baseline + training smoke tests.

## Project Structure

```
tracer-sre-rl/
├── config/
│   ├── training.yaml       ← Data sources, agent, episodes, eval
│   ├── reward.yaml         ← Reward weights and thresholds
│   ├── taxonomy.yaml       ← 35 failure types
│   ├── baselines.yaml      ← Metric baselines, log templates
│   └── crawlers.yaml       ← Crawler endpoints + classification patterns
├── data/
│   └── groundtruth-all.csv ← 241 Aiops-Dataset faults (in repo)
├── scenarios/              ← 5 hand-authored failure scenarios
├── app/
│   ├── main.py             # Entry point: python -m app.main <command>
│   ├── cli/                # CLI commands (train, check-reward, crawl, finetune)
│   ├── models/             # Domain models (telemetry, scenario, agent, taxonomy)
│   ├── incidents/          # Fetch and process real incidents
│   ├── telemetry/          # Generate synthetic metrics, logs, traces
│   ├── rl_env/             # Gymnasium RL environment (7 actions)
│   ├── evaluation/         # Reward scoring (4 components)
│   ├── integration/        # opensre bridge (SimulatedAction, AgentState)
│   └── training/           # Episode runner, trajectory export, validation
├── tests/                  # 169 tests
├── .github/workflows/      # CI
├── Makefile                # All commands
├── Dockerfile
└── docker-compose.yml
```

## Docs

| Doc | Content |
|-----|---------|
| [Part 1: Research](docs/part1_research.md) | RL for SWE agents, SRE constraints (25 citations) |
| [Part 2: Architecture](docs/part2_architecture.md) | 5 pillars, learning flow, opensre mapping |
| [Part 3: Implementation](docs/part3_implementation.md) | MVP walkthrough, compute estimates |
| [Part 4: Reflection](docs/part4_reflection.md) | Results, limitations, open questions |
| [Data Sources](docs/data_sources.md) | Training data pipeline, available datasets |

---

[1] bbyldebb, "Aiops-Dataset," 2022. https://github.com/bbyldebb/Aiops-Dataset

Built with Claude Code. Python 3.11+. 169 tests. Deterministic per seed.
