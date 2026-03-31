# SRE Reinforcement Learning Environment

A training environment for AI SRE agents to learn incident investigation, diagnosis, and remediation. Built for the [Tracer open-sre-agent](https://github.com/Tracer-Cloud/open-sre-agent) ecosystem.

## Get Started

```bash
git clone <repo-url> && cd tracer-sre-rl
make quickstart     # install + test + demo + learn — everything in one command
```

Or step by step: `make install` → `make test` → `make demo` → `make learn`

## What You Can Do

| Command | What it does | Time |
|---------|-------------|------|
| `make demo` | Compare 3 agents (random, heuristic, oracle) on 5 scenarios | 2s |
| `make learn` | Train Q-learning agent — shows reward improving with live progress | 15s |
| `make train` | Collect trajectories with EDA, accuracy matrix, and eval report | 10s |
| `make export` | Export trajectories as JSONL for LLM fine-tuning | 10s |
| `make finetune` | Fine-tune LLM on trajectories (dry run without GPU) | 2s |
| `make validate` | Measure sim-to-real gap against Aiops-Dataset telemetry | 2s |
| `make test` | Run 161 tests | 6s |
| `make crawl` | Fetch latest incidents from public APIs | 30s |
| `make train-all` | Crawl + train on everything | 40s |
| `make help` | Show all commands | instant |

## Configure Training

Edit `config/training.yaml` — no code changes needed:

```yaml
sources:
  builtin_scenarios: scenarios/                   # 5 hand-authored scenarios
  aiops_groundtruth: data/groundtruth-all.csv     # 241 real faults (in repo)
  # incident_db: data/incidents.db                # uncomment after: make crawl

episodes: 100
max_difficulty: null          # null = all, 0.4 = easy only
eval_episodes: 20             # held-out evaluation episodes

agent:
  type: random                # random | heuristic | oracle
```

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

## Q-Learning Agent (make learn)

```
Training Q-learning agent on 5 scenarios, 200 episodes...
 Episode    Reward   Avg(50)   Epsilon  Progress
-----------------------------------------------------------------
      50     0.316     0.393     0.778  █████░░░░░░░░░░░░░░░ 25%
     100     0.800     0.478     0.606  ██████████░░░░░░░░░░ 50%
     150     0.495     0.405     0.471  ███████████████░░░░░ 75%
     200     0.245     0.416     0.367  ████████████████████ 100%

  First 50 episodes avg:  0.393
  Last 50 episodes avg:   0.416
  Improvement:            +6.0%

  The agent learned. Reward improved over training.
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
make learn          # 1. Prove the reward signal drives learning (Q-learning)
make export         # 2. Export trajectories as JSONL
make finetune       # 3. Fine-tune LLM (dry run, or real with GPU + trl)
make validate       # 4. Measure sim-to-real fidelity gap
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
  → ScenarioGenerator → topology + timeline + gold standard
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
docker compose up demo          # Baseline comparison
docker compose up test          # 161 tests
docker compose up train         # Train (reads config/training.yaml)
docker compose up train-export  # Train + export JSONL
docker compose up crawl         # Fetch incidents
```

## CI

GitHub Actions on every PR to `main`: tests on Python 3.11 + 3.12, demo + training smoke tests.

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
├── src/
│   ├── crawler/            # Crawlers + scenario generator
│   ├── generators/         # Synthetic telemetry (metrics, logs, traces)
│   ├── environment/        # Gymnasium RL environment (7 actions)
│   ├── evaluation/         # Reward (4 scorers)
│   ├── integration/        # opensre bridge
│   └── training/           # Episode runner, trajectory export, validation, metrics
├── tests/                  # 161 tests
├── .github/workflows/      # CI
├── Makefile                # All commands
├── demo.py                 # Baseline comparison
├── run_training.py         # Training (EDA, progress, eval, accuracy matrix)
├── run_learning.py         # Q-learning demo (live progress)
├── run_finetune.py         # LLM fine-tuning (GRPO/DPO)
├── run_crawler.py          # Incident crawler
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

Built with Claude Code. Python 3.11+. 161 tests. Deterministic per seed.
