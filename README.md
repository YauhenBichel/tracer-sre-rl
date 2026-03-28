# SRE Reinforcement Learning Environment

A training environment for AI SRE agents to learn incident investigation, diagnosis, and remediation. Built for the [Tracer open-sre-agent](https://github.com/Tracer-Cloud/open-sre-agent) ecosystem.

## Get Started

```bash
git clone <repo-url> && cd tracer-sre-rl
make install
make demo
```

## What You Can Do

| Command | What happens | Time |
|---------|-------------|------|
| `make demo` | Compare 3 agents (random, heuristic, oracle) on 5 scenarios | 2 sec |
| `make train` | Train on 5 builtin scenarios | 3 sec |
| `make train-real` | Train on 241 real Aiops-Dataset faults | 5 sec |
| `make train-all` | Train on everything (builtins + real faults + crawled) | 30 sec |
| `make test` | Run 155 tests | 3 sec |
| `make crawl` | Fetch latest incidents from public APIs | 30 sec |
| `make update-data` | Refresh all data sources | 30 sec |
| `make export` | Export trajectories for LLM fine-tuning | 10 sec |
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

agent:
  type: random                # random | heuristic | oracle
```

Then: `python run_training.py`

## Train opensre

The main use case — opensre investigates simulated incidents, gets scored, trajectories are saved for LLM fine-tuning:

```bash
# 1. Install opensre (one time)
git clone https://github.com/Tracer-Cloud/open-sre-agent ../opensre
cd ../opensre && python3 -m venv .venv && source .venv/bin/activate
make install && pip install gymnasium numpy pyyaml requests
opensre onboard    # configure LLM provider

# 2. Run opensre against simulated scenarios
cd ../tracer-sre-rl
PYTHONPATH="../opensre:." python -m src.integration.opensre_runner

# 3. With opensre's full LLM pipeline
PYTHONPATH="../opensre:." python -m src.integration.opensre_runner --use-opensre

# 4. Export trajectories
make export        # → training_data/trajectories.jsonl
```

## How It Works

```
Real incidents (Aiops-Dataset, GCP, Cloudflare, GitHub)
  → ScenarioGenerator → service topology + event timeline + gold standard
    → TelemetryGenerator → correlated metrics, logs, traces
      → SREEnvironment → agent investigates (7 tool-based actions)
        → RewardCalculator → score (diagnosis × efficiency + remediation + safety)
          → Trajectory export → JSONL for LLM fine-tuning
```

Each episode perturbed per seed. 246 scenarios × unlimited seeds = unlimited training data.

## Baseline Results

| Agent | Reward | Diagnosis | Efficiency | Remediation | Safety |
|-------|--------|-----------|------------|-------------|--------|
| Random | 0.321 | 0.244 | 0.244 | 0.100 | 1.000 |
| Heuristic | 0.464 | 0.160 | 0.077 | 0.940 | 1.000 |
| Oracle | 0.875 | 0.960 | 0.480 | 0.980 | 1.000 |

## Training Data

| Source | Size | In Repo | Command |
|--------|------|---------|---------|
| 5 builtin scenarios | 5 YAML files | Yes | `make train` |
| 241 Aiops-Dataset faults [1] | 16 KB CSV | Yes | `make train-real` |
| 261 crawled incidents | 288 KB SQLite | Generated | `make crawl` |

See [docs/data_sources.md](docs/data_sources.md) for large datasets (LogHub, AIOpsLab).

## Project Structure

```
tracer-sre-rl/
├── config/training.yaml    ← Configure training (data, agent, episodes)
├── config/reward.yaml      ← Configure reward weights
├── data/                   ← Training data (CSV in repo, DB generated)
├── scenarios/              ← 5 hand-authored failure scenarios
├── src/
│   ├── crawler/            # Incident crawlers + scenario generator
│   ├── generators/         # Synthetic telemetry (metrics, logs, traces)
│   ├── environment/        # Gymnasium RL environment
│   ├── evaluation/         # Reward (4 scorers)
│   ├── integration/        # opensre bridge
│   └── training/           # Episode runner + trajectory export
├── docs/                   # Assessment deliverables (Parts 1-4)
├── tests/                  # 155 tests
├── Makefile                # All commands
├── demo.py                 # Baseline comparison
├── run_training.py         # Training CLI
└── run_crawler.py          # Incident crawler CLI
```

## Docs

| Doc | Content |
|-----|---------|
| [Part 1: Research](docs/part1_research.md) | RL for SWE agents, SRE constraints (25 citations) |
| [Part 2: Architecture](docs/part2_architecture.md) | 5 pillars, learning flow, opensre mapping |
| [Part 3: Implementation](docs/part3_implementation.md) | MVP walkthrough, compute estimates |
| [Part 4: Reflection](docs/part4_reflection.md) | Results, limitations, open questions |
| [Data Sources](docs/data_sources.md) | Training data pipeline, datasets |

---

[1] bbyldebb, "Aiops-Dataset," 2022. https://github.com/bbyldebb/Aiops-Dataset

Built with Claude Code. Python 3.11+. 155 tests. Deterministic per seed.
