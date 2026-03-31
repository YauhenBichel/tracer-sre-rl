# Part 2: Core Architecture Design

## Objective

A scalable system that continuously generates high-fidelity failure scenarios for training AI SRE agents, integrated with the Tracer open-sre-agent ecosystem. The architecture addresses the five key constraints identified in Part 1: feedback latency, state fragmentation, non-determinism, ambiguous correctness, and environment cost.

## System Architecture Overview

```
DATA PLANE:     Crawlers (GCP, Cloudflare, GitHub, Aiops-Dataset)
                  → Taxonomy classifier (35 leaf nodes)
                    → ScenarioGenerator (topology + timeline + gold standard)

SIMULATION:     Layer 1: Synthetic telemetry (1,000 eps/hr, $0.001/ep)
                Layer 2: AIOpsLab real microservices (10 eps/hr, $0.10/ep)

TRAINING:       SREEnvironment (Gymnasium) ←→ opensre (LangGraph)
                  → RewardCalculator (diagnosis × efficiency + remediation + safety)
                    → Trajectory export (JSONL) → GRPO/DPO fine-tuning
                      → Coverage gap → new scenarios → repeat
```

## Data Flow

**Ingestion → Classification → Scenario Generation → Training → Evaluation**

**1. Ingestion.** Four crawlers fetch real incident data: VOID (~10K incidents from ~590 organisations), Aiops-Dataset (labeled fault scenarios from a 46-instance microservice system), GCP (5-year incident feed), Cloudflare (Statuspage API), GitHub post-mortems (~200 curated incidents). Each incident is normalised into a `NormalisedIncident` with: title, summary, timeline, root causes, affected services, taxonomy labels, quality score. Stored in SQLite.

**2. Classification.** Incidents are classified against the hierarchical taxonomy (35 leaf nodes across infrastructure, application, operational, external). Classification uses keyword pattern matching from `config/crawlers.yaml` and taxonomy label inheritance from the source data. Coverage gaps are tracked per taxonomy node.

**3. Scenario Generation.** The `ScenarioGenerator` converts each classified incident into a playable scenario YAML: selects a service topology template (5 options by failure category), builds an event timeline (8 templates by failure type), sets gold-standard root causes and remediations. The `IncidentReplaySource` combines generated scenarios with 5 hand-authored builtin scenarios. Each scenario is perturbed per seed (±15% timing, ±20% magnitudes) for unlimited variation.

**4. Training.** The `EpisodeRunner` samples scenarios (weighted by difficulty for curriculum learning), runs episodes through the `SREEnvironment`, and collects full trajectories (observation, action, reward per step). Trajectories are exported as JSONL for LLM fine-tuning (GRPO) or as preference pairs for DPO. The open-sre-agent's LangGraph pipeline can run against the environment via `SimulatedAction` objects that replace real Grafana/Datadog tool calls.

**5. Evaluation.** The `RewardCalculator` scores each episode across 4 dimensions: diagnosis accuracy (hierarchical taxonomy matching with partial credit), efficiency (steps × diagnosis coupling), remediation quality (gold-standard matching), and safety (investigation thoroughness). Results feed back into the coverage matrix to prioritise gap-filling.

---

## Pillar 1: Crawling & Indexing

### Data Pipeline for Continuous Ingestion

```
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Source Adapters  │────▶│  Normaliser  │────▶│  Dedup &     │────▶│  Index   │
│                  │     │              │     │  Quality     │     │  (SQLite/│
│ - GitHub scraper │     │ - Extract    │     │              │     │  Postgres)│
│ - Statuspage API │     │   structured │     │ - Embedding  │     │          │
│ - GCP JSON feed  │     │   fields     │     │   similarity │     │          │
│ - VOID exporter  │     │ - Normalise  │     │ - Min quality│     │          │
│ - RSS feeds      │     │   timestamps │     │   threshold  │     │          │
└──────────────────┘     │ - Map to     │     │ - Staleness  │     │          │
                         │   schema     │     │   window     │     │          │
                         └──────────────┘     └──────────────┘     └──────────┘
```

**Source Adapters (all implemented):**
1. **GitHub post-mortem crawler**: Scrapes curated collections (danluu/post-mortems, kubernetes-failure-stories). Follows links to original articles, extracts text via readability parsers.
2. **Statuspage API poller**: Polls JSON APIs from major providers (Cloudflare, GitHub, Atlassian) on a schedule. Structured data — no parsing needed.
3. **GCP incident feed**: Downloads the 5-year JSON incident history from `status.cloud.google.com`.
4. **VOID database connector**: Interfaces with the Verica Open Incident Database (~10K incidents from ~590 organisations) [VOID].
5. **Aiops-Dataset loader**: Imports labeled fault scenarios (log/metric/trace triplets with ground-truth root causes) from a 46-instance microservice e-commerce system [Aiops-Dataset].

**External datasets integrated for training data enrichment:**
- **LogHub** (300M+ real production log lines from HDFS, BGL, Thunderbird, OpenStack) — used to validate and improve synthetic log template realism [LogHub].
- **LitmusChaos ChaosHub** (50+ Kubernetes fault experiments) — used to expand taxonomy coverage and map fault types to scenario templates [LitmusChaos].
- **Microsoft AIOpsLab** (MLSys'25) — deploys real microservices with fault injection and telemetry export; serves as the reference implementation for Layer 2/3 validation [AIOpsLab].
- **RCAEval** — standardised benchmark for root cause analysis with metrics/logs/traces; used for evaluation comparison [RCAEval].

**Normalised Schema:**

```json
{
  "id": "uuid",
  "source": "github|statuspage|gcp|void",
  "source_url": "https://...",
  "title": "Database connection pool exhaustion during traffic spike",
  "summary": "...",
  "timeline": [
    {"timestamp": "2024-01-15T14:30:00Z", "event": "Alert fired: p99 latency > 500ms"},
    {"timestamp": "2024-01-15T14:35:00Z", "event": "On-call paged, began investigation"}
  ],
  "root_causes": ["connection_pool_exhaustion", "traffic_spike"],
  "affected_services": ["api-gateway", "user-service", "postgres"],
  "impact": {"duration_minutes": 45, "severity": "major", "users_affected": "~50K"},
  "remediation": ["Increased pool size from 50 to 200", "Added connection pool metrics"],
  "taxonomy_labels": ["infrastructure.database.connection_pool", "operational.capacity"],
  "ingested_at": "2025-03-27T10:00:00Z",
  "quality_score": 0.85
}
```

**Data Quality:** Each incident gets a quality score (0-1) based on field completeness (timeline, root cause, remediation). Deduplication uses text similarity on summaries. Re-crawl weekly.

---

## Pillar 2: Failure Classification & Taxonomy

### Hierarchical Taxonomy

```
Root
├── Infrastructure
│   ├── Network
│   │   ├── Partition (split-brain, cross-AZ)
│   │   ├── DNS (resolution failure, propagation delay, cache poisoning)
│   │   ├── TLS/Certificate (expiry, misconfiguration, CA outage)
│   │   ├── BGP (route leak, hijack, misconfiguration)
│   │   └── Load Balancer (misconfiguration, health check failure, capacity)
│   ├── Compute
│   │   ├── OOM (memory leak, heap exhaustion, container limits)
│   │   ├── CPU Saturation (runaway process, GC storms, thread contention)
│   │   ├── Instance Failure (hardware, hypervisor, spot termination)
│   │   └── Container/Pod (CrashLoopBackOff, image pull failure, resource limits)
│   ├── Storage
│   │   ├── Disk Full (log accumulation, temp files, data growth)
│   │   ├── IOPS Throttling (noisy neighbour, burst credit exhaustion)
│   │   ├── Data Corruption (bit rot, partial write, replication lag)
│   │   └── Volume Detachment (cloud provider, AZ failure)
│   └── Database
│       ├── Connection Pool Exhaustion
│       ├── Replication Lag (async replication, cross-region)
│       ├── Lock Contention (deadlock, long-running transaction)
│       ├── Schema Migration Failure
│       └── Split-Brain (failover failure, consensus loss)
├── Application
│   ├── Memory
│   │   ├── Leak (gradual, event-driven)
│   │   ├── Cache Stampede (thundering herd on cache expiry)
│   │   └── Buffer Overflow
│   ├── Concurrency
│   │   ├── Deadlock (resource ordering, distributed lock)
│   │   ├── Race Condition (TOCTOU, double-submit)
│   │   └── Thread Pool Exhaustion
│   ├── Dependency
│   │   ├── Cascading Failure (retry storms, circuit breaker failure)
│   │   ├── Upstream Timeout (slow dependency, missing timeout)
│   │   ├── API Breaking Change (version mismatch, schema drift)
│   │   └── Third-Party Outage (SaaS dependency, CDN)
│   └── Data
│       ├── Data Inconsistency (eventual consistency violation)
│       ├── Poison Pill (malformed message blocking queue)
│       └── Hotspot (partition key skew, hot shard)
├── Operational
│   ├── Deployment
│   │   ├── Bad Deploy (code bug, misconfiguration pushed)
│   │   ├── Rollback Failure (incompatible state, migration irreversible)
│   │   ├── Canary Failure (insufficient traffic, wrong metrics)
│   │   └── Feature Flag Misconfiguration
│   ├── Configuration
│   │   ├── Resource Limits (too low/too high)
│   │   ├── Security Group / Firewall
│   │   ├── Environment Variable (wrong value, missing)
│   │   └── Autoscaling Policy (too aggressive, too conservative)
│   └── Capacity
│       ├── Traffic Spike (organic growth, viral event, DDoS)
│       ├── Resource Quota (cloud provider limits)
│       └── Queue Backlog (consumer lag, producer burst)
└── External
    ├── Cloud Provider (AZ outage, service degradation)
    ├── DNS Provider (global resolution failure)
    └── CDN / Edge (cache invalidation, origin overload)
```

### Classification Pipeline (Implemented)

Keyword pattern matching from `config/crawlers.yaml` (25 patterns mapping to taxonomy labels). Multi-label supported — incidents can have multiple taxonomy labels. Coverage tracked via `EpisodeRunner.stats.rewards_by_scenario` per taxonomy node.

---

## Pillar 3: Simulation of Distributed Infrastructure

### Position: Hybrid — Real Data Drives Synthetic Generation

**Explicit position:** Real-data-driven synthetic telemetry (Layer 1) for bulk training, AIOpsLab [AIOpsLab] (Layer 2) for transfer validation.

Why not pure end-to-end? 100K episodes at $5/episode = $500K and 11 years. Why not pure synthetic? Hand-tuned baselines don't match real Grafana dashboards — the agent won't transfer. The hybrid: real incident data (Aiops-Dataset [Aiops-Dataset], VOID [VOID]) drives scenario generation; synthetic rendering produces the telemetry at 38ms/episode.

### The Architecture

| Layer | Data Source | Fidelity | Speed | Cost | When Used |
|-------|-----------|----------|-------|------|-----------|
| **L1: Real-Data-Driven Synthetic** | VOID + Aiops-Dataset → scenarios; LogHub → log calibration | Medium-High | ~1,000 eps/hr | ~$0.001/ep | Bulk RL training (95% of episodes) |
| **L2: AIOpsLab** | DeathStarBench microservices with real fault injection | High | ~10 eps/hr | ~$0.10/ep | Transfer validation (5% of episodes) |

#### Layer 1 — Real-Data-Driven Synthetic Telemetry (MVP, implemented)

The pipeline:

```
Real Incident Data                    Synthetic Telemetry Generation
─────────────────                    ────────────────────────────────

VOID (~10K incidents)  ──┐
Aiops-Dataset (labeled)──┤──► ScenarioGenerator ──► Scenario YAML
GCP/Cloudflare feeds  ──┘    (topology, timeline,    (parameterised)
                              gold standard)              │
                                                          │ + perturbation per seed
                                                          ▼
LogHub (300M+ real logs)──► Log template ──► TelemetryGenerator
                            calibration       │
                                              ├── Metrics (time-series with effect handlers)
                                              ├── Logs (event-correlated, pattern-matched)
                                              └── Traces (error rate from event params)
                                                          │
                                                          ▼
                                                    SREEnvironment
                                                    (Gymnasium API)
```

**What comes from real data:**
- **Scenario timelines** — extracted from VOID incident reports (what failed, when, in what order)
- **Root causes and remediations** — from VOID/Aiops-Dataset ground-truth labels
- **Service topologies** — mapped from Aiops-Dataset's 46-instance architecture
- **Failure patterns** — from Aiops-Dataset's labeled fault types (CPU saturation, memory leak, network partition, etc.)
- **Log message patterns** — calibrated against LogHub's 300M+ real production log lines from HDFS, BGL, Thunderbird, OpenStack

**What is generated synthetically:**
- **Metric time-series values** — produced by the metrics generator with noise and effect handlers, not replayed from recordings
- **Log timestamps and placeholder values** — generated per-episode with event correlation
- **Trace span durations and error status** — derived from active events, not recorded spans
- **Perturbation** — ±15% timing jitter, ±20% magnitude variation per seed

**Defence by criterion:**

| Criterion | Assessment |
|-----------|-----------|
| **Latency** | Sub-millisecond. Telemetry served from memory. Episodes complete in ~50ms. |
| **Cost** | ~$0.001/episode. 100K episodes = $100 (CPU only, no infrastructure). |
| **Determinism** | Same scenario + same seed = identical telemetry. Essential for RL gradient stability. |
| **Coverage** | Any fault in the taxonomy. Can generate scenarios for rare failures (split-brain, data corruption) that are dangerous or impossible to reproduce in real infrastructure. |
| **Realism** | Medium-high. Real incident patterns drive the scenarios. Log/metric/trace signals are correlated (an error_spike event produces matching ERROR logs, elevated error metrics, and ERROR trace spans simultaneously). Weakness: metric values in logs are approximate, not numerically identical to metric time-series. |

**Key limitation:** The agent does not execute remediation actions against real systems. It proposes remediations and is scored against gold standards. This is acceptable because SRE investigation (reading telemetry, correlating signals, forming hypotheses) is ~80% of the work, and it's the skill that benefits most from RL training. Remediation execution is validated in Layer 2.

#### Layer 2 — AIOpsLab Validation (recommended for transfer testing)

Microsoft's AIOpsLab [AIOpsLab] (MLSys'25) provides:
- **Real microservices** from DeathStarBench (social network, hotel reservation — 10-30 services each)
- **Real fault injection** at multiple levels (network, resource, application)
- **Real telemetry** — Jaeger traces, Prometheus metrics, Filebeat logs
- **Standardised evaluation** — detection, localisation, root cause diagnosis, mitigation tasks
- **Reproducible benchmarks** — other researchers can compare against the same setup

**How Layer 2 composes with Layer 1:**

```
Phase 1: Train on Layer 1 (95K episodes, ~48 hours)
  Agent learns investigation strategy from real-data-driven synthetic telemetry.

Phase 2: Validate on Layer 2 (500 episodes, ~50 hours)
  Run the trained agent against AIOpsLab's real microservices.
  Measure transfer gap: synthetic-trained performance vs real-telemetry performance.

Phase 3: Fine-tune on Layer 1 (5K episodes, ~5 hours)
  For failure modes where Layer 2 showed poor transfer,
  generate more Layer 1 scenarios with LogHub-calibrated templates
  that closer match the real telemetry format.

Phase 4: Final validation on Layer 2 (200 episodes, ~20 hours)
  Confirm transfer gap has narrowed.
```

This loop converges because each iteration identifies *specific* fidelity gaps (e.g., "the agent doesn't recognise Prometheus-style metric names" or "real logs have stack traces, synthetic ones don't") that can be addressed by improving the Layer 1 generators.

#### Layer 3 — Kubernetes + Chaos Engineering (future, for production validation)

Full Kubernetes deployment with LitmusChaos [LitmusChaos] (CNCF, 50+ pre-built fault experiments mapped to our taxonomy) and Chaos Mesh [Chaos Mesh]. Used only for final evaluation against production-fidelity failure modes and coverage gap testing. Not part of MVP — deferred until Layer 1 → Layer 2 transfer is validated.

---

## Pillar 4: Evaluation & Reward Design

The core challenge: in coding, "correct" means tests pass. In SRE, "correct" is ambiguous (multiple valid root causes), delayed (remediation effects unfold over time), and context-dependent (the right action depends on what the agent has already observed). The reward signal must be usable despite these properties.

### Question 1: What Constitutes a "Correct" Diagnosis?

**The problem.** The VOID database [VOID] shows that only ~25% of real incidents have a single definitive root cause. The majority involve multiple contributing factors. A database connection pool exhaustion might be caused by a traffic spike (proximate) AND an undersized pool configuration (underlying) AND a missing circuit breaker (contributing). All three are valid diagnoses at different levels of depth.

**Our approach: multi-label gold standard with relevance-weighted scoring.**

Each scenario defines multiple gold-standard root causes with relevance weights:

```yaml
gold_standard:
  root_causes:
    - taxonomy_label: "infrastructure.database.connection_pool"
      relevance: 1.0     # primary root cause
    - taxonomy_label: "operational.capacity.traffic_spike"
      relevance: 0.7     # contributing factor
```

The agent's diagnosis is scored against all gold-standard root causes, and the **best match wins**:

```
R_diagnosis = max_i( similarity(d, gᵢ) × relevance(gᵢ) )
```

This means diagnosing the traffic spike (relevance 0.7) still earns substantial credit even if the "primary" root cause is the connection pool. The agent is rewarded for identifying *any valid contributing factor*, weighted by how central it is.

**Handling novel diagnoses.** When the agent produces a diagnosis not in the gold standard, the hierarchical taxonomy provides graceful degradation. An agent that diagnoses `infrastructure.database.replication_lag` for a connection pool issue gets 0.7 (correct subcategory `infrastructure.database`) rather than 0.0 (wrong answer). This gradient signal is critical for RL — it tells the agent "you're in the right area, keep looking."

### Question 2: How Do You Score Partial Progress?

**Hierarchical taxonomy matching with configurable depth scores.**

The taxonomy tree provides a natural similarity metric. Deeper common prefixes mean closer diagnoses:

| Agent's Diagnosis | Gold Standard | Common Prefix | Depth | Score |
|---|---|---|---|---|
| `infrastructure.database.connection_pool` | `infrastructure.database.connection_pool` | exact match | 3 | **1.0** |
| `infrastructure.database.replication_lag` | `infrastructure.database.connection_pool` | `infrastructure.database` | 2 | **0.7** |
| `infrastructure.network.dns` | `infrastructure.database.connection_pool` | `infrastructure` | 1 | **0.4** |
| `application.memory.leak` | `infrastructure.database.connection_pool` | (none) | 0 | **0.0** |

The depth scores (0.0 → 0.4 → 0.7 → 1.0) are configured in `config/reward.yaml` and can be tuned. For depth ≥ 3, a continuous formula is used: `score = 0.9 + 0.1 × (common_depth / max_depth)`.

**Why this works for RL:** The gradient from 0.0 to 0.4 to 0.7 to 1.0 means the agent always has a direction to improve. Even a random agent that guesses the right top-level category (`infrastructure`) gets 0.4 instead of 0.0 — enough signal to reinforce that direction. This is analogous to how SWE-RL [SWE-RL] uses continuous patch similarity instead of binary pass/fail.

**Efficiency is coupled to diagnosis accuracy.** Raw efficiency (fewer steps = higher score) is multiplied by the diagnosis score:

```
R_efficiency = R_raw_efficiency × R_diagnosis
```

This prevents the degenerate policy where the agent diagnoses immediately (perfect efficiency, zero diagnosis accuracy). Being fast is only rewarded when the diagnosis is correct. The ideal step count of 5 (list alerts + 2 queries + diagnose + remediate) aligns with the safety scorer's requirement of ≥2 queries before diagnosis.

### Question 3: Temporal Credit Assignment

**The problem.** In a real incident, an agent might propose a remediation at minute 5, but the fix takes 10 minutes to take effect. At minute 6, a different metric degrades. Was that caused by the fix (side effect) or already in progress? The agent needs credit for the fix but shouldn't be blamed for coincidental degradation.

**Our approach: time compression eliminates the delay.**

The key architectural decision: the telemetry generator pre-computes the *entire episode timeline* at reset. The agent doesn't wait for effects to propagate — it queries any time range instantly. A 15-minute incident timeline is queryable in milliseconds. This collapses the temporal credit assignment problem into a spatial one: the agent can observe both "before" and "after" the failure onset within a single investigation step.

**Implemented (MVP):**

1. **Pre-generated timeline.** All telemetry (metrics, logs, traces) for the full episode duration is generated upfront. The agent queries `time_start=40, time_end=89` and sees the relevant window immediately — no waiting for real-time effects.

2. **End-of-episode sparse reward.** The four-component reward is computed once when the agent has both diagnosed and proposed remediation. Sparse rewards are harder for RL to learn from than dense rewards, but they avoid the credit assignment errors that come with intermediate rewards in non-deterministic environments.

3. **Truncation penalty.** If the agent runs out of steps before diagnosing, it receives a halved reward (`truncation_factor = 0.5`). If it diagnosed but didn't remediate, it receives `diagnosis_reward × 0.5`. This provides gradient signal even for incomplete episodes.

**Future extensions (not implemented):** potential-based reward shaping (Ng et al., 1999) for denser gradient signal, and counterfactual remediation scoring (simulate two futures with/without fix).

### Question 4: Human-in-the-Loop Evaluations

Humans are needed for: novel root causes not in taxonomy, taxonomy expansion, and reward calibration. To minimise cost: LLM-as-judge handles ~90% of scoring; active learning requests human review only for uncertain cases (disagreement between judge prompts). Estimated cost: ~$4K for a 100K-episode run.

**Estimated cost:** At an active learning rate of 10% and a batch review rate of 50 episodes/hour, a 100K-episode training run requires ~2,000 human-reviewed episodes = ~40 hours of expert time. At $100/hour for an SRE consultant, that's $4,000 — a small fraction of the total training cost.

### Reward Function Summary

```
R = 0.40 × R_diagnosis + 0.20 × R_efficiency + 0.25 × R_remediation + 0.15 × R_safety

R_diagnosis:    Hierarchical taxonomy similarity × gold relevance weight
R_efficiency:   (Linear decay from ideal_steps to max_steps) × R_diagnosis
R_remediation:  Gold-standard effectiveness score, or partial credit for generic actions
R_safety:       1.0 - penalties for hasty diagnosis (-0.3) and tunnel vision (-0.2)
```

### Validated with Baseline Agents

The reward function has been validated to discriminate between agent quality levels:

| Agent | Avg Reward | Diagnosis | Efficiency | Remediation | Safety |
|-------|-----------|-----------|------------|-------------|--------|
| Random | 0.321 | 0.244 | 0.244 | 0.100 | 1.000 |
| Heuristic (no gold labels) | 0.464 | 0.160 | 0.077 | 0.940 | 1.000 |
| Oracle (gold labels) | 0.875 | 0.960 | 0.480 | 0.980 | 1.000 |

The 2.7× gap between oracle (0.875) and random (0.321) provides clear RL gradient signal. The heuristic agent's low diagnosis score (0.160) confirms that diagnosis accuracy is the primary bottleneck — the skill that benefits most from training.

---

## Pillar 5: Test Case Generation & Coverage

The taxonomy has 35 leaf failure types. The MVP covers 18 of them (51%) across 5 hand-authored scenarios + 241 Aiops-Dataset faults + 64 crawled incidents.

### Question 1: From Indexed Incidents to Reproducible Test Cases

**The implemented pipeline (3 stages):**

```
Aiops-Dataset (241 faults)─┐
GCP/Cloudflare/GitHub    ──┤──► ScenarioGenerator ──► scenario.yaml ──► SREEnvironment
(261 crawled incidents)  ──┘    (topology + timeline     (perturbation per seed
                                 + gold standard)         + telemetry generation)
```

**Stage 1: Crawl and normalise.** Three crawlers (`GCPIncidentCrawler`, `CloudflareIncidentCrawler`, `GitHubPostmortemCrawler`) fetch real incidents from public APIs and normalise them into `NormalisedIncident` objects stored in SQLite (261 incidents, 288 KB). Additionally, `load_aiops_groundtruth()` reads the Aiops-Dataset groundtruth CSV — 241 labeled fault scenarios from a real 46-instance microservice system, included in the repo at `data/groundtruth-all.csv` (16 KB). Each incident has: title, summary, taxonomy labels, quality score.

**Stage 2: Generate scenario YAML.** The `ScenarioGenerator` converts each `NormalisedIncident` into a playable scenario:

- **Topology selection.** Maps the incident's taxonomy label to a service topology template. `infrastructure.database.*` incidents get a web-server → application → database topology. `application.*` incidents get a topology with cache and queue services. 5 topology templates cover the major categories.

- **Timeline construction.** Maps taxonomy labels to event timeline templates. A `connection_exhaustion` incident gets: `normal_traffic → traffic_ramp → connection_exhaustion → error_spike → alert`. 8 event templates cover the most common failure patterns. Incidents without a matching template get a generic `error_spike → latency_spike → alert` timeline.

- **Gold standard.** Root causes come from the incident's taxonomy labels with relevance weights. Remediations come from the incident's remediation field (if populated) or a generic fallback.

- **Difficulty estimation.** Derived from incident severity: critical → 0.6, major → 0.5, minor → 0.3.

**Stage 3: Play in the RL environment.** The generated scenario is perturbed per seed (±15% timing, ±20% magnitudes) and run through the `SREEnvironment`.

### Question 2: Collecting a Corpus for 80% Coverage

**Current state:** 18 of 35 taxonomy leaves covered (51%) from builtins + Aiops-Dataset + crawled incidents.

**Path to 80%:** Four tiers: (1) auto-generate from VOID/Aiops-Dataset, (2) map LitmusChaos ChaosHub experiments [LitmusChaos] to taxonomy nodes, (3) LLM-assisted gap-filling for rare failures (split_brain, data_corruption), (4) community-contributed scenarios.

### Question 3: Novel Scenarios for Generalisation

An agent that memorises 5 scenarios is useless. It must generalise to failure patterns it hasn't seen. Four mechanisms:

**1. Perturbation (implemented).** Each seed produces a different variation of the same scenario. The `perturb_scenario()` function jitters event timing (±15%) and effect magnitudes (±20%). Same connection pool exhaustion scenario, seed 42 has the alert at 540s with 3x traffic — seed 43 has the alert at 580s with 2.5x traffic. 5 scenarios × unlimited seeds = unlimited variations.

**2. Compositional scenarios (designed).** Combine two failure modes into one scenario. Example: memory leak + traffic spike. The agent must determine which is the primary root cause and which is a contributing factor. Implementation: merge two scenario timelines onto the same service topology, assign the primary root cause higher relevance weight.

**3. Topology variation (designed).** Same failure pattern, different service graph. A connection pool exhaustion in a 3-service topology (web → app → db) behaves differently than in an 8-service topology (web → 3 app services → db + cache + queue). The `ScenarioGenerator` has 5 topology templates; training samples from all of them.

**4. Held-out evaluation set.** Reserve 20% of scenarios for evaluation only — never used during training. Measure generalisation gap: `(train_reward - eval_reward) / train_reward`. A gap > 20% indicates overfitting; trigger more perturbation diversity or compositional scenarios.

### Question 4: Coverage Measurement and Gap-Filling Priority

**Coverage metric per taxonomy node:**

```
Coverage(node) = min(
    incident_count(node) / 5,           # at least 5 real incidents
    scenario_count(node) / 3,           # at least 3 playable scenarios
    1.0
)
```

**Overall coverage:**

```
Overall = Σ( Coverage(leaf) × frequency_weight(leaf) ) / Σ( frequency_weight(leaf) )
```

Frequency weights come from real incident data: a taxonomy node that represents 10% of VOID incidents gets 10× the weight of a node representing 1%. This ensures the agent trains hardest on the failures that happen most in production.

**Gap-filling priority.** Nodes are prioritised for scenario generation by:

```
priority(node) = frequency_weight(node) × (1 - Coverage(node)) × (1 - agent_score(node))
```

A high-frequency, low-coverage, low-agent-score node gets the highest priority. This means: (a) failures that happen often in production, (b) that we don't have enough training data for, (c) that the agent is currently bad at — get filled first.

**Implemented tracking.** The `EpisodeRunner.stats.rewards_by_scenario` tracks per-scenario reward averages. The `ScenarioGenerator` tracks which taxonomy labels have been generated. Combining these gives the coverage matrix needed for gap-filling.

---

## RL Loop Closure

The learning loop: (1) sample scenario from config, (2) opensre investigates via SimulatedActions, (3) RewardCalculator scores against gold standard, (4) trajectory saved as JSONL, (5) GRPO/DPO fine-tuning updates the LLM, (6) improved model deployed back to opensre.

RL improves two opensre decisions: `plan_actions` (which tools to call) and `root_cause_diagnosis` (which root cause to propose). Generalisation comes from perturbation (different seed = different telemetry), curriculum (easy → hard), and scenario diversity (246 scenarios across 18 taxonomy leaves).

```bash
make train          # collect trajectories
make export         # → training_data/trajectories.jsonl
make finetune       # GRPO fine-tuning (dry run without GPU)
```

---

## opensre Integration (Verified End-to-End)

A custom LangGraph graph replaces opensre's `plan_actions` and `investigate` nodes with simulated versions that use our environment. Action names match opensre's `EVIDENCE_MAPPERS` (`query_grafana_*`). Verified: opensre's LLM investigated a simulated scenario, planned 4 tool calls across 3 loops, and correctly diagnosed the root cause with 100% validity.

---

## MVP Scope vs. Deferred

**Built:** Telemetry generator, Gymnasium environment, 5 builtin + 241 Aiops-Dataset scenarios, 4-component reward, 3 baseline agents, crawlers, scenario generator, opensre integration (custom LangGraph), Q-learning agent, training loop with EDA/accuracy matrix/eval, trajectory export, sim-to-real validation, Docker, CI. 161 tests.

**Deferred:** Layer 2/3 real infrastructure, LLM-as-judge, milestone rewards, automatic curriculum, distributed training.
