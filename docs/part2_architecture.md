# Part 2: Core Architecture Design

## Objective

A scalable system that continuously generates high-fidelity failure scenarios for training AI SRE agents, integrated with the Tracer open-sre-agent ecosystem. The architecture addresses the five key constraints identified in Part 1: feedback latency, state fragmentation, non-determinism, ambiguous correctness, and environment cost.

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA PLANE                                      │
│                                                                          │
│  ┌────────────────────┐   ┌──────────────────┐   ┌───────────────────┐  │
│  │  Incident Crawlers │   │ Failure Taxonomy  │   │ Scenario          │  │
│  │                    │──▶│ & Classifier      │──▶│ Generator         │  │
│  │  - VOID  (~10K)    │   │                   │   │                   │  │
│  │  - Aiops-Dataset   │   │  - 35 leaf nodes  │   │  - Topology       │  │
│  │  - GCP incidents   │   │  - Rule + keyword │   │    templates (5)  │  │
│  │  - Cloudflare      │   │    classification │   │  - Event timeline │  │
│  │  - GitHub PMs      │   │  - Coverage       │   │    templates (8)  │  │
│  │                    │   │    gap tracking    │   │  - Gold standard  │  │
│  └────────────────────┘   └──────────────────┘   └────────┬──────────┘  │
│                                                            │             │
│          SQLite DB                                         │             │
│          (normalised incidents)                            ▼             │
│                                                  ┌────────────────────┐ │
│                                                  │ IncidentReplay     │ │
│                                                  │ Source             │ │
│                                                  │  + 5 builtin YAML │ │
│                                                  │  + N generated    │ │
│                                                  └────────┬──────────┘ │
└───────────────────────────────────────────────────────────┼────────────┘
                                                            │
┌───────────────────────────────────────────────────────────┼────────────┐
│                       SIMULATION PLANE                    │            │
│                                                           ▼            │
│  ┌───────────────────────────────────┐   ┌─────────────────────────┐  │
│  │ Layer 1: Real-Data-Driven         │   │ Layer 2: AIOpsLab       │  │
│  │ Synthetic Telemetry               │   │ (MLSys'25)              │  │
│  │                                   │   │                         │  │
│  │  Scenario + seed                  │   │  DeathStarBench         │  │
│  │    → perturbation (±15%/±20%)     │   │  microservices          │  │
│  │    → MetricsGenerator (effects)   │   │    + fault injection    │  │
│  │    → LogGenerator (event-aware)   │   │    + Jaeger traces      │  │
│  │    → TraceGenerator (correlated)  │   │    + Prometheus metrics  │  │
│  │                                   │   │    + Filebeat logs      │  │
│  │  ~1,000 eps/hr · $0.001/ep        │   │  ~10 eps/hr · $0.10/ep  │  │
│  └──────────────┬────────────────────┘   └─────────────────────────┘  │
│                 │                                                      │
└─────────────────┼──────────────────────────────────────────────────────┘
                  │
┌─────────────────┼──────────────────────────────────────────────────────┐
│                 │          TRAINING PLANE                               │
│                 ▼                                                       │
│  ┌────────────────────┐    ┌─────────────────────┐                     │
│  │ SREEnvironment     │    │ open-sre-agent      │                     │
│  │ (Gymnasium API)    │◄──▶│ (LangGraph pipeline)│                     │
│  │                    │    │                     │                     │
│  │  obs: text telemetry│    │  extract_alert      │                     │
│  │  act: tool calls   │    │  → plan_actions     │                     │
│  │  7 actions         │    │  → investigate      │                     │
│  └────────┬───────────┘    │  → diagnose         │                     │
│           │                │  → publish           │                     │
│           │                └─────────────────────┘                     │
│           ▼                                                            │
│  ┌────────────────────┐    ┌─────────────────────┐                     │
│  │ Reward Engine      │    │ Training Loop       │                     │
│  │                    │───▶│ (EpisodeRunner)     │                     │
│  │  R = 0.40·diag     │    │                     │                     │
│  │    + 0.20·eff      │    │  - Trajectory       │                     │
│  │    + 0.25·rem      │    │    collection       │                     │
│  │    + 0.15·safety   │    │  - JSONL/DPO export │                     │
│  └────────────────────┘    │  - Curriculum        │                     │
│                            │  - Coverage tracking │                     │
│                            └─────────────────────┘                     │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     FEEDBACK LOOP                                │   │
│  │  Episode → Reward → Trajectory export → Policy update (GRPO) → │   │
│  │  Coverage gap → Gap-filling priority → New scenarios → Repeat   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

**Ingestion → Classification → Scenario Generation → Training → Evaluation**

**1. Ingestion.** Five crawlers fetch real incident data: VOID (~10K incidents from ~590 organisations), Aiops-Dataset (labeled fault scenarios from a 46-instance microservice system), GCP (5-year incident feed), Cloudflare (Statuspage API), GitHub post-mortems (~200 curated incidents). Each incident is normalised into a `NormalisedIncident` with: title, summary, timeline, root causes, affected services, taxonomy labels, quality score. Stored in SQLite.

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

### Classification Pipeline

1. **Rule-based pre-filter**: Pattern matching on keywords (e.g., "OOM" → Infrastructure.Compute.OOM, "connection pool" → Infrastructure.Database.Connection_Pool_Exhaustion).
2. **LLM classifier**: For incidents that don't match rules, use an LLM with the full taxonomy as context to classify. Prompt includes the incident summary, timeline, and root cause, and asks the model to select taxonomy paths and confidence scores.
3. **Multi-label**: Incidents can have multiple taxonomy labels (e.g., a deployment that triggers a cascading failure → Operational.Deployment.Bad_Deploy + Application.Dependency.Cascading_Failure).

### Coverage Gap Detection

Maintain a **coverage matrix**: for each taxonomy leaf node, track (a) number of indexed incidents, (b) number of generated scenarios, (c) agent performance score. Nodes with low incident count or low agent performance are flagged for targeted crawling or synthetic scenario generation.

---

## Pillar 3: Simulation of Distributed Infrastructure

### Position: Real-Data-Driven Synthetic Telemetry + Validation

**Explicit position:** Hybrid multi-layer, with Layer 1 (real-data-driven synthetic telemetry) as the workhorse and Layer 2 (AIOpsLab [AIOpsLab]) for transfer validation.

This is *not* pure synthetic generation — the scenarios, failure patterns, and telemetry characteristics are derived from real incident databases (VOID [VOID], Aiops-Dataset [Aiops-Dataset], LogHub [LogHub]). What is synthetic is the *rendering* — we generate metrics, logs, and traces programmatically rather than running real infrastructure, because RL training requires thousands of episodes per hour.

### Why Not Pure End-to-End Testing?

End-to-end testing (deploy services, inject faults, observe) is the gold standard for fidelity. But RL training requires 100,000+ episodes. At $5/episode and 1 episode/hour (the realistic cost of a Kubernetes chaos experiment), a single training run would take 11 years and cost $500,000. The economics force a layered approach.

### Why Not Pure Synthetic Generation?

Pure synthetic generation (hand-tuned baselines, made-up log templates) produces telemetry that doesn't match real production systems. An agent trained on synthetic-only data may fail when confronted with real Grafana dashboards because the noise profiles, metric names, and log formats differ. The solution: **calibrate synthetic generation against real data**.

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

**Designed for future implementation:**

4. **Potential-based reward shaping.** Following Ng et al. (1999), define a potential function Φ(s) over investigation states. Award shaped reward `F(s, s') = γΦ(s') - Φ(s)` at each step. Example: Φ increases when the agent queries a service mentioned in an alert (signal that investigation is on track). This preserves the optimal policy while providing denser gradient signal. The first candidate: `Φ = number_of_services_queried / total_services × 0.1`.

5. **Counterfactual remediation scoring.** After the agent proposes a fix, simulate two futures — with and without the remediation — and reward the *difference* in system health metrics. This isolates the agent's causal contribution from background noise. Requires extending the telemetry generator with post-remediation timeline branching.

### Question 4: Human-in-the-Loop Evaluations

**Where humans are needed:**

1. **Novel root causes.** When the agent produces a diagnosis that doesn't match any gold-standard label, it could be wrong — or it could be a valid alternative the scenario designer didn't anticipate. Only a human SRE can distinguish. Example: diagnosing "application.concurrency.thread_pool_exhaustion" for a scenario labeled "infrastructure.database.connection_pool" — this might be the upstream symptom, not the root cause, but it's arguable.

2. **Taxonomy expansion.** When crawled incidents from VOID [VOID] don't fit existing taxonomy categories, a human must decide whether to create a new category or reclassify under an existing one.

3. **Reward calibration.** Periodically compare the automated reward scores against human expert judgments to detect drift. Does a score of 0.7 (correct subcategory) feel right to an experienced SRE, or should it be higher/lower?

**How to minimise cost (active learning):**

The cost of human evaluation scales with the number of episodes reviewed. To minimise it:

1. **LLM-as-judge for routine scoring.** Use an LLM to evaluate free-text diagnoses against gold-standard labels semantically. "The database ran out of connections" should match `infrastructure.database.connection_pool` even without the exact label. This handles ~90% of episodes without human intervention.

2. **Active learning for the remaining 10%.** Only request human review when the LLM-as-judge confidence is low — measured by disagreement between multiple judge prompts or when the similarity score falls in the ambiguous range (0.3–0.6). This focuses human attention on genuinely uncertain cases.

3. **Batch evaluation with feedback loops.** Collect uncertain episodes during a training run, present them in batches to human reviewers (5-10 episodes at a time), and use the feedback to update gold-standard labels and taxonomy rules. Each human review improves future automated scoring.

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

The taxonomy has 35 leaf failure types. The MVP covers 10 of them (29%) across 5 hand-authored scenarios. The question is how to scale from 29% to 80%+ without hand-authoring hundreds of YAML files.

### Question 1: From Indexed Incidents to Reproducible Test Cases

**The implemented pipeline (3 stages):**

```
Stage 1: Crawl              Stage 2: Generate           Stage 3: Play
─────────────               ─────────────               ──────────────

VOID (~10K incidents)──┐
Aiops-Dataset (labeled)┤    ScenarioGenerator           SREEnvironment
GCP/Cloudflare feeds ──┤──► .generate(incident) ──► scenario.yaml ──► episode
GitHub post-mortems  ──┘    │                           │
                            ├── pick topology           ├── perturbation per seed
                            ├── build timeline          ├── telemetry generation
                            ├── estimate difficulty     └── reward computation
                            └── set gold standard
```

**Stage 1: Crawl and normalise.** Five crawlers (`GCPIncidentCrawler`, `CloudflareIncidentCrawler`, `GitHubPostmortemCrawler`, `VOIDIncidentCrawler`, `load_aiops_groundtruth`) fetch real incidents and normalise them into `NormalisedIncident` objects with: title, summary, timeline, root causes, affected services, taxonomy labels, quality score. Stored in SQLite.

**Stage 2: Generate scenario YAML.** The `ScenarioGenerator` converts each `NormalisedIncident` into a playable scenario:

- **Topology selection.** Maps the incident's taxonomy label to a service topology template. `infrastructure.database.*` incidents get a web-server → application → database topology. `application.*` incidents get a topology with cache and queue services. 5 topology templates cover the major categories.

- **Timeline construction.** Maps taxonomy labels to event timeline templates. A `connection_exhaustion` incident gets: `normal_traffic → traffic_ramp → connection_exhaustion → error_spike → alert`. 8 event templates cover the most common failure patterns. Incidents without a matching template get a generic `error_spike → latency_spike → alert` timeline.

- **Gold standard.** Root causes come from the incident's taxonomy labels with relevance weights. Remediations come from the incident's remediation field (if populated) or a generic fallback.

- **Difficulty estimation.** Derived from incident severity: critical → 0.6, major → 0.5, minor → 0.3.

**Stage 3: Play in the RL environment.** The generated YAML is loaded by `ScenarioLoader`, perturbed per seed (±15% timing, ±20% magnitudes), and run through the `SREEnvironment`. The `IncidentReplaySource` handles the full pipeline:

```python
source = IncidentReplaySource()
source.load_from_db("data/incidents.db")       # VOID + Aiops-Dataset
source.load_builtin_scenarios()                 # 5 hand-authored
scenarios = source.get_all()                    # sorted by difficulty

runner = EpisodeRunner(scenarios=scenarios)
results = runner.run_batch(10_000)
```

### Question 2: Collecting a Corpus for 80% Coverage

**Current state:** 10 of 35 taxonomy leaves covered (29%). Target: 28 of 35 (80%).

**The 25 uncovered leaves and how to reach them:**

| Gap Category | Uncovered Leaves | Source to Fill |
|---|---|---|
| **Infrastructure compute** | `cpu_saturation`, `oom`, `instance_failure`, `container_crash` | Aiops-Dataset has labeled CPU/memory/pod faults. LitmusChaos has `pod-cpu-hog`, `pod-memory-hog`, `container-kill` experiments [LitmusChaos]. |
| **Infrastructure database** | `replication_lag`, `lock_contention`, `split_brain` | VOID database incidents from companies running PostgreSQL, MySQL, MongoDB. Chaos engineering literature describes split-brain injection [Chaos Mesh]. |
| **Infrastructure storage** | `iops_throttling`, `data_corruption` | Aiops-Dataset disk fault scenarios. AWS FIS provides EBS throttling injection. |
| **Infrastructure network** | `partition`, `load_balancer`, `tls_certificate` | LitmusChaos has `pod-network-partition`, `pod-network-loss`. VOID contains TLS certificate expiry incidents (Let's Encrypt, Cloudflare). |
| **Application** | `deadlock`, `thread_pool_exhaustion`, `cache_stampede`, `poison_pill`, `hotspot`, `api_breaking_change` | Aiops-Dataset has concurrency and dependency faults. VOID contains cache stampede incidents (Facebook Memcached, 2010). |
| **Operational** | `bad_deploy`, `rollback_failure`, `feature_flag`, `quota_exhaustion`, `queue_backlog` | VOID is rich in deployment failures. GitHub post-mortems contain feature flag incidents (Knight Capital, GitHub). |
| **External** | `cloud_provider`, `dns_provider`, `cdn` | GCP incident feed contains cloud provider outages. VOID has DNS provider incidents (Dyn, 2016). |

**The plan (4 tiers):**

1. **Tier 1: Automated from existing data (~15 leaves).** Run the crawler pipeline against VOID + Aiops-Dataset. The `ScenarioGenerator` auto-classifies incidents against the taxonomy and generates scenario YAMLs. Expected yield: ~15 new taxonomy leaves covered from the ~10K VOID incidents alone.

2. **Tier 2: LitmusChaos mapping (~5 leaves).** Map LitmusChaos ChaosHub experiments [LitmusChaos] to taxonomy nodes. Each experiment (pod-cpu-hog, pod-network-loss, node-drain, etc.) defines a fault injection specification that maps directly to our event timeline format. 50+ experiments cover most infrastructure failure types.

3. **Tier 3: LLM-assisted gap filling (~5 leaves).** For leaves with <5 real incidents (e.g., `split_brain`, `data_corruption`), prompt an LLM with: (a) the taxonomy node description, (b) 2-3 real incidents from adjacent nodes, (c) the scenario YAML schema. The LLM generates plausible scenario templates that a human SRE reviews before inclusion.

4. **Tier 4: Community contribution (~remaining).** Publish the taxonomy and scenario format as open-source. Accept community-contributed scenarios — following the LitmusChaos ChaosHub model where practitioners contribute experiments from their own production experience.

**Projected coverage:**

| Stage | Leaves Covered | Coverage |
|---|---|---|
| MVP (hand-authored) | 10 / 35 | 29% |
| + Tier 1 (VOID/Aiops auto-generation) | 25 / 35 | 71% |
| + Tier 2 (LitmusChaos mapping) | 30 / 35 | 86% |
| + Tier 3 (LLM gap-filling) | 33 / 35 | 94% |

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

## RL Loop Closure: How the Agent Improves

```
1. Sample scenario from training set (weighted by coverage gaps + agent weakness)
2. Instantiate environment (L1: generate synthetic telemetry)
3. Agent receives initial observation (alert + system topology)
4. Agent takes actions (query metrics, search logs, inspect traces)
5. Environment returns observations (telemetry for queried services/time ranges)
6. Agent proposes root cause + remediation
7. Evaluation engine scores against gold standard → reward
8. Policy update (GRPO/PPO on the trajectory)
9. Log episode results → update coverage matrix + agent performance by taxonomy node
10. Curriculum learning: increase difficulty over time (more services, more subtle failures,
    compound scenarios)
```

The agent improves against unseen scenarios through:
- **Generalisation from parameterised training**: Seeing many variations of "connection pool exhaustion" with different topologies and parameters teaches the pattern, not the specific instance.
- **Curriculum learning**: Starting with simple single-failure scenarios and progressing to compound failures.
- **Adversarial generation**: LLM-generated scenarios that probe the agent's weaknesses (identified from the coverage matrix).

---

## Mapping to open-sre-agent Pipeline

The environment is designed to mirror the open-sre-agent's existing LangGraph investigation pipeline. Each environment action corresponds to a stage or tool in the production agent:

| RL Environment Action | open-sre-agent Pipeline Stage | AgentState Fields |
|----------------------|------------------------------|-------------------|
| Initial alert observation | `extract_alert` node | `alert_data`, `alert_name`, `severity` |
| `LIST_SERVICES` | `resolve_integrations` node | `integrations`, `evidence_sources` |
| `QUERY_METRICS` / `QUERY_LOGS` / `QUERY_TRACES` | `investigate` node (tool actions) | `evidence`, `executed_hypotheses` |
| `LIST_ALERTS` | Alert context from Slack/PagerDuty | `slack_context`, `alert_data` |
| `DIAGNOSE` | `root_cause_diagnosis` node | `root_cause`, `root_cause_category`, `validated_claims` |
| `REMEDIATE` | `publish_findings` node | `remediation_steps` |
| Episode loop (investigate → diagnose → repeat) | `route_investigation_loop` (max 5 iterations) | `total_loops`, `investigation_recommendations` |

This mapping means an RL-trained policy can be transferred to the production agent by replacing the LLM's prompt-driven decision-making in `plan_actions` and `root_cause_diagnosis` with the learned policy — the tool interface is the same.

---

## MVP Scope vs. Deferred

### MVP (Part 3 Implementation)
- **Synthetic telemetry generator** for Layer 1 simulation — correlated metrics, logs, and traces that tell a consistent causal story
- **Gymnasium RL environment** with defined observation/action/reward spaces
- **5 scenario templates** covering distinct failure classes (disk full, connection pool, memory leak, cascading failure, DNS failure)
- **Scenario perturbation** — each seed produces ±15% timing jitter and ±20% magnitude variation, preventing memorisation
- **Evaluation harness** with hierarchical reward scoring (4 components: diagnosis × efficiency, remediation, safety)
- **Three baseline agents** (random, heuristic, oracle) demonstrating the reward signal discriminates correctly
- **Incident data crawler** for GCP, Cloudflare, and GitHub post-mortems (Pillar 1)
- **Crawler→scenario pipeline** that converts crawled incidents into playable YAML scenarios (Pillar 5)
- **Tool adapter** (`SREToolAdapter`) exposing the environment as callable tools for LLM agents, compatible with Claude tool_use and the open-sre-agent tool interface
- **Training loop** (`EpisodeRunner`) with scenario sampling, curriculum support via difficulty filtering, and trajectory collection
- **Docker Compose** setup with demo, crawler, tests, and training services
- **149 passing tests** across all components

### Deferred
- Layer 2/3 simulation (real services, Kubernetes, Chaos Mesh)
- LLM-as-judge evaluation (requires LLM API integration)
- Milestone rewards and counterfactual evaluation (designed above, not implemented)
- Automatic curriculum progression based on agent performance (difficulty filtering is supported but not auto-adaptive)
- Full LangGraph integration with open-sre-agent's AgentState TypedDict (tool interface is compatible; wiring is deferred)
- Distributed training infrastructure
