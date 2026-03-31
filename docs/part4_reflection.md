# Part 4: Results & Reflection

## What I Built

### Architecture Decisions

**1. Real-data-driven synthetic telemetry — not pure synthetic, not pure real.**

The central insight: RL training needs 100K+ episodes, but real infrastructure costs $5/episode (11 years and $500K for a training run). Pure synthetic telemetry is cheap but unrealistic. The solution: use real incident data from crawled incidents (~10K incidents), Aiops-Dataset (labeled microservice faults), and public status pages to *drive* the synthetic generation. The scenarios come from real failures; only the rendering is synthetic.

This gives us the speed of synthetic generation (~120K episodes/hour) with the pattern diversity of real production incidents.

**2. The reward signal is the highest-risk component — that's what I targeted.**

For coding agents, the reward is workable (tests pass or fail). For distributed systems, "correct" is ambiguous, delayed, and context-dependent. If the reward signal doesn't work, no amount of environment fidelity matters. Three specific design decisions made it work:

- **Hierarchical partial credit** (0.0 → 0.4 → 0.7 → 1.0 by taxonomy depth) provides learning signal for partially correct diagnoses, instead of binary right/wrong.
- **Efficiency × diagnosis coupling** prevents the broken strategy where the agent diagnoses immediately without investigating (fast + wrong = 0 efficiency).
- **Multi-label gold standard** with relevance weights handles incidents with multiple valid root causes (which the incident data shows is ~75% of real incidents).

**3. Tool-based investigation, not flat text.**

The RL environment exposes the same tool interface as opensre's production pipeline: `query_metrics`, `query_logs`, `query_traces`, `list_services`, `list_alerts`, `diagnose`, `remediate`. The agent must *choose* what to query, not receive all data at once. This trains investigation *strategy* — the skill that actually matters for reducing MTTR.

### MVP Implementation

| Component | What It Does | Key Metric |
|-----------|-------------|------------|
| **4 incident crawlers** | GCP, Cloudflare, GitHub post-mortems + Aiops-Dataset CSV → SQLite | ~10K incidents accessible |
| **Scenario generator** | Converts real incidents → playable YAML scenarios | 15/35 taxonomy leaves (43%) covered |
| **Telemetry generator** | Event-correlated metrics + logs + traces | 38ms/episode, 2.5MB memory |
| **RL environment** | Gymnasium API with 7 actions, tool-based observation space | 120K episodes/hour |
| **Reward engine** | 4-component score with hierarchical partial credit | 2.7× gap between oracle and random |
| **Scenario perturbation** | ±15% timing, ±20% magnitudes per seed | Same scenario, infinite variations |
| **opensre integration** | SimulatedAction matching InvestigationAction interface | 5/5 actions pass through real execute_actions |
| **Training loop** | EpisodeRunner with trajectory collection + JSONL/DPO export | Curriculum support via difficulty filtering |
| **3 baseline agents** | Random, heuristic (no known correct answers), oracle | Validates reward discrimination |
| **169 tests** | Unit + integration + end-to-end | 3.2 seconds, all passing |

### Baseline Agent Results (Measured)

| Agent | Avg Reward | Diagnosis | Efficiency | Remediation | Safety |
|-------|-----------|-----------|------------|-------------|--------|
| Random | 0.321 | 0.244 | 0.244 | 0.100 | 1.000 |
| Heuristic (no known correct answers) | 0.464 | 0.160 | 0.077 | 0.940 | 1.000 |
| Oracle (known correct answers) | 0.875 | 0.960 | 0.480 | 0.980 | 1.000 |

Key observations:
- **The reward discriminates clearly.** Oracle (0.875) >> heuristic (0.464) >> random (0.321). The 2.7× gap is learnable — there's room for an RL agent to improve.
- **Diagnosis is the bottleneck.** The heuristic agent scores 0.160 on diagnosis (picks alphabetically first option) vs 0.960 for oracle. This is the skill that benefits most from training.
- **Fast + wrong = low reward.** The random agent takes few steps but gets wrong answers. After the `efficiency × diagnosis` fix, its efficiency score dropped from 1.0 to 0.244 — the reward correctly penalises speed without accuracy.
- **Safety is easy to satisfy.** All agents score 1.0. The safety penalties catch truly broken behaviour (diagnose on step 1), not normal investigation patterns.

### opensre End-to-End Integration (Verified)

opensre's LLM successfully investigated a simulated "Disk Full" scenario through the full LangGraph pipeline:

```
  ● Planning       Planned: ['query_grafana_alert_rules', 'query_grafana_metrics']
  ● Gathering      grafana:1 alert rules, grafana:1 metric series
  ● Diagnosing     confidence 62% — needs more evidence
  ● Planning       Planned: ['query_grafana_service_names', 'query_grafana_logs']
  ● Gathering      grafana:1 services, grafana:1 logs (13 errors)
  ● Diagnosing     confidence 100% — found root cause

  Root cause: "high disk usage on postgres-primary caused cascading failures"
  Validity: 100%, Investigation loops: 3
```

The integration builds a custom LangGraph graph with simulated `plan_actions` and `investigate` nodes. opensre's LLM autonomously decided what to query, gathered synthetic evidence, and correctly identified the root cause. Action names match opensre's `EVIDENCE_MAPPERS` (`query_grafana_*`) so evidence flows correctly through the processing pipeline.

### Trajectory Export

```bash
make export  # → training_data/trajectories.jsonl
```

Each line contains: initial alert (prompt), all observations, all tool calls, and the final reward — ready for GRPO/DPO fine-tuning.

### Key Metrics (Measured)

- **Throughput**: 120,000 episodes/hour (single core, random agent). With LLM: ~1,800 eps/hr.
- **Per episode**: 38ms generation, 2.5MB memory, deterministic per seed.
- **Taxonomy coverage**: 15/35 leaves (43%) from builtins + Aiops-Dataset.
- **opensre integration**: 5/5 `execute_actions` calls pass against real opensre codebase.

See Part 3 for detailed compute estimates and reasoning.

## What I Didn't Build

### Intentionally Deferred

**RL policy update.** The training loop collects trajectories and exports them as JSONL/preference pairs, but does not run GRPO/PPO to actually update model weights. This requires a GPU cluster and a training framework (trl, DeepSpeed) — it's infrastructure, not research. The environment and reward signal are the research contributions; the optimiser is a known algorithm applied on top.

**LLM-as-judge evaluation.** The current reward uses exact taxonomy matching. An LLM judge would evaluate free-text diagnoses semantically ("the database ran out of connections" should match `infrastructure.database.connection_pool`). Deferred because exact matching is sufficient to validate the architecture, and LLM-as-judge adds API cost and latency.

**Real infrastructure simulation (Layers 2-3).** AIOpsLab (Microsoft, MLSys'25) provides the reference implementation for Layer 2. LitmusChaos ChaosHub provides 50+ fault experiments for Layer 3. These are engineering integrations, not research problems. The architecture is designed for them; the wiring is deferred.

**Automatic curriculum progression.** `ScenarioLoader.load_all(max_difficulty=...)` supports difficulty-based filtering, but the training loop doesn't auto-adapt difficulty based on agent performance. This is a training pipeline optimisation that depends on having a working policy update loop first.

### Harder Than Expected

**Telemetry correlation.** Making metrics, logs, and traces tell a consistent story is the core fidelity challenge. Early versions had a connection pool exhaustion scenario showing "Deadlock detected" and "Disk space critically low" in the logs — because log templates were sampled randomly from all ERROR templates, not correlated with the active timeline events.

The fix was fundamentally restructuring the log generator: `EVENT_LOG_TEMPLATES` maps each event type (connection_exhaustion, error_spike, cascade, etc.) to specific log templates. During generation, only templates matching the active events are selected. Additionally, placeholder values (`{active}`, `{max}`, `{duration}`, `{status}`) are now event-correlated — `{active}` shows 85-100 during connection exhaustion, not random 10-100.

**Reward function interaction effects.** The initial design had `ideal_steps = 3` for efficiency and `min_queries_before_diagnosis = 2` for safety. These conflicted: 2 queries + 1 diagnose + 1 remediate = 4 steps minimum for safety, but efficiency rewarded ≤3 steps. The agent couldn't satisfy both. Fix: raised `ideal_steps` to 5 and coupled efficiency to diagnosis accuracy (`efficiency × diagnosis`), so speed is only rewarded when the diagnosis is correct.

**opensre interface contracts.** The initial integration code had 6 breaking issues against opensre's actual codebase:
- `parameter_extractor` was `None` (opensre's `execute_actions` rejects this at line 47)
- `resolved_integrations` had `{"available": True}` instead of `{"endpoint": "...", "api_key": "..."}`
- Action return format was `{output, step, done}` instead of `{source, available, data}`

These were only discovered by reading opensre's actual source code (`execute_actions.py`, `state.py`, `detect_sources.py`) — the documentation didn't specify these contracts. The lesson: integration testing against the real codebase is essential, not just interface matching by reading docs.

### What I'd Build Next (One More Week)

1. **Expand data sources beyond 51% taxonomy coverage.** Add more crawlers (VOID when API available, more statuspage providers) and improve keyword classification patterns. The pipeline exists (`ScenarioGenerator.batch_generate()`) — more data = more scenarios with zero code changes.

2. **Reward variance measurement.** Run 1,000 episodes per scenario with different seeds and measure reward variance. If `std/mean > 0.3`, the reward signal is too noisy for stable RL training. This is the most important validation missing.

3. **LLM-as-judge for free-text diagnosis.** Replace exact taxonomy matching with semantic evaluation. The agent should be able to diagnose in natural language, not just pick from a fixed list.

4. **AIOpsLab Layer 2 integration.** Deploy DeathStarBench via AIOpsLab, run the trained agent against real Prometheus/Jaeger/Filebeat telemetry, and measure the synthetic-to-real transfer gap. This is the biggest risk validation.

5. **Compound failure scenarios.** Create scenarios with 2+ simultaneous failures (memory leak + traffic spike + DNS blip). These are the incidents that separate competent SRE agents from expert ones.

6. **MLflow experiment tracking.** The training pipeline currently prints results to stdout. With multiple experiments (different reward weights, agent types, data sources), comparing runs becomes unwieldy. MLflow would add: experiment comparison dashboards, metric history across runs, model registry for trained checkpoints, artifact storage for trajectory JSONL files. Integration is straightforward — `mlflow.log_params()` + `mlflow.log_metrics()` in `python -m app.main train` — but adds a dependency and server process that isn't justified until there are 10+ experiments to compare.

## Open Questions

### Biggest Unsolved Problems

**1. Sim-to-real transfer.** The agent trains on synthetic telemetry with metric names like `cpu_percent` and `latency_p99_ms`. Real Grafana dashboards use `node_cpu_seconds_total` and `http_request_duration_seconds_bucket`. Real logs are 10x noisier with irrelevant entries. Will investigation skills trained on clean synthetic data transfer to messy production telemetry? This is the biggest risk. The fastest validation would be: train on synthetic, evaluate on Aiops-Dataset's real log/metric/trace data, measure the performance gap.

**2. Reward for novel root causes.** The current reward requires gold-standard labels. But real incidents have novel root causes the scenario designer didn't anticipate. An agent that discovers a valid root cause not in the gold standard currently gets 0. Options: (a) LLM-as-judge to evaluate novel diagnoses semantically, (b) a "confidence calibration" reward where the agent is scored on how well its confidence matches its accuracy, (c) accept the limitation and rely on taxonomy coverage breadth to approximate.

**3. Does RL training actually improve open-sre-agent?** The environment and reward signal are necessary but not sufficient. The final test is: does an RL-tuned opensre agent diagnose faster and more accurately than the current prompt-engineered version? This requires: (a) a benchmark of 50+ real incidents with known root causes, (b) running both agents, (c) comparing MTTR and diagnosis accuracy. This is a product validation, not a research question.

### Where I'd Challenge My Own Assumptions

**"80% of SRE work is investigation."** This assumption justifies training on synthetic telemetry (investigation doesn't need real infrastructure). But if remediation skill matters more than investigation skill — if the bottleneck is knowing *which button to press*, not *which dashboard to read* — then the architecture should invest more in Layers 2-3 (real infrastructure for remediation training) and less in Layer 1 (synthetic for investigation).

**"Hierarchical taxonomy is the right abstraction."** Real incidents are messier than clean taxonomy trees. A "database connection pool exhaustion caused by a traffic spike triggered by a bad deploy" doesn't map cleanly to one taxonomy leaf. An alternative: learn a continuous failure embedding space where the taxonomy provides initialisation but the model can discover new clusters. This would require abandoning the discrete diagnosis action in favour of a continuous representation.

**"More scenarios = better agent."** Scaling from 5 to 500 scenarios might not improve the agent if the 5 existing scenarios already cover the core investigation patterns (check alerts → query affected service → query dependencies → diagnose). Diminishing returns may set in early. The counter-argument: each scenario teaches a different telemetry *pattern* (what connection pool exhaustion looks like vs what a memory leak looks like), and pattern diversity is essential for generalisation.

### What Would Change the Architecture Entirely

**Access to real production telemetry at scale.** If Tracer customers consented to anonymised incident replay — real metrics, logs, and traces from real incidents — we could skip synthetic generation entirely. The architecture would become: replay and perturb real telemetry, with the reward signal calibrated against the actual resolution that worked.

**Proof that LLM agents don't need RL.** If few-shot prompting with retrieval over a large enough incident knowledge base (e.g., 10K VOID incidents as context) achieves comparable investigation quality to RL-trained agents, the entire training environment becomes unnecessary. The investment should shift to building the retrieval system and the knowledge base.

**Breakthrough in long-horizon credit assignment.** The current design uses end-of-episode end-of-episode reward because intermediate rewards for SRE investigation are hard to define correctly. If credit assignment over 50+ step trajectories became reliable (e.g., through world models or hindsight experience replay), we could train on richer, longer investigation episodes with step-by-step rewards at each step.

### Questions I'd Ask Tracer

1. **How does opensre currently measure investigation quality?** The `validity_score` (ratio of validated to non-validated claims) is a proxy — is it reliable enough for RL, or too noisy?
2. **Do you have labeled incident data from customer investigations?** Even 100 real opensre sessions with human ratings would be more valuable for reward calibration than 10K synthetic episodes.
3. **Which opensre node should RL improve first — `plan_actions` or `root_cause_diagnosis`?** They require different training data formats.
4. **What LLM does opensre use in production?** Open-weight (Llama) enables GRPO fine-tuning; API-only (Claude/GPT-4) limits training to prompt optimisation.
5. **Is the `investigation_loop_count` cap of 5 a hard product constraint?** The RL agent might benefit from 10 loops on complex incidents.
