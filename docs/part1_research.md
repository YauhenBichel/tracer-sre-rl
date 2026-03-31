# Part 1: Research & Constraints

## 1A. RL Environments for SWE Agents

### The RL Training Loop for Coding Agents

The architecture has converged across SWE-bench [1], SWE-RL [2], DeepSWE [3], and Codex [4]. I studied these systems to understand which patterns transfer to SRE and which break.

**Environment.** A Docker container with a repository snapshot at a specific commit, pre-installed with dependencies and tests. Each episode = one GitHub issue. The key property: the environment is fully deterministic — same code + same tests = same result, every time. This is what makes RL workable for coding. For SRE, we don't have this: same deployment can fail or succeed depending on load and timing.

**Observations.** Text-based: issue description, file contents, terminal output, conversation history. The agent reads and writes files, runs commands, and sees results. This is directly analogous to SRE investigation — the agent reads metrics dashboards, log queries, and trace visualisations. Our environment mirrors this by returning text observations from tool queries (metrics tables, log entries, trace summaries).

**Actions.** Tool invocations: bash commands, file edits, searches, patch submission. The agent decides what to do, not what to output. This is the pattern we adopted — opensre's `plan_actions` node decides which tools to call (`query_grafana_metrics`, `query_grafana_logs`), and our simulated environment returns the results.

**Reward.** Binary and sparse: tests pass (1) or fail (0), delivered at episode end. SWE-RL [2] introduced continuous reward via patch similarity to provide learning signal for partial solutions. This is where SRE diverges most — there's no test suite to run. We address this with hierarchical partial credit (right category = 0.4, right subcategory = 0.7, exact match = 1.0) and multi-dimensional scoring (diagnosis, efficiency, remediation, safety).

**Training loop.** Sample task → spin up environment → agent acts → collect trajectory → compute reward → update policy. DeepSWE [3] ran this across 4,500 tasks on 64 H100s for 6 days using GRPO. Our loop is the same structure but faster: synthetic telemetry at 38ms/episode vs Docker sandboxes at seconds/episode.

### Infrastructure at Scale

Training requires separating LLM inference (GPU-bound) from environment execution (CPU-bound). DeepSWE [3] used 64 H100 GPUs for inference and separate CPU workers running Docker sandboxes across 4,500 tasks over 6 days. The bottleneck is LLM inference latency (~200ms per step), not sandbox execution (~50ms per episode). Environment throughput scales linearly with GPU count; cheaper CPU workers handle sandbox orchestration independently.

For SRE environments, the infrastructure is simpler: synthetic telemetry generation needs only CPU (~38ms per episode in our implementation), so the GPU cost is entirely LLM inference. A single A100 can serve ~1,800 episodes/hour when the agent uses 10 steps per episode at 200ms/step.

### Test Case Generation and Curation

**SWE-bench** [1] (the correct answers) mines real GitHub PRs: snapshot the repository at the pre-PR commit, extract the linked issue, and use the "fail-to-pass" tests as the evaluation oracle. SWE-bench Verified added human validation by 93 developers to remove ambiguous tasks [6].

**Self-play** (Meta's SSR, 2025) eliminates the data bottleneck: one LLM injects bugs, another fixes them, generating unlimited training pairs from any codebase [7]. **R2E-Gym** procedurally generates 8,100+ tasks from commits without requiring human-written PRs [8].

For distributed systems, equivalent datasets are emerging: VOID [14] (~10K incidents), Aiops-Dataset [22] (labeled microservice faults), LogHub [19] (300M+ log lines), AIOpsLab [18] (real microservice benchmarks), and LitmusChaos [21] (50+ K8s fault experiments). Our MVP uses Aiops-Dataset groundtruth (241 labeled faults, included in the repo) and 3 live crawlers (GCP, Cloudflare, GitHub).

### Why the Reward Signal Works — and Why It Breaks

For coding agents, the reward signal works because of five properties:

1. **Deterministic verification**: same code + same tests = same pass/fail result, every time.
2. **Cheap binary oracle**: run `pytest`, check exit code. Cost is CPU-seconds.
3. **Automated correctness checkers**: unit tests, compilers, linters — decades of tooling.
4. **Fast feedback**: a test suite runs in seconds to minutes.
5. **Infinite data generation**: self-play can produce unlimited bug/fix pairs [7].

Every property above is violated in distributed production systems:

| Property | Coding Agents | Distributed SRE |
|----------|--------------|-----------------|
| Determinism | Same input → same output | Race conditions, network timing, clock skew make outcomes unpredictable |
| Verification cost | CPU-seconds (run tests) | Minutes to hours (deploy, observe, wait for cascading effects) |
| Automated oracles | Tests, compilers, linters | No equivalent — "is this system healthy?" requires judgment |
| Feedback speed | Seconds | Minutes to days (cascading failures unfold over time) |
| Data generation | Self-play on code repos | Requires expensive infrastructure simulation |

The core problem: coding has cheap, fast, deterministic verification. Distributed systems have expensive, slow, unpredictable verification. RL needs thousands of reward signals per training step. When each signal takes minutes instead of seconds and has 5–10% noise, the training loop becomes impractical without architectural innovation.

**Connection to open-sre-agent.** Tracer's open-sre-agent [9] already follows an agentic pattern structurally similar to SWE-bench agents — it has a tool-based action space (query Grafana, search Datadog, inspect EKS), a multi-step investigation loop (plan → investigate → diagnose, up to 5 iterations), and a claim-validation mechanism that could serve as a reward signal proxy. The gap is that there is no training loop — the agent relies entirely on pre-trained LLM capabilities plus prompt engineering. Building an RL environment that mirrors the open-sre-agent's tool interface would enable fine-tuning the investigation policy without changing the production architecture.

---

## 1B. Constraints for Distributed SRE

### 1. Feedback Latency

**The constraint.** In coding, feedback is immediate — run the test, get the result in seconds. In distributed systems, the effects of a failure (or a remediation action) may take minutes to hours to fully manifest. Cloudflare's 2019 regex outage took 27 minutes from deployment to global impact because CPU consumption grew gradually [10]. GitLab's 2017 database incident unfolded over 18 hours as operators attempted multiple recovery strategies, each with delayed feedback [11].

**Why it changes RL design.** Standard RL assumes episodes that complete in reasonable time. When a single episode takes 30+ minutes of simulated time, training throughput drops by orders of magnitude.

**Architectural implication.** The environment must support time compression — either through synthetic telemetry replay (skip to the interesting moments) or multi-fidelity simulation where fast, low-fidelity environments handle the bulk of training and slow, high-fidelity environments validate critical scenarios.

### 2. State Fragmentation

**The constraint.** A coding agent sees a unified workspace — files, terminal output, test results, all in one place. An SRE agent must correlate signals scattered across dozens of services: Prometheus metrics, Grafana dashboards, application logs (structured and unstructured), distributed traces (Jaeger/Zipkin), Kubernetes events, cloud provider health status, deployment manifests, and alert histories [12]. No single view gives the full picture.

**Why it changes RL design.** The observation space is mixed — time-series metrics, log streams, trace graphs, Kubernetes events. The agent must learn to query the right sources; information retrieval itself is an action.

**Architectural implication.** The environment must expose a realistic tool interface (not a flat observation) — the agent queries metrics APIs, searches logs, inspects traces. This mirrors how a real SRE works with Grafana, Datadog, and kubectl, and aligns directly with open-sre-agent's tool-based architecture [9].

### 3. Non-Determinism

**The constraint.** The same deployment can succeed or fail depending on current load, network conditions, JVM warmup state, connection pool saturation, and timing. Knight Capital's 2012 trading loss ($440M in 45 minutes) was triggered by a deployment that interacted with old code on one server — a sequence of events that would likely not reproduce under different timing [13]. A cascading failure triggered by a specific interleaving of events may not reproduce when replayed.

**Why it changes RL design.** The reward signal becomes noisy — the agent cannot learn stable state-action values because the transition function is random. You cannot simply "replay" a real incident; you must model the distribution of outcomes.

**Architectural implication.** The environment needs controlled randomity — adding random variation for robust training, with fixed random seeds for reproducible evaluation. Scenario definitions must specify distributions, not point values (e.g., "latency increases by 50–200ms" not "latency increases by 100ms"). This is implemented in our MVP via the perturbation module, which jitters event timing (±15%) and effect magnitudes (±20%) per seed.

### 4. Ambiguous Correctness

**The constraint.** A coding task has a clear oracle: tests pass or they don't. An incident often has multiple valid root causes and multiple valid remediation paths. Was the outage caused by the bad deploy, or by the database connection pool that was already at 95% capacity? Both are defensible root causes. Analysis of the VOID database (~10,000 incidents from 600+ organisations) found that only approximately 25% of incidents identified a single definitive root cause — the majority involved contributing factors that are difficult to rank [14].

**Why it changes RL design.** A binary reward cannot capture the richness of incident resolution. You need a multi-dimensional reward scoring root cause accuracy, investigation efficiency, remediation quality, and safety. Each dimension may have multiple valid answers.

**Architectural implication.** Reward design must use a hierarchical scoring model with partial credit: coarse credit for the right service layer, finer credit for the exact component. Correct diagnoses should include all acceptable root causes ranked by relevance. LLM-as-judge can scale human judgment at reduced cost [15]. Our MVP implements this via hierarchical taxonomy matching with configurable depth scores.

### 5. Environment Cost

**The constraint.** A Docker container for a coding task costs pennies. Simulating a production-fidelity distributed system costs dollars per episode. 100,000 episodes at full fidelity = ~$10,000 and weeks of compute.

**Why it changes RL design.** Full-fidelity training at scale is prohibitive. This forces a multi-layer strategy: cheap synthetic telemetry (~$0.001/episode) for bulk training, real infrastructure (AIOpsLab [18], ~$0.10/episode) for transfer validation. See Part 2, Pillar 3.

### 6. Safety

**The constraint.** An SRE agent that takes remediation actions in training must not cause real damage. Restarting a primary database during a failover is catastrophic. But an agent that never acts is useless.

**Why it changes RL design.** The reward must penalise reckless actions (hasty diagnosis, tunnel vision) while rewarding decisive investigation. Our safety scorer penalises diagnosing before querying ≥2 sources and querying fewer than 2 services. The environment uses "dry-run" mode — matching opensre's production behaviour where remediations are proposed, not executed [9].

---

## References

[1] C. E. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?," *ICLR 2024*. https://arxiv.org/abs/2310.06770

[2] W. Wei et al., "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Engineering Tasks," *arXiv preprint*, 2025. https://arxiv.org/abs/2502.18449

[3] B. Gu et al., "DeepSWE: Scaling Automated Software Engineering with DeepSeek-R1-Distilled Models," *arXiv preprint*, 2025.

[4] OpenAI, "Codex," 2025. https://openai.com/index/codex/

[5] Martian, "ARES: Automated Research Engineering System." https://www.withmartian.com/blog/ares

[6] N. Chowdhery et al., "SWE-bench Verified: A Stricter Subset of SWE-bench with Human Validation," 2024.

[7] Meta, "Self-Synthesized Rehearsal (SSR) for Code Generation," 2025.

[8] J. Jain et al., "R2E-Gym: Procedural Task Generation for RL-Based Code Agents," 2025.

[9] Tracer Cloud, "open-sre-agent: Open-source AI agent for automated incident investigation." https://github.com/Tracer-Cloud/open-sre-agent

[10] Cloudflare, "Details of the Cloudflare outage on July 2, 2019." https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/

[11] GitLab, "Postmortem of database outage of January 31, 2017." https://about.gitlab.com/blog/2017/02/10/postmortem-of-database-outage-of-january-31/

[12] Charity Majors, Liz Fong-Jones, George Miranda, *Observability Engineering*, O'Reilly Media, 2022.

[13] SEC, "In the Matter of Knight Capital Americas LLC — Administrative Proceeding File No. 3-15570," 2013.

[14] Verica, "The VOID (Verica Open Incident Database)." https://www.thevoid.community/

[15] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *NeurIPS 2023*. https://arxiv.org/abs/2306.05685

[16] A. Basiri et al., "Chaos Engineering," *IEEE Software*, vol. 33, no. 3, pp. 35–41, 2016.

[17] Chaos Mesh, "A Powerful Chaos Engineering Platform for Kubernetes." https://chaos-mesh.org/

[18] Y. Chen et al., "AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds," *MLSys 2025*. https://arxiv.org/abs/2501.06706

[19] J. Zhu et al., "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics," *IEEE ISSRE 2023*. https://github.com/logpai/loghub

[20] P. Q. Luan et al., "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data," 2024. https://arxiv.org/abs/2412.17015

[21] LitmusChaos, "ChaosHub — Open-source Chaos Experiments for Kubernetes," CNCF. https://hub.litmuschaos.io

[22] bbyldebb, "Aiops-Dataset: Labeled Fault Scenarios from a Microservice E-Commerce System," 2022. https://github.com/bbyldebb/Aiops-Dataset

[23] D. Lu et al., "MicroServo: A Scenario-Oriented Benchmark for Assessing AIOps Algorithms in Microservice Management," 2024. https://arxiv.org/abs/2407.14532

[24] AIT, "Anomaly Detection Log Datasets — Analysis Scripts for Log Data Sets," 2023. https://github.com/ait-aecid/anomaly-detection-log-datasets

[25] mooselab, "DevOpsDataCollection — A Collection of DevOps Datasets for Research," 2024. https://github.com/mooselab/DevOpsDataCollection
