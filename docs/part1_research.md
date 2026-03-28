# Part 1: Research & Constraints

## 1A. RL Environments for SWE Agents

### The RL Training Loop for Coding Agents

Modern software engineering agents are trained through reinforcement learning loops that follow a standard Markov Decision Process (MDP) structure. The architecture has converged across leading systems [1, 2, 3]:

**Environment.** A sandboxed Docker container containing a repository snapshot checked out at a specific commit. The container includes all dependencies, build tools, and test suites pre-installed. Each training episode corresponds to one task instance — typically a real GitHub issue paired with the repository state before the fix was merged [1].

**Observations.** The agent observes: (1) the issue description or bug report, (2) current file contents it has opened, (3) terminal output from commands it has executed, and (4) its conversation history. The observation space is text — a sequence of tokens representing the workspace state.

**Actions.** The agent selects from a discrete set of tool invocations: execute a bash command, search files, read a file, edit a file, or submit a patch. Each action transforms the environment state. This tool-based action space is shared across SWE-bench [1], Codex [4], and Claude Code.

**Reward Signal.** The reward is typically binary and sparse — delivered only at the end of an episode. The submitted patch is applied, then the "fail-to-pass" test suite is executed. If all previously-failing tests pass without regressions, reward is 1; otherwise 0. SWE-RL introduced a continuous reward variant based on patch similarity (difflib.SequenceMatcher) to provide gradient signal for partial solutions [2].

**Training loop.** The procedure is: (1) sample a task instance, (2) spin up a containerised environment, (3) run the agent policy for up to N steps or a time budget, (4) collect the trajectory, (5) compute reward via test execution, (6) update the policy. DeepSWE uses GRPO (Group Relative Policy Optimisation) across 4,500+ tasks on 64 H100 GPUs for 6 days [3].

### Infrastructure at Scale

Training requires a decoupled architecture: GPU clusters for LLM inference and CPU clusters for sandbox execution (1,000+ cores running isolated Docker containers). These communicate asynchronously — agentic workloads are I/O-bound (waiting for sandbox execution), not GPU-bound. Kubernetes orchestrates container lifecycle, with Docker images built in layers to minimise provisioning time. The ARES evaluation framework can evaluate all of SWE-bench Verified (~500 instances) in approximately 20 minutes using this architecture [5].

### Test Case Generation and Curation

**SWE-bench** [1] (the gold standard) mines real GitHub PRs: snapshot the repository at the pre-PR commit, extract the linked issue, and use the "fail-to-pass" tests as the evaluation oracle. SWE-bench Verified added human validation by 93 developers to remove ambiguous tasks [6].

**Self-play** (Meta's SSR, 2025) eliminates the data bottleneck: one LLM injects bugs, another fixes them, generating unlimited training pairs from any codebase [7]. **R2E-Gym** procedurally generates 8,100+ tasks from commits without requiring human-written PRs [8].

For distributed systems, equivalent datasets are emerging. The VOID database [14] contains ~10,000 real incidents from ~590 organisations. The Aiops-Dataset [22] provides labeled fault scenarios (log/metric/trace triplets) from a 46-instance microservice system with ground-truth root causes. LogHub [19] offers 300M+ log lines from HDFS, BGL, Thunderbird, and OpenStack with anomaly labels. Microsoft's AIOpsLab [18] deploys real microservices with fault injection and telemetry export, enabling benchmarking of autonomous AIOps agents. LitmusChaos ChaosHub [21] catalogs 50+ fault experiments for Kubernetes. These resources collectively address the data scarcity problem for SRE agent training.

### Why the Reward Signal is Tractable — and Why It Breaks

For coding agents, the reward signal works because of five properties:

1. **Deterministic verification**: same code + same tests = same pass/fail result, every time.
2. **Cheap binary oracle**: run `pytest`, check exit code. Cost is CPU-seconds.
3. **Automated correctness checkers**: unit tests, compilers, linters — decades of tooling.
4. **Fast feedback**: a test suite runs in seconds to minutes.
5. **Infinite data generation**: self-play can produce unlimited bug/fix pairs [7].

Every property above is violated in distributed production systems:

| Property | Coding Agents | Distributed SRE |
|----------|--------------|-----------------|
| Determinism | Same input → same output | Race conditions, network timing, clock skew make outcomes non-deterministic |
| Verification cost | CPU-seconds (run tests) | Minutes to hours (deploy, observe, wait for cascading effects) |
| Automated oracles | Tests, compilers, linters | No equivalent — "is this system healthy?" requires judgment |
| Feedback speed | Seconds | Minutes to days (cascading failures unfold over time) |
| Data generation | Self-play on code repos | Requires expensive infrastructure simulation |

The fundamental asymmetry: coding has cheap, fast, deterministic verification. Distributed systems have expensive, slow, non-deterministic verification. RL needs thousands of reward signals per training step. When each signal takes minutes instead of seconds and has 5–10% noise, the training loop becomes impractical without architectural innovation.

**Connection to open-sre-agent.** Tracer's open-sre-agent [9] already follows an agentic pattern structurally similar to SWE-bench agents — it has a tool-based action space (query Grafana, search Datadog, inspect EKS), a multi-step investigation loop (plan → investigate → diagnose, up to 5 iterations), and a claim-validation mechanism that could serve as a reward signal proxy. The gap is that there is no training loop — the agent relies entirely on pre-trained LLM capabilities plus prompt engineering. Building an RL environment that mirrors the open-sre-agent's tool interface would enable fine-tuning the investigation policy without changing the production architecture.

---

## 1B. Constraints for Distributed SRE

### 1. Feedback Latency

**The constraint.** In coding, feedback is immediate — run the test, get the result in seconds. In distributed systems, the effects of a failure (or a remediation action) may take minutes to hours to fully manifest. Cloudflare's 2019 regex outage took 27 minutes from deployment to global impact because CPU consumption grew gradually [10]. GitLab's 2017 database incident unfolded over 18 hours as operators attempted multiple recovery strategies, each with delayed feedback [11].

**Why it changes RL design.** Standard RL assumes episodes that complete in reasonable time. When a single episode takes 30+ minutes of simulated time, training throughput drops by orders of magnitude.

**Architectural implication.** The environment must support time compression — either through synthetic telemetry replay (skip to the interesting moments) or multi-fidelity simulation where fast, low-fidelity environments handle the bulk of training and slow, high-fidelity environments validate critical scenarios.

### 2. State Fragmentation

**The constraint.** A coding agent sees a unified workspace — files, terminal output, test results, all in one place. An SRE agent must correlate signals scattered across dozens of services: Prometheus metrics, Grafana dashboards, application logs (structured and unstructured), distributed traces (Jaeger/Zipkin), Kubernetes events, cloud provider health status, deployment manifests, and alert histories [12]. No single view gives the full picture.

**Why it changes RL design.** The observation space is heterogeneous — time-series metrics, log streams, trace graphs, Kubernetes events. The agent must learn to query the right sources; information retrieval itself is an action.

**Architectural implication.** The environment must expose a realistic tool interface (not a flat observation) — the agent queries metrics APIs, searches logs, inspects traces. This mirrors how a real SRE works with Grafana, Datadog, and kubectl, and aligns directly with open-sre-agent's tool-based architecture [9].

### 3. Non-Determinism

**The constraint.** The same deployment can succeed or fail depending on current load, network conditions, JVM warmup state, connection pool saturation, and timing. Knight Capital's 2012 trading loss ($440M in 45 minutes) was triggered by a deployment that interacted with old code on one server — a sequence of events that would likely not reproduce under different timing [13]. A cascading failure triggered by a specific interleaving of events may not reproduce when replayed.

**Why it changes RL design.** The reward signal becomes noisy — the agent cannot learn stable state-action values because the transition function is stochastic. You cannot simply "replay" a real incident; you must model the distribution of outcomes.

**Architectural implication.** The environment needs controlled stochasticity — parameterised noise injection for robust training, with fixed random seeds for reproducible evaluation. Scenario definitions must specify distributions, not point values (e.g., "latency increases by 50–200ms" not "latency increases by 100ms"). This is implemented in our MVP via the perturbation module, which jitters event timing (±15%) and effect magnitudes (±20%) per seed.

### 4. Ambiguous Correctness

**The constraint.** A coding task has a clear oracle: tests pass or they don't. An incident often has multiple valid root causes and multiple valid remediation paths. Was the outage caused by the bad deploy, or by the database connection pool that was already at 95% capacity? Both are defensible root causes. Analysis of the VOID database (~10,000 incidents from 600+ organisations) found that only approximately 25% of incidents identified a single definitive root cause — the majority involved contributing factors that are difficult to rank [14].

**Why it changes RL design.** A binary reward cannot capture the richness of incident resolution. You need a multi-dimensional reward scoring root cause accuracy, investigation efficiency, remediation quality, and safety. Each dimension may have multiple valid answers.

**Architectural implication.** Reward design must use a hierarchical scoring model with partial credit: coarse credit for the right service layer, finer credit for the exact component. Gold-standard diagnoses should include all acceptable root causes ranked by relevance. LLM-as-judge can scale human judgment at reduced cost [15]. Our MVP implements this via hierarchical taxonomy matching with configurable depth scores.

### 5. Environment Cost

**The constraint.** A Docker container for a coding task costs pennies and provisions in seconds. Simulating a production-fidelity distributed system costs dollars per episode and takes minutes to provision. Running 100,000 episodes (a modest RL training run) with full Docker Compose infrastructure would cost approximately $10,000 and take weeks.

**Why it changes RL design.** Full-fidelity training at scale is prohibitive. But low-fidelity risks teaching behaviours that don't transfer to production.

**Architectural implication.** A multi-layer simulation strategy, following the fidelity spectrum proposed by chaos engineering literature [16]: Layer 1 (synthetic telemetry, ~$0.001/episode, 1,000+ episodes/hour) for bulk RL training on investigation strategy; Layer 2 (Docker Compose with fault injection or Microsoft AIOpsLab [18], ~$0.10/episode) for validating that learned policies transfer to real services; Layer 3 (Kubernetes + Chaos Mesh [17] + LitmusChaos [21], ~$5/episode) for final evaluation against production-fidelity failure modes. RCAEval [20] and MicroServo [23] provide standardised benchmarks for comparing agent performance across these layers.

### 6. Safety

**The constraint.** An SRE agent that takes remediation actions in training must not cause real damage — but a purely simulated environment may not teach production-relevant safety constraints. In production, the cost of a wrong action (e.g., restarting a primary database during a failover) can be catastrophic. Conversely, an agent that never acts is useless.

**Why it changes RL design.** The agent must learn a safety policy alongside investigation. Restarting a service is riskier than reading a log; the environment must penalise high-risk actions taken without sufficient evidence.

**Architectural implication.** The action space includes safety metadata (risk level, reversibility). The reward function includes a safety penalty that penalises hasty diagnosis (diagnosing before querying sufficient telemetry) and tunnel vision (only investigating one service). The environment supports a "dry-run" mode where the agent proposes but does not execute remediation — matching open-sre-agent's production behaviour where remediation steps are suggested to the human operator, not executed autonomously [9].

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
