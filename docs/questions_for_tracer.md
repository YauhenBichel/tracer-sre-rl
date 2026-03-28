# Questions for Tracer

Questions I'd ask before or during a technical discussion about this architecture. Grouped by what they would change in the design.

## Reward Signal & Evaluation

1. **How does the open-sre-agent currently measure investigation quality?** The `validity_score` in AgentState (ratio of validated to non-validated claims) is a proxy — do you consider this a reliable reward signal, or is it too noisy for RL training?

2. **What does "correct" mean for your customers?** Is it identifying the exact root cause, or is reducing MTTR the real metric? If MTTR is the goal, the reward function should weight remediation speed more heavily than diagnosis precision.

3. **Do you have labeled incident data from customer investigations?** Even 100 real open-sre-agent sessions with human ratings ("this investigation was good/bad") would be more valuable for reward calibration than thousands of synthetic episodes.

4. **Would you accept a reward signal that's right 80% of the time?** LLM-as-judge will disagree with human experts on edge cases. What error rate is tolerable before the RL training signal becomes too noisy?

## Data & Coverage

5. **Can I access the VOID database programmatically?** The website exists but I couldn't find a public API. Do you have a data partnership or should I scrape the web interface?

6. **Do you have internal incident data beyond what's public?** Customer incidents (anonymised) would dramatically improve scenario realism. 10K VOID incidents are useful but they lack telemetry — they're narrative descriptions, not metrics/logs/traces.

7. **Which failure types do your customers hit most often?** The taxonomy has 35 leaf nodes but real-world frequency is heavily skewed. If 60% of customer incidents are deployment failures and connection pool issues, the training corpus should reflect that — not uniform coverage.

8. **How many distinct failure patterns would you consider "good enough" for a first training run?** The current 5 scenarios cover 29% of the taxonomy. Is 80% (28/35) the right target, or would 15 well-calibrated scenarios (43%) covering the highest-frequency failures be more valuable sooner?

## opensre Integration

9. **Which opensre node would you want RL to improve first — `plan_actions` or `root_cause_diagnosis`?** They're different skills. `plan_actions` decides what tools to call (investigation strategy). `root_cause_diagnosis` interprets evidence (reasoning). The training data format differs for each.

10. **How do you currently evaluate open-sre-agent's performance?** Do you have a benchmark suite of incidents you replay, or is evaluation ad-hoc? If a benchmark exists, I should align the RL evaluation against it.

11. **Is the `investigation_loop_count` cap of 5 a hard product constraint or tunable?** The RL agent might learn better with 10 investigation loops on complex incidents. If opensre limits it to 5, the training environment should match.

12. **What LLM provider does opensre use in production?** The RL fine-tuning approach (GRPO vs DPO vs SFT) depends on whether the model weights are accessible (open-source like Llama) or only available via API (Claude/GPT-4). API-only models limit training to prompt optimisation and tool selection policy.

## Simulation Fidelity

13. **Have you tested whether investigation skills transfer from synthetic to real telemetry?** This is the existential risk. If an agent trained on synthetic Prometheus-style metrics can't parse real Grafana dashboard output, the whole approach fails. Has anyone at Tracer tried this, even informally?

14. **What does real telemetry look like in your customers' environments?** My synthetic metrics use names like `cpu_percent`, `latency_p99_ms`. If real Grafana dashboards use `node_cpu_seconds_total` or `http_request_duration_seconds_bucket`, the agent needs to handle that mapping.

15. **Would AIOpsLab (Microsoft, MLSys'25) be acceptable as the Layer 2 validation environment?** It provides real DeathStarBench microservices with fault injection. The alternative is building a custom Docker Compose stack, which is more work but more aligned with opensre's actual integration targets.

## Product & Prioritisation

16. **What's the first customer-facing use case for RL-trained improvements?** Is it faster alert triage, better root cause accuracy, or fewer false positive investigations? This determines which reward component to weight highest.

17. **How much compute budget would you allocate for an initial training run?** The estimates show ~$150 for 100K synthetic episodes + ~$100 for Docker validation. Is that within budget, or should I design for a cheaper first iteration?

18. **Do you want the RL training to run continuously (retrain weekly on new incidents) or as a one-time fine-tuning step?** Continuous training requires the crawler→scenario→training pipeline to be automated. One-time is simpler but doesn't improve as new failure patterns emerge.

19. **Who would review and approve generated scenarios before they enter the training set?** The ScenarioGenerator auto-generates YAML from crawled incidents, but the gold-standard root causes are inferred, not verified. Is there an SRE on the team who would QA these?

20. **What's the timeline expectation?** Is this a "show results in 2 weeks" project or a "build the right foundation over 2 months" project? The architecture supports both, but the MVP scope changes significantly.
