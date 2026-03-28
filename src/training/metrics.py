"""Training metrics — accuracy matrix and data analysis.

Provides:
  - Confusion-style accuracy matrix (predicted vs gold taxonomy labels)
  - Per-taxonomy-leaf performance breakdown
  - Data distribution analysis (EDA) for loaded scenarios
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from src.models import ScenarioDefinition
from src.training.episode_runner import EpisodeResult

logger = logging.getLogger(__name__)


def print_data_summary(scenarios: list[ScenarioDefinition]) -> None:
    """Print exploratory data analysis of the loaded scenario set."""
    print(f"\n  Scenarios loaded: {len(scenarios)}")

    # Taxonomy distribution
    label_counts: Counter[str] = Counter()
    for s in scenarios:
        for label in s.taxonomy_labels:
            label_counts[label] += 1

    print(f"  Taxonomy leaves covered: {len(label_counts)}")
    print("  Taxonomy distribution:")
    for label, count in label_counts.most_common():
        bar = "█" * min(count, 40)
        print(f"    {label:<50} {count:>3} {bar}")

    # Difficulty distribution
    difficulties = [s.difficulty for s in scenarios]
    print(f"\n  Difficulty range: {min(difficulties):.1f} - {max(difficulties):.1f}")
    easy = sum(1 for d in difficulties if d <= 0.4)
    medium = sum(1 for d in difficulties if 0.4 < d <= 0.6)
    hard = sum(1 for d in difficulties if d > 0.6)
    print(f"  Easy (<=0.4): {easy}  Medium (0.4-0.6): {medium}  Hard (>0.6): {hard}")

    # Service count distribution
    svc_counts = Counter(len(s.services) for s in scenarios)
    print(f"\n  Services per scenario: {dict(sorted(svc_counts.items()))}")

    # Source distribution
    sources: Counter[str] = Counter()
    for s in scenarios:
        if s.id.startswith("generated-aiops"):
            sources["aiops_dataset"] += 1
        elif s.id.startswith("generated-"):
            sources["crawled"] += 1
        else:
            sources["builtin"] += 1
    print(f"  Sources: {dict(sources)}")


def print_accuracy_matrix(results: list[EpisodeResult], scenarios: list[ScenarioDefinition]) -> None:
    """Print accuracy breakdown by taxonomy label.

    Shows per-label: episode count, avg reward, avg diagnosis score.
    This is the accuracy matrix — how well does the agent perform
    on each type of failure?
    """
    # Build scenario lookup
    scenario_map = {s.id: s for s in scenarios}

    # Aggregate by taxonomy label
    by_label: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        scenario = scenario_map.get(result.scenario_id)
        if scenario:
            for label in scenario.taxonomy_labels:
                by_label[label].append(result)

    if not by_label:
        print("\n  No results to analyze.")
        return

    print(f"\n  {'Taxonomy Label':<50} {'Episodes':>8} {'Reward':>8} {'Diag':>6} {'Eff':>6} {'Rem':>6}")
    print(f"  {'-' * 88}")

    for label in sorted(by_label.keys()):
        eps = by_label[label]
        n = len(eps)
        avg_reward = sum(r.reward for r in eps) / n
        avg_diag = sum(r.diagnosis_score for r in eps) / n
        avg_eff = sum(r.efficiency_score for r in eps) / n
        avg_rem = sum(r.remediation_score for r in eps) / n
        print(f"  {label:<50} {n:>8} {avg_reward:>8.3f} {avg_diag:>6.3f} {avg_eff:>6.3f} {avg_rem:>6.3f}")

    # Overall
    n = len(results)
    if n > 0:
        avg_r = sum(r.reward for r in results) / n
        avg_d = sum(r.diagnosis_score for r in results) / n
        avg_e = sum(r.efficiency_score for r in results) / n
        avg_m = sum(r.remediation_score for r in results) / n
        print(f"  {'-' * 88}")
        print(f"  {'OVERALL':<50} {n:>8} {avg_r:>8.3f} {avg_d:>6.3f} {avg_e:>6.3f} {avg_m:>6.3f}")
