"""Scores diagnosis accuracy using hierarchical taxonomy matching.

The agent's diagnosis label is compared to each gold-standard root cause
using taxonomy path similarity. Partial credit is given for identifying
the correct category even if the exact failure type is wrong.
"""

from __future__ import annotations

from src.config import reward_config
from src.models import ScenarioDefinition


class DiagnosisScorer:
    def __init__(self):
        cfg = reward_config()["diagnosis"]
        self._depth_scores: dict[int, float] = {int(k): v for k, v in cfg["depth_scores"].items()}
        self._deep_match_base: float = cfg["deep_match_base"]
        self._deep_match_scale: float = cfg["deep_match_scale"]

    def score(self, label: str, scenario: ScenarioDefinition) -> float:
        """Best similarity-weighted score across all gold-standard root causes."""
        if not scenario.gold_root_causes:
            return 0.0
        return max(self._similarity(label, gold.taxonomy_label) * gold.relevance for gold in scenario.gold_root_causes)

    def _similarity(self, diagnosed: str, gold: str) -> float:
        """Hierarchical similarity based on depth of common taxonomy prefix.

        Example: "infrastructure.database.connection_pool" vs "infrastructure.database.replication_lag"
          - Common prefix depth = 2 ("infrastructure.database")
          - Configured score for depth 2 = 0.7
        """
        if diagnosed == gold:
            return 1.0

        diagnosed_parts = diagnosed.split(".")
        gold_parts = gold.split(".")

        common_depth = 0
        for a, b in zip(diagnosed_parts, gold_parts, strict=False):
            if a == b:
                common_depth += 1
            else:
                break

        if common_depth in self._depth_scores:
            return self._depth_scores[common_depth]

        max_depth = max(len(diagnosed_parts), len(gold_parts))
        if max_depth == 0:
            return 0.0
        return self._deep_match_base + self._deep_match_scale * (common_depth / max_depth)
