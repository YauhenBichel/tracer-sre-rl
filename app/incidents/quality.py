"""Quality scoring for crawled incidents."""

from __future__ import annotations


def quality_score(item: dict, field_specs: list[str]) -> float:
    """Score 0-1 based on how many configured fields are present.

    Supports OR fields via pipe: "updates|most-recent-update" means either counts.
    """
    present = 0
    for spec in field_specs:
        if any(bool(item.get(f)) for f in spec.split("|")):
            present += 1
    return present / max(len(field_specs), 1)
