"""Pure training-example selection helpers."""

from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timezone
from typing import Any, Dict, List

ExampleStats = namedtuple("ExampleStats", "confirmations first_seen")
TrainingSelectionKnobs = namedtuple(
    "TrainingSelectionKnobs",
    "trainset_cap mmr_lambda low_confirmation_threshold downweight_age_days downweight_factor",
)
TRAINING_SELECTION_DEFAULTS = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)

__all__ = [
    "ExampleStats",
    "TRAINING_SELECTION_DEFAULTS",
    "TrainingSelectionKnobs",
    "confirmation_stats",
    "decay_weight",
]


def _parse_created_at(created_at: Any) -> datetime:
    """Parse ledger timestamps into datetimes."""
    parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def confirmation_stats(lineage: List[dict]) -> Dict[str, ExampleStats]:
    """Aggregate confirmations and first-seen timestamps per example id."""
    buckets: Dict[str, list[Any]] = {}
    for version in lineage:
        created_at = _parse_created_at(version["created_at"])
        consumed_example_ids = version.get("consumed_example_ids") or []
        promote = version.get("decision") == "promote"
        for example_id in consumed_example_ids:
            entry = buckets.get(example_id)
            if entry is None:
                buckets[example_id] = [1 if promote else 0, created_at]
                continue
            if promote:
                entry[0] += 1
            if created_at < entry[1]:
                entry[1] = created_at

    return {
        example_id: ExampleStats(entry[0], entry[1])
        for example_id, entry in buckets.items()
    }


def decay_weight(
    stats: Dict[str, ExampleStats],
    example_id: str,
    *,
    now: datetime,
    knobs: TrainingSelectionKnobs,
) -> float:
    """Return the confirmation-aware decay multiplier for one example id."""
    example_stats = stats.get(example_id)
    if example_stats is None:
        return 1.0

    if (
        example_stats.confirmations < knobs.low_confirmation_threshold
        and (now - example_stats.first_seen).days > knobs.downweight_age_days
    ):
        return knobs.downweight_factor

    return 1.0
