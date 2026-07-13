"""Persist search-result relevance annotations to the telemetry backend.

The relevance signal is what the embedding-triplet miner (TripletExtractor)
reads back as positives.
"""

from __future__ import annotations

from typing import Any

# Relevance radio label -> score. result_relevance >= 0.7 counts as a positive
# in TripletExtractor._get_clicked_results.
RELEVANCE_SCORES = {
    "Highly Relevant": 1.0,
    "Somewhat Relevant": 0.5,
    "Not Relevant": 0.0,
}


async def persist_result_relevance(
    provider: Any,
    project: str,
    span_id: str | None,
    result_id: str,
    relevance_label: str,
) -> float:
    """Write a ``result_relevance`` annotation on the search span.

    Raises on a missing span_id (telemetry off) or an unknown label so the
    caller surfaces the failure instead of reporting a save that wrote nothing.
    """
    if not span_id:
        raise ValueError(
            "no search span_id — cannot annotate this result "
            "(telemetry disabled or span not captured)"
        )
    if relevance_label not in RELEVANCE_SCORES:
        raise ValueError(f"unknown relevance label: {relevance_label!r}")

    score = RELEVANCE_SCORES[relevance_label]
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="result_relevance",
        label=relevance_label,
        score=score,
        metadata={"result_id": str(result_id)},
        project=project,
    )
    return score
