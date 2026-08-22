"""Pure training-example selection helpers."""

from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence

from cogniverse_core.common.models.semantic_embedder import get_semantic_embedder

ExampleStats = namedtuple("ExampleStats", "confirmations first_seen")
SelectionReport = namedtuple(
    "SelectionReport",
    "pool deduped cap mmr_applied decayed_count selected_ids",
)
TrainingSelectionKnobs = namedtuple(
    "TrainingSelectionKnobs",
    "trainset_cap mmr_lambda low_confirmation_threshold downweight_age_days "
    "downweight_factor confirmation_score_threshold",
    defaults=(None,),
)
TRAINING_SELECTION_DEFAULTS = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)

__all__ = [
    "ExampleStats",
    "TRAINING_SELECTION_DEFAULTS",
    "SelectionReport",
    "TrainingSelectionKnobs",
    "confirmation_stats",
    "decay_weight",
    "embed_texts",
    "select_training_records",
]


def _parse_created_at(created_at: Any) -> datetime:
    """Parse ledger timestamps into datetimes."""
    parsed = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def confirmation_stats(
    lineage: List[dict], *, score_threshold: float | None = None
) -> Dict[str, ExampleStats]:
    """Aggregate confirmations and first-seen timestamps per example id.

    When ``score_threshold`` is set, only promoted versions with a recorded
    score at or above the threshold count as confirmations. Legacy rows with no
    recorded score stay unconfirmed under the thresholded path.
    """
    buckets: Dict[str, list[Any]] = {}
    for version in lineage:
        created_at = _parse_created_at(version["created_at"])
        consumed_example_ids = version.get("consumed_example_ids") or []
        promote = version.get("decision") == "promote"
        score = version.get("score")
        confirmed = (
            promote
            if score_threshold is None
            else promote and score is not None and score >= score_threshold
        )
        for example_id in consumed_example_ids:
            entry = buckets.get(example_id)
            if entry is None:
                buckets[example_id] = [1 if confirmed else 0, created_at]
                continue
            if confirmed:
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


def _dedupe_records(records: List[dict]) -> List[dict]:
    seen_queries: set[str] = set()
    deduped: List[dict] = []
    for record in records:
        query_key = str(record["query"]).casefold()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        deduped.append(record)
    return deduped


def _vector_norm(vector: Sequence[float]) -> float:
    return sum(component * component for component in vector) ** 0.5


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"embedding dimensions must match: got {len(left)} and {len(right)}"
        )
    return sum(lhs * rhs for lhs, rhs in zip(left, right))


def _validate_embeddings(
    example_ids: List[str],
    embeddings: Sequence[Sequence[float]],
) -> List[Sequence[float]]:
    if len(embeddings) != len(example_ids):
        raise ValueError(
            "embed_fn returned "
            f"{len(embeddings)} embeddings for {len(example_ids)} records"
        )

    validated: List[Sequence[float]] = []
    for example_id, embedding in zip(example_ids, embeddings):
        if _vector_norm(embedding) == 0.0:
            raise ValueError(f"embedding for {example_id} has zero norm")
        validated.append(embedding)
    return validated


def embed_texts(
    endpoint: str,
    texts: List[str],
) -> List[List[float]]:
    """Embed query texts through the production DenseOn embedder.

    Delegates to ``get_semantic_embedder`` so the canonical model name, the
    DenseOn query prompt, token truncation, auth headers, and normalization
    all match the vectors production stores; a private transport here would
    drift from that contract.
    """
    try:
        embedder = get_semantic_embedder(remote_url=endpoint)
        matrix = embedder.encode(list(texts), is_query=True)
        normalized: List[List[float]] = []
        for index, row in enumerate(matrix):
            vector = [float(component) for component in row]
            if _vector_norm(vector) == 0.0:
                raise ValueError(f"embedding {index} has zero norm")
            normalized.append(vector)
        return normalized
    except Exception as exc:
        raise RuntimeError(
            f"training-selection embedder at {endpoint} failed: {exc}"
        ) from exc


def select_training_records(
    records: List[dict],
    *,
    weights: Dict[str, float],
    knobs: TrainingSelectionKnobs,
    embed_fn: Callable[[List[str]], List[List[float]]],
) -> tuple[List[dict], SelectionReport]:
    pool = len(records)
    deduped_records = _dedupe_records(records)
    deduped = len(deduped_records)
    decayed_count = sum(
        1 for record in deduped_records if weights[record["example_id"]] < 1.0
    )

    if deduped <= knobs.trainset_cap:
        selected_ids = [record["example_id"] for record in deduped_records]
        return (
            deduped_records,
            SelectionReport(
                pool,
                deduped,
                knobs.trainset_cap,
                False,
                decayed_count,
                selected_ids,
            ),
        )

    query_texts = [record["query"] for record in deduped_records]
    embeddings = _validate_embeddings(
        [record["example_id"] for record in deduped_records],
        embed_fn(query_texts),
    )

    selected_indices: List[int] = []
    selected_vectors: List[Sequence[float]] = []
    remaining_indices = list(range(deduped))

    seed_index = min(
        remaining_indices,
        key=lambda index: (
            -weights[deduped_records[index]["example_id"]],
            deduped_records[index]["example_id"],
        ),
    )
    selected_indices.append(seed_index)
    selected_vectors.append(embeddings[seed_index])
    remaining_indices.remove(seed_index)

    while remaining_indices and len(selected_indices) < knobs.trainset_cap:

        def _candidate_key(index: int) -> tuple[float, str]:
            example_id = deduped_records[index]["example_id"]
            similarity = max(
                _dot_product(embeddings[index], selected_vector)
                for selected_vector in selected_vectors
            )
            score = (
                knobs.mmr_lambda * weights[example_id]
                - (1.0 - knobs.mmr_lambda) * similarity
            )
            return (-score, example_id)

        next_index = min(remaining_indices, key=_candidate_key)
        selected_indices.append(next_index)
        selected_vectors.append(embeddings[next_index])
        remaining_indices.remove(next_index)

    selected_records = [deduped_records[index] for index in selected_indices]
    selected_ids = [record["example_id"] for record in selected_records]
    return (
        selected_records,
        SelectionReport(
            pool,
            deduped,
            knobs.trainset_cap,
            True,
            decayed_count,
            selected_ids,
        ),
    )
