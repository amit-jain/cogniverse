"""Self-consistency sampling of the entity-extraction teacher.

The optimizer draws the teacher several times for one query and keeps the
entities every draw agreed on. A mention the teacher only sometimes produces
is the one a human is asked about.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable, Mapping, Sequence

SELF_CONSISTENCY_SAMPLES = 3
"""Teacher draws per training example."""

SELF_CONSISTENCY_TEMPERATURE = 0.7
"""Sampling temperature for those draws; zero would return one answer thrice."""

AGREEMENT_KEY = "agreement"
NEEDS_REVIEW_KEY = "needs_review"
ENTITY_TEXT_KEY = "text"
ENTITY_TYPE_KEY = "type"

SELF_CONSISTENCY_ENTITY_KEYS = (
    ENTITY_TEXT_KEY,
    ENTITY_TYPE_KEY,
    AGREEMENT_KEY,
    NEEDS_REVIEW_KEY,
)
"""Every key one entity record in a review row carries."""

SAMPLES_KEY = "samples"
ENTITIES_KEY = "entities"
QUERY_KEY = "query"

SELF_CONSISTENCY_METADATA_KEY = "self_consistency"
SELF_CONSISTENCY_METADATA_KEYS = (SAMPLES_KEY, ENTITIES_KEY)
"""Every key the review item's self-consistency metadata block carries."""

NO_UNANIMOUS_KEY = "no_unanimous_examples"
"""Report key listing the queries no mention was unanimous for."""

UNANIMOUS_AGREEMENT = 1.0


def canonical_pair(text: Any, entity_type: Any) -> tuple[str, str]:
    """The identity a mention is counted under across samples."""
    return (str(text).strip().casefold(), str(entity_type).strip())


def _sample_pairs(sample: Iterable[Any]) -> dict[tuple[str, str], str]:
    """Canonical pair -> verbatim text, one entry per distinct mention."""
    pairs: dict[tuple[str, str], str] = {}
    for entity in sample:
        text, entity_type = _entity_fields(entity)
        pairs.setdefault(canonical_pair(text, entity_type), str(text).strip())
    return pairs


def _entity_fields(entity: Any) -> tuple[Any, Any]:
    if isinstance(entity, Mapping):
        return entity[ENTITY_TEXT_KEY], entity[ENTITY_TYPE_KEY]
    if isinstance(entity, (tuple, list)):
        text, entity_type = entity
        return text, entity_type
    return getattr(entity, ENTITY_TEXT_KEY), getattr(entity, ENTITY_TYPE_KEY)


def entity_agreement(
    samples: Sequence[Iterable[Any]],
) -> dict[tuple[str, str], float]:
    """Fraction of ``samples`` carrying each canonical mention.

    Raises ``ValueError`` on an empty sample sequence: a fraction of no draws
    is not an agreement, and returning an empty mapping would read as
    "the teacher agreed on nothing".
    """
    if not samples:
        raise ValueError("entity agreement needs at least one sample")
    total = len(samples)
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        for pair in _sample_pairs(sample):
            counts[pair] = counts.get(pair, 0) + 1
    return {pair: count / total for pair, count in counts.items()}


def agreement_entities(samples: Sequence[Iterable[Any]]) -> list[dict[str, Any]]:
    """One record per mention, in order of first appearance across samples."""
    agreement = entity_agreement(samples)
    verbatim: dict[tuple[str, str], str] = {}
    for sample in samples:
        for pair, text in _sample_pairs(sample).items():
            verbatim.setdefault(pair, text)
    return [
        {
            ENTITY_TEXT_KEY: verbatim[pair],
            ENTITY_TYPE_KEY: pair[1],
            AGREEMENT_KEY: agreement[pair],
            NEEDS_REVIEW_KEY: agreement[pair] < UNANIMOUS_AGREEMENT,
        }
        for pair in verbatim
    ]


def unanimous_entities(entities: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """The mentions every draw produced, as training-example entities."""
    return [
        {
            ENTITY_TEXT_KEY: entity[ENTITY_TEXT_KEY],
            ENTITY_TYPE_KEY: entity[ENTITY_TYPE_KEY],
        }
        for entity in entities
        if not entity[NEEDS_REVIEW_KEY]
    ]


def sample_entity_extraction(
    module_factory: Callable[[], Any],
    query: str,
    *,
    lm,
    samples: int = SELF_CONSISTENCY_SAMPLES,
) -> list[list[dict[str, str]]]:
    """Draw ``samples`` independent teacher extractions for one query.

    Each draw builds its own module and runs under ``lm``; the caller owns
    ``lm``'s sampling settings. Any draw that raises propagates, so a caller
    never sees agreement computed over fewer draws than it asked for.
    """
    import dspy

    drawn: list[list[dict[str, str]]] = []
    for _ in range(samples):
        with dspy.context(lm=lm):
            prediction = module_factory()(query=query)
        drawn.append(
            [
                {
                    ENTITY_TEXT_KEY: str(text).strip(),
                    ENTITY_TYPE_KEY: str(entity_type).strip(),
                }
                for text, entity_type in (
                    _entity_fields(entity) for entity in prediction.entities
                )
            ]
        )
    return drawn


def review_row(query: str, samples: Sequence[Iterable[Any]]) -> dict[str, Any]:
    """The approval-queue payload for one sampled example.

    ``data`` is the training example a human is asked to accept — the
    unanimous mentions only. ``metadata`` carries what every draw produced
    and how often, so the reviewer sees the disagreement the row hides.
    """
    entities = agreement_entities(samples)
    return {
        "data": {
            QUERY_KEY: query,
            ENTITIES_KEY: unanimous_entities(entities),
            "relationships": [],
        },
        "metadata": {
            SAMPLES_KEY: len(samples),
            ENTITIES_KEY: entities,
        },
    }


def row_confidence(row: Mapping[str, Any]) -> float:
    """Mean agreement over the row's mentions."""
    entities = row["metadata"][ENTITIES_KEY]
    return sum(entity[AGREEMENT_KEY] for entity in entities) / len(entities)


def row_needs_review(row: Mapping[str, Any]) -> bool:
    """True when at least one mention was not unanimous."""
    return any(entity[NEEDS_REVIEW_KEY] for entity in row["metadata"][ENTITIES_KEY])


async def collect_self_consistency_rows(
    records: Sequence[Mapping[str, Any]],
    module_factory: Callable[[], Any],
    *,
    lm,
    samples: int = SELF_CONSISTENCY_SAMPLES,
    record_cause: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Sample every record concurrently and return one review row per success.

    A record whose draws did not all complete contributes no row; its cause is
    recorded instead, so a partial draw can never be read as agreement.
    """

    async def one(record: Mapping[str, Any]) -> dict[str, Any] | None:
        query = record[QUERY_KEY]
        try:
            drawn = await asyncio.to_thread(
                sample_entity_extraction,
                module_factory,
                query,
                lm=lm,
                samples=samples,
            )
        except Exception as exc:  # noqa: BLE001
            record_cause(
                f"self-consistency sampling failed for query {query!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        row = review_row(query, drawn)
        row["example_id"] = record.get("example_id")
        return row

    results = await asyncio.gather(*(one(record) for record in records))
    return [row for row in results if row is not None]
