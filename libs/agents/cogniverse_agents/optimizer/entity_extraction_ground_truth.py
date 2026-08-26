"""Entity-extraction ground truth stored as a tenant-owned blob.

The runtime admin upload path canonicalizes rows here, and the loader reads the
active blob through the same seam. Missing artifacts surface as a dedicated
status, while store failures remain faults.
"""

from __future__ import annotations

import json
from typing import Any

from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KIND = "config"
ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KEY = "entity_extraction_ground_truth"


class EntityExtractionGroundTruthError(RuntimeError):
    """Base error for entity-extraction ground-truth loading."""

    status = ""
    retryable = False

    def to_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "retryable": self.retryable,
            "error": str(self),
        }
        cause = self.__cause__
        if cause is not None:
            result["cause"] = {
                "type": type(cause).__name__,
                "message": str(cause),
            }
        return result


class EntityExtractionGroundTruthMissingError(EntityExtractionGroundTruthError):
    status = "entity_extraction_ground_truth_missing"
    retryable = False


class EntityExtractionGroundTruthStoreUnavailableError(
    EntityExtractionGroundTruthError
):
    status = "entity_extraction_ground_truth_store_unavailable"
    retryable = True


def _normalize_non_empty_string(
    value: Any,
    *,
    row_index: int,
    field_name: str,
    entity_index: int | None = None,
) -> str:
    if not isinstance(value, str):
        prefix = (
            f"entity_extraction_ground_truth row {row_index} "
            f"entities entry {entity_index} "
            if entity_index is not None
            else f"entity_extraction_ground_truth row {row_index} "
        )
        raise ValueError(f"{prefix}{field_name} must be a string")

    normalized = value.strip()
    if normalized:
        return normalized

    prefix = (
        f"entity_extraction_ground_truth row {row_index} entities entry {entity_index} "
        if entity_index is not None
        else f"entity_extraction_ground_truth row {row_index} "
    )
    raise ValueError(
        f"{prefix}{field_name} must be non-empty after stripping whitespace"
    )


def canonicalize_entity_extraction_ground_truth_rows(
    rows: Any,
) -> list[dict[str, Any]]:
    """Validate and normalize an uploaded ground-truth payload."""

    if not isinstance(rows, list):
        raise ValueError("entity_extraction_ground_truth upload must be a JSON array")
    if not rows:
        raise ValueError("entity_extraction_ground_truth must contain at least one row")

    normalized_rows: list[dict[str, Any]] = []
    seen_queries: dict[str, int] = {}
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"entity_extraction_ground_truth row {row_index} must be an object"
            )
        if "query" not in row:
            raise ValueError(
                f"entity_extraction_ground_truth row {row_index} missing query"
            )
        if "entities" not in row:
            raise ValueError(
                f"entity_extraction_ground_truth row {row_index} missing entities"
            )

        query = _normalize_non_empty_string(
            row["query"], row_index=row_index, field_name="query"
        )
        prior_row = seen_queries.get(query)
        if prior_row is not None:
            raise ValueError(
                "entity_extraction_ground_truth row "
                f"{row_index} query duplicates row {prior_row}"
            )
        seen_queries[query] = row_index

        entities_value = row["entities"]
        if not isinstance(entities_value, list) or not entities_value:
            raise ValueError(
                "entity_extraction_ground_truth row "
                f"{row_index} entities must be a non-empty array"
            )

        normalized_entities: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for entity_index, entity in enumerate(entities_value, start=1):
            if not isinstance(entity, dict):
                raise ValueError(
                    "entity_extraction_ground_truth row "
                    f"{row_index} entities entry {entity_index} must be an object"
                )
            if "text" not in entity:
                raise ValueError(
                    "entity_extraction_ground_truth row "
                    f"{row_index} entities entry {entity_index} missing text"
                )
            if "type" not in entity:
                raise ValueError(
                    "entity_extraction_ground_truth row "
                    f"{row_index} entities entry {entity_index} missing type"
                )

            text = _normalize_non_empty_string(
                entity["text"],
                row_index=row_index,
                entity_index=entity_index,
                field_name="text",
            )
            entity_type = _normalize_non_empty_string(
                entity["type"],
                row_index=row_index,
                entity_index=entity_index,
                field_name="type",
            )

            pair = (text.casefold(), entity_type)
            if pair in seen_pairs:
                raise ValueError(
                    "entity_extraction_ground_truth row "
                    f"{row_index} entities entry {entity_index} "
                    "duplicates a prior text/type pair"
                )
            seen_pairs.add(pair)

            normalized_entity = dict(entity)
            normalized_entity["text"] = text
            normalized_entity["type"] = entity_type
            normalized_entities.append(normalized_entity)

        normalized_row = dict(row)
        normalized_row["query"] = query
        normalized_row["entities"] = normalized_entities
        normalized_rows.append(normalized_row)

    return normalized_rows


def serialize_entity_extraction_ground_truth_rows(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


async def load_entity_extraction_ground_truth_rows(
    artifact_manager: Any,
) -> list[dict[str, Any]]:
    """Load the active tenant artifact and return canonicalized rows."""

    tenant_id = getattr(artifact_manager, "_tenant_id", "unknown")
    try:
        raw = await artifact_manager.load_blob(
            ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KIND,
            ENTITY_EXTRACTION_GROUND_TRUTH_BLOB_KEY,
        )
    except DatasetNotFoundError as exc:
        raise EntityExtractionGroundTruthMissingError(
            f"entity_extraction_ground_truth is not configured for tenant {tenant_id}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise EntityExtractionGroundTruthStoreUnavailableError(
            "entity_extraction_ground_truth store unavailable"
        ) from exc

    if raw is None:
        raise EntityExtractionGroundTruthMissingError(
            f"entity_extraction_ground_truth is not configured for tenant {tenant_id}"
        )

    loaded = json.loads(raw)
    return canonicalize_entity_extraction_ground_truth_rows(loaded)
