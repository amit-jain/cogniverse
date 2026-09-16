"""Profile-selection ground truth stored as a tenant-owned blob.

The runtime admin upload path canonicalizes rows here, and the optimizer loads
the active blob through the same seam. The loader renders the shared
ground-truth contract, so a tenant that never uploaded one skips the step while
a store outage fails it.
"""

from __future__ import annotations

import json
from typing import Any

from cogniverse_agents.optimizer.ground_truth_blob import (
    GroundTruthError,
    GroundTruthInvalidError,
    GroundTruthMissingError,
    GroundTruthStoreUnavailableError,
    load_ground_truth_rows,
)

PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND = "config"
PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY = "profile_selection_ground_truth"


class ProfileSelectionGroundTruthError(GroundTruthError):
    """Base error for profile-selection ground-truth loading."""


class ProfileSelectionGroundTruthMissingError(
    ProfileSelectionGroundTruthError, GroundTruthMissingError
):
    reason = "profile_selection_ground_truth_missing"


class ProfileSelectionGroundTruthStoreUnavailableError(
    ProfileSelectionGroundTruthError, GroundTruthStoreUnavailableError
):
    reason = "profile_selection_ground_truth_store_unavailable"


class ProfileSelectionGroundTruthInvalidError(
    ProfileSelectionGroundTruthError, GroundTruthInvalidError
):
    reason = "profile_selection_ground_truth_invalid"


def _normalize_expected_videos(value: Any) -> list[str]:
    if isinstance(value, str):
        return [video.strip() for video in value.split(",") if video.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(video).strip() for video in value if str(video).strip()]
    return []


def canonicalize_profile_selection_ground_truth_rows(
    rows: Any,
) -> list[dict[str, Any]]:
    """Validate and normalize an uploaded ground-truth payload."""

    if not isinstance(rows, list):
        raise ValueError("profile_selection_ground_truth upload must be a JSON array")
    if not rows:
        raise ValueError("profile_selection_ground_truth must contain at least one row")

    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"profile_selection_ground_truth row {index} must be an object"
            )
        if "query" not in row:
            raise ValueError(
                f"profile_selection_ground_truth row {index} missing query"
            )
        if "expected_videos" not in row:
            raise ValueError(
                f"profile_selection_ground_truth row {index} missing expected_videos"
            )

        query_value = row["query"]
        if not isinstance(query_value, str):
            raise ValueError(
                f"profile_selection_ground_truth row {index} query must be a string"
            )
        query = query_value.strip()
        if not query:
            raise ValueError(
                "profile_selection_ground_truth row "
                f"{index} query must be non-empty after stripping whitespace"
            )

        expected_videos = _normalize_expected_videos(row["expected_videos"])
        if not expected_videos:
            raise ValueError(
                "profile_selection_ground_truth row "
                f"{index} expected_videos must contain at least one non-empty id "
                "after normalization"
            )

        normalized_row = dict(row)
        normalized_row["query"] = query
        normalized_row["expected_videos"] = expected_videos
        normalized_rows.append(normalized_row)

    return normalized_rows


def serialize_profile_selection_ground_truth_rows(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


async def load_profile_selection_ground_truth_rows(
    artifact_manager: Any,
) -> list[dict[str, Any]]:
    """Load the active tenant artifact and return canonicalized rows."""

    return await load_ground_truth_rows(
        artifact_manager,
        kind=PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND,
        key=PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY,
        canonicalize=canonicalize_profile_selection_ground_truth_rows,
        missing_error=ProfileSelectionGroundTruthMissingError,
        unavailable_error=ProfileSelectionGroundTruthStoreUnavailableError,
        invalid_error=ProfileSelectionGroundTruthInvalidError,
    )
