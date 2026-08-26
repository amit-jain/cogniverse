"""Profile-selection ground truth stored as a tenant-owned blob.

The runtime admin upload path canonicalizes rows here, and the optimizer loads
the active blob through the same seam. The loader distinguishes a missing
tenant artifact from a store failure so callers can surface a precise status.
"""

from __future__ import annotations

import json
from typing import Any

PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND = "config"
PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY = "profile_selection_ground_truth"


class ProfileSelectionGroundTruthError(RuntimeError):
    """Base error for profile-selection ground-truth loading."""

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


class ProfileSelectionGroundTruthMissingError(ProfileSelectionGroundTruthError):
    status = "profile_selection_ground_truth_missing"
    retryable = False


class ProfileSelectionGroundTruthStoreUnavailableError(
    ProfileSelectionGroundTruthError
):
    status = "profile_selection_ground_truth_store_unavailable"
    retryable = True


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

    tenant_id = getattr(artifact_manager, "_tenant_id", "unknown")
    try:
        raw = await artifact_manager.load_blob(
            PROFILE_SELECTION_GROUND_TRUTH_BLOB_KIND,
            PROFILE_SELECTION_GROUND_TRUTH_BLOB_KEY,
        )
    except Exception as exc:  # noqa: BLE001
        raise ProfileSelectionGroundTruthStoreUnavailableError(
            "profile_selection_ground_truth store unavailable"
        ) from exc

    if raw is None:
        raise ProfileSelectionGroundTruthMissingError(
            f"profile_selection_ground_truth is not configured for tenant {tenant_id}"
        )

    try:
        loaded = json.loads(raw)
        return canonicalize_profile_selection_ground_truth_rows(loaded)
    except ValueError as exc:
        raise ProfileSelectionGroundTruthStoreUnavailableError(
            "profile_selection_ground_truth store unavailable"
        ) from exc
