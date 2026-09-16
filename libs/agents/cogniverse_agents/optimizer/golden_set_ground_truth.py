"""Golden-set ground truth stored as a tenant-owned blob.

The runtime admin upload path canonicalizes rows here, and the quality monitor
loads the active blob through the same seam. The loader renders the shared
ground-truth contract, so a tenant that never uploaded one skips the step while
a store outage fails it.
"""

from __future__ import annotations

from typing import Any

from cogniverse_agents.optimizer.ground_truth_blob import (
    GroundTruthError,
    GroundTruthInvalidError,
    GroundTruthMissingError,
    GroundTruthStoreUnavailableError,
    load_ground_truth_rows,
)
from cogniverse_agents.optimizer.profile_selection_ground_truth import (
    canonicalize_profile_selection_ground_truth_rows as canonicalize_golden_set_ground_truth_rows,
)
from cogniverse_agents.optimizer.profile_selection_ground_truth import (
    serialize_profile_selection_ground_truth_rows as serialize_golden_set_ground_truth_rows,
)

__all__ = [
    "GOLDEN_SET_GROUND_TRUTH_BLOB_KEY",
    "GOLDEN_SET_GROUND_TRUTH_BLOB_KIND",
    "GoldenSetGroundTruthError",
    "GoldenSetGroundTruthInvalidError",
    "GoldenSetGroundTruthMissingError",
    "GoldenSetGroundTruthStoreUnavailableError",
    "canonicalize_golden_set_ground_truth_rows",
    "load_golden_set_ground_truth_rows",
    "serialize_golden_set_ground_truth_rows",
]

GOLDEN_SET_GROUND_TRUTH_BLOB_KIND = "config"
GOLDEN_SET_GROUND_TRUTH_BLOB_KEY = "golden_set_ground_truth"


class GoldenSetGroundTruthError(GroundTruthError):
    """Base error for golden-set ground-truth loading."""


class GoldenSetGroundTruthMissingError(
    GoldenSetGroundTruthError, GroundTruthMissingError
):
    reason = "golden_set_missing"


class GoldenSetGroundTruthStoreUnavailableError(
    GoldenSetGroundTruthError, GroundTruthStoreUnavailableError
):
    reason = "golden_set_store_unavailable"


class GoldenSetGroundTruthInvalidError(
    GoldenSetGroundTruthError, GroundTruthInvalidError
):
    reason = "golden_set_invalid"


async def load_golden_set_ground_truth_rows(
    artifact_manager: Any,
) -> list[dict[str, Any]]:
    """Load the active tenant artifact and return canonicalized rows."""

    return await load_ground_truth_rows(
        artifact_manager,
        kind=GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
        key=GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
        canonicalize=canonicalize_golden_set_ground_truth_rows,
        missing_error=GoldenSetGroundTruthMissingError,
        unavailable_error=GoldenSetGroundTruthStoreUnavailableError,
        invalid_error=GoldenSetGroundTruthInvalidError,
    )
