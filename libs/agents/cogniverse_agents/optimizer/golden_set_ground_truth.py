"""Golden-set ground truth stored as a tenant-owned blob.

The runtime admin upload path canonicalizes rows here, and the quality monitor
loads the active blob through the same seam. Missing artifacts surface as a
dedicated status, while store failures remain faults.
"""

from __future__ import annotations

import json
from typing import Any

from cogniverse_agents.optimizer.profile_selection_ground_truth import (
    canonicalize_profile_selection_ground_truth_rows as canonicalize_golden_set_ground_truth_rows,
    serialize_profile_selection_ground_truth_rows as serialize_golden_set_ground_truth_rows,
)
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

GOLDEN_SET_GROUND_TRUTH_BLOB_KIND = "config"
GOLDEN_SET_GROUND_TRUTH_BLOB_KEY = "golden_set_ground_truth"


class GoldenSetGroundTruthError(RuntimeError):
    """Base error for golden-set ground-truth loading."""

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


class GoldenSetGroundTruthMissingError(GoldenSetGroundTruthError):
    status = "golden_set_missing"
    retryable = False


class GoldenSetGroundTruthStoreUnavailableError(GoldenSetGroundTruthError):
    status = "golden_set_store_unavailable"
    retryable = True


async def load_golden_set_ground_truth_rows(
    artifact_manager: Any,
) -> list[dict[str, Any]]:
    """Load the active tenant artifact and return canonicalized rows."""

    tenant_id = getattr(artifact_manager, "_tenant_id", "unknown")
    try:
        raw = await artifact_manager.load_blob(
            GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
            GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
        )
    except DatasetNotFoundError as exc:
        raise GoldenSetGroundTruthMissingError(
            f"golden_set_ground_truth is not configured for tenant {tenant_id}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise GoldenSetGroundTruthStoreUnavailableError(
            "golden_set_ground_truth store unavailable"
        ) from exc

    if raw is None:
        raise GoldenSetGroundTruthMissingError(
            f"golden_set_ground_truth is not configured for tenant {tenant_id}"
        )

    loaded = json.loads(raw)
    return canonicalize_golden_set_ground_truth_rows(loaded)
