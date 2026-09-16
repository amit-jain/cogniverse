"""The contract every tenant ground-truth blob loader renders.

A step that needs ground truth has three outcomes the CLI exit classifier and
the workflow must tell apart: the tenant never uploaded one, so there is
nothing to do; the store could not answer, which is a fault worth retrying; or
the uploaded payload is not usable, which is a fault no retry fixes.
``status`` carries which, ``reason`` carries the typed name of the blob it was.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

GROUND_TRUTH_SKIPPED = "skipped"
GROUND_TRUTH_FAILED = "failed"


class GroundTruthError(RuntimeError):
    """Base error for loading a tenant-owned ground-truth blob."""

    status = ""
    reason = ""
    retryable = False

    def to_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
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


class GroundTruthMissingError(GroundTruthError):
    """No blob was ever uploaded for this tenant; the step has no work."""

    status = GROUND_TRUTH_SKIPPED
    retryable = False


class GroundTruthStoreUnavailableError(GroundTruthError):
    """The store could not answer for the blob."""

    status = GROUND_TRUTH_FAILED
    retryable = True


class GroundTruthInvalidError(GroundTruthError):
    """The uploaded blob is not a payload the loader can canonicalize."""

    status = GROUND_TRUTH_FAILED
    retryable = False


async def load_ground_truth_rows(
    artifact_manager: Any,
    *,
    kind: str,
    key: str,
    canonicalize: Callable[[Any], list[dict[str, Any]]],
    missing_error: type[GroundTruthMissingError],
    unavailable_error: type[GroundTruthStoreUnavailableError],
    invalid_error: type[GroundTruthInvalidError],
) -> list[dict[str, Any]]:
    """Load the tenant's active blob ``kind/key`` and return canonicalized rows."""

    tenant_id = getattr(artifact_manager, "_tenant_id", "unknown")
    try:
        raw = await artifact_manager.load_blob(kind, key)
    except DatasetNotFoundError as exc:
        raise missing_error(f"{key} is not configured for tenant {tenant_id}") from exc
    except Exception as exc:  # noqa: BLE001
        raise unavailable_error(f"{key} store unavailable") from exc

    if raw is None:
        raise missing_error(f"{key} is not configured for tenant {tenant_id}")

    try:
        return canonicalize(json.loads(raw))
    except ValueError as exc:
        raise invalid_error(f"{key} payload is not usable") from exc
