"""One contract for every tenant ground-truth blob loader.

A step that needs ground truth has three outcomes, and they are not the same
workflow outcome: nothing uploaded is no work (``skipped``, exit 0), a store
that cannot answer is a retryable fault (``failed``), and an uploaded payload
that will not canonicalize is a fault no retry fixes (``failed``). The three
loaders rendered three different shapes before, so the profile step failed on
an absent blob while the entity step succeeded on one.
"""

from __future__ import annotations

import json

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
    EntityExtractionGroundTruthError,
    load_entity_extraction_ground_truth_rows,
)
from cogniverse_agents.optimizer.golden_set_ground_truth import (
    GoldenSetGroundTruthError,
    load_golden_set_ground_truth_rows,
)
from cogniverse_agents.optimizer.profile_selection_ground_truth import (
    ProfileSelectionGroundTruthError,
    load_profile_selection_ground_truth_rows,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

LOADERS = {
    "profile_selection_ground_truth": (
        load_profile_selection_ground_truth_rows,
        ProfileSelectionGroundTruthError,
        "profile_selection_ground_truth",
    ),
    "entity_extraction_ground_truth": (
        load_entity_extraction_ground_truth_rows,
        EntityExtractionGroundTruthError,
        "entity_extraction_ground_truth",
    ),
    "golden_set_ground_truth": (
        load_golden_set_ground_truth_rows,
        GoldenSetGroundTruthError,
        "golden_set",
    ),
}


class _Datasets:
    """A dataset store whose blob slots answer, are empty, or fail."""

    def __init__(self, *, rows=None, get_error=None):
        self.rows = rows
        self.get_error = get_error

    async def get_dataset(self, name):
        if self.get_error is not None:
            raise self.get_error
        if self.rows is None:
            return None
        import pandas as pd

        return pd.DataFrame([{"content": json.dumps(self.rows), "blob_revision": "0"}])


class _Provider:
    def __init__(self, datasets):
        self.datasets = datasets


def _manager(datasets) -> ArtifactManager:
    return ArtifactManager(_Provider(datasets), "acme:prod")


@pytest.mark.parametrize("blob_key", sorted(LOADERS))
@pytest.mark.asyncio
async def test_nothing_uploaded_skips_the_step(blob_key):
    load, error_type, reason_prefix = LOADERS[blob_key]

    with pytest.raises(error_type) as caught:
        await load(_manager(_Datasets(rows=None)))

    assert caught.value.to_result() == {
        "status": "skipped",
        "reason": f"{reason_prefix}_missing",
        "retryable": False,
        "error": f"{blob_key} is not configured for tenant acme:prod",
    }


@pytest.mark.parametrize("blob_key", sorted(LOADERS))
@pytest.mark.asyncio
async def test_store_outage_fails_the_step_and_is_retryable(blob_key):
    load, error_type, reason_prefix = LOADERS[blob_key]

    with pytest.raises(error_type) as caught:
        await load(_manager(_Datasets(get_error=ConnectionError("blob store down"))))

    assert caught.value.to_result() == {
        "status": "failed",
        "reason": f"{reason_prefix}_store_unavailable",
        "retryable": True,
        "error": f"{blob_key} store unavailable",
        "cause": {"type": "ConnectionError", "message": "blob store down"},
    }


@pytest.mark.parametrize("blob_key", sorted(LOADERS))
@pytest.mark.asyncio
async def test_unusable_payload_fails_the_step_without_retry(blob_key):
    load, error_type, reason_prefix = LOADERS[blob_key]

    with pytest.raises(error_type) as caught:
        await load(_manager(_Datasets(rows=[{"query": "no expectations"}])))

    result = caught.value.to_result()
    assert {key: value for key, value in result.items() if key != "cause"} == {
        "status": "failed",
        "reason": f"{reason_prefix}_invalid",
        "retryable": False,
        "error": f"{blob_key} payload is not usable",
    }
    assert result["cause"]["type"] == "ValueError"


@pytest.mark.parametrize("blob_key", sorted(LOADERS))
@pytest.mark.asyncio
async def test_uploaded_rows_load(blob_key):
    load, _, _ = LOADERS[blob_key]
    if blob_key == "entity_extraction_ground_truth":
        rows = [
            {
                "query": "red kite over the field",
                "entities": [{"text": "red kite", "type": "CONCEPT"}],
            }
        ]
    else:
        rows = [{"query": "red kite over the field", "expected_videos": ["kite1"]}]

    assert await load(_manager(_Datasets(rows=rows))) == rows
