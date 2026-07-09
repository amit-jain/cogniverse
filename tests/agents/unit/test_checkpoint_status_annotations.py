"""_get_checkpoint_current_status must honor status-update annotations.

Status transitions are written as ``checkpoint_status_update`` span
annotations; the read path previously ignored them and always returned the
span's original attribute, so an approval marking a checkpoint COMPLETED or
SUPERSEDED was invisible and the workflow saw stale status.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cogniverse_agents.orchestrator.checkpoint_storage import WorkflowCheckpointStorage
from cogniverse_agents.orchestrator.checkpoint_types import CheckpointStatus


def _storage(annotations_df):
    st = object.__new__(WorkflowCheckpointStorage)
    st.full_project_name = "proj"
    st.provider = MagicMock()
    st.provider.annotations.get_annotations = AsyncMock(return_value=annotations_df)
    return st


@pytest.mark.asyncio
async def test_latest_annotation_overrides_default():
    df = pd.DataFrame(
        {
            "result.label": ["superseded", "completed"],
            "created_at": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        }
    )
    st = _storage(df)
    status = await st._get_checkpoint_current_status("span-1", "active")
    assert status == CheckpointStatus.COMPLETED


@pytest.mark.asyncio
async def test_no_annotation_falls_back_to_default():
    st = _storage(pd.DataFrame())
    status = await st._get_checkpoint_current_status("span-1", "superseded")
    assert status == CheckpointStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_backend_failure_falls_back_to_default_not_crash(caplog):
    import logging

    st = object.__new__(WorkflowCheckpointStorage)
    st.full_project_name = "proj"
    st.provider = MagicMock()
    st.provider.annotations.get_annotations = AsyncMock(
        side_effect=ConnectionError("phoenix down")
    )
    with caplog.at_level(logging.WARNING):
        status = await st._get_checkpoint_current_status("span-1", "active")
    assert status == CheckpointStatus.ACTIVE
    assert any(
        "status-annotation read failed" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_unknown_label_coerces_to_active():
    df = pd.DataFrame({"result.label": ["nonsense"], "created_at": ["2026-01-01Z"]})
    st = _storage(df)
    status = await st._get_checkpoint_current_status("span-1", "active")
    assert status == CheckpointStatus.ACTIVE
