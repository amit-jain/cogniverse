"""Real-Phoenix round-trip for the stage-pipeline durable-execution store.

Long-running optimization/eval workflows checkpoint their progress as
telemetry spans so a killed pod can resume from the last completed stage
instead of re-running expensive DSPy compiles. This exercises the store
against a real Phoenix Docker (no mocked boundary): save a checkpoint and
read it back, assert every field survives the span round-trip, that
get_latest returns the most recent checkpoint, and that a status annotation
overrides the embedded status.

Each test uses a unique workflow_id so the shared Phoenix container can't
cross-contaminate, and polls for the span/annotation to index (the exporter
flush guarantees export, not query-visibility).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from cogniverse_core.durable import (
    PipelineCheckpoint,
    PipelineCheckpointStatus,
    PipelineCheckpointStorage,
)

pytestmark = pytest.mark.integration


def _storage(phoenix_container, tenant_id="acme:acme") -> PipelineCheckpointStorage:
    return PipelineCheckpointStorage(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
    )


def _checkpoint(workflow_id: str, **overrides) -> PipelineCheckpoint:
    base = dict(
        checkpoint_id="ckpt_1",
        workflow_id=workflow_id,
        tenant_id="acme:acme",
        status="active",
        phases=["load", "compile", "distill", "eval"],
        phase_index=1,
        completed_units={
            "search_agent": {
                "status": "completed",
                "result_ref": "artifact_abc",
                "training_examples": 42,
            }
        },
        cursor=None,
        metadata={
            "trigger_dataset": "trigger-x",
            "agents": ["search_agent", "kg_agent"],
            "lookback_hours": 24,
        },
        created_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
        resume_count=0,
    )
    base.update(overrides)
    return PipelineCheckpoint(**base)


async def _await_latest(storage, workflow_id, predicate, tries=15, delay=1.0):
    """Poll get_latest until predicate holds — spans index after export."""
    latest = None
    for _ in range(tries):
        latest = await storage.get_latest_checkpoint(workflow_id)
        if latest is not None and predicate(latest):
            return latest
        await asyncio.sleep(delay)
    return latest


@pytest.mark.asyncio
async def test_save_and_get_latest_roundtrip(phoenix_container):
    storage = _storage(phoenix_container)
    wf = "wf_roundtrip"
    await storage.save_checkpoint(_checkpoint(wf))

    loaded = await _await_latest(storage, wf, lambda c: c.checkpoint_id == "ckpt_1")
    assert loaded is not None
    assert loaded.checkpoint_id == "ckpt_1"
    assert loaded.workflow_id == wf
    assert loaded.tenant_id == "acme:acme"
    assert loaded.status == "active"
    assert loaded.phases == ["load", "compile", "distill", "eval"]
    assert loaded.phase_index == 1
    # Resume skips the done phase and re-runs from "compile".
    assert loaded.pending_phases() == ["compile", "distill", "eval"]
    # Per-agent compile result survives the span round-trip intact.
    assert loaded.completed_units["search_agent"]["result_ref"] == "artifact_abc"
    assert loaded.completed_units["search_agent"]["training_examples"] == 42
    assert loaded.completed_unit_keys() == {"search_agent"}
    # Free-form metadata (trigger dataset, agent list, lookback) round-trips.
    assert loaded.metadata["trigger_dataset"] == "trigger-x"
    assert loaded.metadata["agents"] == ["search_agent", "kg_agent"]
    assert loaded.metadata["lookback_hours"] == 24


@pytest.mark.asyncio
async def test_get_latest_returns_most_recent_by_created_at(phoenix_container):
    storage = _storage(phoenix_container)
    wf = "wf_most_recent"
    await storage.save_checkpoint(
        _checkpoint(
            wf,
            checkpoint_id="ckpt_early",
            phase_index=1,
            created_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
    )
    await storage.save_checkpoint(
        _checkpoint(
            wf,
            checkpoint_id="ckpt_late",
            phase_index=3,
            completed_units={
                "search_agent": {"status": "completed", "result_ref": "a1"},
                "kg_agent": {"status": "completed", "result_ref": "a2"},
            },
            created_at=datetime(2026, 7, 15, 12, 5, 0, tzinfo=timezone.utc),
        )
    )

    loaded = await _await_latest(storage, wf, lambda c: c.checkpoint_id == "ckpt_late")
    assert loaded is not None
    assert loaded.checkpoint_id == "ckpt_late"
    assert loaded.phase_index == 3
    assert loaded.completed_unit_keys() == {"search_agent", "kg_agent"}


@pytest.mark.asyncio
async def test_mark_status_annotation_overrides_embedded_status(phoenix_container):
    storage = _storage(phoenix_container)
    wf = "wf_mark_status"
    await storage.save_checkpoint(
        _checkpoint(wf, checkpoint_id="ckpt_done", status="active")
    )
    # Ensure the span is indexed before annotating it.
    await _await_latest(storage, wf, lambda c: c.checkpoint_id == "ckpt_done")

    assert await storage.mark_status("ckpt_done", PipelineCheckpointStatus.COMPLETED)

    loaded = await _await_latest(storage, wf, lambda c: c.status == "completed")
    assert loaded is not None
    # The annotation wins over the span's embedded "active".
    assert loaded.status == "completed"


@pytest.mark.asyncio
async def test_get_latest_none_for_unknown_workflow(phoenix_container):
    storage = _storage(phoenix_container)
    await storage.save_checkpoint(_checkpoint("wf_something"))
    # A real Phoenix that simply has no matching workflow returns None —
    # distinct from a backend outage (which raises; see the unit tests).
    assert await storage.get_latest_checkpoint("wf_does_not_exist") is None
