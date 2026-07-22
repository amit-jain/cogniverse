"""Pure record serialization + storage backend-outage contract.

Real-Phoenix round-trips live in
``tests/core/integration/test_pipeline_checkpoint_storage_real.py``. These
pin the resume math (pending phases, completed-unit skipping), the exact
span-attribute serialization, and the storage's raise-on-outage guarantee: a
backend read failure must propagate, never be read as "no checkpoint" (which
would silently restart a long-running workflow from scratch).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_core.durable import (
    PipelineCheckpoint,
    PipelineCheckpointStatus,
    PipelineCheckpointStorage,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _cp(**overrides) -> PipelineCheckpoint:
    base = dict(
        checkpoint_id="ckpt_1",
        workflow_id="wf",
        tenant_id="acme:acme",
        status="active",
        phases=["load", "compile", "distill", "eval"],
        phase_index=2,
        completed_units={
            "a": {"status": "completed", "result_ref": "r1", "training_examples": 7},
            "b": {"status": "failed", "error": "boom"},
        },
        metadata={"trigger_dataset": "t-x", "agents": ["a", "b"], "lookback_hours": 24},
        created_at=datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
        cursor={"golden_offset": 5},
        resume_count=1,
    )
    base.update(overrides)
    return PipelineCheckpoint(**base)


class TestRecord:
    def test_to_from_dict_roundtrip_exact(self):
        cp = _cp()
        assert PipelineCheckpoint.from_dict(cp.to_dict()) == cp

    def test_to_dict_flattens_complex_fields_to_json_strings(self):
        d = _cp().to_dict()
        assert d["phases"] == '["load", "compile", "distill", "eval"]'
        assert d["phase_index"] == 2
        assert d["resume_count"] == 1
        assert d["cursor"] == '{"golden_offset": 5}'
        assert '"trigger_dataset": "t-x"' in d["metadata"]

    def test_none_cursor_serializes_empty_and_restores_none(self):
        cp = _cp(cursor=None)
        assert cp.to_dict()["cursor"] == ""
        assert PipelineCheckpoint.from_dict(cp.to_dict()).cursor is None

    def test_pending_phases(self):
        assert _cp(phase_index=2).pending_phases() == ["distill", "eval"]
        assert _cp(phase_index=0).pending_phases() == [
            "load",
            "compile",
            "distill",
            "eval",
        ]
        assert _cp(phase_index=4).pending_phases() == []

    def test_completed_unit_keys_excludes_failed(self):
        # Only 'completed' units are skippable on resume; a failed unit re-runs.
        assert _cp().completed_unit_keys() == {"a"}


def _storage_with_failing_provider() -> PipelineCheckpointStorage:
    tm = MagicMock()
    tm.config.provider_config = {"http_endpoint": "h"}
    provider = MagicMock()
    provider.traces.get_spans = AsyncMock(side_effect=RuntimeError("phoenix down"))
    tm.get_provider.return_value = provider
    return PipelineCheckpointStorage(
        grpc_endpoint="g",
        http_endpoint="h",
        tenant_id="acme:acme",
        telemetry_manager=tm,
    )


class TestBackendOutageRaises:
    @pytest.mark.asyncio
    async def test_get_latest_reraises_backend_error(self):
        st = _storage_with_failing_provider()
        with pytest.raises(RuntimeError, match="phoenix down"):
            await st.get_latest_checkpoint("wf")

    @pytest.mark.asyncio
    async def test_mark_status_reraises_backend_error(self):
        st = _storage_with_failing_provider()
        with pytest.raises(RuntimeError, match="phoenix down"):
            await st.mark_status("ckpt_1", PipelineCheckpointStatus.COMPLETED)
