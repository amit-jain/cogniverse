"""Unit tests for C.4 — snapshot-on-promote and rollback to a prior version."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


class FakeStore:
    """Captures dataset writes; tracks calls to assert snapshot semantics."""

    def __init__(self):
        self.created: dict[str, pd.DataFrame] = {}
        self.create_calls: list[str] = []
        self.append_calls: list[tuple[str, pd.DataFrame]] = []

    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Treat re-create of an existing name as overwrite — the active
        # path uses a fixed dataset name on each save, mirroring how the
        # un-versioned save_prompts works.
        self.created[name] = data.copy()
        self.create_calls.append(name)
        return f"id::{name}"

    async def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.created:
            raise KeyError(name)
        return self.created[name]

    async def append_to_dataset(self, name: str, data: pd.DataFrame) -> None:
        if name not in self.created:
            raise KeyError(name)
        self.append_calls.append((name, data))
        self.created[name] = pd.concat([self.created[name], data], ignore_index=True)


class FakeProvider:
    def __init__(self):
        self.datasets = FakeStore()
        self.experiments = AsyncMock()


def _ds_prompts(agent: str) -> str:
    return f"dspy-prompts-acme-{agent}"


def _ds_demos(agent: str) -> str:
    return f"dspy-demos-acme-{agent}"


def _ds_prompts_v(agent: str, v: int) -> str:
    return f"dspy-prompts-acme-{agent}-v{v}"


def _ds_demos_v(agent: str, v: int) -> str:
    return f"dspy-demos-acme-{agent}-v{v}"


@pytest.fixture
def manager_provider():
    provider = FakeProvider()
    return ArtifactManager(provider, tenant_id="acme"), provider


@pytest.mark.asyncio
class TestSnapshotOnPromote:
    async def test_first_promote_has_no_snapshot_to_take(self, manager_provider):
        mgr, provider = manager_provider
        record = await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=None,
            baseline_score=0.0,
            candidate_score=0.5,
        )
        # Active prompts persisted; no versioned prompts dataset exists yet.
        assert _ds_prompts("agent_a") in provider.datasets.created
        assert _ds_prompts_v("agent_a", 1) not in provider.datasets.created
        # extra_metrics records that no snapshot was taken (key absent).
        assert "pre_promote_snapshot" not in record.extra_metrics

    async def test_second_promote_snapshots_active_to_versioned(self, manager_provider):
        mgr, provider = manager_provider
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=[{"input": "q1", "output": "a1"}],
            baseline_score=0.0,
            candidate_score=0.5,
        )

        record = await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v2"},
            candidate_demos=[{"input": "q2", "output": "a2"}],
            baseline_score=0.5,
            candidate_score=0.7,
        )

        # Snapshot of v1 lives at version 1 (prompts + demos).
        assert _ds_prompts_v("agent_a", 1) in provider.datasets.created
        assert _ds_demos_v("agent_a", 1) in provider.datasets.created
        # Active is now v2's content.
        active = await mgr.load_prompts("agent_a")
        assert active == {"system": "v2"}
        # Record carries the snapshot info.
        snap = record.extra_metrics["pre_promote_snapshot"]
        assert snap["prompts_version"] == 1
        assert snap["demos_version"] == 1

    async def test_snapshot_can_be_disabled(self, manager_provider):
        mgr, _ = manager_provider
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=None,
            baseline_score=0.0,
            candidate_score=0.5,
        )
        record = await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v2"},
            candidate_demos=None,
            baseline_score=0.5,
            candidate_score=0.6,
            snapshot_before_promote=False,
        )
        # No pre-promote snapshot recorded.
        assert "pre_promote_snapshot" not in record.extra_metrics


@pytest.mark.asyncio
class TestRollback:
    async def test_rollback_restores_versioned_prompts_to_active(
        self, manager_provider
    ):
        mgr, provider = manager_provider
        # Two successive promotes — produces a v1 snapshot for prompts.
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=None,
            baseline_score=0.0,
            candidate_score=0.5,
        )
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v2"},
            candidate_demos=None,
            baseline_score=0.5,
            candidate_score=0.7,
        )

        # Sanity: active is v2.
        assert (await mgr.load_prompts("agent_a")) == {"system": "v2"}

        # Rollback to v1.
        out = await mgr.rollback_to_version("agent_a", prompts_version=1)
        assert out["restored"]["prompts_version"] == 1

        # Active prompts are now v1 content.
        assert (await mgr.load_prompts("agent_a")) == {"system": "v1"}

        # Rollback also created a backup snapshot of the just-overwritten v2
        # (so the rollback itself is reversible).
        assert "prompts_version" in out["backup_versions"]

    async def test_rollback_unknown_version_raises(self, manager_provider):
        mgr, _ = manager_provider
        await mgr.promote_if_better(
            agent_type="agent_a",
            candidate_prompts={"system": "v1"},
            candidate_demos=None,
            baseline_score=0.0,
            candidate_score=0.5,
        )

        with pytest.raises(ValueError, match="prompts version 99"):
            await mgr.rollback_to_version("agent_a", prompts_version=99)

    async def test_rollback_with_no_args_raises(self, manager_provider):
        mgr, _ = manager_provider
        with pytest.raises(ValueError, match="nothing to restore"):
            await mgr.rollback_to_version("agent_a")
