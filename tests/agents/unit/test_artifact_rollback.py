"""Unit tests for snapshot-on-promote and rollback to a prior version."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


class FakeStore:
    """Captures dataset writes; tracks calls to assert snapshot semantics."""

    def __init__(self):
        self.created: dict[str, pd.DataFrame] = {}
        self.create_calls: list[str] = []
        self.append_calls: list[tuple[str, pd.DataFrame]] = []

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

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

    async def append_to_dataset(
        self, name: str, data: pd.DataFrame, metadata: dict | None = None
    ) -> None:
        if name not in self.created:
            raise KeyError(name)
        self.append_calls.append((name, data))
        self.created[name] = pd.concat([self.created[name], data], ignore_index=True)


class FakeProvider:
    def __init__(self):
        self.datasets = FakeStore()


def _ds_prompts(agent: str) -> str:
    return f"dspy-prompts-acme:acme-{agent}"


def _ds_demos(agent: str) -> str:
    return f"dspy-demos-acme:acme-{agent}"


def _ds_prompts_v(agent: str, v: int) -> str:
    return f"dspy-prompts-acme:acme-{agent}-v{v}"


def _ds_demos_v(agent: str, v: int) -> str:
    return f"dspy-demos-acme:acme-{agent}-v{v}"


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


class _BlobStore:
    """Blob-focused fake: delete commits immediately, create can be armed
    to fail with specific exceptions (in order) to exercise torn writes."""

    def __init__(self):
        self.datasets: dict[str, pd.DataFrame] = {}
        self.create_failures: list[Exception] = []

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.create_failures:
            raise self.create_failures.pop(0)
        self.datasets[name] = data.copy()
        return f"id::{name}"

    async def delete_dataset(self, name: str) -> bool:
        self.datasets.pop(name, None)
        return True

    async def get_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.datasets:
            raise KeyError(name)
        return self.datasets[name]


class _BlobProvider:
    def __init__(self):
        self.datasets = _BlobStore()


@pytest.mark.asyncio
class TestSaveBlobCompensation:
    """A failed overwrite must not destroy the previously-saved blob.

    save_blob deletes the dataset before re-creating it (last-write-wins,
    one row per blob); a create failure after the committed delete would
    otherwise leave NO blob at all, and the next load would silently serve
    defaults as if the tenant had never been optimized."""

    async def test_create_failure_restores_previous_content(self):
        provider = _BlobProvider()
        manager = ArtifactManager(provider, tenant_id="acme")

        await manager.save_blob("config", "gateway_thresholds", "OLD")
        provider.datasets.create_failures = [ConnectionError("boom-new")]

        with pytest.raises(ConnectionError, match="boom-new"):
            await manager.save_blob("config", "gateway_thresholds", "NEW")

        assert await manager.load_blob("config", "gateway_thresholds") == "OLD"

    async def test_create_failure_with_no_previous_blob_re_raises(self):
        provider = _BlobProvider()
        manager = ArtifactManager(provider, tenant_id="acme")
        provider.datasets.create_failures = [ConnectionError("boom-new")]

        with pytest.raises(ConnectionError, match="boom-new"):
            await manager.save_blob("config", "gateway_thresholds", "NEW")

        assert await manager.load_blob("config", "gateway_thresholds") is None

    async def test_restore_failure_still_raises_the_original_error(self):
        provider = _BlobProvider()
        manager = ArtifactManager(provider, tenant_id="acme")

        await manager.save_blob("config", "gateway_thresholds", "OLD")
        provider.datasets.create_failures = [
            ConnectionError("boom-new"),
            ConnectionError("boom-restore"),
        ]

        with pytest.raises(ConnectionError, match="boom-new"):
            await manager.save_blob("config", "gateway_thresholds", "NEW")

    async def test_happy_path_keeps_single_row_overwrite(self):
        provider = _BlobProvider()
        manager = ArtifactManager(provider, tenant_id="acme")

        await manager.save_blob("config", "k", "v1")
        await manager.save_blob("config", "k", "v2")

        assert await manager.load_blob("config", "k") == "v2"
        name = manager._blob_dataset_name("config", "k")
        assert len(provider.datasets.datasets[name]) == 1
