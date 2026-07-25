"""Real-Phoenix coverage for learning-corpus replacement.

``save_executions`` / ``save_agent_profiles`` must honour "replace with the
empty set": an empty list clears the stored demonstrations so a later load
returns nothing. Without that, ``save_learning_corpus``'s failure compensation
cannot roll an empty-prior corpus back — a forward-written profile survives the
rollback and the orchestrator reads an agent profile whose executions never
persisted.

Exercises the real ``TelemetryWorkflowStore`` resolved through the registry over
a real Phoenix Docker instance, including empty patterns, same-tenant
concurrency, and a provider failure during the executions step.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from cogniverse_agents.workflow.telemetry_workflow_store import (
    _EXECUTIONS_KIND,
    _PROFILES_KIND,
)
from cogniverse_core.registries import WorkflowStoreRegistry
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture
def real_provider(telemetry_manager_with_phoenix):
    return telemetry_manager_with_phoenix.get_provider(tenant_id="wf-clear-test")


def _store(real_provider):
    WorkflowStoreRegistry.clear_cache()
    return WorkflowStoreRegistry.get(
        name="telemetry", config={"telemetry_provider": real_provider}
    )


def _execution(workflow_id: str) -> WorkflowExecution:
    return WorkflowExecution(
        workflow_id=workflow_id,
        query="find cats",
        query_type="video_search",
        execution_time=1.0,
        success=True,
        agent_sequence=["video_search_agent"],
        task_count=1,
        parallel_efficiency=1.0,
        confidence_score=0.9,
        user_satisfaction=0.7,
        error_details=None,
        timestamp=datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc),
        metadata={},
    )


def _profile(name: str) -> AgentPerformance:
    return AgentPerformance(
        agent_name=name,
        total_executions=1,
        successful_executions=1,
        average_execution_time=1.0,
        average_confidence=0.9,
        error_rate=0.0,
        preferred_query_types=["video_search"],
        performance_trend="stable",
        last_updated=datetime(2026, 5, 26, 9, 0, 0, tzinfo=timezone.utc),
    )


class TestEmptyReplacementClears:
    """An empty list must clear the stored set on the real dataset backend."""

    @pytest.mark.asyncio
    async def test_save_empty_profiles_clears_stored(self, real_provider):
        tenant = f"wf-clear-prof-{uuid.uuid4().hex[:8]}"
        store = _store(real_provider)

        await store.save_agent_profiles(tenant, [_profile("agent_a")])
        assert [p.agent_name for p in await store.load_agent_profiles(tenant)] == [
            "agent_a"
        ]

        await store.save_agent_profiles(tenant, [])
        assert await store.load_agent_profiles(tenant) == []

    @pytest.mark.asyncio
    async def test_save_empty_executions_clears_stored(self, real_provider):
        tenant = f"wf-clear-exec-{uuid.uuid4().hex[:8]}"
        store = _store(real_provider)

        await store.save_executions(tenant, [_execution("wf-1")])
        assert [e.workflow_id for e in await store.load_executions(tenant)] == ["wf-1"]

        await store.save_executions(tenant, [])
        assert await store.load_executions(tenant) == []

    @pytest.mark.asyncio
    async def test_empty_patterns_replace_stale_patterns(self, real_provider):
        tenant = f"wf-clear-pattern-{uuid.uuid4().hex[:8]}"
        store = _store(real_provider)

        await store.save_learning_corpus(
            tenant,
            [_execution("wf-old")],
            [_profile("agent_old")],
            {"video_search": ["find old"]},
        )
        await store.save_learning_corpus(
            tenant,
            [_execution("wf-new")],
            [_profile("agent_new")],
            {},
        )

        assert await store.load_query_patterns(tenant) == {}
        assert [item.workflow_id for item in await store.load_executions(tenant)] == [
            "wf-new"
        ]
        assert [
            item.agent_name for item in await store.load_agent_profiles(tenant)
        ] == ["agent_new"]


class TestConcurrentCorpusReplacement:
    @pytest.mark.asyncio
    async def test_same_tenant_concurrent_saves_finish_as_one_corpus(
        self, real_provider
    ):
        tenant = f"wf-concurrent-{uuid.uuid4().hex[:8]}"
        store = _store(real_provider)
        a_profile_written = asyncio.Event()
        release_a = asyncio.Event()
        b_execution_written = asyncio.Event()
        real_save_profiles = store.save_agent_profiles
        real_save_executions = store.save_executions

        async def controlled_profiles(tenant_id, profiles):
            await real_save_profiles(tenant_id, profiles)
            if profiles and profiles[0].agent_name == "agent_a":
                a_profile_written.set()
                await release_a.wait()

        async def observed_executions(tenant_id, executions):
            await real_save_executions(tenant_id, executions)
            if executions and executions[0].workflow_id == "wf-b":
                b_execution_written.set()

        store.save_agent_profiles = controlled_profiles
        store.save_executions = observed_executions
        save_a = asyncio.create_task(
            store.save_learning_corpus(
                tenant,
                [_execution("wf-a")],
                [_profile("agent_a")],
                {"a": ["pattern-a"]},
            )
        )
        await asyncio.wait_for(a_profile_written.wait(), timeout=10)
        save_b = asyncio.create_task(
            store.save_learning_corpus(
                tenant,
                [_execution("wf-b")],
                [_profile("agent_b")],
                {"b": ["pattern-b"]},
            )
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(b_execution_written.wait(), timeout=2)
        release_a.set()
        await asyncio.gather(save_a, save_b)

        assert [item.workflow_id for item in await store.load_executions(tenant)] == [
            "wf-b"
        ]
        assert [
            item.agent_name for item in await store.load_agent_profiles(tenant)
        ] == ["agent_b"]
        assert await store.load_query_patterns(tenant) == {"b": ["pattern-b"]}


class TestEmptyPriorRestoreClearsProfile:
    """A mid-write failure on a first-run tenant rolls the forward writes back to
    the empty prior — no orphan profile survives."""

    @pytest.mark.asyncio
    async def test_mid_write_failure_rolls_back_forward_profile(
        self, real_provider, monkeypatch
    ):
        tenant = f"wf-empty-prior-{uuid.uuid4().hex[:8]}"
        store = _store(real_provider)

        # First-run tenant: nothing stored on any channel.
        assert await store.load_agent_profiles(tenant) == []
        assert await store.load_query_patterns(tenant) == {}
        assert await store.load_executions(tenant) == []

        # Fail exactly the forward executions write at the provider boundary; the
        # profiles write, the patterns write, and the whole restore stay real.
        am = store._am(tenant)
        executions_dataset = am._demo_dataset_name(_EXECUTIONS_KIND)
        real_replace = real_provider.datasets.replace_dataset

        async def replace_or_fail(name, data, metadata=None):
            if name == executions_dataset:
                raise ConnectionError("phoenix down on executions replace")
            return await real_replace(name=name, data=data, metadata=metadata)

        monkeypatch.setattr(real_provider.datasets, "replace_dataset", replace_or_fail)

        with pytest.raises(ConnectionError, match="executions replace"):
            await store.save_learning_corpus(
                tenant,
                [_execution("wf-new")],
                [_profile("agent_leak")],
                {"video_search": ["find *"]},
            )

        # Compensation restores the empty prior on every channel: the
        # forward-written profile is cleared (not left as an orphan), the
        # forward-written patterns are cleared, and no executions persisted.
        assert await store.load_agent_profiles(tenant) == []
        assert await store.load_query_patterns(tenant) == {}
        assert await store.load_executions(tenant) == []

    @pytest.mark.asyncio
    async def test_restore_failure_surfaces_both_errors_and_continues_compensation(
        self, real_provider, monkeypatch
    ):
        tenant = f"wf-restore-fail-{uuid.uuid4().hex[:8]}"
        store = _store(real_provider)
        await store.save_learning_corpus(
            tenant,
            [_execution("wf-old")],
            [_profile("agent_old")],
            {"old": ["pattern-old"]},
        )

        am = store._am(tenant)
        executions_dataset = am._demo_dataset_name(_EXECUTIONS_KIND)
        profiles_dataset = am._demo_dataset_name(_PROFILES_KIND)
        real_replace = real_provider.datasets.replace_dataset
        calls = {"executions": 0, "profiles": 0}

        async def replace_or_fail(name, data, metadata=None):
            if name == executions_dataset:
                calls["executions"] += 1
                if calls["executions"] == 1:
                    raise ConnectionError("forward executions failed")
            if name == profiles_dataset:
                calls["profiles"] += 1
                if calls["profiles"] == 2:
                    raise RuntimeError("profile restore failed")
            return await real_replace(name=name, data=data, metadata=metadata)

        monkeypatch.setattr(real_provider.datasets, "replace_dataset", replace_or_fail)

        with pytest.raises(ExceptionGroup) as exc_info:
            await store.save_learning_corpus(
                tenant,
                [_execution("wf-new")],
                [_profile("agent_new")],
                {"new": ["pattern-new"]},
            )

        assert [type(error) for error in exc_info.value.exceptions] == [
            ConnectionError,
            RuntimeError,
        ]
        assert [str(error) for error in exc_info.value.exceptions] == [
            "forward executions failed",
            "profile restore failed",
        ]
        assert calls == {"executions": 2, "profiles": 2}
        assert [item.workflow_id for item in await store.load_executions(tenant)] == [
            "wf-old"
        ]
        assert await store.load_query_patterns(tenant) == {"old": ["pattern-old"]}
        assert [
            item.agent_name for item in await store.load_agent_profiles(tenant)
        ] == ["agent_new"]
