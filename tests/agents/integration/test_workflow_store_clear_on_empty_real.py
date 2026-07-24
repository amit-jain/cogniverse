"""Real-Phoenix coverage for empty-set replacement in the workflow store.

``save_executions`` / ``save_agent_profiles`` must honour "replace with the
empty set": an empty list clears the stored demonstrations so a later load
returns nothing. Without that, ``save_learning_corpus``'s failure compensation
cannot roll an empty-prior corpus back — a forward-written profile survives the
rollback and the orchestrator reads an agent profile whose executions never
persisted.

Exercises the real ``TelemetryWorkflowStore`` resolved through the registry over
a real Phoenix Docker instance; the only injected fault is the outermost
provider call for the executions step.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from cogniverse_agents.workflow.telemetry_workflow_store import _EXECUTIONS_KIND
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
        timestamp=datetime(2026, 5, 26, 12, 0, 0),
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
        last_updated=datetime(2026, 5, 26, 9, 0, 0),
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

        with pytest.raises(ConnectionError):
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
