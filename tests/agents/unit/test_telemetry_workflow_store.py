"""Unit tests for TelemetryWorkflowStore over its demonstration+blob layout.

The store's save/load logic is exercised against an in-memory ArtifactManager
double that mirrors the real channels: ``save_demonstrations``/
``load_demonstrations`` (executions, agent profiles) and ``save_blob``/
``load_blob`` (query patterns, templates), keyed by tenant. The real
Phoenix/ArtifactManager round-trip is covered by the integration suite.
"""

from datetime import datetime

import pytest

from cogniverse_agents.workflow.telemetry_workflow_store import TelemetryWorkflowStore
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
    WorkflowTemplate,
)


class _FakeArtifactManager:
    """Async demonstration + blob store backed by shared dicts, per tenant."""

    def __init__(self, tenant_id, demos, blobs):
        self._tenant = tenant_id
        self._demos = demos  # {(tenant, kind): [demo, ...]}
        self._blobs = blobs  # {(tenant, kind, key): content}

    async def save_demonstrations(self, kind, demos):
        self._demos[(self._tenant, kind)] = list(demos)
        return f"ds_{kind}"

    async def load_demonstrations(self, kind):
        return self._demos.get((self._tenant, kind))

    async def save_blob(self, kind, key, content):
        self._blobs[(self._tenant, kind, key)] = content
        return f"ds_{kind}_{key}"

    async def load_blob(self, kind, key):
        return self._blobs.get((self._tenant, kind, key))


@pytest.fixture
def store():
    s = TelemetryWorkflowStore(telemetry_provider=object())
    demos: dict = {}
    blobs: dict = {}
    s._am = lambda tenant_id: _FakeArtifactManager(tenant_id, demos, blobs)
    return s


def _execution(
    workflow_id="wf_1", query="find cats", query_type="search"
) -> WorkflowExecution:
    return WorkflowExecution(
        workflow_id=workflow_id,
        query=query,
        query_type=query_type,
        execution_time=1.5,
        success=True,
        agent_sequence=["routing", "video_search"],
        task_count=2,
        parallel_efficiency=0.8,
        confidence_score=0.91,
        user_satisfaction=0.75,
        error_details=None,
        timestamp=datetime(2026, 5, 26, 12, 0, 0),
        metadata={"source": "test"},
    )


class TestExecutions:
    async def test_save_then_load_round_trip_exact(self, store):
        original = _execution()
        await store.save_executions("acme:prod", [original])

        loaded = await store.load_executions("acme:prod")
        assert loaded == [original]
        # Spot-check the typed fields survived the demonstration round-trip.
        assert loaded[0].confidence_score == 0.91
        assert loaded[0].agent_sequence == ["routing", "video_search"]
        assert loaded[0].timestamp == datetime(2026, 5, 26, 12, 0, 0)
        assert loaded[0].metadata == {"source": "test"}

    async def test_save_replaces_previous_set(self, store):
        await store.save_executions("t:t", [_execution(workflow_id="old")])
        await store.save_executions("t:t", [_execution(workflow_id="new")])
        loaded = await store.load_executions("t:t")
        assert [e.workflow_id for e in loaded] == ["new"]

    async def test_tenant_isolation(self, store):
        await store.save_executions("acme:prod", [_execution()])
        assert await store.load_executions("globex:prod") == []

    async def test_load_missing_returns_empty(self, store):
        assert await store.load_executions("t:t") == []


class TestAgentProfiles:
    async def test_round_trip_exact(self, store):
        profile = AgentPerformance(
            agent_name="video_search",
            total_executions=10,
            successful_executions=9,
            average_execution_time=2.3,
            average_confidence=0.88,
            error_rate=0.1,
            preferred_query_types=["visual", "temporal"],
            performance_trend="improving",
            last_updated=datetime(2026, 5, 26, 9, 30, 0),
        )
        await store.save_agent_profiles("t:t", [profile])
        loaded = await store.load_agent_profiles("t:t")
        assert loaded == [profile]
        assert loaded[0].preferred_query_types == ["visual", "temporal"]
        assert loaded[0].performance_trend == "improving"

    async def test_load_missing_returns_empty(self, store):
        assert await store.load_agent_profiles("t:t") == []


class TestQueryPatterns:
    async def test_round_trip_exact(self, store):
        patterns = {"search": ["find", "show me"], "summary": ["summarize"]}
        await store.save_query_patterns("t:t", patterns)
        assert await store.load_query_patterns("t:t") == patterns

    async def test_load_missing_returns_empty_dict(self, store):
        assert await store.load_query_patterns("t:t") == {}


class TestTemplates:
    def _template(self, template_id="fast_path") -> WorkflowTemplate:
        return WorkflowTemplate(
            template_id=template_id,
            name="Fast Path",
            description="single-agent search",
            query_patterns=["find *"],
            task_sequence=[{"agent": "video_search"}],
            expected_execution_time=1.2,
            success_rate=0.95,
            usage_count=3,
            created_at=datetime(2026, 5, 1, 0, 0, 0),
            last_used=datetime(2026, 5, 20, 0, 0, 0),
        )

    async def test_save_load_delete_round_trip(self, store):
        tid = await store.save_template("acme:prod", self._template())
        assert tid == "fast_path"

        loaded = await store.load_templates("acme:prod")
        assert loaded == [self._template()]
        assert loaded[0].query_patterns == ["find *"]
        assert loaded[0].task_sequence == [{"agent": "video_search"}]

        assert await store.delete_template("acme:prod", "fast_path") is True
        assert await store.load_templates("acme:prod") == []

    async def test_delete_missing_returns_false(self, store):
        assert await store.delete_template("acme:prod", "nope") is False

    async def test_two_templates_indexed(self, store):
        await store.save_template("t:t", self._template("a"))
        await store.save_template("t:t", self._template("b"))
        loaded = await store.load_templates("t:t")
        assert sorted(t.template_id for t in loaded) == ["a", "b"]

    async def test_torn_delete_tombstones_blob_before_index(self, store):
        """delete_template tombstones the blob BEFORE removing it from the
        index; a failed index write must leave the template skipped by
        load_templates and its blob an empty tombstone, never a non-empty
        orphan the index still points at."""
        from cogniverse_agents.workflow import telemetry_workflow_store as tws

        await store.save_template("acme:prod", self._template("keep"))
        await store.save_template("acme:prod", self._template("drop"))

        real_am = store._am
        fail = {"on": False}

        def _am(tenant_id):
            am = real_am(tenant_id)
            real_save_blob = am.save_blob

            async def _save_blob(kind, key, content):
                if fail["on"] and key == tws._TEMPLATE_INDEX_KEY:
                    raise ConnectionError("phoenix down on index write")
                return await real_save_blob(kind, key, content)

            am.save_blob = _save_blob
            return am

        store._am = _am
        fail["on"] = True
        with pytest.raises(ConnectionError):
            await store.delete_template("acme:prod", "drop")
        fail["on"] = False

        # The dropped template is not listed (its blob is an empty tombstone the
        # loader skips), and no live content is orphaned under it.
        loaded = await store.load_templates("acme:prod")
        assert [t.template_id for t in loaded] == ["keep"]
        drop_blob = await store._am("acme:prod").load_blob(
            tws._BLOB_KIND, tws._template_key("drop")
        )
        assert drop_blob == ""


async def test_health_and_stats(store):
    assert store.health_check() is True
    await store.save_executions("t:t", [_execution()])
    stats = store.get_stats()
    assert stats["backend"] == "telemetry"
    assert stats["tenants_cached"] == 0  # fixture overrides _am, never populating cache


def _profile(name: str) -> AgentPerformance:
    return AgentPerformance(
        agent_name=name,
        total_executions=5,
        successful_executions=4,
        average_execution_time=1.1,
        average_confidence=0.8,
        error_rate=0.2,
        preferred_query_types=["search"],
        performance_trend="stable",
        last_updated=datetime(2026, 5, 26, 9, 30, 0),
    )


class TestSaveLearningCorpus:
    """save_learning_corpus writes the three corpora as one unit: executions
    (the only corpus referencing agents) go last, and any write failure restores
    the previous corpus so the orchestrator never reads a torn learning set."""

    async def test_write_failure_restores_previous_corpus(self, store):
        tenant = "acme:prod"
        # Seed a prior corpus.
        await store.save_learning_corpus(
            tenant,
            [_execution(workflow_id="old")],
            [_profile("agent_a")],
            {"s": ["q0"]},
        )

        # Fail only the FORWARD executions save (the last write); the restore's
        # executions save succeeds.
        real_save_exec = store.save_executions
        calls = {"n": 0}

        async def failing_then_ok(tenant_id, executions):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("phoenix down on executions save")
            return await real_save_exec(tenant_id, executions)

        store.save_executions = failing_then_ok
        try:
            with pytest.raises(ConnectionError):
                await store.save_learning_corpus(
                    tenant,
                    [_execution(workflow_id="new")],
                    [_profile("agent_b")],
                    {"s": ["q1"]},
                )
        finally:
            store.save_executions = real_save_exec

        # The whole corpus is back at the prior state — no torn write.
        assert [e.workflow_id for e in await store.load_executions(tenant)] == ["old"]
        assert [p.agent_name for p in await store.load_agent_profiles(tenant)] == [
            "agent_a"
        ]
        assert await store.load_query_patterns(tenant) == {"s": ["q0"]}
