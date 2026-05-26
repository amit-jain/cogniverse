"""Unit tests for TelemetryWorkflowStore CRUD logic.

The store's record/list/get/aggregate/template logic is exercised against an
in-memory ArtifactManager double (keyed by tenant+kind+key, mirroring the real
blob store). The real Phoenix/ArtifactManager round-trip is covered separately
once the store is wired into WorkflowIntelligence.
"""

import pytest

from cogniverse_agents.workflow.telemetry_workflow_store import TelemetryWorkflowStore


class _FakeArtifactManager:
    """Async save_blob/load_blob backed by a shared dict, per tenant."""

    def __init__(self, tenant_id, shared):
        self._tenant = tenant_id
        self._shared = shared

    async def save_blob(self, kind, key, content):
        self._shared[(self._tenant, kind, key)] = content
        return f"ds_{kind}_{key}"

    async def load_blob(self, kind, key):
        return self._shared.get((self._tenant, kind, key))


@pytest.fixture
def store():
    s = TelemetryWorkflowStore(telemetry_provider=object())
    shared: dict = {}
    s._am = lambda tenant_id, _s=shared: _FakeArtifactManager(tenant_id, _s)
    return s


class TestExecutions:
    def test_record_then_get_and_list_round_trip(self, store):
        eid = store.record_execution(
            "acme:prod", "search_then_summarize", "completed", {"steps": 3}
        )
        assert eid.startswith("acme:prod|exec|")

        rec = store.get_execution(eid)
        assert rec is not None
        assert rec.execution_id == eid
        assert rec.tenant_id == "acme:prod"
        assert rec.workflow_name == "search_then_summarize"
        assert rec.status == "completed"
        assert rec.metrics == {"steps": 3}

        listed = store.list_executions("acme:prod")
        assert [r.execution_id for r in listed] == [eid]

    def test_list_filters_by_workflow_name_and_limit(self, store):
        store.record_execution("t:t", "alpha", "completed", {})
        store.record_execution("t:t", "beta", "completed", {})
        store.record_execution("t:t", "alpha", "failed", {})

        alpha = store.list_executions("t:t", workflow_name="alpha")
        assert len(alpha) == 2
        assert {r.workflow_name for r in alpha} == {"alpha"}
        assert len(store.list_executions("t:t", limit=1)) == 1

    def test_tenant_isolation(self, store):
        store.record_execution("acme:prod", "wf", "completed", {})
        assert store.list_executions("globex:prod") == []

    def test_get_missing_execution_returns_none(self, store):
        assert store.get_execution("acme:prod|exec|deadbeef") is None


class TestAgentPerformance:
    def test_stats_aggregate_exact(self, store):
        store.record_agent_performance("t:t", "search", 100.0, True, {})
        store.record_agent_performance("t:t", "search", 300.0, False, {})
        store.record_agent_performance("t:t", "summary", 50.0, True, {})

        stats = store.get_agent_stats("t:t", "search")
        assert stats is not None
        assert stats.total_executions == 2
        assert stats.avg_duration_ms == 200.0
        assert stats.success_rate == 0.5
        assert stats.last_execution is not None

    def test_stats_none_when_no_data(self, store):
        assert store.get_agent_stats("t:t", "search") is None

    def test_list_filters_by_agent_type(self, store):
        store.record_agent_performance("t:t", "search", 1.0, True, {})
        store.record_agent_performance("t:t", "summary", 1.0, True, {})
        assert len(store.list_agent_performance("t:t", agent_type="search")) == 1
        assert len(store.list_agent_performance("t:t")) == 2


class TestTemplates:
    def test_save_get_list_delete_round_trip(self, store):
        tid = store.save_template("acme:prod", "fast_path", {"steps": ["search"]})
        assert tid == "acme:prod:tmpl:fast_path"

        tmpl = store.get_template("acme:prod", "fast_path")
        assert tmpl is not None
        assert tmpl.template_name == "fast_path"
        assert tmpl.config == {"steps": ["search"]}

        assert [t.template_name for t in store.list_templates("acme:prod")] == [
            "fast_path"
        ]

        assert store.delete_template("acme:prod", "fast_path") is True
        assert store.get_template("acme:prod", "fast_path") is None
        assert store.list_templates("acme:prod") == []

    def test_delete_missing_returns_false(self, store):
        assert store.delete_template("acme:prod", "nope") is False

    def test_save_preserves_created_at_on_update(self, store):
        store.save_template("t:t", "x", {"v": 1})
        first = store.get_template("t:t", "x")
        store.save_template("t:t", "x", {"v": 2})
        second = store.get_template("t:t", "x")
        assert second.config == {"v": 2}
        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at


def test_health_and_stats(store):
    assert store.health_check() is True
    store.record_execution("t:t", "wf", "completed", {})
    assert store.get_stats()["backend"] == "telemetry"
