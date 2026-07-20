"""In-process tests for the knowledge-agent admin routes.

Each test mounts the real router and executes the live handler body with
the agent class stubbed at its import seam (recorder classes), asserting
the exact agent-input object the handler constructs — including the
route-field → agent-field renames (``relation_filter`` →
``relation_allowlist``, ``max_nodes`` → ``max_edges``, ``top_k`` →
``top_k_per_tenant``) — and the exact response mapping. The memory /
graph / registry wiring helpers are stubbed as recorders so the tests
also pin which agent_name namespace each route injects.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import knowledge as knowledge_router

TENANT = "acme:prod"


@pytest.fixture
def harness(monkeypatch):
    """Real router + stubbed memory/graph/registry seams."""
    app = FastAPI()
    app.include_router(knowledge_router.router, prefix="/admin")
    cm = MagicMock(name="config_manager")
    app.dependency_overrides[knowledge_router._get_config_manager] = lambda: cm

    mm_sentinel = MagicMock(name="memory_manager")
    registry_sentinel = MagicMock(name="knowledge_registry")
    calls = {"factory": [], "inject": [], "bind": []}

    def _factory(tid):
        calls["factory"].append(tid)
        return mm_sentinel

    def _inject(agent, tenant_id, agent_name):
        calls["inject"].append((agent, tenant_id, agent_name))
        return mm_sentinel

    def _bind(agent, tenant_id):
        calls["bind"].append((agent, tenant_id))

    monkeypatch.setattr(knowledge_router, "_build_factory", _factory)
    monkeypatch.setattr(knowledge_router, "_inject_memory", _inject)
    monkeypatch.setattr(knowledge_router, "_bind_graph", _bind)
    monkeypatch.setattr(
        knowledge_router, "_build_default_registry", lambda: registry_sentinel
    )

    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            cm=cm,
            mm=mm_sentinel,
            registry=registry_sentinel,
            factory=_factory,
            calls=calls,
        )


def _install_recorder(monkeypatch, dotted_class, out_payload, raise_exc=None):
    """Replace the agent class at its import seam with a recorder whose
    ``_process_impl`` captures the (real) input model instance."""
    recorded: dict = {}

    class _RecorderAgent:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs
            recorded["agent"] = self

        async def _process_impl(self, input_obj):
            recorded["input"] = input_obj
            if raise_exc is not None:
                raise raise_exc
            return SimpleNamespace(model_dump=lambda: out_payload)

    monkeypatch.setattr(dotted_class, _RecorderAgent)
    return recorded


@pytest.mark.unit
@pytest.mark.ci_fast
class TestKGTraverse:
    def test_remaps_filter_and_bounds_exactly(self, harness, monkeypatch):
        from cogniverse_agents.kg_traversal_agent import KGTraversalInput

        payload = {
            "nodes": [{"subject_key": "policy:refunds", "label": "Refund policy"}],
            "edges": [{"relation": "mentions"}],
            "truncated": False,
        }
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.kg_traversal_agent.KnowledgeGraphTraversalAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/kg/traverse",
            json={
                "start_subject_key": "policy:refunds",
                "relation_filter": ["mentions", "cites"],
                "max_depth": 4,
                "max_nodes": 77,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, KGTraversalInput)
        assert inp.tenant_id == TENANT
        assert inp.start_subject_key == "policy:refunds"
        assert inp.relation_allowlist == ["mentions", "cites"]
        assert inp.max_depth == 4
        assert inp.max_edges == 77
        # The route-level names must NOT leak through to the agent input.
        assert not hasattr(inp, "relation_filter")
        assert not hasattr(inp, "max_nodes")

        assert set(recorded["init"]) == {"deps"}
        assert recorded["init"]["deps"].tenant_id == TENANT
        assert harness.calls["inject"] == [
            (recorded["agent"], TENANT, "kg_traversal_agent")
        ]
        assert harness.calls["bind"] == [(recorded["agent"], TENANT)]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestFederatedQuery:
    def test_translates_top_k_and_threads_actor(self, harness, monkeypatch):
        from cogniverse_agents.federated_query_agent import FederatedQueryInput

        payload = {"results_by_tenant": {"acme:a": [], "acme:b": []}, "total": 0}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.federated_query_agent.FederatedQueryAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/federated/query",
            json={
                "query": "refund policy",
                "tenant_ids": ["acme:a", "acme:b"],
                "actor_role": "org_admin",
                "actor_id": "auditor-7",
                "top_k": 25,
                "agent_name_filter": "_promoted",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, FederatedQueryInput)
        assert inp.tenant_id == TENANT
        assert inp.query == "refund policy"
        assert inp.tenant_ids == ["acme:a", "acme:b"]
        assert inp.actor_role == "org_admin"
        assert inp.actor_id == "auditor-7"
        assert inp.top_k_per_tenant == 25
        assert inp.agent_name_filter == "_promoted"
        assert not hasattr(inp, "top_k")

        assert set(recorded["init"]) == {"deps", "memory_manager_factory", "registry"}
        assert recorded["init"]["deps"].tenant_id == TENANT
        assert recorded["init"]["memory_manager_factory"] is harness.factory
        assert recorded["init"]["registry"] is harness.registry


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMultiDocSynthesize:
    def test_builds_documents_and_threads_llm_config(
        self, harness, monkeypatch, tmp_path
    ):
        from cogniverse_agents.multi_document_synthesis_agent import (
            DocumentRef,
            MultiDocSynthesisInput,
        )
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "llm_config": {
                        "primary": {
                            "model": "openai/test-model",
                            "api_base": "http://vllm.local:8000/v1",
                            "api_key": "sk-local",
                            "temperature": 0.0,
                            "max_tokens": 512,
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("COGNIVERSE_CONFIG", str(cfg_file))

        payload = {"answer": "alpha beats beta", "citations": [{"memory_id": "mem-9"}]}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.multi_document_synthesis_agent.MultiDocumentSynthesisAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/synthesis/multi_doc",
            json={
                "query": "compare the two designs",
                "documents": [
                    {"content": "alpha", "label": "d1"},
                    {"memory_id": "mem-9"},
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, MultiDocSynthesisInput)
        assert inp.tenant_id == TENANT
        assert inp.query == "compare the two designs"
        assert inp.documents == [
            DocumentRef(content="alpha", label="d1"),
            DocumentRef(memory_id="mem-9"),
        ]
        assert inp.rlm is None

        assert set(recorded["init"]) == {"deps", "llm_config"}
        assert recorded["init"]["deps"].tenant_id == TENANT
        assert recorded["init"]["llm_config"] == LLMEndpointConfig(
            model="openai/test-model",
            api_base="http://vllm.local:8000/v1",
            api_key="sk-local",
            temperature=0.0,
            max_tokens=512,
        )
        assert harness.calls["inject"] == [
            (recorded["agent"], TENANT, "multi_document_synthesis_agent")
        ]
        assert harness.calls["bind"] == [(recorded["agent"], TENANT)]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestContradictionReconcile:
    def test_wires_memory_attrs_and_builds_input(self, harness, monkeypatch):
        from cogniverse_agents.contradiction_reconciliation_agent import (
            ContradictionReconciliationInput,
        )

        payload = {"resolution": "prefer_newest", "winner_id": "m2"}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.contradiction_reconciliation_agent."
            "ContradictionReconciliationAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/contradictions/reconcile",
            json={
                "target_kind": "entity_fact",
                "conflict_member_ids": ["m1", "m2"],
                "policy_override": "prefer_newest",
                "subject_key": "policy:refunds",
                "predicate": "allows",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, ContradictionReconciliationInput)
        assert inp.tenant_id == TENANT
        assert inp.target_kind == "entity_fact"
        assert inp.conflict_member_ids == ["m1", "m2"]
        assert inp.policy_override == "prefer_newest"
        assert inp.subject_key == "policy:refunds"
        assert inp.predicate == "allows"

        assert set(recorded["init"]) == {"deps", "registry"}
        assert recorded["init"]["registry"] is harness.registry
        # This agent takes no factory arg — the handler wires the mixin
        # attributes directly from the per-tenant factory.
        agent = recorded["agent"]
        assert agent.memory_manager is harness.mm
        assert agent._memory_initialized is True
        assert agent._memory_tenant_id == TENANT
        assert agent._memory_agent_name == "contradiction_reconciliation_agent"
        assert harness.calls["factory"] == [TENANT]
        assert harness.calls["bind"] == [(agent, TENANT)]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCrossTenantCompare:
    def test_builds_input_exactly(self, harness, monkeypatch):
        from cogniverse_agents.cross_tenant_comparison_agent import (
            CrossTenantComparisonInput,
        )

        payload = {"comparison": {"acme:a": "x", "acme:b": "y"}}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.cross_tenant_comparison_agent.CrossTenantComparisonAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/cross_tenant/compare",
            json={
                "subject_key": "policy:refunds",
                "tenant_ids": ["acme:a", "acme:b"],
                "actor_role": "org_admin",
                "actor_id": "auditor-7",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, CrossTenantComparisonInput)
        assert inp.tenant_id == TENANT
        assert inp.subject_key == "policy:refunds"
        assert inp.tenant_ids == ["acme:a", "acme:b"]
        assert inp.actor_role == "org_admin"
        assert inp.actor_id == "auditor-7"

        assert set(recorded["init"]) == {"deps", "memory_manager_factory", "registry"}
        assert recorded["init"]["memory_manager_factory"] is harness.factory
        assert recorded["init"]["registry"] is harness.registry

    def test_acl_rejection_maps_to_403(self, harness, monkeypatch):
        from cogniverse_core.memory.federation import ACLRejected

        _install_recorder(
            monkeypatch,
            "cogniverse_agents.cross_tenant_comparison_agent.CrossTenantComparisonAgent",
            {},
            raise_exc=ACLRejected("actor_role 'user' cannot read across tenants"),
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/cross_tenant/compare",
            json={
                "subject_key": "policy:refunds",
                "tenant_ids": ["acme:a", "acme:b"],
                "actor_role": "user",
                "actor_id": "intruder",
            },
        )
        assert resp.status_code == 403
        assert resp.json() == {"detail": "actor_role 'user' cannot read across tenants"}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestTemporalReason:
    def test_builds_windows_exactly(self, harness, monkeypatch):
        from cogniverse_agents.temporal_reasoning_agent import (
            TemporalReasoningInput,
            TimeWindow,
        )

        payload = {"trajectory": "expanded", "windows_compared": 2}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.temporal_reasoning_agent.TemporalReasoningAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/temporal/reason",
            json={
                "subject_key": "policy:refunds",
                "windows": [
                    {
                        "label": "Q1",
                        "start": "2026-01-01T00:00:00+00:00",
                        "end": "2026-04-01T00:00:00+00:00",
                    },
                    {"label": "Q2", "start": "2026-04-01T00:00:00+00:00"},
                ],
                "agent_name_filter": "_promoted",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, TemporalReasoningInput)
        assert inp.tenant_id == TENANT
        assert inp.subject_key == "policy:refunds"
        assert inp.windows == [
            TimeWindow(
                label="Q1",
                start="2026-01-01T00:00:00+00:00",
                end="2026-04-01T00:00:00+00:00",
            ),
            TimeWindow(label="Q2", start="2026-04-01T00:00:00+00:00"),
        ]
        assert inp.agent_name_filter == "_promoted"

        assert set(recorded["init"]) == {"deps", "memory_manager_factory"}
        assert recorded["init"]["deps"].tenant_id == TENANT
        assert recorded["init"]["memory_manager_factory"] is harness.factory
        assert harness.calls["bind"] == [(recorded["agent"], TENANT)]

    def test_single_window_rejected_at_validation(self, harness):
        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/temporal/reason",
            json={
                "subject_key": "policy:refunds",
                "windows": [{"label": "Q1", "start": "2026-01-01T00:00:00+00:00"}],
            },
        )
        assert resp.status_code == 422


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAuditExplain:
    def test_builds_input_and_threads_factory(self, harness, monkeypatch):
        from cogniverse_agents.audit_explanation_agent import AuditExplanationInput

        payload = {"explanation": "because X", "trust_score": 0.8}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.audit_explanation_agent.AuditExplanationAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/audit/explain",
            json={
                "answer_memory_id": "mem-answer-1",
                "include_trust": True,
                "include_contradictions": False,
                "max_chain_depth": 7,
                "max_chain_nodes": 40,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, AuditExplanationInput)
        assert inp.tenant_id == TENANT
        assert inp.answer_memory_id == "mem-answer-1"
        assert inp.include_contradictions is False
        assert inp.max_chain_depth == 7
        assert inp.max_chain_nodes == 40

        assert set(recorded["init"]) == {"deps", "memory_manager_factory"}
        assert recorded["init"]["deps"].tenant_id == TENANT
        assert recorded["init"]["memory_manager_factory"] is harness.factory


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCitationTrace:
    def test_builds_input_and_injects_namespace(self, harness, monkeypatch):
        from cogniverse_agents.citation_tracing_agent import CitationTracingInput

        payload = {"primary_sources": ["mem-src-1"], "kg_primary_sources": []}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.citation_tracing_agent.CitationTracingAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/citations/trace",
            json={
                "memory_id": "mem-claim-1",
                "claim_id": "edge-9",
                "max_depth": 6,
                "max_nodes": 30,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, CitationTracingInput)
        assert inp.tenant_id == TENANT
        assert inp.memory_id == "mem-claim-1"
        assert inp.claim_id == "edge-9"
        assert inp.max_depth == 6

        assert set(recorded["init"]) == {"deps"}
        # Memory injected under the citation namespace; graph bound.
        assert (recorded["agent"], TENANT, "citation_tracing_agent") in harness.calls[
            "inject"
        ]
        assert (recorded["agent"], TENANT) in harness.calls["bind"]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestKnowledgeSummarize:
    def test_builds_input_and_threads_deps(self, harness, monkeypatch):
        from cogniverse_agents.knowledge_summarization_agent import (
            KnowledgeSummarizationInput,
        )

        payload = {"summary": "distilled", "promoted_to_org_trunk": False}
        recorded = _install_recorder(
            monkeypatch,
            "cogniverse_agents.knowledge_summarization_agent.KnowledgeSummarizationAgent",
            payload,
        )

        resp = harness.client.post(
            f"/admin/tenants/{TENANT}/knowledge/summarize",
            json={
                "subject_keys": ["policy:refunds"],
                "kinds": ["external_doc"],
                "title": "Refund policy",
                "actor_role": "tenant_admin",
                "actor_id": "tadm",
                "promote": False,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        inp = recorded["input"]
        assert isinstance(inp, KnowledgeSummarizationInput)
        assert inp.tenant_id == TENANT
        assert inp.subject_keys == ["policy:refunds"]
        assert inp.kinds == ["external_doc"]

        assert set(recorded["init"]) == {
            "deps",
            "memory_manager_factory",
            "registry",
            "config_manager",
        }
        assert recorded["init"]["deps"].tenant_id == TENANT
        assert recorded["init"]["registry"] is harness.registry
        assert recorded["init"]["config_manager"] is harness.cm
        assert (recorded["agent"], TENANT) in harness.calls["bind"]
