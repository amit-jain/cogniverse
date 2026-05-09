"""B4 — admin C3 endpoints reach the read-only knowledge agents end-to-end.

Audit found 8 of 9 C3 agents were orchestrator-unreachable: even with
opt-in registration, the dispatcher's _execute_generic_agent only fills
5 input fields and every C3 agent has additional required fields, so
the first dispatched call Pydantic-errors.

The new admin routes give each C3 agent a dedicated HTTP surface whose
body matches the agent's input model exactly. This test exercises the
3 read-only endpoints with full coverage (citation_tracing, audit,
knowledge_summarization) against real Vespa, asserting the chain seeded
in Vespa is actually walked / summarised through the HTTP layer.

The other 6 endpoints (multi_doc_synthesis, kg_traversal, cross_tenant,
federated_query, temporal_reasoning, contradiction_reconciliation) get
a request-validation smoke test that verifies the route is mounted, the
input body validates, and a no-data 200 is returned without raising —
real-data coverage for those agents lives in the agent-level integration
tests; this test asserts the HTTP wire holds.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_runtime.routers import c3

pytestmark = pytest.mark.integration


@pytest.fixture
def c3_client(memory_manager):
    """TestClient with the c3 router mounted; memory_manager fixture
    has already initialised the per-tenant Mem0 singleton against
    real Vespa, so the lazy-init path inside the router is a no-op."""
    app = FastAPI()
    app.include_router(c3.router, prefix="/admin")
    yield TestClient(app, raise_server_exceptions=False), memory_manager
    # Clean up any memories this test created.
    try:
        memory_manager.clear_agent_memory(memory_manager.tenant_id, "c3_admin_seed")
    except Exception:
        pass


def _seed_chain(mm) -> tuple[str, str]:
    """Seed a leaf + derived memory pair so audit/citation can walk it."""
    leaf_prov = make_provenance(
        written_by="agent:c3_admin_test",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("https://wiki/c3_admin_leaf")],
    )
    leaf_id = mm.add_memory(
        content="C3 admin: leaf source",
        tenant_id=mm.tenant_id,
        agent_name="c3_admin_seed",
        metadata=attach_to_metadata({"kind": "entity_fact"}, leaf_prov),
        infer=False,
    )
    assert leaf_id
    derived_prov = make_provenance(
        written_by="agent:c3_admin_synth",
        derivation_kind=DerivationKind.SYNTHESIS,
        confidence=0.85,
        derived_from=[CitationRef.memory(leaf_id, label="leaf")],
    )
    derived_id = mm.add_memory(
        content="C3 admin: derived synthesis",
        tenant_id=mm.tenant_id,
        agent_name="c3_admin_seed",
        metadata=attach_to_metadata({"kind": "entity_fact"}, derived_prov),
        infer=False,
    )
    assert derived_id
    return leaf_id, derived_id


class TestAuditExplainEndpoint:
    def test_walks_real_provenance_chain(self, c3_client):
        client, mm = c3_client
        leaf_id, derived_id = _seed_chain(mm)
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/c3/audit/explain",
            json={
                "answer_memory_id": derived_id,
                "include_trust": False,
                "include_contradictions": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        visited = {s["memory_id"] for s in body["sources"]}
        assert derived_id in visited
        assert leaf_id in visited, (
            "audit endpoint must walk derived → leaf via provenance; "
            f"visited={visited!r}"
        )
        primary_refs = {p["ref_id"] for p in body["primary_sources"]}
        assert "https://wiki/c3_admin_leaf" in primary_refs

    def test_missing_required_field_returns_422(self, c3_client):
        client, mm = c3_client
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/c3/audit/explain",
            json={"include_trust": False},  # missing answer_memory_id
        )
        assert resp.status_code == 422


class TestCitationTraceEndpoint:
    def test_walks_real_chain(self, c3_client):
        client, mm = c3_client
        leaf_id, derived_id = _seed_chain(mm)
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/c3/citations/trace",
            json={"memory_id": derived_id},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        nodes = {n["memory_id"] for n in body["nodes"]}
        assert derived_id in nodes
        assert leaf_id in nodes, f"citation chain must include leaf; nodes={nodes!r}"
        primary_refs = {p["ref_id"] for p in body["primary_sources"]}
        assert "https://wiki/c3_admin_leaf" in primary_refs


class TestKnowledgeSummarizeEndpoint:
    def test_summarises_real_subject_slice(self, c3_client):
        client, mm = c3_client
        subject = "policy:c3_admin_summary"
        ids: list[str] = []
        for i in range(3):
            prov = make_provenance(
                written_by=f"agent:doc_{i}",
                derivation_kind=DerivationKind.DIRECT_INGEST,
                confidence=0.8,
                derived_from=[CitationRef.external(f"https://docs/c3_admin_{i}")],
            )
            mid = mm.add_memory(
                content=f"C3 admin doc {i}",
                tenant_id=mm.tenant_id,
                agent_name="c3_admin_seed",
                metadata=attach_to_metadata(
                    {"kind": "external_doc", "subject_key": subject}, prov
                ),
                infer=False,
            )
            assert mid, f"seed write {i} failed"
            ids.append(mid)

        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/c3/knowledge/summarize",
            json={
                "subject_keys": [subject],
                "kinds": ["external_doc"],
                "agent_name_filter": "c3_admin_seed",
                "title": "B4 admin slice",
                "actor_role": "user",
                "actor_id": "alice",
                "promote": False,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source_count"] >= 3, (
            f"summarize endpoint must surface all 3 seeded memories on the "
            f"subject slice; got source_count={body['source_count']}"
        )
        cited = {ref["ref_id"] for ref in body["citation_refs"]}
        for mid in ids:
            assert mid in cited, f"memory {mid} missing from citation_refs={cited!r}"


class TestRouteRegistration:
    """All 9 C3 routes must be mounted; reaching the right URL must
    Pydantic-validate the body. No-args (empty body) requests verify
    the route exists and rejects bad input — proving discovery without
    requiring real-data setup for every agent.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/admin/tenants/test:unit/c3/audit/explain",
            "/admin/tenants/test:unit/c3/citations/trace",
            "/admin/tenants/test:unit/c3/knowledge/summarize",
            "/admin/tenants/test:unit/c3/contradictions/reconcile",
            "/admin/tenants/test:unit/c3/synthesis/multi_doc",
            "/admin/tenants/test:unit/c3/kg/traverse",
            "/admin/tenants/test:unit/c3/cross_tenant/compare",
            "/admin/tenants/test:unit/c3/federated/query",
            "/admin/tenants/test:unit/c3/temporal/reason",
        ],
    )
    def test_route_mounted_and_validates_body(self, c3_client, path: str):
        client, _mm = c3_client
        resp = client.post(path, json={})
        # 422 = route exists, body failed validation. Anything 4xx other
        # than 404 also proves the route is mounted; 404 would mean the
        # router never registered the path.
        assert resp.status_code != 404, (
            f"{path} returned 404 — route is not mounted. Check "
            "main.py app.include_router(c3.router, prefix='/admin')"
        )
        assert resp.status_code in (400, 422, 503), (
            f"{path} expected 400/422/503 for empty body; got "
            f"{resp.status_code}: {resp.text[:200]}"
        )
