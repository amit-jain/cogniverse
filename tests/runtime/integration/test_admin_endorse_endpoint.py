"""admin endorse endpoint records a trust delta on a memory.

apply_endorsement was a public API with no caller. The
audit/explanation agent reads endorsement counts off TrustRecords but
nothing in the runtime ever incremented them.

This test exercises the new POST .../memories/{id}/endorse endpoint
end-to-end against real Vespa: an org_admin endorsement bumps the
trust score by +0.20 and increments the endorsements count from 0
to 1, both round-tripped through Vespa metadata.
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
from cogniverse_core.memory.trust import extract_trust
from cogniverse_runtime.routers import admin

pytestmark = pytest.mark.integration


@pytest.fixture
def endorse_client(memory_manager):
    """Mount the admin router. Wire the knowledge_registry onto the
    runtime conftest's memory_manager so writes auto-attach trust —
    the conftest fixture initialises Mem0 without a registry by
    default.
    """
    from cogniverse_core.memory.schema import build_default_registry

    memory_manager._knowledge_registry = build_default_registry()
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    yield TestClient(app, raise_server_exceptions=False), memory_manager
    try:
        memory_manager.clear_agent_memory(memory_manager.tenant_id, "h9_endorse")
    except Exception:
        pass
    memory_manager._knowledge_registry = None


def _seed_with_trust(mm) -> str:
    """Seed a memory whose write path attaches an initial trust record."""
    prov = make_provenance(
        written_by="agent:h9",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("https://wiki/h9")],
    )
    return mm.add_memory(
        content="endorse target",
        tenant_id=mm.tenant_id,
        agent_name="h9_endorse",
        metadata=attach_to_metadata({"kind": "entity_fact"}, prov),
        infer=False,
    )


class TestEndorseEndpoint:
    def test_org_admin_endorsement_round_trips(self, endorse_client):
        client, mm = endorse_client
        mid = _seed_with_trust(mm)
        assert mid

        baseline_rows = mm.get_all_memories(
            tenant_id=mm.tenant_id, agent_name="h9_endorse"
        )
        baseline_row = next((r for r in baseline_rows if str(r.get("id")) == mid), None)
        assert baseline_row is not None
        baseline_trust = extract_trust(baseline_row)
        assert baseline_trust is not None, (
            "schema enforcement should attach an initial trust on entity_fact "
            "writes — without baseline the endorse endpoint has nothing to bump"
        )
        assert baseline_trust.endorsements == 0

        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/memories/{mid}/endorse",
            json={"endorser_role": "org_admin", "actor_id": "alice"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["endorsements"] == 1
        # org_admin delta is +0.20 (clamped at 1.0).
        assert body["new_score"] == pytest.approx(
            min(1.0, baseline_trust.score + 0.20), abs=0.001
        )

        # Real round-trip: re-read via get_all_memories, the new trust
        # must be persisted in metadata.
        post_rows = mm.get_all_memories(tenant_id=mm.tenant_id, agent_name="h9_endorse")
        post_row = next((r for r in post_rows if str(r.get("id")) == mid), None)
        assert post_row is not None
        post_trust = extract_trust(post_row)
        assert post_trust is not None, (
            "trust record was lost on round-trip; the endorse update path is broken"
        )
        assert post_trust.endorsements == 1, (
            f"endorsement count must persist; got {post_trust.endorsements}"
        )
        assert post_trust.score == pytest.approx(body["new_score"], abs=0.001)

    def test_unknown_role_returns_400(self, endorse_client):
        client, mm = endorse_client
        mid = _seed_with_trust(mm)
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/memories/{mid}/endorse",
            json={"endorser_role": "kingmaker", "actor_id": "x"},
        )
        assert resp.status_code == 400

    def test_unknown_memory_returns_404(self, endorse_client):
        client, mm = endorse_client
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/memories/no-such-id/endorse",
            json={"endorser_role": "user", "actor_id": "x"},
        )
        assert resp.status_code == 404

    def test_memory_without_trust_returns_422(self, endorse_client):
        """When a memory is missing a trust record, endorsement returns
        422. We bypass the schema-enforcement write path (which would
        attach a trust record automatically) by writing directly via
        mem0.memory.add with a kind that has no schema entry.
        """
        client, mm = endorse_client
        # Direct write bypasses _enforce_schema_on_write so no trust
        # record is attached.
        result = mm.memory.add(
            "trustless content",
            user_id=mm.tenant_id,
            agent_id="h9_endorse",
            metadata={"kind": "trustless_h9"},
            infer=False,
        )
        rows = (result.get("results") or []) if isinstance(result, dict) else result
        mid = next(
            (str(r.get("id")) for r in rows if isinstance(r, dict) and r.get("id")),
            None,
        )
        if mid is None:
            pytest.skip("Mem0 dropped the no-trust write; cannot test the 422 path")
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/memories/{mid}/endorse",
            json={"endorser_role": "user", "actor_id": "x"},
        )
        assert resp.status_code == 422, resp.text
        assert "no trust record" in resp.json()["detail"]
