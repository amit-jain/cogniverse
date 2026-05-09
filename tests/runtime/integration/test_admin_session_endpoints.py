"""Admin session-cleanup endpoints — round-trip through real Vespa.

Wire-up coverage for the EPHEMERAL_SESSION lifecycle:

  * Per-tenant DELETE /admin/tenants/{tenant_id}/sessions/{session_id}
    drops only the matching session's EPHEMERAL_SESSION rows for that
    one tenant.
  * Cross-tenant POST /admin/sessions/{session_id}/close fans the drop
    out across every warm Mem0 instance.
  * Both endpoints honour the schema gate (a non-EPHEMERAL_SESSION
    memory tagged with the same session_id is left alone).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_runtime.routers import admin

pytestmark = pytest.mark.integration


def _seed_session_memory(
    mm: Mem0MemoryManager, *, session_id: str, content: str = "scratch"
) -> str:
    """Write a session_scratch row tagged with session_id."""
    return mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name="session_scratch_writer",
        metadata={"kind": "session_scratch", "session_id": session_id},
        infer=False,
    )


def _seed_permanent_memory(
    mm: Mem0MemoryManager, *, session_id: str, content: str = "permanent"
) -> str:
    """Write a permanent memory tagged with the SAME session_id.

    The schema gate must NOT delete this on session close — only
    EPHEMERAL_SESSION-kind rows are eligible.
    """
    return mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name="session_scratch_writer",
        metadata={"kind": "tenant_instruction", "session_id": session_id},
        infer=False,
    )


@pytest.fixture
def admin_session_client(memory_manager):
    """TestClient with the admin router mounted on a real Mem0 manager."""
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin._reset_admin_overrides_for_tests()
    yield TestClient(app), memory_manager
    # Clean up any rows this test wrote so the next test starts empty.
    try:
        memory_manager.clear_agent_memory(
            memory_manager.tenant_id, "session_scratch_writer"
        )
    except Exception:
        pass


class TestPerTenantDeleteEndpoint:
    def test_drops_session_kind_keeps_permanent(self, admin_session_client):
        client, mm = admin_session_client
        tenant = mm.tenant_id
        session_a = "s_alpha"
        session_b = "s_beta"

        scratch_a = [_seed_session_memory(mm, session_id=session_a) for _ in range(3)]
        scratch_b = [_seed_session_memory(mm, session_id=session_b) for _ in range(2)]
        permanent = _seed_permanent_memory(mm, session_id=session_a)

        resp = client.delete(f"/admin/tenants/{tenant}/sessions/{session_a}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "dropped"
        assert body["tenant_id"] == tenant
        assert body["session_id"] == session_a
        assert body["deleted_by_kind"] == {"session_scratch": 3}
        assert body["total_deleted"] == 3

        # session_a's scratch rows must be gone; session_b's untouched;
        # the permanent row tagged with session_a must survive (only
        # EPHEMERAL_SESSION-kind rows are eligible).
        surviving = {
            m["id"] for m in mm.get_all_memories(tenant, "session_scratch_writer")
        }
        for sid in scratch_a:
            assert sid not in surviving, f"scratch row {sid} must be hard-deleted"
        for sid in scratch_b:
            assert sid in surviving, f"session_b scratch {sid} must survive"
        assert permanent in surviving, (
            "permanent-kind row must survive the session-end sweep"
        )

    def test_empty_session_id_returns_400(self, admin_session_client):
        client, mm = admin_session_client
        # Path-level empty string is treated as a 404 by FastAPI; a single
        # space hits the body-level validator.
        resp = client.delete(f"/admin/tenants/{mm.tenant_id}/sessions/%20")
        assert resp.status_code == 400
        assert "non-empty" in resp.text.lower()

    def test_unknown_session_returns_zero(self, admin_session_client):
        client, mm = admin_session_client
        resp = client.delete(
            f"/admin/tenants/{mm.tenant_id}/sessions/never_existed_session"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_deleted"] == 0
        assert body["deleted_by_kind"] == {}


class TestFanoutCloseEndpoint:
    def test_close_sweeps_warm_tenants(self, admin_session_client):
        client, mm = admin_session_client
        tenant = mm.tenant_id
        session_id = "s_fanout"

        ids = [_seed_session_memory(mm, session_id=session_id) for _ in range(2)]

        resp = client.post(f"/admin/sessions/{session_id}/close")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "closed"
        assert body["session_id"] == session_id
        # The warm-set sweep must have caught this tenant.
        assert tenant in body["per_tenant"], body
        assert body["per_tenant"][tenant] == {"session_scratch": 2}
        assert body["total_deleted"] == 2

        surviving = {
            m["id"] for m in mm.get_all_memories(tenant, "session_scratch_writer")
        }
        for sid in ids:
            assert sid not in surviving

    def test_close_with_no_session_rows_returns_zero(self, admin_session_client):
        client, _mm = admin_session_client
        resp = client.post("/admin/sessions/no_one_uses_this/close")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_deleted"] == 0
        assert body["per_tenant"] == {}
