"""admin pin/unpin/list endpoints.

Audit caught: PinService was reachable only via direct Python API. The
runtime exposed pin-quota CRUD but no way to actually pin a memory.
This test exercises the new HTTP endpoints end-to-end against real
Vespa, then confirms the pinned memory survives a lifecycle tick (i.e.
the wire from HTTP → PinService → Vespa → lifecycle scheduler holds).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.pinning import PinService
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_runtime.routers import admin

pytestmark = pytest.mark.integration


@pytest.fixture
def admin_pin_client(memory_manager):
    """TestClient with the admin router mounted.

    Mem0MemoryManager is a tenant-keyed singleton — once the
    ``memory_manager`` fixture initialises it for ``test:unit``, any later
    ``Mem0MemoryManager(tenant_id)`` returns that same instance, so the
    admin endpoint's lazy-init path is a no-op here.
    """
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin._reset_admin_overrides_for_tests()
    yield TestClient(app), memory_manager
    # Clean up any pin records this test created so the next test sees
    # an empty pin list.
    try:
        memory_manager.clear_agent_memory(memory_manager.tenant_id, "_pinning")
    except Exception:
        pass


def _seed_memory(mm: Mem0MemoryManager, kind: str = "tenant_instruction") -> str:
    return mm.add_memory(
        content=f"B2 target memory of kind={kind}",
        tenant_id=mm.tenant_id,
        agent_name="b2_pin_target",
        metadata={"kind": kind},
        infer=False,
    )


class TestPinRoundTrip:
    def test_pin_then_list_then_unpin(self, admin_pin_client):
        client, mm = admin_pin_client
        tenant = mm.tenant_id

        target_id = _seed_memory(mm, kind="tenant_instruction")
        assert target_id, "seed memory must persist before pinning"

        # Pin via admin endpoint.
        resp = client.post(
            f"/admin/tenants/{tenant}/memories/{target_id}/pin",
            json={
                "target_kind": "tenant_instruction",
                "pinned_by": "tenant_admin",
                "actor_id": "admin_alpha",
            },
        )
        assert resp.status_code == 200, (
            f"pin endpoint returned {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["target_memory_id"] == target_id
        assert body["pinned_by"] == "tenant_admin"
        assert body["pinned_by_actor"] == "admin_alpha", (
            "pinned_by_actor must round-trip through Vespa metadata; if "
            "this returns 'unknown' the actor_id key is colliding with "
            "Mem0's reserved-keys set again"
        )

        # List endpoint must surface the new pin record.
        resp = client.get(f"/admin/tenants/{tenant}/pins")
        assert resp.status_code == 200
        listed = resp.json()
        assert listed["tenant_id"] == tenant
        target_records = [
            p for p in listed["pins"] if p["target_memory_id"] == target_id
        ]
        assert len(target_records) == 1, (
            f"expected exactly one pin record for {target_id}; "
            f"got pins={listed['pins']!r}"
        )
        assert target_records[0]["pinned_by"] == "tenant_admin"
        assert target_records[0]["pinned_by_actor"] == "admin_alpha"

        # Unpin via admin endpoint.
        resp = client.request(
            "DELETE",
            f"/admin/tenants/{tenant}/memories/{target_id}/pin",
            json={"requester_role": "tenant_admin", "actor_id": "admin_alpha"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["removed"] == 1

        # And the list endpoint reflects the removal.
        resp = client.get(f"/admin/tenants/{tenant}/pins")
        assert resp.status_code == 200
        remaining = [
            p for p in resp.json()["pins"] if p["target_memory_id"] == target_id
        ]
        assert remaining == [], (
            "after DELETE pin, list must no longer include the target; "
            f"remaining={remaining!r}"
        )


class TestAuthorityAndQuota:
    def test_user_pinning_admin_kind_returns_403(self, admin_pin_client):
        """tenant_instruction has pinnable_by=TENANT_ADMIN; user role denied."""
        client, mm = admin_pin_client
        target_id = _seed_memory(mm, kind="tenant_instruction")
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/memories/{target_id}/pin",
            json={
                "target_kind": "tenant_instruction",
                "pinned_by": "user",
                "actor_id": "user_bob",
            },
        )
        assert resp.status_code == 403, (
            f"user attempting to pin tenant_instruction must be rejected; "
            f"got {resp.status_code}: {resp.text}"
        )

    def test_invalid_role_returns_400(self, admin_pin_client):
        client, mm = admin_pin_client
        target_id = _seed_memory(mm, kind="tenant_instruction")
        resp = client.post(
            f"/admin/tenants/{mm.tenant_id}/memories/{target_id}/pin",
            json={
                "target_kind": "tenant_instruction",
                "pinned_by": "superadmin",
                "actor_id": "x",
            },
        )
        assert resp.status_code == 400, resp.text
        assert "invalid role" in resp.json()["detail"]

    def test_unpin_nonexistent_returns_404(self, admin_pin_client):
        client, mm = admin_pin_client
        resp = client.request(
            "DELETE",
            f"/admin/tenants/{mm.tenant_id}/memories/no-such-id/pin",
            json={"requester_role": "tenant_admin", "actor_id": "admin_alpha"},
        )
        assert resp.status_code == 404


class TestLifecycleSurvival:
    """The whole point of pinning: the lifecycle scheduler must skip
    pinned memories. This wires the same pin_lookup callback the
    runtime's main.py constructs for production (PinService.list_pins
    against the per-tenant manager), so a regression that breaks the
    HTTP-pin path also breaks this test.
    """

    def test_pinned_memory_survives_lifecycle_tick(self, admin_pin_client):
        client, mm = admin_pin_client
        tenant = mm.tenant_id
        registry = build_default_registry()

        target_id = _seed_memory(mm, kind="tenant_instruction")

        # Pin via the same HTTP endpoint a real operator would hit.
        resp = client.post(
            f"/admin/tenants/{tenant}/memories/{target_id}/pin",
            json={
                "target_kind": "tenant_instruction",
                "pinned_by": "tenant_admin",
                "actor_id": "admin_alpha",
            },
        )
        assert resp.status_code == 200, resp.text

        def _pin_lookup(manager) -> set:
            return {
                rec.target_memory_id
                for rec in PinService(manager, registry).list_pins(manager.tenant_id)
            }

        scheduler = LifecycleScheduler(
            get_warm_managers=lambda: [mm],
            registry=registry,
            interval_seconds=3600,  # never auto-fires; we drive the tick.
            pin_lookup=_pin_lookup,
        )
        asyncio.run(scheduler.tick_once())

        # The pinned memory must still be retrievable.
        rows = mm.get_all_memories(tenant_id=tenant, agent_name="b2_pin_target")
        survivors = [r for r in rows if str(r.get("id")) == target_id]
        assert survivors, (
            f"pinned memory {target_id} was deleted by lifecycle tick; "
            "the pin → lifecycle wire is broken. "
            "Pin record returned by HTTP endpoint vs what lifecycle saw "
            "must match."
        )
