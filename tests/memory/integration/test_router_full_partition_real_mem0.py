"""Real Vespa-mem0 round-trips proving the messaging and tenant memory
routes walk the whole partition, not just the store's first 100-row page.

Drives the REAL admin and tenant routers over ASGITransport backed by real
Vespa-backed Mem0. Seeds past the 100-row page and asserts exact, complete
results: every linked chat resolves and enqueues, and list/clear by category
returns/deletes exactly the matches sitting beyond the page.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.messaging_auth import GATEWAY_AGENT_NAME, UserTenantMapper
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime import messaging
from cogniverse_runtime.routers import admin, tenant
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

# Canonical org:tenant form so the derived schema matches the single suffix
# the shared fixture deploys (agent_memories_test_tenant); the bare form
# derives the never-deployed double suffix.
_TENANT = "test:tenant"
_USER_MEMORY_AGENT = "_user_memories"


def _poll_and_return(list_fn, expected: int, timeout: float = 90.0):
    """Poll ``list_fn`` until it returns at least ``expected`` rows (or times out).

    Absorbs Vespa indexing latency for a large seed so the route reads a
    fully-visible partition; the returned list length is asserted so a
    genuine shortfall (a dropped-row pagination bug) surfaces as an exact
    number rather than as flakiness.
    """
    deadline = time.monotonic() + timeout
    rows = list_fn()
    while len(rows) < expected and time.monotonic() < deadline:
        time.sleep(3.0)
        rows = list_fn()
    return rows


def _init_manager(tenant_id: str, *, shared_memory_vespa, shared_denseon, cm):
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    return mm


@pytest.fixture(scope="module")
def wired_routers(shared_memory_vespa, shared_denseon):
    """Admin + tenant routers wired to real Vespa-mem0 via ASGITransport."""
    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()

    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
        )
    )
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )

    system_mgr = _init_manager(
        SYSTEM_TENANT_ID,
        shared_memory_vespa=shared_memory_vespa,
        shared_denseon=shared_denseon,
        cm=cm,
    )
    tenant_mgr = _init_manager(
        _TENANT,
        shared_memory_vespa=shared_memory_vespa,
        shared_denseon=shared_denseon,
        cm=cm,
    )

    # Admin messaging routes resolve the SYSTEM manager through this factory;
    # a None module config selects the in-pod outbound queue.
    prev_admin_cm = admin._config_manager
    prev_tenant_cm = tenant._config_manager
    admin.set_system_memory_factory(lambda: system_mgr)
    admin._config_manager = None
    messaging.reset_outbound_queue_for_testing()
    # Tenant memory routes build their manager per request; the pre-initialized
    # singleton for _TENANT is reused, and set_config_manager lets a cold one
    # lazy-init against real Vespa.
    tenant.set_config_manager(cm)

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tenant.router)

    try:
        yield app, system_mgr, tenant_mgr
    finally:
        try:
            tenant_mgr.clear_agent_memory(_TENANT, _USER_MEMORY_AGENT)
        except Exception:
            pass
        admin.set_system_memory_factory(None)
        admin._config_manager = prev_admin_cm
        tenant.set_config_manager(prev_tenant_cm)
        messaging.reset_outbound_queue_for_testing()
        Mem0MemoryManager._instances.clear()


@pytest.mark.integration
def test_send_enqueues_every_linked_chat_past_the_page(wired_routers):
    """A tenant's linked chats all resolve even when other tenants' mappings
    fill the shared SYSTEM partition ahead of them, past Mem0's 100-row page."""
    app, system_mgr, _ = wired_routers
    run = uuid.uuid4().hex[:8]
    target_tenant = f"acme:{run}"
    mapper = UserTenantMapper(system_mgr)

    target_chat_ids = [f"{run}-t{i}" for i in range(5)]
    filler_chat_ids = [f"{run}-f{i}" for i in range(100)]

    # Target mappings first (oldest), then 100 fillers for another tenant —
    # a capped newest-100 read would drop the older target rows entirely.
    for cid in target_chat_ids:
        assert mapper.register_user("telegram", cid, target_tenant) is True
    for cid in filler_chat_ids:
        assert mapper.register_user("telegram", cid, f"globex:{run}") is True

    def _run_rows():
        return [
            row
            for row in system_mgr.get_all_memories(
                tenant_id=SYSTEM_TENANT_ID, agent_name=GATEWAY_AGENT_NAME, limit=None
            )
            if run in str((row.get("metadata") or {}).get("external_user_id", ""))
        ]

    # The full walk must recover every seeded mapping, target and filler.
    assert len(_poll_and_return(_run_rows, 105)) == 105
    seeded_ids = {str(r.get("id")) for r in _run_rows()}
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/admin/messaging/send",
            json={"tenant_id": target_tenant, "message": "job done"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"enqueued": 5}

        drained = client.get("/admin/messaging/outbound/drain").json()["messages"]
        assert sorted(m["chat_id"] for m in drained) == sorted(target_chat_ids)
        assert all(m["text"] == "job done" for m in drained)
    finally:
        for mid in seeded_ids:
            system_mgr.delete_memory(mid, SYSTEM_TENANT_ID, GATEWAY_AGENT_NAME)


@pytest.mark.integration
def test_list_and_clear_category_cover_every_match_past_the_page(wired_routers):
    """Listing and clearing a category must reach every match past the page —
    both scan the partition and filter category in Python, so a capped read
    would silently drop the tail while reporting success."""
    app, _, tenant_mgr = wired_routers
    run = uuid.uuid4().hex[:8]
    bulk_cat = f"bulk-{run}"
    keep_cat = f"keep-{run}"

    for i in range(103):
        assert (
            tenant_mgr.add_memory(
                content=f"bulk memory {run} number {i}",
                tenant_id=_TENANT,
                agent_name=_USER_MEMORY_AGENT,
                metadata={"category": bulk_cat},
                infer=False,
            )
            is not None
        )
    for i in range(4):
        assert (
            tenant_mgr.add_memory(
                content=f"keep memory {run} number {i}",
                tenant_id=_TENANT,
                agent_name=_USER_MEMORY_AGENT,
                metadata={"category": keep_cat},
                infer=False,
            )
            is not None
        )

    def _bulk_rows():
        return [
            r
            for r in tenant_mgr.get_all_memories(
                tenant_id=_TENANT, agent_name=_USER_MEMORY_AGENT, limit=None
            )
            if (r.get("metadata") or {}).get("category") == bulk_cat
        ]

    # Every seeded bulk row must be recoverable by the full walk before the
    # route reads it, or the count assertions below would race indexing.
    assert len(_poll_and_return(_bulk_rows, 103)) == 103

    client = TestClient(app, raise_server_exceptions=False)

    listed = client.get(
        f"/{_TENANT}/memories",
        params={"type": "preference", "category": bulk_cat, "limit": 200},
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()["count"] == 103

    cleared = client.request(
        "DELETE", f"/{_TENANT}/memories", params={"category": bulk_cat}
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json() == {
        "status": "cleared",
        "category": bulk_cat,
        "deleted": 103,
    }
    # Wait for the deletes to become visible before re-reading.
    _deadline = time.monotonic() + 30.0
    while _bulk_rows() and time.monotonic() < _deadline:
        time.sleep(3.0)
    assert _bulk_rows() == []

    after = client.get(
        f"/{_TENANT}/memories",
        params={"type": "preference", "category": bulk_cat, "limit": 200},
    )
    assert after.json()["count"] == 0
    survivors = client.get(
        f"/{_TENANT}/memories",
        params={"type": "preference", "category": keep_cat, "limit": 200},
    )
    assert survivors.json()["count"] == 4
