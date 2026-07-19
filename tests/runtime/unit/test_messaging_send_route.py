"""POST /messaging/send fans a message out to a tenant's linked chats.

Chat ids are reversed out of the SYSTEM mem0 partition (the user↔tenant mapping
the gateway wrote). The route enqueues one OutboundMessage per linked telegram
chat and returns the count; GET /messaging/outbound/drain returns-then-clears
the queue for the gateway to deliver. A backend outage while resolving chats
must surface as 503 — never be read as "no linked chats" and enqueue nothing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime import messaging
from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _mapping(external_user_id, tenant_id, platform="telegram", type_="user_mapping"):
    return {
        "id": f"m_{external_user_id}",
        "memory": "mapping",
        "metadata": {
            "type": type_,
            "platform": platform,
            "external_user_id": external_user_id,
            "tenant_id": tenant_id,
        },
    }


@pytest.fixture
def app(monkeypatch):
    messaging.reset_outbound_queue_for_testing()
    # Force the in-pod queue: _resolve_outbound_queue reads the module-level
    # _config_manager, and an empty/None one selects the in-memory singleton.
    monkeypatch.setattr(admin, "_config_manager", None)
    application = FastAPI()
    application.include_router(admin.router, prefix="/admin")
    application.dependency_overrides[admin.get_config_manager_dependency] = lambda: (
        MagicMock()
    )
    yield application
    messaging.reset_outbound_queue_for_testing()


def _patch_mapping(monkeypatch, rows=None, raise_outage=False):
    mgr = MagicMock()
    mgr.memory = MagicMock()  # already initialized
    if raise_outage:
        mgr.get_all_memories.side_effect = RuntimeError("vespa unreachable")
    else:
        mgr.get_all_memories.return_value = rows or []
    monkeypatch.setattr(
        "cogniverse_core.memory.manager.Mem0MemoryManager", lambda tid: mgr
    )
    return mgr


def test_send_resolves_chats_and_enqueues_one_per_chat(app, monkeypatch):
    _patch_mapping(
        monkeypatch,
        rows=[
            _mapping("111", "acme:acme"),
            _mapping("222", "acme:acme"),
            _mapping("999", "globex:globex"),  # other org — excluded
            _mapping("333", "acme:acme", platform="slack"),  # other platform
        ],
    )
    with TestClient(app) as c:
        resp = c.post(
            "/admin/messaging/send",
            json={"tenant_id": "acme:acme", "message": "job done"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"enqueued": 2}

        drained = c.get("/admin/messaging/outbound/drain").json()["messages"]
        assert sorted(m["chat_id"] for m in drained) == ["111", "222"]
        assert all(m["text"] == "job done" for m in drained)
        assert all(m["tenant_id"] == "acme:acme" for m in drained)
        # Drain cleared the queue.
        assert c.get("/admin/messaging/outbound/drain").json()["messages"] == []


def test_send_with_no_linked_chats_enqueues_zero(app, monkeypatch):
    _patch_mapping(monkeypatch, rows=[])
    with TestClient(app) as c:
        resp = c.post(
            "/admin/messaging/send", json={"tenant_id": "acme:acme", "message": "x"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"enqueued": 0}


def test_send_surfaces_backend_outage_as_503_not_zero(app, monkeypatch):
    _patch_mapping(monkeypatch, raise_outage=True)
    with TestClient(app) as c:
        resp = c.post(
            "/admin/messaging/send", json={"tenant_id": "acme:acme", "message": "x"}
        )
        assert resp.status_code == 503
        # The failure path enqueued nothing.
        assert c.get("/admin/messaging/outbound/drain").json()["messages"] == []


@pytest.mark.asyncio
async def test_job_executor_delivery_hits_the_real_mounted_send_route(app, monkeypatch):
    """job_executor's delivery URL must match the real mounted route.

    The admin router mounts at /admin, so the caller has to POST
    /admin/messaging/send. A stub-path test would pass while production
    404-skips; this drives the REAL route and asserts the message was
    actually enqueued (not the 404-skip branch).
    """
    import cogniverse_runtime.job_executor as je

    _patch_mapping(monkeypatch, rows=[_mapping("111", "acme:acme")])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt") as client:
        await je._deliver_to_telegram(client, "http://rt", "acme:acme", "job done")
        drained = (await client.get("http://rt/admin/messaging/outbound/drain")).json()[
            "messages"
        ]

    assert [m["chat_id"] for m in drained] == ["111"]
    assert drained[0]["text"] == "job done"
