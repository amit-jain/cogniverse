"""HTTP route contract for ``POST /agents/{name}/message``.

Locks the route's observable behaviour: 202 with explicit shape on
active session, 404 with explicit detail on unknown session, 400 on
already-past deadline (intake validation), 422 on invalid role
(Pydantic validation). Every assertion pins exact-value: status
code + exact ``detail`` text + exact body shape.

End-to-end behaviour (message reaches agent's drain) is covered by
``test_iterative_loop_inbound.py``; this file scopes only the HTTP
boundary.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.messaging import (
    QueueClosedError,
    get_inbound_queue_registry,
    reset_inbound_queue_registry_for_testing,
)
from cogniverse_runtime.routers import agents as agents_router

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """FastAPI TestClient with only the agents router mounted.

    No agent registry / dispatcher dependencies — the message route
    is independent of the dispatcher (it only touches the inbound
    queue registry).
    """
    reset_inbound_queue_registry_for_testing()
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    return TestClient(app)


@pytest.fixture
async def _registered_session():
    """Register a known session in the singleton registry, yield its id."""
    reset_inbound_queue_registry_for_testing()
    registry = get_inbound_queue_registry()
    await registry.get_or_create_queue("sess-active", "test_tenant")
    yield "sess-active"
    await registry.close_queue("sess-active")


def _register_sync(session_id: str, tenant_id: str) -> None:
    """Sync wrapper around the async get_or_create — TestClient runs
    its own event loop and can't await fixtures cleanly here."""
    import asyncio

    asyncio.run(get_inbound_queue_registry().get_or_create_queue(session_id, tenant_id))


def _close_sync(session_id: str) -> None:
    import asyncio

    asyncio.run(get_inbound_queue_registry().close_queue(session_id))


# --------------------------------------------------------------------- #
# Active session — 202 + explicit body shape                             #
# --------------------------------------------------------------------- #


def test_post_to_active_session_returns_202_with_message_id_and_queued_at(
    client,
):
    _register_sync("sess-active", "test_tenant")
    try:
        resp = client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-active",
                "tenant_id": "test_tenant",
                "role": "user",
                "content": "only sources from 2024",
                "tags": ["constraint"],
            },
        )
        assert resp.status_code == 202

        body = resp.json()
        # Exact key set — no other fields, no missing fields.
        assert set(body.keys()) == {"message_id", "queued_at"}
        # Locked prefix so a future change to the id scheme trips this test.
        assert body["message_id"].startswith("msg_")
        # 16-char hex tail after "msg_" — UUID4 prefix length is the
        # contract; any change to length must update the test in
        # the same commit.
        assert len(body["message_id"]) == len("msg_") + 16
        # queued_at is ISO-8601 UTC; minimal but exact shape check
        # (trailing "+00:00" or "Z" — datetime.isoformat() emits +00:00).
        assert "T" in body["queued_at"]
        assert body["queued_at"].endswith("+00:00")
    finally:
        _close_sync("sess-active")


# --------------------------------------------------------------------- #
# Unknown session — 404 + exact detail string                            #
# --------------------------------------------------------------------- #


def test_post_to_unknown_session_returns_404_with_exact_detail(client):
    resp = client.post(
        "/agents/orchestrator/message",
        json={
            "session_id": "never-existed",
            "tenant_id": "test_tenant",
            "role": "user",
            "content": "hi",
            "tags": [],
        },
    )
    assert resp.status_code == 404
    # Exact detail string — a future refactor that changes the
    # phrasing must update this test in the same commit.
    assert resp.json()["detail"] == "session 'never-existed' not active"


# --------------------------------------------------------------------- #
# Past deadline — 400 at intake (not buffered + dropped at drain)        #
# --------------------------------------------------------------------- #


def test_post_with_past_deadline_returns_400_at_intake(client):
    _register_sync("sess-active", "test_tenant")
    try:
        past_ms = int(time.time() * 1000) - 1000
        resp = client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-active",
                "tenant_id": "test_tenant",
                "role": "user",
                "content": "stale",
                "tags": [],
                "deadline_ms": past_ms,
            },
        )
        assert resp.status_code == 400
        # Detail mentions the offending field + value so the caller
        # can debug clock skew without reading server logs.
        detail = resp.json()["detail"]
        assert "deadline_ms" in detail
        assert str(past_ms) in detail
    finally:
        _close_sync("sess-active")


# --------------------------------------------------------------------- #
# Invalid role — 422 with field locator                                  #
# --------------------------------------------------------------------- #


def test_post_with_invalid_role_returns_422(client):
    _register_sync("sess-active", "test_tenant")
    try:
        resp = client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-active",
                "tenant_id": "test_tenant",
                "role": "wizard",  # not in {user, system, agent}
                "content": "hi",
                "tags": [],
            },
        )
        assert resp.status_code == 422
        # The pydantic 422 body always has detail as a list with the
        # field locator; we lock 'role' as the offending field.
        loc_paths = [tuple(item["loc"]) for item in resp.json()["detail"]]
        assert ("body", "role") in loc_paths
    finally:
        _close_sync("sess-active")


# --------------------------------------------------------------------- #
# Empty session_id — 422 (min_length=1) with field locator               #
# --------------------------------------------------------------------- #


def test_post_with_empty_session_id_returns_422(client):
    resp = client.post(
        "/agents/orchestrator/message",
        json={
            "session_id": "",
            "tenant_id": "test_tenant",
            "role": "user",
            "content": "hi",
            "tags": [],
        },
    )
    assert resp.status_code == 422
    loc_paths = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in loc_paths


# --------------------------------------------------------------------- #
# Missing required field — 422 names the missing field                    #
# --------------------------------------------------------------------- #


def test_post_with_missing_session_id_returns_422(client):
    resp = client.post(
        "/agents/orchestrator/message",
        json={"role": "user", "content": "hi", "tags": []},
    )
    assert resp.status_code == 422
    loc_paths = [tuple(item["loc"]) for item in resp.json()["detail"]]
    assert ("body", "session_id") in loc_paths


# --------------------------------------------------------------------- #
# Message reaches the queue — proves the route DELIVERS, not just 202s   #
# --------------------------------------------------------------------- #


def test_successful_post_message_is_observable_in_queue_drain(client):
    """A 202 means the message was buffered; this test proves it.
    Without this, the route could silently 202 and drop messages —
    a failure mode the agent integration would only catch downstream.
    """
    import asyncio

    _register_sync("sess-active", "test_tenant")
    try:
        resp = client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-active",
                "tenant_id": "test_tenant",
                "role": "user",
                "content": "constraint-payload-XYZ",
                "tags": ["constraint", "interrupt"],
            },
        )
        assert resp.status_code == 202

        # Drain the queue and assert the message is present byte-equal.
        queue = asyncio.run(get_inbound_queue_registry().get_queue("sess-active"))
        drained = asyncio.run(queue.drain())
        assert len(drained) == 1
        msg = drained[0]
        assert msg.session_id == "sess-active"
        assert msg.role == "user"
        assert msg.content == "constraint-payload-XYZ"
        assert msg.tags == ("constraint", "interrupt")
    finally:
        _close_sync("sess-active")


# --------------------------------------------------------------------- #
# After close_queue — same POST returns 404 (lifecycle boundary)         #
# --------------------------------------------------------------------- #


def test_post_after_close_queue_returns_404(client):
    """The boundary is close_queue(), not a TTL — POST 100 ms after
    close vs 10 s after close both return 404 with the same detail.
    Proven here by closing immediately.
    """
    _register_sync("sess-active", "test_tenant")
    _close_sync("sess-active")
    resp = client.post(
        "/agents/orchestrator/message",
        json={
            "session_id": "sess-active",
            "tenant_id": "test_tenant",
            "role": "user",
            "content": "post-close",
            "tags": [],
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session 'sess-active' not active"


# --------------------------------------------------------------------- #
# Race: close happens between get_queue and enqueue → still 404           #
# --------------------------------------------------------------------- #


def test_race_close_during_enqueue_surfaces_404(client, monkeypatch):
    """If the agent's ``finally`` closes the queue between the
    route's ``get_queue`` (success) and its ``enqueue`` call (which
    sees ``QueueClosedError``), the route MUST surface 404 — not
    500. The caller's mental model is "session ended."
    """
    _register_sync("sess-active", "test_tenant")
    queue = _resolve_queue("sess-active")
    # Pre-close the queue but DON'T close-and-remove via the registry,
    # so get_queue still returns the (now-closed) queue and enqueue
    # raises QueueClosedError.
    queue.close()

    resp = client.post(
        "/agents/orchestrator/message",
        json={
            "session_id": "sess-active",
            "tenant_id": "test_tenant",
            "role": "user",
            "content": "racing",
            "tags": [],
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session 'sess-active' not active"
    _close_sync("sess-active")


def _resolve_queue(session_id: str):
    import asyncio

    return asyncio.run(get_inbound_queue_registry().get_queue(session_id))


# --------------------------------------------------------------------- #
# Session-poll route — 200 with metadata when active                      #
# --------------------------------------------------------------------- #


def test_get_session_returns_metadata_when_active(client):
    _register_sync("sess-poll", "test_tenant")
    try:
        resp = client.get(
            "/agents/orchestrator/sessions/sess-poll",
            params={"tenant_id": "test_tenant"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-poll"
        assert body["tenant_id"] == "test_tenant"
        assert "T" in body["created_at"]
        assert body["created_at"].endswith("+00:00")
    finally:
        _close_sync("sess-poll")


def test_get_session_returns_404_when_not_active(client):
    resp = client.get(
        "/agents/orchestrator/sessions/never-existed",
        params={"tenant_id": "test_tenant"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session 'never-existed' not active"


# --------------------------------------------------------------------- #
# QueueClosedError import — confirms public symbol stays exposed         #
# --------------------------------------------------------------------- #


def test_queue_closed_error_is_public_for_consumers():
    # Trivial but locked: the HTTP route catches QueueClosedError, so
    # the symbol MUST stay importable from cogniverse_runtime.messaging.
    # A refactor that hides it internally would break the route and
    # surface only as a 500 in production — this catches it at lint time.
    assert QueueClosedError.__module__ == "cogniverse_runtime.messaging"


# --------------------------------------------------------------------- #
# Cross-tenant POST → 404 (mismatched tenant cannot probe sessions)      #
# --------------------------------------------------------------------- #


def test_cross_tenant_post_to_active_session_returns_404(client):
    """Alice registers ``sess-shared`` under tenant ``alice``. Bob
    POSTs to the same session_id with tenant ``bob``. The route MUST
    return 404 with the same detail used for genuinely-unknown
    sessions — a 403 would leak "session exists, just not for you,"
    enabling tenant + session_id enumeration. Locked: same detail
    string for "doesn't exist" and "exists under another tenant."
    """
    _register_sync("sess-shared", "alice")
    try:
        resp = client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-shared",
                "tenant_id": "bob",  # wrong tenant
                "role": "user",
                "content": "probe",
                "tags": [],
            },
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "session 'sess-shared' not active"
    finally:
        _close_sync("sess-shared")


def test_cross_tenant_post_leaves_target_queue_untouched(client):
    """Beyond the 404, Bob's POST MUST NOT leave any message in
    Alice's queue. Otherwise the leak surface is "I can write to
    your queue, you just can't tell it's me" — Alice's drain after
    Bob's POST sees zero of Bob's messages.
    """
    import asyncio

    _register_sync("sess-leak", "alice")
    try:
        # Alice writes her own message first.
        client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-leak",
                "tenant_id": "alice",
                "role": "user",
                "content": "alice-msg-1",
                "tags": ["constraint"],
            },
        )
        # Bob attempts a cross-tenant write.
        client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-leak",
                "tenant_id": "bob",
                "role": "user",
                "content": "bob-leaked-msg",
                "tags": ["constraint"],
            },
        )
        # Alice drains — must see ONLY her message.
        queue = asyncio.run(get_inbound_queue_registry().get_queue("sess-leak"))
        drained = asyncio.run(queue.drain())
        assert [m.content for m in drained] == ["alice-msg-1"]
        assert all(m.content != "bob-leaked-msg" for m in drained)
    finally:
        _close_sync("sess-leak")


def test_cross_tenant_get_session_returns_404(client):
    """GET sessions endpoint mirrors the same cross-tenant guard."""
    _register_sync("sess-peek", "alice")
    try:
        resp = client.get(
            "/agents/orchestrator/sessions/sess-peek",
            params={"tenant_id": "bob"},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "session 'sess-peek' not active"
    finally:
        _close_sync("sess-peek")


# --------------------------------------------------------------------- #
# Missing tenant_id → 422 (required field)                                #
# --------------------------------------------------------------------- #


def test_post_with_missing_tenant_id_returns_422(client):
    _register_sync("sess-active", "test_tenant")
    try:
        resp = client.post(
            "/agents/orchestrator/message",
            json={
                "session_id": "sess-active",
                "role": "user",
                "content": "no tenant",
                "tags": [],
            },
        )
        assert resp.status_code == 422
        locs = [tuple(item["loc"]) for item in resp.json()["detail"]]
        assert ("body", "tenant_id") in locs
    finally:
        _close_sync("sess-active")
