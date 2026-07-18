"""Happy-path coverage for the cancel + queue inspection event routes.

Existing tests only exercised the 404 branches (cancel a non-existent
task; inspect a non-existent queue). The populated-queue path — the
one a real dashboard subscriber actually hits — was never asserted.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.events.backends.memory import (
    get_queue_manager,
    reset_queue_manager,
)
from cogniverse_runtime.routers import events as events_router


@pytest.fixture
def client():
    reset_queue_manager()
    app = FastAPI()
    app.include_router(events_router.router, prefix="/events")
    yield TestClient(app)
    reset_queue_manager()


@pytest.mark.asyncio
async def _seed_queue(task_id: str, tenant_id: str = "acme") -> None:
    qm = get_queue_manager()
    await qm.create_queue(task_id=task_id, tenant_id=tenant_id, ttl_minutes=10)


def test_cancel_workflow_happy_path(client: TestClient) -> None:
    import asyncio

    asyncio.run(_seed_queue("wf-123"))
    r = client.post(
        "/events/workflows/wf-123/cancel", json={"reason": "user requested"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "wf-123"
    assert body["cancelled"] is True
    assert "wf-123" in body["message"]


def test_cancel_ingestion_happy_path(client: TestClient) -> None:
    import asyncio

    asyncio.run(_seed_queue("ing-456"))
    r = client.post("/events/ingestion/ing-456/cancel", json={"reason": "disk full"})
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "ing-456"
    assert body["cancelled"] is True


def test_cancel_unknown_workflow_returns_404(client: TestClient) -> None:
    r = client.post("/events/workflows/missing/cancel")
    assert r.status_code == 404


def test_cancel_unknown_ingestion_returns_404(client: TestClient) -> None:
    r = client.post("/events/ingestion/missing/cancel")
    assert r.status_code == 404


def test_get_queue_info_for_populated_queue_returns_shape(
    client: TestClient,
) -> None:
    import asyncio

    asyncio.run(_seed_queue("wf-info"))
    r = client.get("/events/queues/wf-info")
    assert r.status_code == 200
    body = r.json()
    # Populated queue path — the test_api_e2e tests only ever hit the 404.
    assert body["task_id"] == "wf-info"
    assert body["tenant_id"] == "acme"
    assert body["is_closed"] is False
    assert body["event_count"] >= 0
    assert body["subscriber_count"] >= 0


def test_get_queue_offset_for_populated_queue(client: TestClient) -> None:
    import asyncio

    asyncio.run(_seed_queue("wf-offset"))
    r = client.get("/events/queues/wf-offset/offset")
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "wf-offset"
    assert body["offset"] == 0  # fresh queue starts at offset 0


def test_list_active_queues_maps_every_queue_field(client: TestClient) -> None:
    """GET /events/queues returns one QueueInfo per live queue of the tenant,
    with every field mapped from the backend's stats — other tenants' queues
    excluded, cancelled-but-open queues included with is_cancelled=True."""
    import asyncio

    from cogniverse_core.events import TaskState, create_status_event

    async def _seed() -> dict:
        qm = get_queue_manager()
        q1 = await qm.create_queue(
            task_id="wf-list-1", tenant_id="acme:list", ttl_minutes=10
        )
        await q1.enqueue(
            create_status_event(
                task_id="wf-list-1", tenant_id="acme:list", state=TaskState.WORKING
            )
        )
        await qm.create_queue(
            task_id="ing-list-2", tenant_id="acme:list", ttl_minutes=10
        )
        await qm.cancel_task("ing-list-2", "operator stop")
        await qm.create_queue(
            task_id="wf-other-tenant", tenant_id="globex:list", ttl_minutes=10
        )
        return {q["task_id"]: q for q in await qm.list_active_queues("acme:list")}

    stats = asyncio.run(_seed())

    r = client.get("/events/queues", params={"tenant_id": "acme:list"})
    assert r.status_code == 200
    body = sorted(r.json(), key=lambda q: q["task_id"])
    assert body == [
        {
            "task_id": "ing-list-2",
            "tenant_id": "acme:list",
            "event_count": 0,
            "subscriber_count": 0,
            "is_closed": False,
            "is_cancelled": True,
            "created_at": stats["ing-list-2"]["created_at"],
        },
        {
            "task_id": "wf-list-1",
            "tenant_id": "acme:list",
            "event_count": 1,
            "subscriber_count": 0,
            "is_closed": False,
            "is_cancelled": False,
            "created_at": stats["wf-list-1"]["created_at"],
        },
    ]


def test_list_active_queues_requires_tenant_id(client: TestClient) -> None:
    r = client.get("/events/queues")
    assert r.status_code == 422
