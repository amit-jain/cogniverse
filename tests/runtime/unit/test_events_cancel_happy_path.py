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
    r = client.post(
        "/events/ingestion/ing-456/cancel", json={"reason": "disk full"}
    )
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
