"""Coverage for the cancel and queue inspection event routes.

Each route runs against the real Redis task event store: a cancellation is
recorded for whichever process runs the task, the queue routes read the
shared task state, and every outcome a cancellation can meet has its own
answer.
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis

from cogniverse_core.events import (
    TaskState,
    create_complete_event,
    create_status_event,
)
from cogniverse_runtime.routers import events as events_router
from cogniverse_runtime.task_events import INGESTION, WORKFLOW, TaskEventStore


def _store(redis, **kwargs) -> TaskEventStore:
    prefix = f"test:task-events:{uuid.uuid4().hex}"
    return TaskEventStore(
        redis, key_prefix=prefix, ingestion_stream_prefix=f"{prefix}:ingest:", **kwargs
    )


@pytest.fixture
async def routes(shared_state_redis):
    store = _store(shared_state_redis)
    events_router.set_task_event_store(store)
    app = FastAPI()
    app.include_router(events_router.router, prefix="/events")
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://events"
    )
    try:
        yield client, store
    finally:
        await client.aclose()
        events_router.set_task_event_store(None)


async def test_cancel_workflow_happy_path(routes) -> None:
    client, store = routes
    queue = await store.open_task(WORKFLOW, "wf-123", "acme:acme")

    response = await client.post(
        "/events/workflows/wf-123/cancel", json={"reason": "user requested"}
    )
    await store.poll_once()

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "wf-123",
        "cancelled": True,
        "message": "Workflow wf-123 cancellation requested",
    }
    # The process running the workflow picks the cancellation up.
    assert queue.cancellation_token.is_cancelled is True
    assert queue.cancellation_token.reason == "user requested"


async def test_cancel_ingestion_happy_path(routes) -> None:
    client, store = routes
    await store.register_queued("ing-456", "acme:acme")

    response = await client.post(
        "/events/ingestion/ing-456/cancel", json={"reason": "disk full"}
    )
    claimed = await store.attach(INGESTION, "ing-456", "acme:acme")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "ing-456",
        "cancelled": True,
        "message": "Ingestion job ing-456 cancellation requested",
    }
    assert claimed.cancellation_token.reason == "disk full"


async def test_cancel_unknown_workflow_returns_404(routes) -> None:
    client, store = routes
    await store.register_queued("ing-only", "acme:acme")

    missing = await client.post("/events/workflows/missing/cancel")
    other_kind = await client.post("/events/workflows/ing-only/cancel")

    assert (missing.status_code, missing.json()) == (
        404,
        {"detail": "No active workflow found with ID missing"},
    )
    assert (other_kind.status_code, other_kind.json()) == (
        404,
        {"detail": "No active workflow found with ID ing-only"},
    )


async def test_cancel_unknown_ingestion_returns_404(routes) -> None:
    client, _ = routes
    response = await client.post("/events/ingestion/missing/cancel")

    assert (response.status_code, response.json()) == (
        404,
        {"detail": "No active ingestion job found with ID missing"},
    )


async def test_cancel_of_a_finished_or_silent_task_is_a_conflict(
    routes, shared_state_redis
) -> None:
    client, _ = routes
    store = _store(shared_state_redis, producer_lease_s=1.0, poll_interval_s=0.1)
    events_router.set_task_event_store(store)
    finished = await store.open_task(WORKFLOW, "wf-done", "acme:acme")
    await finished.finish(create_complete_event("wf-done", "acme:acme", result={}))
    silent = await store.open_task(WORKFLOW, "wf-silent", "acme:acme")
    silent.release()
    await asyncio.sleep(1.2)

    done = await client.post("/events/workflows/wf-done/cancel")
    stopped = await client.post("/events/workflows/wf-silent/cancel")

    assert (done.status_code, done.json()) == (
        409,
        {"detail": "Workflow wf-done already finished"},
    )
    assert (stopped.status_code, stopped.json()) == (
        409,
        {"detail": "Workflow wf-silent stopped reporting before it finished"},
    )


async def test_get_queue_info_for_populated_queue_returns_shape(routes) -> None:
    client, store = routes
    queue = await store.open_task(WORKFLOW, "wf-info", "acme:acme")
    await queue.enqueue(
        create_status_event("wf-info", "acme:acme", TaskState.WORKING, phase="plan")
    )
    await store.read("wf-info", subscriber="web")
    read = await store.read("wf-info", count=0)

    response = await client.get("/events/queues/wf-info")
    missing = await client.get("/events/queues/absent")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "wf-info",
        "kind": WORKFLOW,
        "tenant_id": "acme:acme",
        "event_count": 1,
        "subscriber_count": 1,
        "is_closed": False,
        "is_cancelled": False,
        "created_at": read.info("wf-info")["created_at"],
    }
    assert (missing.status_code, missing.json()) == (
        404,
        {"detail": "No queue found for task absent"},
    )


async def test_get_queue_offset_for_populated_queue(routes) -> None:
    client, store = routes
    queue = await store.open_task(WORKFLOW, "wf-offset", "acme:acme")
    fresh = await client.get("/events/queues/wf-offset/offset")
    for phase in ("plan", "run"):
        await queue.enqueue(
            create_status_event(
                "wf-offset", "acme:acme", TaskState.WORKING, phase=phase
            )
        )

    response = await client.get("/events/queues/wf-offset/offset")

    assert fresh.json() == {"task_id": "wf-offset", "offset": 0}
    assert response.status_code == 200
    assert response.json() == {"task_id": "wf-offset", "offset": 2}


async def test_list_active_queues_maps_every_queue_field(routes) -> None:
    """GET /events/queues returns one QueueInfo per running or queued task of
    the tenant, with every field mapped from the shared task state — other
    tenants' and ended tasks excluded, cancelled-but-running tasks included
    with is_cancelled=True."""
    client, store = routes
    first = await store.open_task(WORKFLOW, "wf-list-1", "acme:list")
    await first.enqueue(
        create_status_event("wf-list-1", "acme:list", TaskState.WORKING)
    )
    await store.register_queued("ing-list-2", "acme:list")
    await store.cancel(INGESTION, "ing-list-2", "operator stop")
    await store.open_task(WORKFLOW, "wf-other-tenant", "globex:list")
    ended = await store.open_task(WORKFLOW, "wf-ended", "acme:list")
    await ended.finish(create_complete_event("wf-ended", "acme:list", result={}))
    created = {
        task: (await store.read(task, count=0)).info(task)["created_at"]
        for task in ("wf-list-1", "ing-list-2")
    }

    response = await client.get("/events/queues", params={"tenant_id": "acme:list"})

    assert response.status_code == 200
    assert sorted(response.json(), key=lambda q: q["task_id"]) == [
        {
            "task_id": "ing-list-2",
            "kind": INGESTION,
            "tenant_id": "acme:list",
            "event_count": 0,
            "subscriber_count": 0,
            "is_closed": False,
            "is_cancelled": True,
            "created_at": created["ing-list-2"],
        },
        {
            "task_id": "wf-list-1",
            "kind": WORKFLOW,
            "tenant_id": "acme:list",
            "event_count": 1,
            "subscriber_count": 0,
            "is_closed": False,
            "is_cancelled": False,
            "created_at": created["wf-list-1"],
        },
    ]


async def test_list_active_queues_requires_tenant_id(routes) -> None:
    client, _ = routes
    response = await client.get("/events/queues")
    assert response.status_code == 422


async def test_every_route_answers_503_when_the_store_does_not(routes) -> None:
    client, _ = routes
    dead = Redis.from_url(
        "redis://127.0.0.1:9/0",
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )
    events_router.set_task_event_store(_store(dead))
    try:
        answers = {
            "cancel": await client.post("/events/workflows/wf/cancel"),
            "list": await client.get("/events/queues", params={"tenant_id": "t:t"}),
            "info": await client.get("/events/queues/wf"),
            "offset": await client.get("/events/queues/wf/offset"),
        }
    finally:
        await dead.aclose()

    unavailable = {
        "error": "task_events_unavailable",
        "message": "The task event store did not answer; retry.",
        "failure": "TaskEventsUnavailable",
    }
    assert {name: (r.status_code, r.json()) for name, r in answers.items()} == {
        "cancel": (503, {"detail": {**unavailable, "task_id": "wf"}}),
        "list": (503, {"detail": unavailable}),
        "info": (503, {"detail": {**unavailable, "task_id": "wf"}}),
        "offset": (503, {"detail": {**unavailable, "task_id": "wf"}}),
    }
