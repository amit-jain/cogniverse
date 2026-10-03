"""SSE event stream coverage for /events/workflows/{id} and /events/ingestion/{id}.

Exercises the routes and the ``_event_stream`` generator against the real
Redis task event store. Asserts the documented event shapes: the
``connected`` opener, the published events in order, the stream ending once
the task has ended, the ``error`` event for a missing task and for one that
stopped reporting, heartbeats while idle, and the 503 and ``stream_error``
answers of a store that does not answer.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_core.events import (
    TaskState,
    create_complete_event,
    create_status_event,
)
from cogniverse_runtime.ingestion_worker import queue as ingest_queue
from cogniverse_runtime.routers import events as events_router
from cogniverse_runtime.shared_state import connect_shared_state_redis
from cogniverse_runtime.task_events import WORKFLOW, TaskEventStore

TENANT = "acme:acme"


def _store(redis, **kwargs) -> TaskEventStore:
    prefix = f"test:task-events:{uuid.uuid4().hex}"
    kwargs.setdefault("ingestion_stream_prefix", f"{prefix}:ingest:")
    kwargs.setdefault("read_interval_s", 0.02)
    return TaskEventStore(redis, key_prefix=prefix, **kwargs)


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


def _parse_sse(body: str) -> list[dict]:
    """Parse the ``data:`` lines out of an SSE response body."""
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _working(task_id: str, phase: str):
    return create_status_event(task_id, TENANT, TaskState.WORKING, phase=phase)


async def test_stream_unknown_workflow_emits_error_then_closes(routes) -> None:
    """No task for the id → single error event then the stream closes."""
    client, _ = routes
    response = await client.get("/events/workflows/missing")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _parse_sse(response.text) == [
        {"type": "error", "message": "No active queue for task missing"}
    ]


async def test_stream_emits_connected_then_events_until_the_task_ends(
    routes,
) -> None:
    """Happy path: the subscriber sees ``connected``, every published event
    in order, and the stream closes after the terminal event."""
    client, store = routes
    queue = await store.open_task(WORKFLOW, "wf-sse", TENANT)
    await queue.enqueue(_working("wf-sse", "planning"))
    await queue.enqueue(_working("wf-sse", "execution"))
    await queue.finish(create_complete_event("wf-sse", TENANT, result={"ok": True}))

    response = await client.get("/events/workflows/wf-sse")
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert {key: events[0][key] for key in ("type", "task_id", "offset")} == {
        "type": "connected",
        "task_id": "wf-sse",
        "offset": 0,
    }
    assert [
        (event["event_type"], event.get("phase"), event["task_id"])
        for event in events[1:]
    ] == [
        ("status", "planning", "wf-sse"),
        ("status", "execution", "wf-sse"),
        ("complete", None, "wf-sse"),
    ]
    assert events[3]["result"] == {"ok": True}


async def test_stream_replay_starts_from_offset(routes) -> None:
    """``from_offset`` skips the events before it."""
    client, store = routes
    queue = await store.open_task(WORKFLOW, "wf-replay", TENANT)
    for index in range(2):
        await queue.enqueue(_working("wf-replay", f"phase-{index}"))
    await queue.finish(create_complete_event("wf-replay", TENANT, result={}))

    response = await client.get("/events/workflows/wf-replay?from_offset=1")
    events = _parse_sse(response.text)

    assert events[0]["type"] == "connected"
    assert events[0]["offset"] == 1
    assert [event.get("phase") for event in events[1:]] == ["phase-1", None]
    assert [event["event_type"] for event in events[1:]] == ["status", "complete"]


async def test_ingestion_stream_reports_the_job_statuses_as_task_events(
    routes, shared_state_redis
) -> None:
    """The ingestion route reads the job's status stream, the one
    ``/ingestion/{id}/events`` serves, and reports each status as an event."""
    client, _ = routes
    store = _store(
        shared_state_redis,
        ingestion_stream_prefix=ingest_queue.STATUS_STREAM_KEY_PREFIX,
    )
    events_router.set_task_event_store(store)
    job = f"ing-sse-{uuid.uuid4().hex}"
    await store.register_queued(job, TENANT)
    try:
        for status in (
            {"state": "queued", "ingest_id": job},
            {"state": "running", "ingest_id": job},
            {"state": "complete", "ingest_id": job, "result": {"keyframes": 3}},
        ):
            await ingest_queue.publish_status(shared_state_redis, job, status)
        response = await client.get(f"/events/ingestion/{job}")
        wrong_kind = await client.get(f"/events/workflows/{job}")
    finally:
        await shared_state_redis.delete(ingest_queue._status_stream_key(job))
    events = _parse_sse(response.text)

    assert events[0]["type"] == "connected"
    assert [
        (event["event_type"], event.get("state"), event.get("phase"))
        for event in events[1:]
    ] == [
        ("status", "pending", "queued"),
        ("status", "working", "running"),
        ("complete", None, None),
    ]
    assert events[3]["result"] == {"keyframes": 3}
    assert {event["task_id"] for event in events[1:]} == {job}
    assert _parse_sse(wrong_kind.text) == [
        {"type": "error", "message": f"No active queue for task {job}"}
    ]


async def test_event_stream_emits_heartbeats_while_idle(shared_state_redis) -> None:
    """An idle subscription emits SSE heartbeat comments every
    heartbeat_interval so a proxy/load balancer doesn't drop the connection,
    and counts as a subscriber until it ends."""
    store = _store(shared_state_redis)
    queue = await store.open_task(WORKFLOW, "wf-hb", TENANT)
    subscriber = uuid.uuid4().hex
    first = await store.read("wf-hb", kind=WORKFLOW, subscriber=subscriber)

    chunks: list[str] = []

    async def consume() -> None:
        async for chunk in events_router._event_stream(
            store, WORKFLOW, "wf-hb", first, subscriber, heartbeat_interval=0.05
        ):
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.3)  # idle: no events published, only heartbeats flow
    heartbeats = [c for c in chunks if c.startswith(": heartbeat")]
    reading = (await store.read("wf-hb", kind=WORKFLOW, count=0)).subscribers

    # Ending the task ends the stream cleanly.
    await queue.finish(create_complete_event("wf-hb", TENANT, result={}))
    await asyncio.wait_for(task, timeout=5.0)
    after = (await store.read("wf-hb", kind=WORKFLOW, count=0)).subscribers

    assert len(heartbeats) >= 2, (
        f"expected periodic heartbeats during 0.3s idle, got {len(heartbeats)}"
    )
    assert (reading, after) == (1, 0)
    # The terminal event is still delivered and closes the stream.
    delivered = [
        json.loads(c[len("data: ") :]) for c in chunks if c.startswith("data:")
    ]
    assert [event.get("event_type", event.get("type")) for event in delivered] == [
        "connected",
        "complete",
    ]


async def test_a_task_that_stops_reporting_ends_its_stream_with_an_error(
    shared_state_redis,
) -> None:
    store = _store(shared_state_redis, producer_lease_s=1.0, poll_interval_s=0.1)
    queue = await store.open_task(WORKFLOW, "wf-silent", TENANT)
    await queue.enqueue(_working("wf-silent", "planning"))
    queue.release()
    subscriber = uuid.uuid4().hex
    first = await store.read("wf-silent", kind=WORKFLOW, subscriber=subscriber)

    chunks = [
        chunk
        async for chunk in events_router._event_stream(
            store, WORKFLOW, "wf-silent", first, subscriber
        )
    ]

    assert [_parse_sse(chunk)[0].get("type", "event") for chunk in chunks] == [
        "connected",
        "event",
        "error",
    ]
    assert _parse_sse(chunks[2]) == [
        {
            "type": "error",
            "message": "Task wf-silent stopped reporting before it finished",
        }
    ]


async def test_a_store_that_does_not_answer_is_a_503_before_the_stream(
    own_redis,
) -> None:
    url, pause, resume = own_redis
    redis = await connect_shared_state_redis(url, timeout_seconds=1.0)
    store = _store(redis)
    events_router.set_task_event_store(store)
    app = FastAPI()
    app.include_router(events_router.router, prefix="/events")
    pause()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://events"
        ) as client:
            response = await client.get("/events/workflows/wf-down")
    finally:
        resume()
        events_router.set_task_event_store(None)
        await redis.aclose()

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "error": "task_events_unavailable",
            "message": "The task event store did not answer; retry.",
            "failure": "TaskEventsUnavailable",
            "task_id": "wf-down",
        }
    }


async def test_a_store_lost_mid_stream_ends_it_with_a_stream_error(own_redis) -> None:
    url, pause, resume = own_redis
    redis = await connect_shared_state_redis(url, timeout_seconds=1.0)
    store = _store(redis)
    queue = await store.open_task(WORKFLOW, "wf-lost", TENANT)
    await queue.enqueue(_working("wf-lost", "planning"))
    subscriber = uuid.uuid4().hex
    first = await store.read("wf-lost", kind=WORKFLOW, subscriber=subscriber)
    stream = events_router._event_stream(store, WORKFLOW, "wf-lost", first, subscriber)
    received = [await anext(stream), await anext(stream)]
    pause()
    try:
        received += [chunk async for chunk in stream]
    finally:
        resume()
        await redis.aclose()

    assert [_parse_sse(chunk)[0].get("type", "event") for chunk in received] == [
        "connected",
        "event",
        "stream_error",
    ]
    assert _parse_sse(received[2]) == [
        {
            "type": "stream_error",
            "message": "The event stream for task wf-lost failed.",
            "failure": "TaskEventsUnavailable",
        }
    ]
