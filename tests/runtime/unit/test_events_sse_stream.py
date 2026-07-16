"""SSE event stream coverage for /events/workflows/{id} and /events/ingestion/{id}.

Exercises the ``_event_stream`` generator end-to-end against the live
``InMemoryQueueManager`` boundary. Asserts the documented event shapes:
``connected`` opener, the published payload, the ``complete``/``error``
early break, and the ``error`` event when no queue exists.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.events.backends.memory import (
    get_queue_manager,
    reset_queue_manager,
)
from cogniverse_core.events.types import StatusEvent
from cogniverse_runtime.routers import events as events_router


@pytest.fixture
def client():
    reset_queue_manager()
    app = FastAPI()
    app.include_router(events_router.router, prefix="/events")
    yield TestClient(app)
    reset_queue_manager()


def _parse_sse(body: str) -> list[dict]:
    """Parse the ``data:`` lines out of an SSE response body."""
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_stream_unknown_workflow_emits_error_then_closes(client: TestClient) -> None:
    """No queue for task_id → single error event then stream closes."""
    with client.stream("GET", "/events/workflows/missing") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    events = _parse_sse(body)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "missing" in events[0]["message"]


def test_stream_emits_connected_then_completes_on_complete_event(
    client: TestClient,
) -> None:
    """Happy path: subscriber sees ``connected``, then the published
    completion event, then the stream closes."""

    async def setup_and_publish() -> None:
        qm = get_queue_manager()
        queue = await qm.create_queue(task_id="wf-sse", tenant_id="acme")
        # publish one status event with phase=complete so _event_stream
        # breaks out of the subscribe loop deterministically.
        await queue.enqueue(
            StatusEvent(
                event_type="complete",
                task_id="wf-sse",
                tenant_id="acme",
                phase="done",
                state="completed",
                message="finished",
            )
        )

    asyncio.run(setup_and_publish())

    with client.stream("GET", "/events/workflows/wf-sse") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    events = _parse_sse(body)

    # First event is the connect opener with offset and task_id.
    assert events[0]["type"] == "connected"
    assert events[0]["task_id"] == "wf-sse"
    assert events[0]["offset"] == 0
    # Second event is the published complete event.
    assert events[1]["event_type"] == "complete"
    assert events[1]["task_id"] == "wf-sse"
    # Stream broke out after the complete — no further events.
    assert len(events) == 2


def test_stream_replay_starts_from_offset(client: TestClient) -> None:
    """``from_offset`` query param feeds queue.subscribe."""

    async def setup_and_publish() -> None:
        qm = get_queue_manager()
        queue = await qm.create_queue(task_id="wf-replay", tenant_id="acme")
        # publish two events so we can test offset replay
        for i in range(2):
            await queue.enqueue(
                StatusEvent(
                    event_type="complete" if i == 1 else "status",
                    task_id="wf-replay",
                    tenant_id="acme",
                    phase=f"phase-{i}",
                    state="working" if i == 0 else "completed",
                    message=f"event-{i}",
                )
            )

    asyncio.run(setup_and_publish())

    with client.stream("GET", "/events/workflows/wf-replay?from_offset=1") as resp:
        body = "".join(resp.iter_text())
    events = _parse_sse(body)

    # The connect event echoes the requested offset.
    assert events[0]["type"] == "connected"
    assert events[0]["offset"] == 1
    # And the replay starts at offset 1 — the second event with phase=phase-1.
    assert events[1]["phase"] == "phase-1"


def test_ingestion_stream_uses_same_event_shape(client: TestClient) -> None:
    """Ingestion route is a thin wrapper around the same _event_stream."""

    async def setup() -> None:
        qm = get_queue_manager()
        queue = await qm.create_queue(task_id="ing-sse", tenant_id="acme")
        await queue.enqueue(
            StatusEvent(
                event_type="complete",
                task_id="ing-sse",
                tenant_id="acme",
                phase="done",
                state="completed",
                message="ok",
            )
        )

    asyncio.run(setup())

    with client.stream("GET", "/events/ingestion/ing-sse") as resp:
        body = "".join(resp.iter_text())
    events = _parse_sse(body)
    assert events[0]["type"] == "connected"
    assert events[1]["event_type"] == "complete"
    assert events[1]["task_id"] == "ing-sse"


@pytest.mark.asyncio
async def test_event_stream_emits_heartbeats_while_idle() -> None:
    """An idle subscription emits SSE heartbeat comments every
    heartbeat_interval so a proxy/load balancer doesn't drop the connection.
    The documented heartbeat_interval was accepted but never used, so a
    long-running workflow that went quiet silently lost its stream."""
    reset_queue_manager()
    qm = get_queue_manager()
    queue = await qm.create_queue(task_id="wf-hb", tenant_id="acme")

    chunks: list[str] = []

    async def consume() -> None:
        async for chunk in events_router._event_stream(
            "wf-hb", from_offset=0, heartbeat_interval=0.05
        ):
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)  # idle: no events published, only heartbeats flow
    heartbeats = [c for c in chunks if c.startswith(": heartbeat")]

    # Publishing a completion event ends the stream cleanly.
    await queue.enqueue(
        StatusEvent(
            event_type="complete",
            task_id="wf-hb",
            tenant_id="acme",
            phase="done",
            state="completed",
            message="done",
        )
    )
    await asyncio.wait_for(task, timeout=2.0)
    reset_queue_manager()

    assert len(heartbeats) >= 2, (
        f"expected periodic heartbeats during 0.2s idle, got {len(heartbeats)}"
    )
    # The real completion event is still delivered and closes the stream.
    assert any(c.startswith("data:") and "complete" in c for c in chunks)
