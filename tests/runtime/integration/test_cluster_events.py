"""Admin events reach every subscribed worker through a real Redis.

Each ``ClusterEvents`` instance stands for one runtime worker process: its own
Redis connections, its own subscription and handlers. The publisher's answer
is pinned against the handlers that actually ran, and every failure shape —
a handler that raised, a worker that never answered, no worker subscribed,
Redis unreachable or paused — must raise a typed error naming it.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import threading
import time
import uuid

import pytest

from cogniverse_runtime.cluster_events import (
    ClusterEventIncomplete,
    ClusterEvents,
    ClusterEventUnavailable,
)

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


def _channel() -> str:
    return f"cogniverse:test-events:{uuid.uuid4().hex[:8]}"


def _dead_redis_url() -> str:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        return f"redis://127.0.0.1:{reserved.getsockname()[1]}/0"


class _Recorder:
    def __init__(self, worker: str):
        self.worker = worker
        self.seen: list = []
        self._lock = threading.Lock()

    def __call__(self, payload):
        with self._lock:
            self.seen.append(payload)
        return {"worker": self.worker, "echo": payload["n"]}


async def _workers(redis_url: str, channel: str, handlers_by_worker: dict):
    workers = []
    for worker_id, handlers in handlers_by_worker.items():
        worker = ClusterEvents(redis_url, worker_id, handlers, channel=channel)
        await worker.start()
        workers.append(worker)
    return workers


async def _close(workers) -> None:
    for worker in workers:
        await worker.close()


@pytest.mark.asyncio
async def test_an_event_runs_on_every_worker_and_returns_each_result(
    workflow_state_redis_url,
):
    channel = _channel()
    a, b = _Recorder("a"), _Recorder("b")
    workers = await _workers(
        workflow_state_redis_url,
        channel,
        {"worker-a": {"ping": a}, "worker-b": {"ping": b}},
    )
    try:
        results = await workers[0].publish("ping", {"n": 7}, timeout_s=10)
    finally:
        await _close(workers)

    assert results == {
        "worker-a": {"worker": "a", "echo": 7},
        "worker-b": {"worker": "b", "echo": 7},
    }
    assert (a.seen, b.seen) == ([{"n": 7}], [{"n": 7}])


@pytest.mark.asyncio
async def test_concurrent_events_from_both_workers_each_collect_their_own_answers(
    workflow_state_redis_url,
):
    channel = _channel()
    a, b = _Recorder("a"), _Recorder("b")
    workers = await _workers(
        workflow_state_redis_url,
        channel,
        {"worker-a": {"ping": a}, "worker-b": {"ping": b}},
    )
    try:
        results = await asyncio.gather(
            *[workers[n % 2].publish("ping", {"n": n}, timeout_s=20) for n in range(16)]
        )
    finally:
        await _close(workers)

    assert results == [
        {
            "worker-a": {"worker": "a", "echo": n},
            "worker-b": {"worker": "b", "echo": n},
        }
        for n in range(16)
    ]
    assert sorted(payload["n"] for payload in a.seen) == list(range(16))
    assert sorted(payload["n"] for payload in b.seen) == list(range(16))


@pytest.mark.asyncio
async def test_a_worker_whose_handler_raises_fails_the_event_by_name(
    workflow_state_redis_url,
):
    channel = _channel()

    def broken(payload):
        raise RuntimeError("cache eviction failed")

    workers = await _workers(
        workflow_state_redis_url,
        channel,
        {"worker-a": {"ping": _Recorder("a")}, "worker-b": {"ping": broken}},
    )
    try:
        with pytest.raises(ClusterEventIncomplete) as caught:
            await workers[0].publish("ping", {"n": 1}, timeout_s=10)
    finally:
        await _close(workers)

    assert caught.value.results == {"worker-a": {"worker": "a", "echo": 1}}
    assert caught.value.failures == {"worker-b": "RuntimeError: cache eviction failed"}
    assert str(caught.value) == (
        "1 of 2 workers handled 'ping'; failed: "
        "{'worker-b': 'RuntimeError: cache eviction failed'}"
    )


@pytest.mark.asyncio
async def test_a_worker_that_never_answers_fails_the_event_at_the_timeout(
    workflow_state_redis_url,
):
    channel = _channel()
    release = threading.Event()

    def stuck(payload):
        release.wait(30)
        return {}

    workers = await _workers(
        workflow_state_redis_url,
        channel,
        {"worker-a": {"ping": _Recorder("a")}, "worker-b": {"ping": stuck}},
    )
    try:
        started = time.monotonic()
        with pytest.raises(ClusterEventIncomplete) as caught:
            await workers[0].publish("ping", {"n": 3}, timeout_s=2)
        waited = time.monotonic() - started
    finally:
        release.set()
        await _close(workers)

    assert 2 <= waited < 4
    assert caught.value.results == {"worker-a": {"worker": "a", "echo": 3}}
    assert caught.value.failures == {}
    assert str(caught.value) == "1 of 2 workers handled 'ping'; 1 did not answer"


@pytest.mark.asyncio
async def test_an_event_no_worker_is_subscribed_to_is_refused(
    workflow_state_redis_url,
):
    channel = _channel()
    (publisher,) = await _workers(
        workflow_state_redis_url, channel, {"worker-a": {"ping": _Recorder("a")}}
    )
    # This worker's subscription is down; nothing else listens on the channel.
    publisher._listener.cancel()
    await asyncio.gather(publisher._listener, return_exceptions=True)
    try:
        with pytest.raises(ClusterEventUnavailable) as caught:
            await publisher.publish("ping", {"n": 1}, timeout_s=5)
    finally:
        await publisher.close()

    assert str(caught.value) == (
        f"cluster events: no runtime worker is subscribed to {channel!r}; "
        "'ping' reached none"
    )


@pytest.mark.asyncio
async def test_an_unreachable_redis_refuses_to_start():
    worker = ClusterEvents(
        _dead_redis_url(), "worker-a", {}, channel=_channel(), redis_timeout_s=2
    )

    with pytest.raises(ClusterEventUnavailable) as caught:
        await worker.start()

    assert str(caught.value).startswith("cluster events: cannot reach Redis: ")


@pytest.mark.asyncio
async def test_a_paused_redis_fails_the_publish_within_its_timeout(owned_redis):
    channel = _channel()
    worker = ClusterEvents(
        owned_redis["url"],
        "worker-a",
        {"ping": _Recorder("a")},
        channel=channel,
        redis_timeout_s=2,
    )
    await worker.start()
    subprocess.run(
        ["docker", "pause", owned_redis["name"]],
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        started = time.monotonic()
        with pytest.raises(ClusterEventUnavailable) as caught:
            await worker.publish("ping", {"n": 1}, timeout_s=10)
        waited = time.monotonic() - started
    finally:
        subprocess.run(
            ["docker", "unpause", owned_redis["name"]],
            check=True,
            capture_output=True,
            timeout=30,
        )
        await worker.close()

    assert waited < 5
    assert str(caught.value).startswith("cluster events: cannot publish 'ping': ")
