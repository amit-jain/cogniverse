"""The annotation queue every runtime process shares, against a real Redis.

Each queue under test gets its own key prefix. A second ``AnnotationQueue``
on its own client stands in for another runtime process: it shares nothing
with the first but Redis.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from redis.asyncio import Redis

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)
from cogniverse_agents.routing.annotation_queue import (
    AnnotationCompletionInProgressError,
    AnnotationQueue,
    AnnotationQueueFullError,
    AnnotationQueueUnavailableError,
    EnqueueOutcome,
)
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_runtime.shared_state import connect_shared_state_redis

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
    pytest.mark.asyncio,
]

T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now: datetime = T0):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


def _request(
    span_id: str,
    priority: AnnotationPriority = AnnotationPriority.MEDIUM,
    timestamp: datetime = T0 - timedelta(hours=1),
    **fields,
) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=span_id,
        timestamp=timestamp,
        query=fields.pop("query", "find clips of animals"),
        chosen_agent="search_agent",
        routing_confidence=fields.pop("routing_confidence", 0.42),
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=priority,
        reason="low confidence",
        context=fields.pop("context", {}),
        **fields,
    )


@pytest.fixture
def prefix():
    return f"test:annotation-queue:{uuid.uuid4().hex}"


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def queue(shared_state_redis, prefix, clock):
    return AnnotationQueue(shared_state_redis, key_prefix=prefix, clock=clock)


@pytest.fixture
async def peer_redis(shared_state_redis_url):
    client = await connect_shared_state_redis(shared_state_redis_url)
    yield client
    await client.aclose()


@pytest.fixture
def peer(peer_redis, prefix, clock):
    """The same queue as another process sees it."""
    return AnnotationQueue(peer_redis, key_prefix=prefix, clock=clock)


class TestEnqueue:
    async def test_a_request_round_trips_every_field_to_another_process(
        self, queue, peer
    ):
        request = _request(
            "span-1",
            context={"tags": [], "nested": {"ids": [1, 2]}, "empty": {}},
            routing_confidence=0.1 + 0.2,
            tenant_id="acme:acme",
            agent_type="search",
        )

        assert await queue.enqueue(request) is True

        stored = await peer.get("span-1")
        assert stored.to_dict() == {
            "span_id": "span-1",
            "timestamp": "2026-09-01T11:00:00+00:00",
            "query": "find clips of animals",
            "chosen_agent": "search_agent",
            "routing_confidence": 0.1 + 0.2,
            "outcome": "ambiguous",
            "priority": "medium",
            "reason": "low confidence",
            "context": {"tags": [], "nested": {"ids": [1, 2]}, "empty": {}},
            "status": "pending",
            "assigned_to": None,
            "assigned_at": None,
            "sla_deadline": None,
            "completed_at": None,
            "label": None,
            "agent_type": "search",
            "tenant_id": "acme:acme",
        }

    async def test_an_enqueued_request_enters_pending_unassigned(self, queue):
        request = _request("span-1", label="correct_routing")
        request.status = AnnotationStatus.COMPLETED
        request.assigned_to = "alice"

        await queue.enqueue(request)

        stored = await queue.get("span-1")
        assert (stored.status, stored.assigned_to, stored.label) == (
            AnnotationStatus.PENDING,
            None,
            "correct_routing",
        )

    async def test_a_batch_skips_spans_already_queued_or_repeated(self, queue):
        assert await queue.enqueue(_request("span-0")) is True
        assert await queue.enqueue(_request("span-0")) is False

        outcome = await queue.enqueue_batch(
            [
                _request("span-0"),
                _request("span-1"),
                _request("span-2"),
                _request("span-1"),
            ]
        )

        assert outcome == EnqueueOutcome(enqueued=2, total=3)

    async def test_statistics_count_by_status_and_priority(self, queue):
        await queue.enqueue_batch(
            [
                _request("s1", AnnotationPriority.HIGH),
                _request("s2", AnnotationPriority.MEDIUM),
                _request("s3", AnnotationPriority.LOW),
                _request("s4", AnnotationPriority.LOW),
            ]
        )
        await queue.assign("s1", reviewer="alice")
        await queue.complete("s2")

        assert await queue.statistics() == {
            "total": 4,
            "by_status": {"pending": 2, "assigned": 1, "completed": 1},
            "by_priority": {"high": 1, "medium": 1, "low": 2},
        }

    async def test_an_empty_queue_has_zero_statistics(self, queue):
        snapshot = await queue.snapshot()

        assert snapshot.statistics == {"total": 0, "by_status": {}, "by_priority": {}}
        assert (snapshot.pending, snapshot.assigned, snapshot.expired) == ([], [], [])
        assert await queue.get("missing") is None

    async def test_pending_requests_list_by_priority_then_timestamp(self, queue):
        await queue.enqueue_batch(
            [
                _request("low-old", AnnotationPriority.LOW, T0 - timedelta(hours=9)),
                _request("med-new", AnnotationPriority.MEDIUM, T0 - timedelta(hours=1)),
                _request("high-new", AnnotationPriority.HIGH, T0 - timedelta(hours=1)),
                _request("med-old", AnnotationPriority.MEDIUM, T0 - timedelta(hours=5)),
                _request("high-old", AnnotationPriority.HIGH, T0 - timedelta(hours=3)),
            ]
        )

        snapshot = await queue.snapshot(limit=4)

        assert [r.span_id for r in snapshot.pending] == [
            "high-old",
            "high-new",
            "med-old",
            "med-new",
        ]

    async def test_a_batch_past_the_open_cap_is_refused_whole(
        self, shared_state_redis, prefix, clock
    ):
        queue = AnnotationQueue(
            shared_state_redis, key_prefix=prefix, clock=clock, max_open=3
        )
        await queue.enqueue_batch([_request("open-1"), _request("open-2")])

        with pytest.raises(AnnotationQueueFullError) as refused:
            await queue.enqueue_batch([_request("new-1"), _request("new-2")])

        assert str(refused.value) == (
            "annotation queue holds 2 open requests; adding this batch would "
            "pass the limit of 3"
        )
        assert await queue.get("new-1") is None
        assert (await queue.statistics())["total"] == 2
        assert await queue.enqueue_batch(
            [_request("open-1"), _request("new-1")]
        ) == EnqueueOutcome(enqueued=1, total=3)


class TestAssign:
    async def test_assign_records_reviewer_and_deadline(self, queue, peer):
        await queue.enqueue(_request("span-1"))

        assigned = await queue.assign("span-1", reviewer="bob", sla_hours=48)

        assert assigned.to_dict() == (await peer.get("span-1")).to_dict()
        assert (
            assigned.status,
            assigned.assigned_to,
            assigned.assigned_at,
            assigned.sla_deadline,
        ) == (AnnotationStatus.ASSIGNED, "bob", T0, T0 + timedelta(hours=48))
        snapshot = await peer.snapshot()
        assert [r.span_id for r in snapshot.assigned] == ["span-1"]
        assert snapshot.pending == []

    async def test_the_default_sla_follows_priority(self, queue):
        await queue.enqueue_batch(
            [
                _request("high", AnnotationPriority.HIGH),
                _request("medium", AnnotationPriority.MEDIUM),
                _request("low", AnnotationPriority.LOW),
            ]
        )

        deadlines = [
            (await queue.assign(span, reviewer="alice")).sla_deadline
            for span in ("high", "medium", "low")
        ]

        assert deadlines == [
            T0 + timedelta(hours=4),
            T0 + timedelta(hours=24),
            T0 + timedelta(hours=72),
        ]

    async def test_assign_of_a_missing_span_raises_key_error(self, queue):
        with pytest.raises(KeyError) as missing:
            await queue.assign("nonexistent", reviewer="alice")

        assert missing.value.args == ("Span nonexistent not found in annotation queue",)

    async def test_assign_of_an_assigned_span_raises_value_error(self, queue):
        await queue.enqueue(_request("span-1"))
        await queue.assign("span-1", reviewer="alice")

        with pytest.raises(ValueError) as refused:
            await queue.assign("span-1", reviewer="bob")

        assert str(refused.value) == "Cannot assign span span-1: status is assigned"
        assert (await queue.get("span-1")).assigned_to == "alice"


class TestComplete:
    async def test_complete_from_assigned_records_label_and_time(
        self, queue, peer, clock
    ):
        await queue.enqueue(_request("span-1"))
        await queue.assign("span-1", reviewer="alice")
        clock.advance(minutes=5)

        completed = await queue.complete("span-1", label="correct_routing")

        assert completed.to_dict() == (await peer.get("span-1")).to_dict()
        assert (completed.status, completed.label, completed.completed_at) == (
            AnnotationStatus.COMPLETED,
            "correct_routing",
            T0 + timedelta(minutes=5),
        )
        assert (await peer.statistics())["by_status"] == {"completed": 1}

    async def test_complete_from_pending_without_label(self, queue):
        await queue.enqueue(_request("span-1"))

        completed = await queue.complete("span-1")

        assert (completed.status, completed.label) == (AnnotationStatus.COMPLETED, None)

    async def test_complete_of_a_completed_span_raises(self, queue):
        await queue.enqueue(_request("span-1"))
        await queue.complete("span-1")

        with pytest.raises(ValueError) as refused:
            await queue.complete("span-1")

        assert str(refused.value) == "Cannot complete span span-1: status is completed"

    async def test_complete_of_a_missing_span_raises_key_error(self, queue):
        with pytest.raises(KeyError) as missing:
            await queue.complete("nonexistent")

        assert missing.value.args == ("Span nonexistent not found in annotation queue",)

    async def test_a_held_claim_refuses_a_second_completion(self, queue, peer):
        await queue.enqueue(_request("span-1"))
        claim = await queue.begin_completion("span-1")

        with pytest.raises(AnnotationCompletionInProgressError) as busy:
            await peer.begin_completion("span-1")

        assert str(busy.value) == "Span span-1 is being completed by another request"
        await queue.abandon_completion(claim)
        retry = await peer.begin_completion("span-1")
        completed = await peer.finish_completion(retry, "wrong_routing")
        assert (completed.status, completed.label) == (
            AnnotationStatus.COMPLETED,
            "wrong_routing",
        )

    async def test_a_lapsed_claim_frees_the_request_and_loses_its_finish(
        self, shared_state_redis, peer_redis, prefix, clock
    ):
        first = AnnotationQueue(
            shared_state_redis, key_prefix=prefix, clock=clock, claim_seconds=60
        )
        second = AnnotationQueue(
            peer_redis, key_prefix=prefix, clock=clock, claim_seconds=60
        )
        await first.enqueue(_request("span-1"))
        stale = await first.begin_completion("span-1")
        clock.advance(seconds=61)
        taken = await second.begin_completion("span-1")

        with pytest.raises(AnnotationCompletionInProgressError) as lost:
            await first.finish_completion(stale, "correct_routing")

        assert str(lost.value) == (
            "Span span-1 completion claim lapsed and was taken over"
        )
        completed = await second.finish_completion(taken, "wrong_routing")
        assert completed.label == "wrong_routing"


class TestTimeRules:
    async def test_an_overdue_assignment_expires_on_read(self, queue, peer, clock):
        await queue.enqueue(_request("span-1"))
        await queue.enqueue(_request("span-2"))
        await queue.assign("span-1", reviewer="alice", sla_hours=1)
        await queue.assign("span-2", reviewer="bob", sla_hours=3)
        clock.advance(hours=2)

        snapshot = await peer.snapshot()

        assert [r.span_id for r in snapshot.expired] == ["span-1"]
        assert snapshot.expired[0].status == AnnotationStatus.EXPIRED
        assert [r.span_id for r in snapshot.assigned] == ["span-2"]
        assert snapshot.statistics["by_status"] == {"assigned": 1, "expired": 1}
        with pytest.raises(ValueError) as refused:
            await queue.complete("span-1")
        assert str(refused.value) == "Cannot complete span span-1: status is expired"

    async def test_a_claimed_overdue_assignment_is_not_expired(
        self, shared_state_redis, prefix, clock
    ):
        queue = AnnotationQueue(
            shared_state_redis, key_prefix=prefix, clock=clock, claim_seconds=3 * 3600
        )
        await queue.enqueue(_request("span-1"))
        await queue.assign("span-1", reviewer="alice", sla_hours=1)
        claim = await queue.begin_completion("span-1")
        clock.advance(hours=2)

        snapshot = await queue.snapshot()
        completed = await queue.finish_completion(claim, "correct_routing")

        assert [r.span_id for r in snapshot.assigned] == ["span-1"]
        assert snapshot.expired == []
        assert completed.status == AnnotationStatus.COMPLETED

    async def test_finished_requests_are_removed_after_retention(
        self, shared_state_redis, prefix, clock
    ):
        queue = AnnotationQueue(
            shared_state_redis, key_prefix=prefix, clock=clock, retention_seconds=86400
        )
        await queue.enqueue_batch(
            [
                _request("done", AnnotationPriority.HIGH),
                _request("open", AnnotationPriority.LOW),
            ]
        )
        await queue.complete("done")
        clock.advance(days=1)
        assert (await queue.statistics())["total"] == 2

        clock.advance(milliseconds=1)

        assert await queue.statistics() == {
            "total": 1,
            "by_status": {"pending": 1},
            "by_priority": {"low": 1},
        }
        assert await queue.get("done") is None
        assert (await queue.get("open")).status == AnnotationStatus.PENDING


class TestConcurrentProcesses:
    """Sixteen clients, each with its own connection pool, act at once."""

    CLIENTS = 16

    async def _queues(self, url, prefix, clock):
        clients = [await connect_shared_state_redis(url) for _ in range(self.CLIENTS)]
        return clients, [
            AnnotationQueue(client, key_prefix=prefix, clock=clock)
            for client in clients
        ]

    async def test_overlapping_batches_enqueue_each_span_once(
        self, shared_state_redis_url, prefix, clock
    ):
        clients, queues = await self._queues(shared_state_redis_url, prefix, clock)
        barrier = asyncio.Barrier(self.CLIENTS)
        spans = [f"span-{i}" for i in range(10)]

        async def enqueue(index, queue):
            batch = [
                _request(span) for span in spans[index % 10 :] + spans[: index % 10]
            ]
            await barrier.wait()
            return await queue.enqueue_batch(batch)

        try:
            outcomes = await asyncio.gather(
                *(enqueue(i, q) for i, q in enumerate(queues))
            )
            statistics = await queues[0].statistics()
        finally:
            for client in clients:
                await client.aclose()

        assert sum(outcome.enqueued for outcome in outcomes) == 10
        assert statistics == {
            "total": 10,
            "by_status": {"pending": 10},
            "by_priority": {"medium": 10},
        }

    async def test_concurrent_assigns_admit_exactly_one(
        self, shared_state_redis_url, prefix, clock
    ):
        clients, queues = await self._queues(shared_state_redis_url, prefix, clock)
        await queues[0].enqueue(_request("span-1"))
        barrier = asyncio.Barrier(self.CLIENTS)

        async def assign(index, queue):
            await barrier.wait()
            try:
                return (await queue.assign("span-1", reviewer=f"r{index}")).assigned_to
            except ValueError as refused:
                return str(refused)

        try:
            answers = await asyncio.gather(
                *(assign(i, q) for i, q in enumerate(queues))
            )
            stored = await queues[0].get("span-1")
        finally:
            for client in clients:
                await client.aclose()

        winners = [a for a in answers if not a.startswith("Cannot")]
        assert len(winners) == 1
        assert sorted(a for a in answers if a.startswith("Cannot")) == [
            "Cannot assign span span-1: status is assigned"
        ] * (self.CLIENTS - 1)
        assert stored.assigned_to == winners[0]

    async def test_concurrent_completions_admit_exactly_one_claim(
        self, shared_state_redis_url, prefix, clock
    ):
        clients, queues = await self._queues(shared_state_redis_url, prefix, clock)
        await queues[0].enqueue(_request("span-1"))
        barrier = asyncio.Barrier(self.CLIENTS)

        async def begin(queue):
            await barrier.wait()
            try:
                return await queue.begin_completion("span-1")
            except AnnotationCompletionInProgressError as busy:
                return str(busy)

        try:
            answers = await asyncio.gather(*(begin(q) for q in queues))
            claims = [a for a in answers if not isinstance(a, str)]
            completed = await queues[0].finish_completion(claims[0], "correct_routing")
        finally:
            for client in clients:
                await client.aclose()

        assert len(claims) == 1
        assert [a for a in answers if isinstance(a, str)] == [
            "Span span-1 is being completed by another request"
        ] * (self.CLIENTS - 1)
        assert completed.label == "correct_routing"


class _GatedStorage:
    """Stands in for the telemetry write of a reviewer's label.

    Records every write; a write waits for ``release`` once ``hold`` is set,
    runs ``during`` first when given, and raises when ``fail`` is set.
    """

    writes: list = []
    hold = False
    fail = False
    during = None
    entered: asyncio.Event
    release: asyncio.Event

    def __init__(self, tenant_id, agent_type="routing"):
        self.tenant_id = tenant_id

    @classmethod
    def reset(cls) -> None:
        cls.writes = []
        cls.hold = False
        cls.fail = False
        cls.during = None
        cls.entered = asyncio.Event()
        cls.release = asyncio.Event()

    async def store_human_annotation(
        self, span_id, label, reasoning, suggested_agent=None, annotator_id="human"
    ):
        type(self).entered.set()
        if type(self).hold:
            await type(self).release.wait()
        if type(self).during is not None:
            await asyncio.to_thread(type(self).during)
        if type(self).fail:
            raise RuntimeError("telemetry backend down")
        type(self).writes.append((self.tenant_id, span_id, label.value, annotator_id))
        return True


@pytest.fixture
def storage(monkeypatch):
    _GatedStorage.reset()
    monkeypatch.setattr(
        "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
        _GatedStorage,
    )
    return _GatedStorage


@pytest.fixture
def route():
    """The runtime's agents router; ``route(queue)`` serves ``queue``."""
    import httpx
    from fastapi import FastAPI

    from cogniverse_runtime.routers import agents as agents_router

    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    previous = agents_router._annotation_queue

    def serve(queue: AnnotationQueue) -> httpx.AsyncClient:
        agents_router.set_annotation_queue(queue)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        )

    yield serve
    agents_router._annotation_queue = previous


def _owned(span_id: str) -> AnnotationRequest:
    return _request(span_id, tenant_id="acme:acme")


# Run in a fresh interpreter, where the label types have never been imported:
# completes one request with a label while a ticker measures the longest gap
# between turns of the event loop.
_FIRST_LABEL_PROBE = """
import asyncio, json, sys, time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority, AnnotationRequest,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_runtime.routers import agents
from cogniverse_runtime.shared_state import connect_shared_state_redis

LABEL_TYPES = "cogniverse_agents.routing.llm_auto_annotator"


async def main(url, prefix):
    redis = await connect_shared_state_redis(url)
    queue = AnnotationQueue(redis, key_prefix=prefix)
    await queue.enqueue(AnnotationRequest(
        span_id="span-1",
        timestamp=datetime(2026, 9, 1, 11, tzinfo=timezone.utc),
        query="find clips of animals",
        chosen_agent="search_agent",
        routing_confidence=0.42,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=AnnotationPriority.MEDIUM,
        reason="low confidence",
        context={},
    ))
    agents.set_annotation_queue(queue)
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    loaded_before = LABEL_TYPES in sys.modules
    gaps = []

    async def tick():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.05)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    ) as client:
        answer = await client.post(
            "/agents/annotations/queue/span-1/complete", json={"label": "correct"}
        )
    ticker.cancel()
    await redis.delete(*[key async for key in redis.scan_iter(prefix + ":*")])
    await redis.aclose()
    print(json.dumps({
        "loaded_before": loaded_before,
        "loaded_after": LABEL_TYPES in sys.modules,
        "status": answer.status_code,
        "persisted": answer.json()["persisted"],
        "longest_gap_s": round(max(gaps), 3),
    }))


asyncio.run(main(sys.argv[1], sys.argv[2]))
"""


class TestCompleteRoute:
    """``POST /agents/annotations/queue/{span_id}/complete`` claims the request
    before it writes the reviewer's label."""

    async def test_the_first_labelled_completion_keeps_the_event_loop_turning(
        self, shared_state_redis_url, prefix, tmp_path
    ):
        """The label types load litellm on first use, and litellm's import
        fetches its model cost map over the network with a five-second
        timeout. Here that fetch reaches a peer that never answers; the route
        must not hold the loop that serves every other request meanwhile."""
        probe = tmp_path / "first_label_probe.py"
        probe.write_text(_FIRST_LABEL_PROBE)
        env = {
            k: v for k, v in os.environ.items() if k != "LITELLM_LOCAL_MODEL_COST_MAP"
        }

        with socket.socket() as silent_peer:
            silent_peer.bind(("127.0.0.1", 0))
            silent_peer.listen(8)
            env["LITELLM_MODEL_COST_MAP_URL"] = (
                f"http://127.0.0.1:{silent_peer.getsockname()[1]}/costs.json"
            )
            finished = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, str(probe), shared_state_redis_url, prefix],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )

        assert finished.returncode == 0, finished.stderr[-4000:]
        measured = json.loads(finished.stdout.strip().splitlines()[-1])
        assert measured["longest_gap_s"] < 1.0, measured
        assert {k: v for k, v in measured.items() if k != "longest_gap_s"} == {
            "loaded_before": False,
            "loaded_after": True,
            "status": 200,
            "persisted": False,
        }

    async def test_concurrent_completions_write_one_label(self, queue, route, storage):
        await queue.enqueue(_owned("span-1"))
        storage.hold = True
        body = {"label": "correct", "annotator": "reviewer"}

        async with route(queue) as client:
            winner = asyncio.create_task(
                client.post("/agents/annotations/queue/span-1/complete", json=body)
            )
            await storage.entered.wait()
            contenders = [
                asyncio.create_task(
                    client.post("/agents/annotations/queue/span-1/complete", json=body)
                )
                for _ in range(15)
            ]
            # Each contender answers while the winner still holds the request.
            _, waiting = await asyncio.wait(contenders, timeout=10)
            storage.release.set()
            won = await winner
            losers = await asyncio.gather(*contenders)

        assert waiting == set()
        assert [(r.status_code, r.json()) for r in losers] == [
            (409, {"detail": "Span span-1 is being completed by another request"})
        ] * 15
        assert won.status_code == 200, won.text
        assert (won.json()["persisted"], won.json()["annotation"]["label"]) == (
            True,
            "correct",
        )
        assert storage.writes == [("acme:acme", "span-1", "correct", "reviewer")]
        assert (await queue.get("span-1")).status == AnnotationStatus.COMPLETED

    async def test_a_failed_label_write_frees_the_request_for_a_retry(
        self, queue, route, storage
    ):
        await queue.enqueue(_owned("span-1"))
        storage.fail = True

        async with route(queue) as client:
            failed = await client.post(
                "/agents/annotations/queue/span-1/complete", json={"label": "wrong"}
            )
            status_after_failure = (await queue.get("span-1")).status
            storage.fail = False
            retried = await client.post(
                "/agents/annotations/queue/span-1/complete", json={"label": "wrong"}
            )

        assert (failed.status_code, failed.json()) == (
            502,
            {
                "detail": "Annotation could not be persisted to the telemetry "
                "backend; the item remains open for retry."
            },
        )
        assert status_after_failure == AnnotationStatus.PENDING
        assert retried.status_code == 200, retried.text
        assert retried.json()["annotation"]["status"] == "completed"
        assert storage.writes == [("acme:acme", "span-1", "wrong", "human")]

    async def test_a_queue_outage_after_the_label_write_is_503_until_the_claim_lapses(
        self, own_redis, prefix, route, storage
    ):
        """Redis stops answering between the label write and the completion:
        the route answers 503 and the request stays claimed; once the claim
        lapses a retry completes it, writing the same label again."""
        url, pause, resume = own_redis
        claim_seconds = 10
        client = await connect_shared_state_redis(url, timeout_seconds=1.5)
        queue = AnnotationQueue(client, key_prefix=prefix, claim_seconds=claim_seconds)
        await queue.enqueue(_owned("span-1"))
        storage.during = pause
        body = {"label": "correct"}

        try:
            async with route(queue) as http:
                claimed_by = time.monotonic()
                down = await http.post(
                    "/agents/annotations/queue/span-1/complete", json=body
                )
                storage.during = None
                resume()
                held = await http.post(
                    "/agents/annotations/queue/span-1/complete", json=body
                )
                held_after = time.monotonic() - claimed_by
                await asyncio.sleep(claimed_by + claim_seconds + 2 - time.monotonic())
                retried = await http.post(
                    "/agents/annotations/queue/span-1/complete", json=body
                )
        finally:
            await client.aclose()

        assert (down.status_code, down.json()) == (
            503,
            {"detail": "annotation queue unavailable: complete span span-1"},
        )
        assert held_after < claim_seconds, held_after
        assert (held.status_code, held.json()) == (
            409,
            {"detail": "Span span-1 is being completed by another request"},
        )
        assert retried.status_code == 200, retried.text
        assert retried.json()["annotation"]["status"] == "completed"
        assert storage.writes == [("acme:acme", "span-1", "correct", "human")] * 2


class TestRedisFailures:
    async def test_every_operation_raises_when_redis_is_unreachable(
        self, dead_redis_url, prefix
    ):
        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        queue = AnnotationQueue(client, key_prefix=prefix)
        operations = {
            "enqueue requests": lambda: queue.enqueue_batch([_request("span-1")]),
            "get span span-1": lambda: queue.get("span-1"),
            "claim span span-1": lambda: queue.begin_completion("span-1"),
            "read queue": lambda: queue.snapshot(),
        }
        try:
            for operation, call in operations.items():
                with pytest.raises(AnnotationQueueUnavailableError) as failed:
                    await call()
                assert str(failed.value) == f"annotation queue unavailable: {operation}"
            with pytest.raises(AnnotationQueueUnavailableError) as failed:
                await queue.assign("span-1", reviewer="alice")
            assert str(failed.value) == "annotation queue unavailable: get span span-1"
        finally:
            await client.aclose()

    async def test_a_hung_redis_fails_within_the_command_timeout(
        self, own_redis, prefix
    ):
        url, pause, _ = own_redis
        client = await connect_shared_state_redis(url, timeout_seconds=1.5)
        queue = AnnotationQueue(client, key_prefix=prefix)
        await queue.enqueue(_request("span-1"))
        pause()
        started = time.monotonic()
        try:
            with pytest.raises(AnnotationQueueUnavailableError) as failed:
                await queue.snapshot()
            elapsed = time.monotonic() - started
        finally:
            await client.aclose()

        assert str(failed.value) == "annotation queue unavailable: read queue"
        assert 1.5 <= elapsed < 4.0, elapsed
