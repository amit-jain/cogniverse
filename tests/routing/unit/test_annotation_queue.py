"""
Tests for AnnotationQueue — the shared queue with reviewer assignment and SLA.

The queue keeps its requests in Redis; each test gets its own key prefix on
the test-owned Redis and a fixed clock.

Tests:
1. Queue state transitions: PENDING → ASSIGNED → COMPLETED
2. SLA deadline enforcement and expiration
3. Batch enqueue deduplication
4. Assignment validation (cannot assign non-PENDING)
5. Completion validation (cannot complete EXPIRED)
6. Statistics tracking
7. Priority sorting
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue, EnqueueOutcome
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome

pytestmark = [pytest.mark.unit]

T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def queue(shared_state_redis, clock):
    return AnnotationQueue(
        shared_state_redis,
        key_prefix=f"test:annotation-queue:{uuid.uuid4().hex}",
        clock=clock,
    )


def _make_request(
    span_id: str = "span-1",
    priority: AnnotationPriority = AnnotationPriority.MEDIUM,
    confidence: float = 0.5,
    timestamp: datetime = T0 - timedelta(hours=1),
) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=span_id,
        timestamp=timestamp,
        query="test query",
        chosen_agent="search_agent",
        routing_confidence=confidence,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=priority,
        reason="test reason",
        context={},
    )


class TestAnnotationQueueBasicOperations:
    async def test_enqueue_and_get_pending(self, queue):
        req = _make_request("span-1")
        await queue.enqueue(req)

        pending = (await queue.snapshot()).pending
        assert len(pending) == 1
        assert pending[0].span_id == "span-1"
        assert pending[0].status == AnnotationStatus.PENDING
        assert pending[0].to_dict() == req.to_dict()

    async def test_enqueue_deduplication(self, queue):
        assert await queue.enqueue(_make_request("span-1")) is True
        assert await queue.enqueue(_make_request("span-1")) is False
        assert (await queue.statistics())["total"] == 1

    async def test_enqueue_batch(self, queue):
        requests = [_make_request(f"span-{i}") for i in range(5)]
        added = await queue.enqueue_batch(requests)
        assert added == EnqueueOutcome(enqueued=5, total=5)
        assert (await queue.statistics())["total"] == 5

    async def test_enqueue_batch_deduplication(self, queue):
        await queue.enqueue(_make_request("span-0"))
        requests = [_make_request(f"span-{i}") for i in range(5)]
        added = await queue.enqueue_batch(requests)
        assert added.enqueued == 4  # span-0 already exists
        assert added.total == 5

    async def test_size_and_statistics(self, queue):
        await queue.enqueue(_make_request("s1", AnnotationPriority.HIGH))
        await queue.enqueue(_make_request("s2", AnnotationPriority.MEDIUM))
        await queue.enqueue(_make_request("s3", AnnotationPriority.LOW))

        stats = await queue.statistics()
        assert stats["total"] == 3
        assert stats["by_status"]["pending"] == 3
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["medium"] == 1
        assert stats["by_priority"]["low"] == 1
        assert stats == {
            "total": 3,
            "by_status": {"pending": 3},
            "by_priority": {"high": 1, "medium": 1, "low": 1},
        }

    async def test_get_returns_none_for_missing(self, queue):
        assert await queue.get("nonexistent") is None


class TestAnnotationQueueStateTransitions:
    async def test_assign_sets_status_and_metadata(self, queue):
        await queue.enqueue(_make_request("span-1"))

        result = await queue.assign("span-1", reviewer="alice")
        assert result.status == AnnotationStatus.ASSIGNED
        assert result.assigned_to == "alice"
        assert result.assigned_at == T0
        assert result.sla_deadline == T0 + timedelta(hours=24)
        assert result.assigned_at.tzinfo == timezone.utc
        assert result.sla_deadline.tzinfo == timezone.utc

    async def test_assign_with_custom_sla(self, queue):
        await queue.enqueue(_make_request("span-1"))

        result = await queue.assign("span-1", reviewer="bob", sla_hours=48)
        assert result.sla_deadline == T0 + timedelta(hours=48)

    async def test_assign_with_zero_hour_sla(self, queue):
        await queue.enqueue(_make_request("span-1"))

        result = await queue.assign("span-1", reviewer="bob", sla_hours=0)

        assert result.sla_deadline == T0
        assert result.assigned_at == T0

    async def test_assign_missing_span_raises(self, queue):
        with pytest.raises(KeyError, match="not found"):
            await queue.assign("nonexistent", reviewer="alice")

    async def test_assign_non_pending_raises(self, queue):
        await queue.enqueue(_make_request("span-1"))
        await queue.assign("span-1", reviewer="alice")

        with pytest.raises(ValueError, match="Cannot assign"):
            await queue.assign("span-1", reviewer="bob")
        assert (await queue.get("span-1")).assigned_to == "alice"

    async def test_complete_from_assigned(self, queue):
        await queue.enqueue(_make_request("span-1"))
        await queue.assign("span-1", reviewer="alice")

        result = await queue.complete("span-1", label="correct_routing")
        assert result.status == AnnotationStatus.COMPLETED
        assert result.completed_at == T0
        assert result.completed_at.tzinfo == timezone.utc
        # The reviewer's label is captured, not silently dropped.
        assert result.label == "correct_routing"
        assert result.to_dict()["label"] == "correct_routing"
        assert (await queue.get("span-1")).to_dict() == result.to_dict()

    async def test_complete_from_pending(self, queue):
        await queue.enqueue(_make_request("span-1"))

        result = await queue.complete("span-1")
        assert result.status == AnnotationStatus.COMPLETED

    async def test_complete_missing_span_raises(self, queue):
        with pytest.raises(KeyError, match="not found"):
            await queue.complete("nonexistent")

    async def test_complete_already_completed_raises(self, queue):
        await queue.enqueue(_make_request("span-1"))
        await queue.complete("span-1")

        with pytest.raises(ValueError, match="Cannot complete"):
            await queue.complete("span-1")


class TestAnnotationQueueSLAExpiration:
    async def test_get_expired_marks_past_deadline(self, queue, clock):
        await queue.enqueue(_make_request("span-1"))
        await queue.assign("span-1", reviewer="alice", sla_hours=0)

        # Move past the deadline
        clock.now = T0 + timedelta(hours=1)

        expired = (await queue.snapshot()).expired
        assert len(expired) == 1
        assert expired[0].span_id == "span-1"
        assert expired[0].status == AnnotationStatus.EXPIRED
        assert (await queue.get("span-1")).status == AnnotationStatus.EXPIRED

    async def test_not_expired_if_within_sla(self, queue, clock):
        await queue.enqueue(_make_request("span-1"))
        await queue.assign("span-1", reviewer="alice", sla_hours=24)
        clock.now = T0 + timedelta(hours=23)

        expired = (await queue.snapshot()).expired
        assert len(expired) == 0

    async def test_pending_items_not_expired(self, queue, clock):
        await queue.enqueue(_make_request("span-1"))
        clock.now = T0 + timedelta(days=30)
        expired = (await queue.snapshot()).expired
        assert len(expired) == 0

    async def test_default_sla_by_priority(self, queue):
        await queue.enqueue(_make_request("s1", AnnotationPriority.HIGH))
        await queue.enqueue(_make_request("s2", AnnotationPriority.LOW))

        await queue.assign("s1", reviewer="alice")
        await queue.assign("s2", reviewer="bob")

        high_req = await queue.get("s1")
        low_req = await queue.get("s2")

        # HIGH gets 4h SLA, LOW gets 72h SLA
        assert high_req.sla_deadline < low_req.sla_deadline
        assert (high_req.sla_deadline, low_req.sla_deadline) == (
            T0 + timedelta(hours=4),
            T0 + timedelta(hours=72),
        )


class TestAnnotationQueuePrioritySorting:
    async def test_pending_sorted_by_priority(self, queue):
        await queue.enqueue(_make_request("s-low", AnnotationPriority.LOW))
        await queue.enqueue(_make_request("s-high", AnnotationPriority.HIGH))
        await queue.enqueue(_make_request("s-med", AnnotationPriority.MEDIUM))

        pending = (await queue.snapshot()).pending
        assert pending[0].span_id == "s-high"
        assert pending[1].span_id == "s-med"
        assert pending[2].span_id == "s-low"


class TestAnnotationRequestSerialization:
    def test_to_dict_includes_queue_fields(self):
        req = _make_request("span-1")
        d = req.to_dict()

        assert d["status"] == "pending"
        assert d["assigned_to"] is None
        assert d["assigned_at"] is None
        assert d["sla_deadline"] is None
        assert d["completed_at"] is None

    async def test_to_dict_after_assignment(self, queue):
        await queue.enqueue(_make_request("span-1"))
        req = await queue.assign("span-1", reviewer="alice")
        d = req.to_dict()

        assert d["status"] == "assigned"
        assert d["assigned_to"] == "alice"
        assert d["assigned_at"] == "2026-09-01T12:00:00+00:00"
        assert d["sla_deadline"] == "2026-09-02T12:00:00+00:00"
        assert d["assigned_at"].endswith("+00:00")
        assert d["sla_deadline"].endswith("+00:00")

    def test_from_dict_rejects_timestamp_without_timezone(self):
        data = _make_request("span-1").to_dict()
        data["timestamp"] = "2026-07-26T12:00:00"

        with pytest.raises(ValueError, match="timestamp must include a timezone"):
            AnnotationRequest.from_dict(data)
