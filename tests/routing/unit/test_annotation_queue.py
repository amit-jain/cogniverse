"""
Unit tests for AnnotationQueue — persistent queue with reviewer assignment and SLA.

Tests:
1. Queue state transitions: PENDING → ASSIGNED → COMPLETED
2. SLA deadline enforcement and expiration
3. Batch enqueue deduplication
4. Assignment validation (cannot assign non-PENDING)
5. Completion validation (cannot complete EXPIRED)
6. Statistics tracking
7. Priority sorting
"""

from datetime import datetime, timedelta

import pytest

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome


def _make_request(
    span_id: str = "span-1",
    priority: AnnotationPriority = AnnotationPriority.MEDIUM,
    confidence: float = 0.5,
) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=span_id,
        timestamp=datetime.now(),
        query="test query",
        chosen_agent="search_agent",
        routing_confidence=confidence,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=priority,
        reason="test reason",
        context={},
    )


class TestAnnotationQueueBasicOperations:
    def test_enqueue_and_get_pending(self):
        queue = AnnotationQueue()
        req = _make_request("span-1")
        queue.enqueue(req)

        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].span_id == "span-1"
        assert pending[0].status == AnnotationStatus.PENDING

    def test_enqueue_deduplication(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        queue.enqueue(_make_request("span-1"))
        assert queue.size() == 1

    def test_enqueue_batch(self):
        queue = AnnotationQueue()
        requests = [_make_request(f"span-{i}") for i in range(5)]
        added = queue.enqueue_batch(requests)
        assert added == 5
        assert queue.size() == 5

    def test_enqueue_batch_deduplication(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-0"))
        requests = [_make_request(f"span-{i}") for i in range(5)]
        added = queue.enqueue_batch(requests)
        assert added == 4  # span-0 already exists
        assert queue.size() == 5

    def test_size_and_statistics(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("s1", AnnotationPriority.HIGH))
        queue.enqueue(_make_request("s2", AnnotationPriority.MEDIUM))
        queue.enqueue(_make_request("s3", AnnotationPriority.LOW))

        assert queue.size() == 3
        stats = queue.statistics()
        assert stats["total"] == 3
        assert stats["by_status"]["pending"] == 3
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["medium"] == 1
        assert stats["by_priority"]["low"] == 1

    def test_get_returns_none_for_missing(self):
        queue = AnnotationQueue()
        assert queue.get("nonexistent") is None


class TestAnnotationQueueStateTransitions:
    def test_assign_sets_status_and_metadata(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))

        result = queue.assign("span-1", reviewer="alice")
        assert result.status == AnnotationStatus.ASSIGNED
        assert result.assigned_to == "alice"
        assert result.assigned_at is not None
        assert result.sla_deadline is not None

    def test_assign_with_custom_sla(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))

        before = datetime.now()
        result = queue.assign("span-1", reviewer="bob", sla_hours=48)
        assert result.sla_deadline > before + timedelta(hours=47)

    def test_assign_missing_span_raises(self):
        queue = AnnotationQueue()
        with pytest.raises(KeyError, match="not found"):
            queue.assign("nonexistent", reviewer="alice")

    def test_assign_non_pending_raises(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        queue.assign("span-1", reviewer="alice")

        with pytest.raises(ValueError, match="Cannot assign"):
            queue.assign("span-1", reviewer="bob")

    def test_complete_from_assigned(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        queue.assign("span-1", reviewer="alice")

        result = queue.complete("span-1", label="correct_routing")
        assert result.status == AnnotationStatus.COMPLETED
        assert result.completed_at is not None

    def test_complete_from_pending(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))

        result = queue.complete("span-1")
        assert result.status == AnnotationStatus.COMPLETED

    def test_complete_missing_span_raises(self):
        queue = AnnotationQueue()
        with pytest.raises(KeyError, match="not found"):
            queue.complete("nonexistent")

    def test_complete_already_completed_raises(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        queue.complete("span-1")

        with pytest.raises(ValueError, match="Cannot complete"):
            queue.complete("span-1")


class TestAnnotationQueueSLAExpiration:
    def test_get_expired_marks_past_deadline(self):
        queue = AnnotationQueue()
        req = _make_request("span-1")
        queue.enqueue(req)
        queue.assign("span-1", reviewer="alice", sla_hours=0)

        # Force deadline to be in the past
        req.sla_deadline = datetime.now() - timedelta(hours=1)

        expired = queue.get_expired()
        assert len(expired) == 1
        assert expired[0].span_id == "span-1"
        assert expired[0].status == AnnotationStatus.EXPIRED

    def test_not_expired_if_within_sla(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        queue.assign("span-1", reviewer="alice", sla_hours=24)

        expired = queue.get_expired()
        assert len(expired) == 0

    def test_pending_items_not_expired(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        expired = queue.get_expired()
        assert len(expired) == 0

    def test_default_sla_by_priority(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("s1", AnnotationPriority.HIGH))
        queue.enqueue(_make_request("s2", AnnotationPriority.LOW))

        queue.assign("s1", reviewer="alice")
        queue.assign("s2", reviewer="bob")

        high_req = queue.get("s1")
        low_req = queue.get("s2")

        # HIGH gets 4h SLA, LOW gets 72h SLA
        assert high_req.sla_deadline < low_req.sla_deadline


class TestAnnotationQueuePrioritySorting:
    def test_pending_sorted_by_priority(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("s-low", AnnotationPriority.LOW))
        queue.enqueue(_make_request("s-high", AnnotationPriority.HIGH))
        queue.enqueue(_make_request("s-med", AnnotationPriority.MEDIUM))

        pending = queue.get_pending()
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

    def test_to_dict_after_assignment(self):
        queue = AnnotationQueue()
        queue.enqueue(_make_request("span-1"))
        req = queue.assign("span-1", reviewer="alice")
        d = req.to_dict()

        assert d["status"] == "assigned"
        assert d["assigned_to"] == "alice"
        assert d["assigned_at"] is not None
        assert d["sla_deadline"] is not None
