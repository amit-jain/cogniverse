"""
Persistent Annotation Queue with reviewer assignment and SLA tracking.

Manages the lifecycle of annotation requests:
  PENDING → ASSIGNED → COMPLETED
                    ↘ EXPIRED (if SLA deadline passes)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_SLA_HOURS = {
    AnnotationPriority.HIGH: 4,
    AnnotationPriority.MEDIUM: 24,
    AnnotationPriority.LOW: 72,
}


class AnnotationQueue:
    """
    In-memory annotation queue with assignment, SLA tracking, and status transitions.

    Provides:
    - enqueue(): Add annotation requests
    - assign(): Assign a request to a reviewer with SLA deadline
    - complete(): Mark a request as completed
    - get_pending(): List unassigned requests
    - get_assigned(): List assigned requests
    - get_expired(): List requests past SLA deadline
    """

    def __init__(
        self,
        sla_hours: Dict[AnnotationPriority, int] | None = None,
    ):
        self._queue: Dict[str, AnnotationRequest] = {}
        self._sla_hours = sla_hours or dict(DEFAULT_SLA_HOURS)

    def enqueue(self, request: AnnotationRequest) -> None:
        """Add an annotation request to the queue."""
        if request.span_id in self._queue:
            logger.debug(f"Span {request.span_id} already in queue, skipping")
            return
        request.status = AnnotationStatus.PENDING
        self._queue[request.span_id] = request

    def enqueue_batch(self, requests: List[AnnotationRequest]) -> int:
        """Add multiple requests, returning count of newly enqueued."""
        added = 0
        for req in requests:
            if req.span_id not in self._queue:
                self.enqueue(req)
                added += 1
        return added

    def assign(
        self,
        span_id: str,
        reviewer: str,
        sla_hours: int | None = None,
    ) -> AnnotationRequest:
        """
        Assign a pending request to a reviewer.

        Args:
            span_id: The span to assign
            reviewer: Reviewer identifier
            sla_hours: Override SLA deadline (hours from now)

        Returns:
            The updated AnnotationRequest

        Raises:
            KeyError: If span_id not in queue
            ValueError: If request is not in PENDING status
        """
        request = self._get_or_raise(span_id)
        if request.status != AnnotationStatus.PENDING:
            raise ValueError(
                f"Cannot assign span {span_id}: status is {request.status.value}"
            )

        now = datetime.now()
        hours = sla_hours or self._sla_hours.get(request.priority, 24)

        request.status = AnnotationStatus.ASSIGNED
        request.assigned_to = reviewer
        request.assigned_at = now
        request.sla_deadline = now + timedelta(hours=hours)
        return request

    def complete(
        self,
        span_id: str,
        label: str | None = None,
    ) -> AnnotationRequest:
        """
        Mark an assigned request as completed.

        Args:
            span_id: The span to complete
            label: Optional annotation label

        Returns:
            The updated AnnotationRequest

        Raises:
            KeyError: If span_id not in queue
            ValueError: If request is not ASSIGNED or PENDING
        """
        request = self._get_or_raise(span_id)
        if request.status not in (AnnotationStatus.ASSIGNED, AnnotationStatus.PENDING):
            raise ValueError(
                f"Cannot complete span {span_id}: status is {request.status.value}"
            )

        request.status = AnnotationStatus.COMPLETED
        request.completed_at = datetime.now()
        return request

    def get_pending(self) -> List[AnnotationRequest]:
        """Return all PENDING requests, sorted by priority then timestamp."""
        pending = [
            r for r in self._queue.values() if r.status == AnnotationStatus.PENDING
        ]
        return self._sort_by_priority(pending)

    def get_assigned(self) -> List[AnnotationRequest]:
        """Return all ASSIGNED requests."""
        return [
            r for r in self._queue.values() if r.status == AnnotationStatus.ASSIGNED
        ]

    def get_completed(self) -> List[AnnotationRequest]:
        """Return all COMPLETED requests."""
        return [
            r for r in self._queue.values() if r.status == AnnotationStatus.COMPLETED
        ]

    def get_expired(self) -> List[AnnotationRequest]:
        """
        Find and mark ASSIGNED requests past their SLA deadline as EXPIRED.
        Returns the list of newly expired requests.
        """
        now = datetime.now()
        expired = []
        for request in self._queue.values():
            if (
                request.status == AnnotationStatus.ASSIGNED
                and request.sla_deadline
                and now > request.sla_deadline
            ):
                request.status = AnnotationStatus.EXPIRED
                expired.append(request)
        return expired

    def get(self, span_id: str) -> AnnotationRequest | None:
        """Get a request by span_id, or None."""
        return self._queue.get(span_id)

    def size(self) -> int:
        """Total number of requests in queue."""
        return len(self._queue)

    def statistics(self) -> Dict:
        """Get queue statistics by status and priority."""
        by_status: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        for r in self._queue.values():
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
            by_priority[r.priority.value] = by_priority.get(r.priority.value, 0) + 1
        return {
            "total": self.size(),
            "by_status": by_status,
            "by_priority": by_priority,
        }

    def _get_or_raise(self, span_id: str) -> AnnotationRequest:
        request = self._queue.get(span_id)
        if request is None:
            raise KeyError(f"Span {span_id} not found in annotation queue")
        return request

    @staticmethod
    def _sort_by_priority(
        requests: List[AnnotationRequest],
    ) -> List[AnnotationRequest]:
        priority_order = {
            AnnotationPriority.HIGH: 0,
            AnnotationPriority.MEDIUM: 1,
            AnnotationPriority.LOW: 2,
        }
        return sorted(
            requests,
            key=lambda r: (priority_order[r.priority], r.timestamp),
        )
