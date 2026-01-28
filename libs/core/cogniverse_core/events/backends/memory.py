"""
In-Memory EventQueue Backend

Lightweight implementation for single-pod deployments and development.
For multi-pod production, use Redis Pub/Sub backend.

Features:
- Multiple subscribers per queue
- Short-term replay (configurable TTL)
- Backpressure handling (max buffer size)
- Cancellation signal support
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from cogniverse_core.events.queue import BaseEventQueue, BaseQueueManager, EventQueue
from cogniverse_core.events.types import TaskEvent

logger = logging.getLogger(__name__)


class InMemoryEventQueue(BaseEventQueue):
    """
    In-memory EventQueue implementation.

    Uses asyncio.Condition for efficient subscriber notification.
    """

    def __init__(
        self,
        task_id: str,
        tenant_id: str,
        ttl_minutes: int = 30,
        max_buffer_size: int = 1000,
    ):
        super().__init__(task_id, tenant_id, ttl_minutes)
        self._events: List[TaskEvent] = []
        self._condition = asyncio.Condition()
        self._subscriber_count = 0
        self._max_buffer_size = max_buffer_size
        self._last_activity = datetime.utcnow()

    async def enqueue(self, event: TaskEvent) -> None:
        """
        Add event to the queue.

        Notifies all waiting subscribers.
        Drops oldest events if buffer is full.
        """
        if self._closed:
            raise RuntimeError(f"Queue {self._task_id} is closed")

        async with self._condition:
            # Handle backpressure - drop oldest events if buffer full
            if len(self._events) >= self._max_buffer_size:
                dropped = self._events.pop(0)
                logger.warning(
                    f"Queue {self._task_id} buffer full, dropped oldest event: "
                    f"{dropped.event_id}"
                )

            self._events.append(event)
            self._last_activity = datetime.utcnow()

            logger.debug(
                f"Queue {self._task_id}: enqueued {event.event_type} "
                f"(offset {len(self._events) - 1})"
            )

            # Notify all waiting subscribers
            self._condition.notify_all()

    async def subscribe(self, from_offset: int = 0) -> AsyncIterator[TaskEvent]:
        """
        Subscribe to events.

        Yields events starting from from_offset.
        Waits for new events when caught up.
        Terminates when queue is closed.
        """
        self._subscriber_count += 1
        current_offset = from_offset

        logger.info(
            f"Queue {self._task_id}: new subscriber (#{self._subscriber_count}) "
            f"from offset {from_offset}"
        )

        try:
            while not self._closed:
                async with self._condition:
                    # Yield all available events
                    while current_offset < len(self._events):
                        event = self._events[current_offset]
                        current_offset += 1
                        yield event

                    # If closed, stop
                    if self._closed:
                        break

                    # Wait for new events or close
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=1.0,  # Check for close periodically
                        )
                    except asyncio.TimeoutError:
                        # Just loop back to check conditions
                        pass

        finally:
            self._subscriber_count -= 1
            logger.info(
                f"Queue {self._task_id}: subscriber disconnected "
                f"({self._subscriber_count} remaining)"
            )

    async def get_latest_offset(self) -> int:
        """Get current event count"""
        async with self._condition:
            return len(self._events)

    async def close(self) -> None:
        """Close queue and notify all subscribers"""
        if self._closed:
            return

        logger.info(f"Closing queue {self._task_id}")

        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "task_id": self._task_id,
            "tenant_id": self._tenant_id,
            "event_count": len(self._events),
            "subscriber_count": self._subscriber_count,
            "is_closed": self._closed,
            "is_cancelled": self._cancellation_token.is_cancelled,
            "is_expired": self.is_expired,
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "ttl_minutes": self._ttl.total_seconds() / 60,
        }


class InMemoryQueueManager(BaseQueueManager):
    """
    In-memory QueueManager implementation.

    Manages lifecycle of InMemoryEventQueue instances.
    """

    def __init__(self, default_ttl_minutes: int = 30, max_buffer_size: int = 1000):
        self._queues: Dict[str, InMemoryEventQueue] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl_minutes
        self._max_buffer_size = max_buffer_size
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_loop(self, interval_seconds: int = 60) -> None:
        """Start background cleanup task"""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    count = await self.cleanup_expired()
                    if count > 0:
                        logger.info(f"Cleaned up {count} expired queues")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup loop error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started queue cleanup loop (interval: {interval_seconds}s)")

    async def stop_cleanup_loop(self) -> None:
        """Stop background cleanup task"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped queue cleanup loop")

    async def create_queue(
        self,
        task_id: str,
        tenant_id: str,
        ttl_minutes: int = 30,
    ) -> EventQueue:
        """Create new queue"""
        async with self._lock:
            if task_id in self._queues:
                raise ValueError(f"Queue already exists for task {task_id}")

            queue = InMemoryEventQueue(
                task_id=task_id,
                tenant_id=tenant_id,
                ttl_minutes=ttl_minutes or self._default_ttl,
                max_buffer_size=self._max_buffer_size,
            )
            self._queues[task_id] = queue

            logger.info(
                f"Created queue for task {task_id} "
                f"(tenant: {tenant_id}, ttl: {ttl_minutes}min)"
            )

            return queue

    async def get_queue(self, task_id: str) -> Optional[EventQueue]:
        """Get existing queue"""
        async with self._lock:
            queue = self._queues.get(task_id)

            # Don't return expired queues
            if queue is not None and queue.is_expired:
                logger.info(f"Queue {task_id} is expired, removing")
                await queue.close()
                del self._queues[task_id]
                return None

            return queue

    async def close_queue(self, task_id: str) -> bool:
        """Close and remove queue"""
        async with self._lock:
            queue = self._queues.get(task_id)
            if queue is None:
                return False

            await queue.close()
            del self._queues[task_id]

            logger.info(f"Closed and removed queue {task_id}")
            return True

    async def list_active_queues(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List active queues"""
        async with self._lock:
            queues_info = []

            for queue in self._queues.values():
                if tenant_id is not None and queue.tenant_id != tenant_id:
                    continue

                if queue.is_expired or queue.is_closed:
                    continue

                queues_info.append(queue.get_stats())

            return queues_info

    async def cleanup_expired(self) -> int:
        """Clean up expired queues"""
        async with self._lock:
            expired_tasks = [
                task_id
                for task_id, queue in self._queues.items()
                if queue.is_expired or queue.is_closed
            ]

            for task_id in expired_tasks:
                queue = self._queues.pop(task_id)
                if not queue.is_closed:
                    await queue.close()
                logger.debug(f"Cleaned up expired queue {task_id}")

            return len(expired_tasks)

    async def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a task by task_id.

        Args:
            task_id: Task to cancel
            reason: Optional cancellation reason

        Returns:
            True if task was cancelled, False if not found
        """
        queue = await self.get_queue(task_id)
        if queue is None:
            return False

        queue.cancel(reason)
        logger.info(f"Cancelled task {task_id}: {reason or 'no reason provided'}")
        return True

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            "total_queues": len(self._queues),
            "active_queues": len(
                [
                    q
                    for q in self._queues.values()
                    if not q.is_closed and not q.is_expired
                ]
            ),
            "closed_queues": len([q for q in self._queues.values() if q.is_closed]),
            "expired_queues": len([q for q in self._queues.values() if q.is_expired]),
            "default_ttl_minutes": self._default_ttl,
            "max_buffer_size": self._max_buffer_size,
        }


# Global singleton instance for convenience
_global_queue_manager: Optional[InMemoryQueueManager] = None


def get_queue_manager(
    default_ttl_minutes: int = 30,
    max_buffer_size: int = 1000,
) -> InMemoryQueueManager:
    """
    Get or create global QueueManager instance.

    Use this for application-wide event queuing.
    """
    global _global_queue_manager

    if _global_queue_manager is None:
        _global_queue_manager = InMemoryQueueManager(
            default_ttl_minutes=default_ttl_minutes,
            max_buffer_size=max_buffer_size,
        )

    return _global_queue_manager


def reset_queue_manager() -> None:
    """Reset global QueueManager (for testing)"""
    global _global_queue_manager
    _global_queue_manager = None
