"""
EventQueue and QueueManager Protocols

Backend-agnostic interfaces for real-time event notification.
EventQueue is for real-time client notifications, NOT durable storage.
Checkpoints handle crash recovery; EventQueue handles live streaming.

Design Principles:
- Backend-agnostic protocols allow swapping implementations
- Multiple subscribers per task (dashboard + CLI can watch same workflow)
- Short-term replay for client reconnection (~30 min TTL)
- Cancellation signal for aborting long-running operations
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

from cogniverse_core.events.types import TaskEvent


class CancellationToken:
    """
    Token for signaling task cancellation.

    Subscribers can check is_cancelled to gracefully terminate.
    """

    def __init__(self):
        self._cancelled = False
        self._cancel_time: Optional[datetime] = None
        self._reason: Optional[str] = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_time(self) -> Optional[datetime]:
        return self._cancel_time

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def cancel(self, reason: Optional[str] = None) -> None:
        """Signal cancellation"""
        self._cancelled = True
        self._cancel_time = datetime.utcnow()
        self._reason = reason


@runtime_checkable
class EventQueue(Protocol):
    """
    Protocol for task event queues.

    Handles event streaming for a single task (workflow or job).
    Supports multiple subscribers and short-term replay.
    """

    @property
    def task_id(self) -> str:
        """Task ID this queue is for"""
        ...

    @property
    def tenant_id(self) -> str:
        """Tenant ID for isolation"""
        ...

    @property
    def cancellation_token(self) -> CancellationToken:
        """Token for checking/signaling cancellation"""
        ...

    async def enqueue(self, event: TaskEvent) -> None:
        """
        Add an event to the queue.

        Args:
            event: Event to enqueue

        All subscribers will receive this event.
        """
        ...

    async def subscribe(self, from_offset: int = 0) -> AsyncIterator[TaskEvent]:
        """
        Subscribe to events from this queue.

        Args:
            from_offset: Start from this offset (0 = from beginning of buffer)

        Yields:
            Events in order. Supports replay from offset for reconnection.

        Multiple subscribers can watch the same queue simultaneously.
        """
        ...

    async def get_latest_offset(self) -> int:
        """
        Get the latest event offset.

        Returns:
            Current offset (number of events in queue)
        """
        ...

    async def close(self) -> None:
        """
        Close the queue.

        Signals to all subscribers that no more events will arrive.
        Cleanup of resources.
        """
        ...

    def cancel(self, reason: Optional[str] = None) -> None:
        """
        Signal cancellation for this task.

        Sets the cancellation token. Producers should check this
        and stop producing events.
        """
        ...


@runtime_checkable
class QueueManager(Protocol):
    """
    Protocol for managing event queues.

    Creates, retrieves, and manages lifecycle of EventQueues.
    """

    async def create_queue(
        self,
        task_id: str,
        tenant_id: str,
        ttl_minutes: int = 30,
    ) -> EventQueue:
        """
        Create a new event queue for a task.

        Args:
            task_id: Task identifier (workflow_id or job_id)
            tenant_id: Tenant identifier
            ttl_minutes: Time-to-live for events (default 30 min)

        Returns:
            New EventQueue instance

        Raises:
            ValueError: If queue already exists for task_id
        """
        ...

    async def get_queue(self, task_id: str) -> Optional[EventQueue]:
        """
        Get an existing queue by task ID.

        Args:
            task_id: Task identifier

        Returns:
            EventQueue if exists, None otherwise
        """
        ...

    async def get_or_create_queue(
        self,
        task_id: str,
        tenant_id: str,
        ttl_minutes: int = 30,
    ) -> EventQueue:
        """
        Get existing queue or create new one.

        Args:
            task_id: Task identifier
            tenant_id: Tenant identifier
            ttl_minutes: TTL for new queue (ignored if queue exists)

        Returns:
            EventQueue (existing or newly created)
        """
        ...

    async def close_queue(self, task_id: str) -> bool:
        """
        Close and remove a queue.

        Args:
            task_id: Task identifier

        Returns:
            True if queue was closed, False if not found
        """
        ...

    async def list_active_queues(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all active queues.

        Args:
            tenant_id: Optional filter by tenant

        Returns:
            List of queue info dicts with task_id, tenant_id, event_count, etc.
        """
        ...

    async def cleanup_expired(self) -> int:
        """
        Clean up expired queues.

        Returns:
            Number of queues cleaned up
        """
        ...


class BaseEventQueue(ABC):
    """
    Abstract base class for EventQueue implementations.

    Provides common functionality and enforces interface.
    """

    def __init__(self, task_id: str, tenant_id: str, ttl_minutes: int = 30):
        self._task_id = task_id
        self._tenant_id = tenant_id
        self._ttl = timedelta(minutes=ttl_minutes)
        self._created_at = datetime.utcnow()
        self._cancellation_token = CancellationToken()
        self._closed = False

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation_token

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self._created_at + self._ttl

    @property
    def is_closed(self) -> bool:
        return self._closed

    def cancel(self, reason: Optional[str] = None) -> None:
        """Signal cancellation"""
        self._cancellation_token.cancel(reason)

    @abstractmethod
    async def enqueue(self, event: TaskEvent) -> None:
        """Add event to queue"""
        ...

    @abstractmethod
    async def subscribe(self, from_offset: int = 0) -> AsyncIterator[TaskEvent]:
        """Subscribe to events"""
        ...

    @abstractmethod
    async def get_latest_offset(self) -> int:
        """Get current offset"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the queue"""
        ...


class BaseQueueManager(ABC):
    """
    Abstract base class for QueueManager implementations.

    Provides common functionality and enforces interface.
    """

    @abstractmethod
    async def create_queue(
        self,
        task_id: str,
        tenant_id: str,
        ttl_minutes: int = 30,
    ) -> EventQueue:
        """Create new queue"""
        ...

    @abstractmethod
    async def get_queue(self, task_id: str) -> Optional[EventQueue]:
        """Get existing queue"""
        ...

    async def get_or_create_queue(
        self,
        task_id: str,
        tenant_id: str,
        ttl_minutes: int = 30,
    ) -> EventQueue:
        """Get existing or create new queue"""
        queue = await self.get_queue(task_id)
        if queue is None:
            queue = await self.create_queue(task_id, tenant_id, ttl_minutes)
        return queue

    @abstractmethod
    async def close_queue(self, task_id: str) -> bool:
        """Close and remove queue"""
        ...

    @abstractmethod
    async def list_active_queues(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List active queues"""
        ...

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired queues"""
        ...
