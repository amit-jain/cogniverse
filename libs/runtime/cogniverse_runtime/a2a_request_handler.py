"""Replica-safe A2A request ownership and cancellation routing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventConsumer, EventQueue, InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import ResultAggregator, TaskManager
from a2a.types import (
    InternalError,
    InvalidParamsError,
    MessageSendParams,
    Task,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskState,
)
from a2a.utils.errors import ServerError

from cogniverse_runtime.a2a_task_store import (
    A2ATaskConflictError,
    A2ATaskOwnershipLostError,
    A2ATaskStoreError,
    CancelCommand,
    RedisTaskStore,
    TaskLease,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class RedisRelayEventQueue(EventQueue):
    """Local SDK queue that mirrors owner events into Redis."""

    def __init__(self, task_id: str, task_store: RedisTaskStore) -> None:
        super().__init__()
        self._task_id = task_id
        self._task_store = task_store
        self._relay_closed = False

    async def enqueue_event(self, event) -> None:
        await super().enqueue_event(event)
        await self._task_store.publish_event(self._task_id, event)

    async def close(self, immediate: bool = False) -> None:
        if not self._relay_closed:
            self._relay_closed = True
            await self._task_store.close_event_stream(self._task_id)
        await super().close(immediate)


class RedisRelayQueueManager(InMemoryQueueManager):
    """Create relay-enabled owner queues while retaining local fan-out."""

    def __init__(self, task_store: RedisTaskStore) -> None:
        super().__init__()
        self._task_store = task_store

    async def create_or_tap(self, task_id: str) -> EventQueue:
        async with self._lock:
            if task_id not in self._task_queue:
                queue = RedisRelayEventQueue(task_id, self._task_store)
                self._task_queue[task_id] = queue
                return queue
            return self._task_queue[task_id].tap()


class RedisRequestHandler(DefaultRequestHandler):
    """Coordinate A2A execution ownership across runtime replicas."""

    def __init__(
        self,
        *,
        task_store: RedisTaskStore,
        replica_id: str,
        lease_seconds: float = 30,
        cancel_timeout_seconds: float = 10,
        drain_timeout_seconds: float = 30,
        **kwargs,
    ) -> None:
        if not replica_id.strip():
            raise ValueError("replica_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError(f"lease_seconds must be > 0, got {lease_seconds}")
        if cancel_timeout_seconds <= 0:
            raise ValueError(
                f"cancel_timeout_seconds must be > 0, got {cancel_timeout_seconds}"
            )
        if drain_timeout_seconds <= 0:
            raise ValueError(
                f"drain_timeout_seconds must be > 0, got {drain_timeout_seconds}"
            )
        kwargs.setdefault("queue_manager", RedisRelayQueueManager(task_store))
        super().__init__(task_store=task_store, **kwargs)
        self.task_store = task_store
        self._replica_id = replica_id
        self._lease_seconds = lease_seconds
        self._cancel_timeout_seconds = cancel_timeout_seconds
        self._drain_timeout_seconds = drain_timeout_seconds
        self._producer_leases: dict[asyncio.Task, TaskLease] = {}
        self._renewal_tasks: dict[asyncio.Task, asyncio.Task] = {}
        self._control_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the owner-addressed cancellation listener."""
        if self._control_task is not None:
            raise RuntimeError("A2A request handler is already started")
        self._control_task = asyncio.create_task(
            self._listen_for_cancels(),
            name=f"a2a-cancel-listener:{self._replica_id}",
        )

    async def close(self) -> None:
        """Stop control intake once served producers have drained or expired.

        A producer that outlives ``drain_timeout_seconds`` is cancelled rather
        than held onto, so shutdown stays bounded; its lease is left to expire
        so a peer reports the task interrupted instead of re-running it.
        """
        producers = list(self._running_agents.values())
        if producers:
            _, pending = await asyncio.wait(
                producers, timeout=self._drain_timeout_seconds
            )
            for producer_task in pending:
                producer_task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        # The SDK finishes a served turn in background tasks that still close
        # relay streams and release leases through Redis. The caller closes the
        # client next, so land them first; a second pass catches the ones a
        # cleanup task spawned while the first pass was running.
        for _ in range(2):
            if not self._background_tasks:
                break
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        if self._control_task is not None:
            self._control_task.cancel()
            await asyncio.gather(self._control_task, return_exceptions=True)
            self._control_task = None

    async def _setup_message_execution(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> tuple[TaskManager, str, EventQueue, ResultAggregator, asyncio.Task]:
        """Acquire task ownership before the SDK reads an existing snapshot."""
        call_context = context or ServerCallContext()
        lease: TaskLease | None = None
        task_manager = TaskManager(
            task_id=params.message.task_id,
            context_id=params.message.context_id,
            task_store=self.task_store,
            initial_message=params.message,
            context=call_context,
        )
        try:
            if params.message.task_id:
                lease = await self._acquire(params.message.task_id)
                self.task_store.attach_execution(call_context, lease)

            task: Task | None = await task_manager.get_task()
            if task:
                if task.status.state in _TERMINAL_STATES:
                    raise ServerError(
                        error=InvalidParamsError(
                            message=(
                                f"Task {task.id} is in terminal state: "
                                f"{task.status.state.value}"
                            )
                        )
                    )
                task = task_manager.update_with_message(params.message, task)
            elif params.message.task_id:
                raise ServerError(
                    error=TaskNotFoundError(
                        message=(
                            f"Task {params.message.task_id} was specified but "
                            "does not exist"
                        )
                    )
                )

            request_context = await self._request_context_builder.build(
                params=params,
                task_id=task.id if task else None,
                context_id=params.message.context_id,
                task=task,
                context=call_context,
            )
            task_id = cast("str", request_context.task_id)
            if lease is None:
                lease = await self._acquire(task_id)
                self.task_store.attach_execution(call_context, lease)

            if (
                self._push_config_store
                and params.configuration
                and params.configuration.push_notification_config
            ):
                await self._push_config_store.set_info(
                    task_id, params.configuration.push_notification_config
                )

            queue = await self._queue_manager.create_or_tap(task_id)
            result_aggregator = ResultAggregator(task_manager)
            producer_task = asyncio.create_task(
                self._run_event_stream(request_context, queue),
                name=f"a2a-producer:{task_id}",
            )
            await self._register_producer(task_id, producer_task)
            self._producer_leases[producer_task] = lease
            self._renewal_tasks[producer_task] = asyncio.create_task(
                self._renew_while_running(lease, producer_task),
                name=f"a2a-lease-renewal:{task_id}",
            )
            return (
                task_manager,
                task_id,
                queue,
                result_aggregator,
                producer_task,
            )
        except BaseException:
            if lease is not None:
                await self.task_store.release_execution(lease)
            raise

    async def _acquire(self, task_id: str) -> TaskLease:
        try:
            return await self.task_store.acquire_execution(
                task_id,
                replica_id=self._replica_id,
                lease_seconds=self._lease_seconds,
            )
        except A2ATaskConflictError as exc:
            owner = await self.task_store.get_execution_lease(task_id)
            owner_name = owner.replica_id if owner else "another replica"
            raise ServerError(
                error=InvalidParamsError(
                    message=f"Task {task_id} is active on {owner_name}; retry"
                )
            ) from exc
        except A2ATaskOwnershipLostError as exc:
            marked = await self.task_store.mark_owner_lost(task_id)
            if not marked:
                return await self.task_store.acquire_execution(
                    task_id,
                    replica_id=self._replica_id,
                    lease_seconds=self._lease_seconds,
                )
            raise ServerError(
                error=InvalidParamsError(
                    message=(
                        f"Task {task_id} was interrupted after its owner stopped; "
                        "inspect the failed task before retrying"
                    )
                )
            ) from exc

    async def _renew_while_running(
        self, lease: TaskLease, producer_task: asyncio.Task
    ) -> None:
        interval = max(0.01, self._lease_seconds / 3)
        try:
            while not producer_task.done():
                await asyncio.sleep(interval)
                if producer_task.done():
                    return
                await self.task_store.renew_execution(
                    lease, lease_seconds=self._lease_seconds
                )
        except asyncio.CancelledError:
            raise
        except A2ATaskStoreError:
            logger.exception(
                "A2A execution lease renewal failed for task %s", lease.task_id
            )
            producer_task.cancel()

    async def _cleanup_producer(
        self,
        producer_task: asyncio.Task,
        task_id: str,
    ) -> None:
        lease = self._producer_leases.pop(producer_task, None)
        renewal = self._renewal_tasks.pop(producer_task, None)
        try:
            await super()._cleanup_producer(producer_task, task_id)
        finally:
            if renewal is not None:
                renewal.cancel()
                await asyncio.gather(renewal, return_exceptions=True)
            # A cancelled producer keeps its lease: the task is still recorded
            # active, and letting the lease expire is what makes a peer report
            # it interrupted instead of silently re-executing its side effects.
            if lease is not None and not producer_task.cancelled():
                await self.task_store.release_execution(lease)

    async def on_cancel_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ) -> Task | None:
        """Cancel locally or route to the active task's owning replica."""
        task = await self.task_store.get(params.id, context)
        if task is None:
            raise ServerError(error=TaskNotFoundError())
        if task.status.state in _TERMINAL_STATES:
            raise ServerError(
                error=TaskNotCancelableError(
                    message=(
                        f"Task cannot be canceled - current state: {task.status.state}"
                    )
                )
            )
        lease = await self.task_store.get_execution_lease(params.id)
        if lease is not None and lease.expires_at_ms <= int(time.time() * 1000):
            await self.task_store.mark_owner_lost(params.id)
            raise ServerError(
                error=TaskNotCancelableError(
                    message=f"Task {params.id} owner expired; task is interrupted"
                )
            )
        # No lease means nothing is executing — an idle task paused in
        # input_required, which is what a completed turn leaves behind and the
        # commonest thing a client cancels. The stock handler cancels any
        # non-terminal task, so cancel it here rather than refusing.
        if lease is None or lease.replica_id == self._replica_id:
            return await self._cancel_owned(params.id)
        return await self.task_store.request_cancel(
            owner_replica_id=lease.replica_id,
            task_id=params.id,
            timeout_seconds=self._cancel_timeout_seconds,
        )

    async def on_resubscribe_to_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ):
        """Relay an active owner's future events to any replica."""
        task = await self.task_store.get(params.id, context)
        if task is None:
            raise ServerError(error=TaskNotFoundError())
        if task.status.state in _TERMINAL_STATES:
            raise ServerError(
                error=InvalidParamsError(
                    message=(
                        f"Task {task.id} is in terminal state: "
                        f"{task.status.state.value}"
                    )
                )
            )
        # The SDK requires a live queue here; the shared relay's equivalent is
        # a live owner, without which there is nothing left to stream.
        if not await self.task_store.has_live_owner(params.id):
            raise ServerError(error=TaskNotFoundError())
        async for event in self.task_store.subscribe_events(params.id):
            yield event

    async def _cancel_owned(self, task_id: str) -> Task:
        cancel_lease = await self.task_store.begin_cancel(
            task_id,
            replica_id=self._replica_id,
            lease_seconds=self._lease_seconds,
        )
        cancel_context = ServerCallContext()
        self.task_store.attach_execution(cancel_context, cancel_lease)
        try:
            task = await self.task_store.get(task_id, cancel_context)
            if task is None:
                raise ServerError(error=TaskNotFoundError())
            queue = RedisRelayEventQueue(task_id, self.task_store)
            await self.agent_executor.cancel(
                RequestContext(
                    None,
                    task_id=task.id,
                    context_id=task.context_id,
                    task=task,
                ),
                queue,
            )
            if producer_task := self._running_agents.get(task.id):
                producer_task.cancel()
            task_manager = TaskManager(
                task_id=task.id,
                context_id=task.context_id,
                task_store=self.task_store,
                initial_message=None,
                context=cancel_context,
            )
            result = await ResultAggregator(task_manager).consume_all(
                EventConsumer(queue)
            )
            if not isinstance(result, Task):
                raise ServerError(
                    error=InternalError(message="Cancel returned no task")
                )
            return result
        finally:
            await self.task_store.release_execution(cancel_lease)

    async def _listen_for_cancels(self) -> None:
        while True:
            command: CancelCommand | None = None
            try:
                command = await self.task_store.next_cancel(replica_id=self._replica_id)
                if command is None:
                    continue
                task = await self._cancel_owned(command.task_id)
                await self.task_store.acknowledge_cancel(command, task=task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "A2A owner cancellation failed on replica %s", self._replica_id
                )
                if command is not None:
                    try:
                        await self.task_store.acknowledge_cancel(
                            command,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    except A2ATaskStoreError:
                        logger.exception(
                            "A2A cancel acknowledgement failed for task %s",
                            command.task_id,
                        )
                await asyncio.sleep(0.1)
