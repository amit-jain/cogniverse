"""Replica-safe A2A request ownership and cancellation routing."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import cast

from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventConsumer, EventQueue, InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import ResultAggregator, TaskManager
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Message,
    MessageSendParams,
    Task,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from a2a.utils.task import apply_history_length

from cogniverse_runtime.a2a_task_store import (
    INTERRUPTED_MESSAGE,
    TERMINAL_STATES,
    A2ACancelCapacityError,
    A2ACancelTimeoutError,
    A2ATaskConflictError,
    A2ATaskOwnershipLostError,
    A2ATaskStoreError,
    CancelCommand,
    RedisTaskStore,
    TaskLease,
)

logger = logging.getLogger(__name__)

# Routed cancels one replica runs at once; more are refused, not queued.
_MAX_CONCURRENT_CANCELS = 16
# Longest a stopped producer waits for its consumers to take its last events
# before closing its queue anyway.
_STOPPED_QUEUE_CLOSE_SECONDS = 5.0
# Resubscriptions one replica serves at once, each holding a pooled Redis
# connection in a blocking read; more are refused, not queued.
_MAX_CONCURRENT_RESUBSCRIPTIONS = 64
# Producer events a relay holds while a cancel takes its generation; a
# producer emitting more waits for the cancel to commit or abort.
_MAX_HELD_EVENTS = 1000


def max_concurrent_cancels_from_env(environ: Mapping[str, str] = os.environ) -> int:
    """``A2A_MAX_CONCURRENT_CANCELS`` as an integer, refusing any other value."""
    raw = environ.get("A2A_MAX_CONCURRENT_CANCELS", str(_MAX_CONCURRENT_CANCELS))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"A2A_MAX_CONCURRENT_CANCELS must be an integer, got {raw!r}"
        ) from None


def _retryable_conflict(exc: Exception) -> ServerError:
    """The retryable conflict a client gets for a cancel that could not run."""
    message = str(exc)
    if not message.endswith("; retry"):
        message = f"{message}; retry"
    return ServerError(error=InvalidParamsError(message=message))


class RedisRelayEventQueue(EventQueue):
    """Local SDK queue that mirrors owner events into Redis.

    Closing it ends the shared relay only while ``generation`` still owns the
    task; a superseded generation closes its local queue alone.
    """

    def __init__(
        self, task_id: str, task_store: RedisTaskStore, generation: int
    ) -> None:
        super().__init__()
        self._task_id = task_id
        self._task_store = task_store
        self._generation = generation
        self._relay_closed = False
        self._relay_lock = asyncio.Lock()
        # open: events flow. pending: a cancel is taking its generation, so
        # events wait in _held until it commits (dropped) or aborts
        # (delivered). committed: only the cancel's own events flow. ended:
        # the cancel is over and later producer events stay superseded.
        self._cancel_state = "open"
        self._state_before_cancel = "open"
        self._held: list = []
        self._hold_released = asyncio.Event()
        self._hold_released.set()
        self._cancel_published = asyncio.Event()
        self._cancel_hold_seconds = 0.0
        # Events a failed release could not publish and no marker names yet.
        self._unmarked_gap = 0
        # Set once the relay refused this generation: another owner took the
        # task, and what this queue still gets stays local.
        self._superseded = False

    def bind_owner(self, generation: int) -> None:
        """Write to the shared relay as ``generation``, the task's owner."""
        self._generation = generation
        self._superseded = False

    def _supersede(self) -> None:
        """Stop writing to a relay another generation owns now."""
        if not self._superseded:
            logger.info(
                "A2A task %s: generation %s no longer owns its relay; its events "
                "stay on the local queue",
                self._task_id,
                self._generation,
            )
        self._superseded = True
        # The new owner's relay is not missing anything of this generation's.
        self._unmarked_gap = 0

    def hold_for_cancel(self, hold_seconds: float) -> bool:
        """Hold producer events and the close while a cancel takes its generation.

        A close waits at most ``hold_seconds`` for the cancel to publish.
        Returns False when another cancel already holds this relay.
        """
        if self._cancel_state in ("pending", "committed"):
            return False
        self._state_before_cancel = self._cancel_state
        self._cancel_state = "pending"
        self._hold_released = asyncio.Event()
        self._cancel_published = asyncio.Event()
        self._cancel_hold_seconds = hold_seconds
        return True

    def commit_cancel(self, generation: int) -> None:
        """The cancel took ``generation``: what the producer emits is superseded.

        A non-cooperative producer keeps running through the cancel, so the
        relay refuses everything but the cancel's own events from here on and
        ends with the cancel, closing as the cancel's generation.
        """
        self.bind_owner(generation)
        if self._held:
            logger.debug(
                "A2A task %s: dropped %d events held while its cancel committed",
                self._task_id,
                len(self._held),
            )
        self._held.clear()
        self._cancel_state = "committed"
        self._hold_released.set()

    async def abort_cancel(self) -> None:
        """Undo a hold whose cancel never took its generation.

        The relay returns to the state it had before the hold; if that was
        open, the held events are delivered in order before anything newer.
        Every held event reaches local consumers. After the first publish
        failure the rest are not published, and the relay stream is marked
        as missing them so a resubscriber does not read past the gap.
        """
        # Events delivered locally without a publish attempt; an event whose
        # publish failed or was cut short is counted by _enqueue itself.
        skipped = 0
        publishing = False
        failure: Exception | None = None
        try:
            if self._state_before_cancel == "open":
                while self._held:
                    event = self._held.pop(0)
                    if failure is not None:
                        skipped += 1
                        await EventQueue.enqueue_event(self, event)
                        continue
                    publishing = True
                    try:
                        await self._enqueue(event)
                    except Exception as exc:
                        # Delivered locally, not published.
                        failure = exc
                    publishing = False
            else:
                # An earlier cancel already ended this relay: still superseded.
                self._held.clear()
        finally:
            dropped = len(self._held)
            self._held.clear()
            self._unmarked_gap += skipped + dropped
            unpublished = skipped + (1 if failure is not None or publishing else 0)
            self._cancel_state = self._state_before_cancel
            self._hold_released.set()
            self._cancel_published.set()
            if unpublished or dropped:
                logger.error(
                    "A2A task %s: of the events held during a failed cancel, %d "
                    "reached local consumers but not the relay publish and %d "
                    "were dropped by an interrupted release%s",
                    self._task_id,
                    unpublished,
                    dropped,
                    f" (first publish failure: {failure})" if failure else "",
                )
        if self._unmarked_gap:
            await self._mark_gap()

    async def _mark_gap(self) -> None:
        """Record the unpublished events on the relay stream, once."""
        async with self._relay_lock:
            if self._relay_closed or self._superseded or not self._unmarked_gap:
                return
            try:
                marked = await self._task_store.mark_event_stream_incomplete(
                    self._task_id,
                    missing=self._unmarked_gap,
                    generation=self._generation,
                )
            except Exception as exc:
                logger.error(
                    "A2A task %s: could not mark its relay as missing %d events; "
                    "nothing more is published until it is: %s",
                    self._task_id,
                    self._unmarked_gap,
                    exc,
                )
                return
            if not marked:
                self._supersede()
                return
            self._unmarked_gap = 0

    def end_cancel(self) -> None:
        """Release a close held for the cancel, published or abandoned."""
        if self._cancel_state == "committed":
            self._cancel_state = "ended"
        self._cancel_published.set()

    async def enqueue_event(self, event) -> None:
        while self._cancel_state == "pending" and len(self._held) >= _MAX_HELD_EVENTS:
            await self._hold_released.wait()
        if self._cancel_state == "pending":
            self._held.append(event)
            return
        if self._cancel_state != "open":
            logger.debug(
                "A2A task %s: dropped %s emitted after its cancel was committed",
                self._task_id,
                type(event).__name__,
            )
            return
        await self._enqueue(event)

    async def enqueue_cancel_event(self, event) -> None:
        await self._enqueue(event)

    async def _enqueue(self, event) -> None:
        published = False
        try:
            await super().enqueue_event(event)
            # An event after the close marker would also PERSIST the drained
            # stream, so nothing is published once the relay is closed.
            async with self._relay_lock:
                if self._relay_closed or self._superseded:
                    published = True
                    return
                if self._unmarked_gap:
                    if not await self._task_store.mark_event_stream_incomplete(
                        self._task_id,
                        missing=self._unmarked_gap,
                        generation=self._generation,
                    ):
                        self._supersede()
                        published = True
                        return
                    self._unmarked_gap = 0
                if not await self._task_store.publish_event(
                    self._task_id, event, generation=self._generation
                ):
                    self._supersede()
                published = True
        finally:
            # Not on the relay, whether the publish failed or was cut short.
            if not published:
                self._unmarked_gap += 1

    async def close(self, immediate: bool = False) -> None:
        if (
            self._cancel_state in ("pending", "committed")
            and not self._cancel_published.is_set()
        ):
            try:
                await asyncio.wait_for(
                    self._cancel_published.wait(), self._cancel_hold_seconds
                )
            except TimeoutError:
                logger.warning(
                    "A2A task %s: closing its relay after %gs without the "
                    "committed cancel",
                    self._task_id,
                    self._cancel_hold_seconds,
                )
        async with self._relay_lock:
            if not self._relay_closed:
                self._relay_closed = True
                closed = await self._task_store.close_event_stream(
                    self._task_id,
                    generation=self._generation,
                    missing=self._unmarked_gap,
                )
                if not closed:
                    logger.info(
                        "A2A task %s: generation %s no longer owns its relay; "
                        "closing the local queue only",
                        self._task_id,
                        self._generation,
                    )
        await super().close(immediate)


class RedisRelayQueueManager(InMemoryQueueManager):
    """Create relay-enabled owner queues while retaining local fan-out."""

    def __init__(self, task_store: RedisTaskStore) -> None:
        super().__init__()
        self._task_store = task_store

    async def create_or_tap(
        self, task_id: str, *, generation: int | None = None
    ) -> EventQueue:
        """The task's relay, created bound to ``generation``, its owner."""
        async with self._lock:
            if task_id not in self._task_queue:
                if generation is None:
                    raise ValueError(
                        f"A2A task {task_id}: a shared relay needs its owning "
                        "generation"
                    )
                queue = RedisRelayEventQueue(task_id, self._task_store, generation)
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
        max_concurrent_cancels: int = _MAX_CONCURRENT_CANCELS,
        max_concurrent_resubscriptions: int = _MAX_CONCURRENT_RESUBSCRIPTIONS,
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
        if max_concurrent_cancels < 1:
            raise ValueError(
                f"max_concurrent_cancels must be >= 1, got {max_concurrent_cancels}"
            )
        if max_concurrent_resubscriptions < 1:
            raise ValueError(
                "max_concurrent_resubscriptions must be >= 1, got "
                f"{max_concurrent_resubscriptions}"
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
        self._abandoned_cancels: set[asyncio.Task] = set()
        self._max_concurrent_cancels = max_concurrent_cancels
        self._routed_cancels: set[asyncio.Task] = set()
        self._inflight_cancels: dict[str, asyncio.Task] = {}
        self._max_concurrent_resubscriptions = max_concurrent_resubscriptions
        self._resubscriptions = 0
        # Producers this handler cancels, with the failure their stream ends
        # on; None ends the stream without one.
        self._stop_reasons: dict[asyncio.Task, str | None] = {}
        # Stopped producers whose task this replica records as ended.
        self._recorded_stops: set[asyncio.Task] = set()

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

        Served producers and running cancels get ``drain_timeout_seconds`` to
        finish; whatever outlives that is cancelled rather than held onto,
        and its cleanup gets at most as long again, so close() ends within
        twice the drain budget. A producer cancelled at the deadline has its
        task saved interrupted, fenced, and its lease released; a task another
        owner took meanwhile is left to it. If the save cannot be made, the
        lease is left to expire and the next reader reports the task
        interrupted instead of anything re-running it.
        """
        loop = asyncio.get_running_loop()
        drained_by = loop.time() + self._drain_timeout_seconds
        cleaned_by = drained_by + self._drain_timeout_seconds

        def until(moment: float) -> float:
            return max(0.0, moment - loop.time())

        producers = list(self._running_agents.values())
        if producers:
            _, pending = await asyncio.wait(producers, timeout=until(drained_by))
            for producer_task in pending:
                self._stop_producer(producer_task, INTERRUPTED_MESSAGE, record=True)
            if pending:
                await asyncio.wait(pending, timeout=until(cleaned_by))
        # The SDK finishes a served turn in background tasks that still close
        # relay streams and release leases through Redis. The caller closes the
        # client next, so land them first; a second pass catches the ones a
        # cleanup task spawned while the first pass was running.
        for _ in range(2):
            if not self._background_tasks:
                break
            _, pending = await asyncio.wait(
                list(self._background_tasks), timeout=until(cleaned_by)
            )
            if pending:
                for background_task in pending:
                    background_task.cancel()
                await asyncio.wait(pending, timeout=until(cleaned_by))
                break
        if self._control_task is not None:
            self._control_task.cancel()
            await asyncio.gather(self._control_task, return_exceptions=True)
            self._control_task = None
        # A cancel run still going at the drain deadline is cancelled, so
        # shutdown never waits out the owner deadline; routed requesters
        # waiting on it get an explicit refusal.
        if self._inflight_cancels:
            runs = set(self._inflight_cancels.values())
            _, unfinished = await asyncio.wait(runs, timeout=until(drained_by))
            for run in unfinished:
                run.cancel()
            if unfinished:
                await asyncio.wait(unfinished, timeout=until(cleaned_by))
        if self._routed_cancels:
            await asyncio.wait(set(self._routed_cancels), timeout=until(cleaned_by))
        if self._abandoned_cancels:
            await asyncio.wait(set(self._abandoned_cancels), timeout=until(cleaned_by))

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
        loop = asyncio.get_running_loop()
        acquired_at = loop.time()
        try:
            if params.message.task_id:
                lease = await self._acquire(params.message.task_id)
                self.task_store.attach_execution(call_context, lease)

            task: Task | None = await task_manager.get_task()
            if task:
                if task.status.state in TERMINAL_STATES:
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
                acquired_at = loop.time()
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

            if isinstance(self._queue_manager, RedisRelayQueueManager):
                queue = await self._queue_manager.create_or_tap(
                    task_id, generation=lease.generation
                )
            else:
                queue = await self._queue_manager.create_or_tap(task_id)
            result_aggregator = ResultAggregator(task_manager)
            producer_task = asyncio.create_task(
                self._run_producer(request_context, queue, lease),
                name=f"a2a-producer:{task_id}",
            )
            await self._register_producer(task_id, producer_task)
            self._producer_leases[producer_task] = lease
            self._renewal_tasks[producer_task] = asyncio.create_task(
                self._renew_while_running(lease, producer_task, acquired_at),
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

    def _stop_producer(
        self, producer_task: asyncio.Task, reason: str | None, *, record: bool = False
    ) -> None:
        """Cancel a producer and record how its stream ends.

        The SDK closes a producer's queue only after ``execute`` returns, and
        its consumer ignores a cancelled producer, so without this a blocking
        send on the stopped execution waits for its client to go away.
        With ``record``, the replica also saves that end as the task's state
        and releases the lease. A producer that has already finished needs
        none of it.
        """
        if producer_task.done():
            return
        self._stop_reasons[producer_task] = reason
        if record:
            self._recorded_stops.add(producer_task)
        producer_task.cancel()

    async def _run_producer(
        self, request_context: RequestContext, queue, lease: TaskLease
    ) -> None:
        """Run the execution; end the stream of one this handler stopped."""
        producer_task = asyncio.current_task()
        try:
            await self._run_event_stream(request_context, queue)
        except asyncio.CancelledError:
            if producer_task in self._stop_reasons:
                recorded = producer_task in self._recorded_stops
                if recorded:
                    # The stop releases the lease; a renewal still running
                    # would then find it gone and stop this producer again.
                    await self._stop_renewal(producer_task)
                await self._end_stopped_stream(
                    request_context,
                    queue,
                    self._stop_reasons[producer_task],
                    lease if recorded else None,
                )
            raise
        finally:
            self._stop_reasons.pop(producer_task, None)
            self._recorded_stops.discard(producer_task)

    async def _end_stopped_stream(
        self,
        request_context: RequestContext,
        queue,
        reason: str | None,
        lease: TaskLease | None = None,
    ) -> None:
        """Hand local consumers a final ``failed`` event, then close the queue.

        The event reaches local consumers only: another replica may own the
        task's relay by now. Saving it is fenced like any write, so a node
        that lost the task records nothing and its send fails instead. With
        ``lease``, the same status is first saved directly, since no consumer
        may be left to save it. While this replica's own cancel of the task
        is under way, the cancel's events end the stream, so no ``failed``
        event goes ahead of them.
        """
        cancelling = request_context.task_id in self._inflight_cancels or (
            isinstance(queue, RedisRelayEventQueue) and queue._cancel_state != "open"
        )
        if reason is not None and not cancelling:
            status = TaskStatus(
                state=TaskState.failed,
                message=new_agent_text_message(reason),
            )
            if lease is not None:
                # Stored before any consumer hears of it.
                await self._record_stopped(lease, status)
            event = TaskStatusUpdateEvent(
                task_id=request_context.task_id or "",
                context_id=request_context.context_id or "",
                final=True,
                status=status,
            )
            try:
                await EventQueue.enqueue_event(queue, event)
            except Exception:
                logger.exception(
                    "A2A task %s: could not hand its stopped stream a final event",
                    request_context.task_id,
                )
        try:
            await asyncio.wait_for(queue.close(), _STOPPED_QUEUE_CLOSE_SECONDS)
        except TimeoutError:
            await queue.close(immediate=True)
        except Exception as exc:
            logger.warning(
                "A2A task %s: closing its stopped stream failed: %s",
                request_context.task_id,
                exc,
            )
            # Close the local queue alone, letting consumers take what is
            # left in it; the relay is already marked closed.
            try:
                await asyncio.wait_for(
                    EventQueue.close(queue), _STOPPED_QUEUE_CLOSE_SECONDS
                )
            except TimeoutError:
                await EventQueue.close(queue, immediate=True)

    async def _stop_renewal(self, producer_task: asyncio.Task) -> None:
        """Stop renewing ``producer_task``'s lease and wait for it to stop."""
        renewal = self._renewal_tasks.pop(producer_task, None)
        if renewal is not None:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)

    async def _record_stopped(self, lease: TaskLease, status: TaskStatus) -> None:
        """Save a stopped execution's end while ``lease`` owns it, then
        release the lease; a task another owner took is left to it."""
        try:
            await self.task_store.interrupt_execution(lease, status)
        except A2ATaskOwnershipLostError:
            logger.info(
                "A2A task %s: another owner took it before this replica stopped "
                "it; nothing recorded",
                lease.task_id,
            )
            return
        except A2ATaskStoreError as exc:
            logger.warning(
                "A2A task %s: could not record its stopped execution; the lease "
                "is left to expire, after which a reader reports it interrupted: "
                "%s",
                lease.task_id,
                exc,
            )
            return
        try:
            await self.task_store.release_execution(lease)
        except A2ATaskStoreError as exc:
            logger.warning(
                "A2A task %s: could not release its lease after recording its "
                "stopped execution; it expires instead: %s",
                lease.task_id,
                exc,
            )

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
        self, lease: TaskLease, producer_task: asyncio.Task, acquired_at: float
    ) -> None:
        """Renew ``lease`` while its producer runs.

        A renewal the store cannot complete is retried until the lease would
        expire; the producer is cancelled only once it has, or when another
        owner took the task. ``acquired_at`` is this loop's time before the
        acquire was sent: Redis dates the lease no earlier, so counting from
        it never outlives the lease.
        """
        loop = asyncio.get_running_loop()
        interval = max(0.01, self._lease_seconds / 3)
        retry_interval = min(1.0, interval)
        held_until = acquired_at + self._lease_seconds
        delay = interval
        try:
            while True:
                await asyncio.sleep(delay)
                if producer_task.done():
                    return
                sent_at = loop.time()
                try:
                    await self.task_store.renew_execution(
                        lease, lease_seconds=self._lease_seconds
                    )
                except A2ATaskOwnershipLostError:
                    raise
                except A2ATaskStoreError as exc:
                    if loop.time() >= held_until:
                        raise
                    logger.warning(
                        "A2A execution lease renewal for task %s failed with "
                        "%.1fs of the lease left; retrying: %s",
                        lease.task_id,
                        held_until - loop.time(),
                        exc,
                    )
                    delay = min(retry_interval, max(0.0, held_until - loop.time()))
                    continue
                held_until = sent_at + self._lease_seconds
                delay = interval
        except asyncio.CancelledError:
            raise
        except A2ATaskStoreError as exc:
            logger.exception(
                "A2A execution lease renewal failed for task %s", lease.task_id
            )
            self._stop_producer(
                producer_task,
                f"Task {lease.task_id} stopped: its execution lease could not be "
                f"kept on replica {self._replica_id} ({exc})",
            )

    async def _cleanup_producer(
        self,
        producer_task: asyncio.Task,
        task_id: str,
    ) -> None:
        lease = self._producer_leases.pop(producer_task, None)
        try:
            await super()._cleanup_producer(producer_task, task_id)
        finally:
            await self._stop_renewal(producer_task)
            # A cancelled producer's lease is not released here; a stop that
            # recorded the task ended released it already. Otherwise letting
            # the lease expire is what makes a peer report it interrupted
            # instead of silently re-executing its side effects.
            if lease is not None and not producer_task.cancelled():
                await self.task_store.release_execution(lease)

    async def on_get_task(
        self, params: TaskQueryParams, context: ServerCallContext | None = None
    ) -> Task | None:
        """Read a task, resolving one whose owner stopped without ending it.

        A non-terminal task whose execution lease expired, judged against
        Redis' clock, gets the interruption a peer's send or cancel records
        (``mark_owner_lost``): atomic, at most once across readers, and never
        applied while a live owner holds or renews the lease.
        """
        task = await self.task_store.get(params.id, context)
        if task is not None and task.status.state not in TERMINAL_STATES:
            lease = await self.task_store.get_execution_lease(params.id)
            if lease is not None and not await self.task_store.has_live_owner(
                params.id
            ):
                await self.task_store.mark_owner_lost(params.id)
                # Read again whichever reader's interruption landed.
                task = await self.task_store.get(params.id, context)
        if task is None:
            raise ServerError(error=TaskNotFoundError())
        return apply_history_length(task, params.history_length)

    async def on_cancel_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ) -> Task | None:
        """Cancel locally or route to the active task's owning replica."""
        task = await self.task_store.get(params.id, context)
        if task is None:
            raise ServerError(error=TaskNotFoundError())
        if task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=TaskNotCancelableError(
                    message=(
                        f"Task cannot be canceled - current state: {task.status.state}"
                    )
                )
            )
        lease = await self.task_store.get_execution_lease(params.id)
        # Expiry is judged against Redis' own clock, never this replica's:
        # ``expires_at_ms`` is computed from ``redis.call('TIME')``, so a pod
        # running ahead would otherwise declare every live owner expired.
        if lease is not None and not await self.task_store.has_live_owner(params.id):
            if await self.task_store.mark_owner_lost(params.id):
                raise ServerError(
                    error=TaskNotCancelableError(
                        message=f"Task {params.id} owner expired; task is interrupted"
                    )
                )
            # The interrupt was declined, so nothing was interrupted: the task
            # is not active and the lease is a record left behind by an owner
            # that is not executing it. Routing a cancel there waits out a
            # replica that will never acknowledge, so treat it as the idle
            # task it is. If Redis does still see a live owner, begin_cancel
            # refuses and the conflict below is what the client gets.
            lease = None
        # No lease means nothing is executing — an idle task paused in
        # input_required, which is what a completed turn leaves behind and the
        # commonest thing a client cancels. The stock handler cancels any
        # non-terminal task, so cancel it here rather than refusing.
        if lease is None or lease.replica_id == self._replica_id:
            run: asyncio.Task | None = None
            try:
                run = self._coalesced_cancel(params.id)
                result = await asyncio.shield(run)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if (
                    run is not None
                    and run.cancelled()
                    and not (current and current.cancelling())
                ):
                    raise ServerError(
                        error=InvalidParamsError(
                            message=(
                                f"replica {self._replica_id} shut down before the "
                                f"cancel of task {params.id} finished; retry"
                            )
                        )
                    ) from None
                raise
            except A2ATaskOwnershipLostError as exc:
                # Same conflict ``message/send`` reports for a lost race, so
                # the client sees a retryable conflict, not an internal error.
                raise ServerError(
                    error=InvalidParamsError(
                        message=(
                            f"Task {params.id} is active on another replica; retry"
                        )
                    )
                ) from exc
            except A2ATaskStoreError as exc:
                # A retryable conflict, as the routed path answers, not the
                # JSON-RPC internal error an unmapped exception becomes.
                raise _retryable_conflict(exc) from exc
        else:
            try:
                result = await self.task_store.request_cancel(
                    owner_replica_id=lease.replica_id,
                    task_id=params.id,
                    timeout_seconds=self._cancel_timeout_seconds,
                )
            except A2ATaskStoreError as exc:
                raise _retryable_conflict(exc) from exc
        if result.status.state != TaskState.canceled:
            raise ServerError(
                error=TaskNotCancelableError(
                    message=(
                        "Task cannot be canceled - current state: "
                        f"{result.status.state}"
                    )
                )
            )
        return result

    async def on_resubscribe_to_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ):
        """Relay an active owner's future events to any replica.

        At most ``max_concurrent_resubscriptions`` run at once on a replica;
        one past that is refused before it touches Redis.
        """
        if self._resubscriptions >= self._max_concurrent_resubscriptions:
            raise ServerError(
                error=InvalidParamsError(
                    message=(
                        f"replica {self._replica_id} is already serving "
                        f"{self._max_concurrent_resubscriptions} resubscriptions; "
                        "retry"
                    )
                )
            )
        self._resubscriptions += 1
        try:
            task = await self.task_store.get(params.id, context)
            if task is None:
                raise ServerError(error=TaskNotFoundError())
            if task.status.state in TERMINAL_STATES:
                raise ServerError(
                    error=InvalidParamsError(
                        message=(
                            f"Task {task.id} is in terminal state: "
                            f"{task.status.state.value}"
                        )
                    )
                )
            # The SDK requires a live queue here; the shared relay's equivalent
            # is a live owner, without which there is nothing left to stream.
            if not await self.task_store.has_live_owner(params.id):
                raise ServerError(error=TaskNotFoundError())
            async for event in self.task_store.subscribe_events(params.id):
                yield event
        finally:
            self._resubscriptions -= 1

    def _coalesced_cancel(self, task_id: str) -> asyncio.Task:
        """The one cancel run for ``task_id``, started if none is in flight.

        Every requester of a task's cancel, local or routed, awaits the same
        run: two runs would fence each other and publish twice.
        """
        running = self._inflight_cancels.get(task_id)
        if running is not None:
            return running
        occupied = len(self._inflight_cancels) + len(self._abandoned_cancels)
        if occupied >= self._max_concurrent_cancels:
            raise A2ACancelCapacityError(
                f"replica {self._replica_id} is already running "
                f"{self._max_concurrent_cancels} cancels; retry"
            )
        run = asyncio.create_task(
            self._cancel_within_deadline(task_id), name=f"a2a-cancel:{task_id}"
        )
        self._inflight_cancels[task_id] = run

        def _finished(done: asyncio.Task) -> None:
            if self._inflight_cancels.get(task_id) is done:
                del self._inflight_cancels[task_id]
            if not done.cancelled():
                # Every waiter re-raises it; retrieve it so a run whose
                # waiters went away is not reported as never retrieved.
                done.exception()

        run.add_done_callback(_finished)
        return run

    async def _cancel_owned(self, task_id: str) -> Task:
        # The cancel's events are persisted under the new generation first and
        # only then handed to the live queue, so this replica's own stream and
        # blocking consumers observe them and find them already stored. With
        # no live queue, a relay queue publishes them to peers'
        # resubscriptions instead. The live relay is committed before the
        # generation moves, so no producer event lands between the two.
        live_queue = await self._queue_manager.get(task_id)
        if live_queue is not None and live_queue.is_closed():
            live_queue = None
        relay = live_queue if isinstance(live_queue, RedisRelayEventQueue) else None
        committed = relay is not None and relay.hold_for_cancel(
            self._cancel_timeout_seconds / 2
        )
        try:
            cancel_lease = await self.task_store.begin_cancel(
                task_id,
                replica_id=self._replica_id,
                lease_seconds=self._lease_seconds,
            )
        except BaseException:
            if committed:
                try:
                    await relay.abort_cancel()
                except Exception:
                    logger.exception(
                        "A2A task %s: releasing the events held for a cancel "
                        "that never began failed",
                        task_id,
                    )
            raise
        if committed:
            relay.commit_cancel(cancel_lease.generation)
        elif relay is not None:
            # An earlier cancel still holds the relay (one abandoned at its
            # deadline); this cancel's generation owns it now.
            relay.bind_owner(cancel_lease.generation)
        cancel_context = ServerCallContext()
        self.task_store.attach_execution(cancel_context, cancel_lease)
        try:
            task = await self.task_store.get(task_id, cancel_context)
            if task is None:
                raise ServerError(error=TaskNotFoundError())
            if task.status.state in TERMINAL_STATES:
                # Already ended, typically by a cancel that finished first:
                # answer with it rather than cancelling and publishing again.
                return task
            queue = (
                EventQueue()
                if live_queue is not None
                else RedisRelayEventQueue(
                    task.id, self.task_store, cancel_lease.generation
                )
            )
            await self.agent_executor.cancel(
                RequestContext(
                    None,
                    task_id=task.id,
                    context_id=task.context_id,
                    task=task,
                ),
                queue,
            )
            task_manager = TaskManager(
                task_id=task.id,
                context_id=task.context_id,
                task_store=self.task_store,
                initial_message=None,
                context=cancel_context,
            )
            events = []
            result: Task | Message | None = None
            async for event in EventConsumer(queue).consume_all():
                events.append(event)
                if isinstance(event, Message):
                    result = event
                    break
                await task_manager.process(event)
            else:
                result = await task_manager.get_task()
            if relay is not None:
                for event in events:
                    await relay.enqueue_cancel_event(event)
                if committed:
                    relay.end_cancel()
            elif live_queue is not None:
                for event in events:
                    await live_queue.enqueue_event(event)
            # Stopped only after the cancel reached the live relay: a stopped
            # producer's cleanup closes that relay.
            if producer_task := self._running_agents.get(task.id):
                producer_task.cancel()
            if not isinstance(result, Task):
                raise ServerError(
                    error=InternalError(message="Cancel returned no task")
                )
            return result
        finally:
            if committed:
                relay.end_cancel()
            await self.task_store.release_execution(cancel_lease)

    async def _cancel_within_deadline(self, task_id: str) -> Task:
        """Run one cancel, abandoning it at the owner deadline.

        Routed and local cancels both wait on this, so an executor cancel
        that never returns holds neither a requester nor the relay's close.
        """
        cancel = asyncio.create_task(
            self._cancel_owned(task_id), name=f"a2a-owner-cancel:{task_id}"
        )
        # A requesting replica waits the same configured timeout, so the owner
        # gives up at half of it for its refusal to arrive before that.
        deadline = self._cancel_timeout_seconds / 2
        try:
            done, _ = await asyncio.wait({cancel}, timeout=deadline)
        except asyncio.CancelledError:
            cancel.cancel()
            self._abandon_cancel(cancel)
            raise
        if not done:
            cancel.cancel()
            self._abandon_cancel(cancel)
            raise A2ACancelTimeoutError(
                f"cancel of task {task_id} did not finish within {deadline:g}s"
            )
        return cancel.result()

    def _abandon_cancel(self, cancel: asyncio.Task) -> None:
        """Track a cancelled cancel run until it ends, so close() waits for it."""
        self._abandoned_cancels.add(cancel)
        cancel.add_done_callback(self._forget_abandoned_cancel)

    def _forget_abandoned_cancel(self, cancel: asyncio.Task) -> None:
        self._abandoned_cancels.discard(cancel)
        if not cancel.cancelled() and cancel.exception() is not None:
            logger.error(
                "Abandoned A2A owner cancellation failed",
                exc_info=cancel.exception(),
            )

    async def _listen_for_cancels(self) -> None:
        while True:
            command: CancelCommand | None = None
            try:
                command = await self.task_store.next_cancel(replica_id=self._replica_id)
                if command is None:
                    continue
                try:
                    run = self._coalesced_cancel(command.task_id)
                except A2ACancelCapacityError as exc:
                    # Queueing past the limit would answer after the
                    # requester's timeout; refuse now so it can retry.
                    logger.warning(
                        "A2A cancel of task %s refused on replica %s: %s",
                        command.task_id,
                        self._replica_id,
                        exc,
                    )
                    await self._refuse_routed_cancel(command, exc)
                    continue
                routed = asyncio.create_task(
                    self._serve_routed_cancel(command, run),
                    name=f"a2a-routed-cancel:{command.task_id}",
                )
                self._routed_cancels.add(routed)
                routed.add_done_callback(self._routed_cancels.discard)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "A2A owner cancellation failed on replica %s", self._replica_id
                )
                if command is not None:
                    await self._refuse_routed_cancel(command, exc)
                else:
                    await asyncio.sleep(0.1)

    async def _serve_routed_cancel(
        self, command: CancelCommand, run: asyncio.Task
    ) -> None:
        try:
            task = await asyncio.shield(run)
            await self.task_store.acknowledge_cancel(command, task=task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if run.cancelled() and not (current and current.cancelling()):
                await self._refuse_routed_cancel(
                    command,
                    A2ACancelTimeoutError(
                        f"replica {self._replica_id} shut down before the cancel "
                        f"of task {command.task_id} finished"
                    ),
                )
                return
            raise
        except Exception as exc:
            logger.exception(
                "A2A owner cancellation of task %s failed on replica %s",
                command.task_id,
                self._replica_id,
            )
            await self._refuse_routed_cancel(command, exc)

    async def _refuse_routed_cancel(
        self, command: CancelCommand, exc: Exception
    ) -> None:
        try:
            await self.task_store.acknowledge_cancel(
                command, error=f"{type(exc).__name__}: {exc}"
            )
        except A2ATaskStoreError:
            logger.exception(
                "A2A cancel acknowledgement failed for task %s", command.task_id
            )
