"""Admin events every runtime worker process acts on, with acknowledgements.

A tenant delete or a session close changes state that each worker process
holds in memory: warm memory managers, cached agents, queued memory writes.
The process serving the request publishes the event on a Redis channel every
worker subscribes to. Each worker runs its handler for the event and pushes
an acknowledgement onto the event's reply list, and the publisher waits until
every worker the publish reached has answered.

``publish`` returns each worker's result only when every receiver
acknowledged success. Redis unreachable, no worker subscribed, a worker that
does not answer within the timeout, and a handler that raised all raise a
:class:`ClusterEventError` subclass naming what is missing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

CLUSTER_EVENT_CHANNEL = "cogniverse:runtime:events"
# Seconds an acknowledgement list outlives its event, so one a publisher
# stopped waiting for does not stay in Redis.
_REPLY_TTL_S = 120
# Longest single blocking read for acknowledgements; it stays under the
# client's socket timeout, so a long wait is a sequence of short reads.
_REPLY_POLL_S = 1.0
_RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 5.0)

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


class ClusterEventError(RuntimeError):
    """An admin event did not reach, or was not handled by, every worker."""


class ClusterEventUnavailable(ClusterEventError):
    """Redis could not carry the event, or no worker is subscribed."""


class ClusterEventIncomplete(ClusterEventError):
    """Some receiving workers did not acknowledge, or reported a failure."""

    def __init__(
        self,
        kind: str,
        expected: int,
        results: Dict[str, Dict[str, Any]],
        failures: Dict[str, str],
    ):
        missing = expected - len(results) - len(failures)
        parts = [f"{len(results)} of {expected} workers handled {kind!r}"]
        if failures:
            parts.append(f"failed: {failures}")
        if missing > 0:
            parts.append(f"{missing} did not answer")
        super().__init__("; ".join(parts))
        self.kind = kind
        self.expected = expected
        self.results = results
        self.failures = failures


class ClusterEvents:
    """Subscribe this worker to admin events and publish events to all workers.

    ``handlers`` maps an event kind to a synchronous function of the event
    payload; it runs on a worker thread and returns this worker's result.
    """

    def __init__(
        self,
        redis_url: str,
        worker_id: str,
        handlers: Dict[str, Handler],
        *,
        channel: str = CLUSTER_EVENT_CHANNEL,
        redis_timeout_s: float = 5.0,
    ):
        if not redis_url.strip():
            raise ValueError("redis_url must be non-empty")
        self._redis_url = redis_url
        self.worker_id = worker_id
        self._handlers = dict(handlers)
        self.channel = channel
        self._redis_timeout_s = redis_timeout_s
        self._client: Optional[Redis] = None
        self._listener: Optional[asyncio.Task] = None
        self._answers: set[asyncio.Task] = set()
        self._subscribed = asyncio.Event()

    def _new_client(self) -> Redis:
        return Redis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_timeout=self._redis_timeout_s,
            socket_connect_timeout=self._redis_timeout_s,
            socket_keepalive=True,
        )

    async def start(self) -> None:
        """Connect, subscribe and start answering events.

        Raises :class:`ClusterEventUnavailable` when Redis cannot be reached.
        """
        self._client = self._new_client()
        try:
            await self._client.ping()
        except RedisError as exc:
            await self._client.aclose()
            self._client = None
            raise ClusterEventUnavailable(
                f"cluster events: cannot reach Redis: {exc}"
            ) from exc
        self._listener = asyncio.create_task(
            self._listen(), name=f"cluster-events-{self.worker_id}"
        )
        try:
            await asyncio.wait_for(
                self._subscribed.wait(), timeout=self._redis_timeout_s
            )
        except asyncio.TimeoutError as exc:
            await self.close()
            raise ClusterEventUnavailable(
                f"cluster events: could not subscribe to {self.channel!r} within "
                f"{self._redis_timeout_s}s"
            ) from exc

    async def close(self) -> None:
        tasks = [*self._answers]
        if self._listener is not None:
            tasks.append(self._listener)
            self._listener = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def publish(
        self, kind: str, payload: Dict[str, Any], *, timeout_s: float
    ) -> Dict[str, Dict[str, Any]]:
        """Run ``kind`` on every subscribed worker; their results by worker id."""
        if self._client is None:
            raise ClusterEventUnavailable("cluster events are not started")
        event_id = uuid.uuid4().hex
        reply_key = f"{self.channel}:replies:{event_id}"
        message = json.dumps(
            {"id": event_id, "kind": kind, "payload": payload, "reply_to": reply_key}
        )
        try:
            expected = int(await self._client.publish(self.channel, message))
        except RedisError as exc:
            raise ClusterEventUnavailable(
                f"cluster events: cannot publish {kind!r}: {exc}"
            ) from exc
        if expected == 0:
            raise ClusterEventUnavailable(
                f"cluster events: no runtime worker is subscribed to "
                f"{self.channel!r}; {kind!r} reached none"
            )
        results: Dict[str, Dict[str, Any]] = {}
        failures: Dict[str, str] = {}
        deadline = time.monotonic() + timeout_s
        try:
            while len(results) + len(failures) < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                reply = await self._client.blpop(
                    [reply_key], timeout=min(_REPLY_POLL_S, remaining)
                )
                if reply is None:
                    continue
                answer = json.loads(reply[1])
                if answer["ok"]:
                    results[answer["worker"]] = answer["result"]
                else:
                    failures[answer["worker"]] = answer["error"]
        except RedisError as exc:
            raise ClusterEventUnavailable(
                f"cluster events: lost Redis while collecting {kind!r} "
                f"acknowledgements ({len(results)} of {expected}): {exc}"
            ) from exc
        finally:
            try:
                await self._client.delete(reply_key)
            except RedisError:
                pass
        if failures or len(results) < expected:
            raise ClusterEventIncomplete(kind, expected, results, failures)
        return results

    async def _listen(self) -> None:
        attempt = 0
        while True:
            pubsub = None
            try:
                pubsub = self._client.pubsub()
                await pubsub.subscribe(self.channel)
                # Subscribed once Redis confirms it: an event published before
                # the confirmation would not count this worker as a receiver.
                while True:
                    confirmation = await pubsub.get_message(timeout=_REPLY_POLL_S)
                    if confirmation is not None and confirmation["type"] == "subscribe":
                        break
                self._subscribed.set()
                attempt = 0
                while True:
                    message = await pubsub.get_message(timeout=_REPLY_POLL_S)
                    if message is not None and message["type"] == "message":
                        # Each event is answered on its own task, so a long
                        # handler does not hold up the next event.
                        task = asyncio.create_task(self._answer(message["data"]))
                        self._answers.add(task)
                        task.add_done_callback(self._answers.discard)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                delay = _RECONNECT_BACKOFF_S[
                    min(attempt, len(_RECONNECT_BACKOFF_S) - 1)
                ]
                attempt += 1
                logger.error(
                    "Cluster events subscription for worker %s lost (%s: %s); "
                    "events published meanwhile do not reach it; resubscribing "
                    "in %.1fs",
                    self.worker_id,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            finally:
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass

    async def _answer(self, data: str) -> None:
        event = json.loads(data)
        handler = self._handlers.get(event["kind"])
        try:
            if handler is None:
                raise LookupError(f"no handler for {event['kind']!r}")
            result = await asyncio.to_thread(handler, event["payload"])
            answer = {"worker": self.worker_id, "ok": True, "result": result}
        except Exception as exc:
            logger.error(
                "Cluster event %s failed on worker %s: %s",
                event["kind"],
                self.worker_id,
                exc,
                exc_info=True,
            )
            answer = {
                "worker": self.worker_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            await self._client.rpush(event["reply_to"], json.dumps(answer))
            await self._client.expire(event["reply_to"], _REPLY_TTL_S)
        except RedisError as exc:
            logger.error(
                "Cluster event %s handled on worker %s but its acknowledgement "
                "was not delivered: %s",
                event["kind"],
                self.worker_id,
                exc,
            )
