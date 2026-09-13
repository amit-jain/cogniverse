"""A tenant-scoped LRU of LM responses with expiring entries and shared calls."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from dspy.clients.cache import Cache

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

logger = logging.getLogger(__name__)

# Secrets never enter key material. Everything else about the request does,
# including ``api_base`` and ``extra_headers``: two endpoints, or two header
# sets, are two different upstreams and must not share an answer.
IGNORED_CACHE_ARGS: tuple[str, ...] = ("api_key",)

# Key derivation only. Both stores are off, so this instance holds nothing.
_KEYER = Cache(enable_disk_cache=False, enable_memory_cache=False, disk_cache_dir="")

_MISS = object()


def failure_note(tenant_id: str, model: str, key: str) -> str:
    """The context a failed call carries away with it.

    Attached to the provider's own exception rather than replacing it: a
    caller that classifies on the exception type, and a client that reads the
    type name off an error event, both need the provider's error, not a
    wrapper that only says an LM call failed.
    """
    return (
        f"cogniverse LM response cache: tenant={tenant_id} model={model} "
        f"key_digest={key_digest(key)} (nothing stored)"
    )


def key_digest(key: str) -> str:
    """The short form of a cache key used in logs and errors."""
    return key.rsplit("|", 1)[-1][:16]


def request_cache_key(tenant_id: str, request: dict[str, Any]) -> str:
    """The key ``request`` is stored under for ``tenant_id``.

    The tenant is a separate component of the key rather than a field of the
    hashed request, so two tenants issuing a byte-identical request still
    address two entries.
    """
    digest = _KEYER.cache_key(
        request, ignored_args_for_cache_key=list(IGNORED_CACHE_ARGS)
    )
    return f"{canonical_tenant_id(tenant_id)}|{digest}"


@dataclass
class _Entry:
    value: Any
    expires_at: float


class TenantScopedLMCache:
    """An LRU of LM responses with a per-entry TTL and one build per key.

    ``ttl_seconds`` and ``max_entries`` come from the shipped
    ``SemanticRouterConfig``; ``clock`` is the monotonic source the TTL is
    measured against.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries}")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()

    def __deepcopy__(self, memo: dict) -> "TenantScopedLMCache":
        """Process state, not LM state: ``dspy.LM.copy()`` deepcopies the LM and
        every copy addresses the same cache."""
        return self

    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _take(self, key: str) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            return _MISS
        if self._clock() >= entry.expires_at:
            del self._entries[key]
            return _MISS
        self._entries.move_to_end(key)
        return entry.value

    def _store(self, key: str, value: Any) -> None:
        self._entries[key] = _Entry(
            value=copy.deepcopy(value), expires_at=self._clock() + self.ttl_seconds
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            evicted, _ = self._entries.popitem(last=False)
            logger.debug(
                "Evicted LM response %s (cap=%d)", key_digest(evicted), self.max_entries
            )

    def _served(self, value: Any) -> Any:
        """A stored response, prepared for a caller that made no LM call."""
        served = copy.deepcopy(value)
        if hasattr(served, "usage"):
            served.usage = {}
        served.cache_hit = True
        return served

    def _report_failure(
        self, exc: BaseException, *, tenant_id: str, model: str, key: str
    ) -> None:
        """Name the failed request, and leave the provider's error alone."""
        logger.error(
            "LM call failed for tenant=%s model=%s key_digest=%s; "
            "nothing cached: %s: %s",
            tenant_id,
            model,
            key_digest(key),
            type(exc).__name__,
            exc,
        )
        exc.add_note(failure_note(tenant_id, model, key))

    def _claim(self, key: str) -> tuple[Future, bool]:
        with self._lock:
            hit = self._take(key)
            if hit is not _MISS:
                ready = Future()
                ready.set_result(hit)
                return ready, False
            pending = self._inflight.get(key)
            if pending is not None:
                return pending, False
            pending = Future()
            self._inflight[key] = pending
            return pending, True

    def _finish(self, key: str, pending: Future, value: Any) -> None:
        with self._lock:
            self._store(key, value)
            stored = self._entries[key].value
            self._inflight.pop(key)
        pending.set_result(stored)

    def _fail(self, key: str, pending: Future, exc: BaseException) -> None:
        with self._lock:
            self._inflight.pop(key)
        pending.set_exception(exc)

    def get_or_call(
        self,
        key: str,
        factory: Callable[[], Any],
        *,
        tenant_id: str,
        model: str,
    ) -> Any:
        """Share one upstream call across synchronous and asynchronous callers."""
        pending, owner = self._claim(key)
        if not owner:
            return self._served(pending.result())
        try:
            value = factory()
            self._finish(key, pending, value)
            return value
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._report_failure(exc, tenant_id=tenant_id, model=model, key=key)
            self._fail(key, pending, exc)
            raise

    async def aget_or_call(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        *,
        tenant_id: str,
        model: str,
    ) -> Any:
        """Await the shared call without blocking a loop or cancelling its owner."""
        pending, owner = self._claim(key)
        if not owner:
            return self._served(await asyncio.shield(asyncio.wrap_future(pending)))
        try:
            value = await factory()
            self._finish(key, pending, value)
            return value
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                self._report_failure(exc, tenant_id=tenant_id, model=model, key=key)
            self._fail(key, pending, exc)
            raise


_PROCESS_CACHE: TenantScopedLMCache | None = None
_PROCESS_CACHE_LOCK = threading.Lock()


def lm_response_cache() -> TenantScopedLMCache:
    """The cache every tenant-bound LM in this process shares."""
    global _PROCESS_CACHE
    if _PROCESS_CACHE is not None:
        return _PROCESS_CACHE
    with _PROCESS_CACHE_LOCK:
        if _PROCESS_CACHE is None:
            from cogniverse_foundation.config.utils import ConfigUtils

            path = ConfigUtils._discover_config_file()
            if path is None:
                raise FileNotFoundError(
                    "LM response cache configuration: no config.json in the "
                    "standard locations"
                )
            try:
                bounds = json.loads(path.read_text())["semantic_router"]
                cache = TenantScopedLMCache(
                    ttl_seconds=bounds["response_cache_ttl_seconds"],
                    max_entries=bounds["response_cache_max_entries"],
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                exc.add_note(f"LM response cache configuration: {path}")
                raise
            _PROCESS_CACHE = cache
        return _PROCESS_CACHE


__all__ = [
    "IGNORED_CACHE_ARGS",
    "TenantScopedLMCache",
    "failure_note",
    "key_digest",
    "lm_response_cache",
    "request_cache_key",
]
