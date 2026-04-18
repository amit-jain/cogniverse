"""Bounded LRU cache keyed by tenant_id.

Every long-running cogniverse process keeps per-tenant state in memory:
Mem0 memory managers, telemetry providers, backend clients, compiled
DSPy modules. Without a bound, a multi-tenant server (or an e2e suite
that creates a fresh tenant per test) accumulates one instance per
tenant indefinitely, which is the dominant OOM driver in this codebase.

`TenantLRUCache` caps the working set at a configurable capacity and
evicts the least-recently-used tenant when the cap is reached. An
optional ``on_evict`` callback lets callers release native resources
(gRPC channels, Vespa sessions) before the instance is dropped.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TenantLRUCache(Generic[T]):
    """Thread-safe LRU cache keyed by tenant id.

    ``get_or_set`` is the primary entry point — it resolves a cached
    value or builds one atomically under a lock, so concurrent callers
    see a single shared instance without racing the factory.
    """

    def __init__(
        self,
        capacity: int,
        on_evict: Optional[Callable[[str, T], None]] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("TenantLRUCache capacity must be >= 1")
        self._capacity = capacity
        self._on_evict = on_evict
        self._data: "OrderedDict[str, T]" = OrderedDict()
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            self._evict_over_capacity()

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        """Return cached value or build + cache one atomically."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            value = factory()
            self._data[key] = value
            self._evict_over_capacity()
            return value

    def pop(self, key: str, default: Optional[T] = None) -> Optional[T]:
        with self._lock:
            return self._data.pop(key, default)

    def clear(self) -> None:
        with self._lock:
            entries = list(self._data.items())
            self._data.clear()
        for key, value in entries:
            self._invoke_on_evict(key, value)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def _evict_over_capacity(self) -> None:
        evicted: list[tuple[str, T]] = []
        while len(self._data) > self._capacity:
            evicted.append(self._data.popitem(last=False))
        for key, value in evicted:
            self._invoke_on_evict(key, value)

    def _invoke_on_evict(self, key: str, value: T) -> None:
        if self._on_evict is None:
            return
        try:
            self._on_evict(key, value)
        except Exception as exc:
            logger.warning("TenantLRUCache on_evict failed for %s: %s", key, exc)
