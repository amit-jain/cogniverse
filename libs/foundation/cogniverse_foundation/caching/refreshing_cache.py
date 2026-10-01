"""Keyed values served from memory while their refresh runs off the caller's thread."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Hashable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class _Entry(Generic[V]):
    value: V
    # When the read that produced ``value`` began: its age counts from here.
    read_at: float
    # Earliest time a caller starts a background refresh of the entry.
    refresh_at: float


class RefreshingCache(Generic[K, V]):
    """Bounded keyed cache whose refreshes never run on the reading thread.

    ``get(key, read)`` answers from an entry younger than ``refresh_after_s``
    without reading. An entry at least that old but younger than
    ``max_staleness_s`` is still answered, and the call starts one background
    ``read`` of the key on a daemon thread; concurrent callers share that read
    and none waits for it. A key with no entry, or one whose entry has reached
    ``max_staleness_s``, is read on the caller's thread; concurrent callers
    wait on that one read, and its failure raises to each of them and caches
    nothing.

    An entry's age counts from when the read that produced it began, so no
    value is answered ``max_staleness_s`` or more after its read started. A
    failed background read is logged, leaves the entry in place, and is not
    retried for ``refresh_after_s``; once the entry reaches
    ``max_staleness_s`` the next caller reads on its own thread and the failure
    raises to it. At most ``max_background_reads`` background reads run at
    once; a stale entry found while all are busy is answered, and its refresh
    starts on a later call.

    ``invalidate`` drops matching entries and detaches their reads in flight:
    a detached read's result reaches the callers already waiting on it and is
    never cached. Entries beyond ``max_entries`` are evicted least recently
    used first.
    """

    def __init__(
        self,
        *,
        name: str,
        refresh_after_s: float,
        max_staleness_s: float,
        max_entries: int,
        max_background_reads: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if refresh_after_s < 0:
            raise ValueError(f"refresh_after_s must be >= 0, got {refresh_after_s}")
        if max_staleness_s < refresh_after_s:
            raise ValueError(
                f"max_staleness_s ({max_staleness_s}) must be >= "
                f"refresh_after_s ({refresh_after_s})"
            )
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        if max_background_reads < 1:
            raise ValueError(
                f"max_background_reads must be >= 1, got {max_background_reads}"
            )
        self._name = name
        self._refresh_after_s = refresh_after_s
        self._max_staleness_s = max_staleness_s
        self._max_entries = max_entries
        self._max_background_reads = max_background_reads
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[K, _Entry[V]]" = OrderedDict()
        self._reads: Dict[K, Future] = {}
        self._background_reads = 0

    @property
    def refresh_after_s(self) -> float:
        return self._refresh_after_s

    @property
    def max_staleness_s(self) -> float:
        return self._max_staleness_s

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> List[K]:
        """Cached keys, least recently used first."""
        with self._lock:
            return list(self._entries)

    def get(self, key: K, read: Callable[[], V]) -> V:
        """Answer ``key`` from memory when allowed, else from ``read``."""
        refresh: Optional[Future] = None
        with self._lock:
            now = self._clock()
            entry = self._entries.get(key)
            answerable = (
                entry is not None and now - entry.read_at < self._max_staleness_s
            )
            if answerable:
                self._entries.move_to_end(key)
                if (
                    now >= entry.refresh_at
                    and key not in self._reads
                    and self._background_reads < self._max_background_reads
                ):
                    refresh = self._reads[key] = Future()
                    self._background_reads += 1
                served = entry.value
            else:
                pending = self._reads.get(key)
                owner = pending is None
                if owner:
                    pending = self._reads[key] = Future()

        if answerable:
            if refresh is not None:
                self._start_refresh(key, read, refresh, now)
            return served
        if not owner:
            return pending.result()
        try:
            value = read()
        except BaseException as exc:
            self._settle(key, pending, now, error=exc)
            raise
        self._settle(key, pending, now, value=value)
        return value

    def invalidate(self, matches: Callable[[K], bool]) -> None:
        """Drop every entry whose key ``matches`` and detach its read in flight."""
        with self._lock:
            for key in [key for key in self._entries if matches(key)]:
                del self._entries[key]
            for key in [key for key in self._reads if matches(key)]:
                del self._reads[key]

    def _start_refresh(self, key: K, read: Callable[[], V], pending, started) -> None:
        try:
            threading.Thread(
                target=self._refresh,
                args=(key, read, pending, started),
                name=f"{self._name}-refresh",
                daemon=True,
            ).start()
        except Exception as exc:
            self._settle(key, pending, started, error=exc, background=True)

    def _refresh(self, key: K, read: Callable[[], V], pending, started) -> None:
        try:
            value = read()
        except BaseException as exc:
            self._settle(key, pending, started, error=exc, background=True)
            if not isinstance(exc, Exception):
                raise
            return
        self._settle(key, pending, started, value=value, background=True)

    def _settle(
        self,
        key: K,
        pending: Future,
        started: float,
        *,
        value: Optional[V] = None,
        error: Optional[BaseException] = None,
        background: bool = False,
    ) -> None:
        served_age: Optional[float] = None
        with self._lock:
            if background:
                self._background_reads -= 1
            if self._reads.get(key) is pending:
                del self._reads[key]
                if error is None:
                    self._entries[key] = _Entry(
                        value, started, started + self._refresh_after_s
                    )
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
                else:
                    entry = self._entries.get(key)
                    if entry is not None:
                        now = self._clock()
                        entry.refresh_at = now + self._refresh_after_s
                        served_age = now - entry.read_at
        if background and error is not None and served_age is not None:
            logger.error(
                "%s: refreshing %r failed with %s: %s; serving the value read "
                "%.1fs ago until it is %.1fs old",
                self._name,
                key,
                type(error).__name__,
                error,
                served_age,
                self._max_staleness_s,
            )
        if error is None:
            pending.set_result(value)
        else:
            pending.set_exception(error)
