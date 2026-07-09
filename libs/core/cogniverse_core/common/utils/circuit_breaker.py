"""Per-dependency circuit breaker.

When a downstream dependency (Vespa, an LM endpoint, Phoenix, the OpenShell
sandbox gateway) is down, dialing it on every request and eating the full
timeout serially stalls the worker pool. A breaker trips ``OPEN`` after enough
failures in a rolling window and then fails fast — raising
:class:`CircuitOpenError` immediately instead of dialing — until a reset window
elapses, when a single ``HALF_OPEN`` trial probes recovery.

Composes with :mod:`.retry`: wrap the outermost boundary call in the breaker and
let ``retry_with_backoff`` sit inside it, so transient blips are retried but a
sustained outage stops being retried once the breaker opens.

State is shared per dependency name via :meth:`CircuitBreaker.get`, so every
caller of the same dependency sees the same breaker.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Deque, Dict, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised while a breaker is OPEN — the call was not attempted."""

    def __init__(self, name: str):
        super().__init__(f"Circuit '{name}' is open; call rejected")
        self.name = name


@dataclass
class BreakerConfig:
    """Per-dependency breaker settings.

    ``failure_threshold`` failures within ``window_s`` trip the breaker OPEN for
    ``reset_timeout_s``; then one HALF_OPEN trial decides CLOSED vs OPEN.
    A ``failure_threshold`` of 0 disables the breaker (calls always pass through).
    """

    name: str
    failure_threshold: int = 5
    window_s: float = 60.0
    reset_timeout_s: float = 30.0
    half_open_max_calls: int = 1
    counted_exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    # Injectable monotonic clock for deterministic tests.
    clock: Callable[[], float] = field(default=time.monotonic)


class CircuitBreaker:
    _registry: Dict[str, "CircuitBreaker"] = {}
    _registry_lock = threading.Lock()

    @classmethod
    def get(cls, config: BreakerConfig) -> "CircuitBreaker":
        """Return the shared breaker for ``config.name`` (created once)."""
        with cls._registry_lock:
            existing = cls._registry.get(config.name)
            if existing is None:
                existing = cls(config)
                cls._registry[config.name] = existing
            return existing

    @classmethod
    def reset_registry(cls) -> None:
        """Drop all breakers — for test isolation."""
        with cls._registry_lock:
            cls._registry.clear()

    def __init__(self, config: BreakerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures: Deque[float] = deque()
        self._opened_at: float = 0.0
        self._half_open_calls: int = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _now(self) -> float:
        return self.config.clock()

    def _maybe_half_open(self) -> None:
        """OPEN → HALF_OPEN once the reset window has elapsed. Caller holds lock."""
        if (
            self._state is CircuitState.OPEN
            and self._now() - self._opened_at >= self.config.reset_timeout_s
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.warning(
                "Circuit '%s' HALF_OPEN — probing recovery", self.config.name
            )

    def _before_call(self) -> None:
        """Admission check. Raises CircuitOpenError if the call must not proceed."""
        if self.config.failure_threshold <= 0:
            return
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                raise CircuitOpenError(self.config.name)
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(self.config.name)
                self._half_open_calls += 1

    def _on_success(self) -> None:
        if self.config.failure_threshold <= 0:
            return
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                logger.warning("Circuit '%s' CLOSED — recovered", self.config.name)
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._half_open_calls = 0

    def _on_failure(self) -> None:
        if self.config.failure_threshold <= 0:
            return
        with self._lock:
            now = self._now()
            if self._state is CircuitState.HALF_OPEN:
                # Trial failed — back to OPEN.
                self._trip(now)
                return
            self._failures.append(now)
            cutoff = now - self.config.window_s
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()
            if len(self._failures) >= self.config.failure_threshold:
                self._trip(now)

    def _trip(self, now: float) -> None:
        """Move to OPEN. Caller holds lock."""
        was = self._state
        self._state = CircuitState.OPEN
        self._opened_at = now
        self._failures.clear()
        self._half_open_calls = 0
        if was is not CircuitState.OPEN:
            logger.warning(
                "Circuit '%s' OPEN — failing fast for %.0fs",
                self.config.name,
                self.config.reset_timeout_s,
            )

    def _counts(self, exc: BaseException) -> bool:
        return isinstance(exc, self.config.counted_exceptions)

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run ``fn`` through the breaker (sync)."""
        self._before_call()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            if self._counts(exc):
                self._on_failure()
            raise
        self._on_success()
        return result

    async def acall(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Run an awaitable ``fn`` through the breaker (async)."""
        self._before_call()
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:
            if self._counts(exc):
                self._on_failure()
            raise
        self._on_success()
        return result


def circuit_breaker(config: BreakerConfig):
    """Decorator form: wrap a function in the shared breaker for ``config.name``."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        breaker = CircuitBreaker.get(config)

        def wrapper(*args, **kwargs) -> T:
            return breaker.call(fn, *args, **kwargs)

        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator


__all__ = [
    "BreakerConfig",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "circuit_breaker",
]
