"""SandboxSessionPool — reuse OpenShell sessions across exec calls.

The original ``SandboxManager.exec_in_sandbox`` created and destroyed a
sandbox session per call. With every non-coding agent's outbound HTTP
wired through the sandbox, that becomes a per-request container churn —
unacceptable cost for a hot path.

This pool keeps one ``(agent_type, session)`` slot per agent type, reuses
it for subsequent calls, and destroys the session when it has been idle
beyond a threshold or when the pool exceeds its capacity.

Pool size is small by design: cogniverse has on the order of 10 agents,
and each session's container holds resources. Operators tune via env vars
(see ``SandboxPoolConfig``).

Concurrency: a single ``threading.Lock`` guards pool mutation. The pool
is sync-only (matches the existing SDK call shape); callers wrap calls
in ``asyncio.to_thread`` when used from async code.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxPoolConfig:
    """Pool sizing + idle-eviction tuning.

    Defaults are conservative. Operators bump these in production after
    measuring real container start latency on their gateway.
    """

    enabled: bool = True
    max_pool_size: int = 8
    max_idle_seconds: float = 60.0

    @classmethod
    def from_environment(cls) -> "SandboxPoolConfig":
        """Honour COGNIVERSE_SANDBOX_POOL_* env vars; fall back to defaults."""
        return cls(
            enabled=os.environ.get("COGNIVERSE_SANDBOX_POOL_ENABLED", "1").lower()
            in {"1", "true", "yes"},
            max_pool_size=int(os.environ.get("COGNIVERSE_SANDBOX_POOL_SIZE", "8")),
            max_idle_seconds=float(
                os.environ.get("COGNIVERSE_SANDBOX_POOL_IDLE_S", "60")
            ),
        )


@dataclass
class _PoolEntry:
    """One slot in the pool: a session plus its last-used timestamp."""

    agent_type: str
    session: Any
    last_used_at: float
    in_use: bool = False


class SandboxSessionPool:
    """Per-agent-type session pool with idle eviction.

    Args:
        client: OpenShell ``SandboxClient`` (already connected). Pool calls
            ``client.create_session()`` and ``client.wait_ready()`` on first
            checkout for an agent type, then reuses the live session.
        config: Pool sizing/idle-eviction config. Defaults from env when
            omitted.
        wait_ready_timeout_s: Timeout passed to ``client.wait_ready``
            on first checkout.
    """

    def __init__(
        self,
        client: Any,
        config: Optional[SandboxPoolConfig] = None,
        wait_ready_timeout_s: int = 120,
    ) -> None:
        self._client = client
        self._config = config or SandboxPoolConfig.from_environment()
        self._wait_ready_timeout = wait_ready_timeout_s
        self._lock = threading.Lock()
        # agent_type -> _PoolEntry. One entry per agent at most; this keeps
        # the pool tiny and predictable. If finer-grained pooling is
        # needed later (e.g. multiple sessions per agent for parallelism),
        # extend the value to a list.
        self._entries: dict[str, _PoolEntry] = {}

    # --- public API -------------------------------------------------------

    @property
    def config(self) -> SandboxPoolConfig:
        return self._config

    def stats(self) -> dict:
        """Snapshot of pool occupancy. Used by tests + dashboards."""
        with self._lock:
            return {
                "pool_size": len(self._entries),
                "max_pool_size": self._config.max_pool_size,
                "in_use": sum(1 for e in self._entries.values() if e.in_use),
                "agents": list(self._entries.keys()),
            }

    def with_session(
        self,
        agent_type: str,
        callback: Callable[[Any], Any],
    ) -> Any:
        """Run ``callback(session)`` against a pooled session for ``agent_type``.

        On first call for an agent, creates a fresh session via
        ``client.create_session`` and waits for readiness. Subsequent calls
        reuse the same session until idle eviction. The session is marked
        ``in_use`` while the callback runs and released back on return.

        Sessions that raise during the callback are destroyed and removed
        from the pool (the next checkout creates a fresh one) — preserves
        the invariant "pooled sessions are healthy."
        """
        if not self._config.enabled:
            # Pool disabled: behave like the un-pooled per-call path.
            session = self._create_with_spans()
            try:
                return callback(session)
            finally:
                self._destroy_with_span(session)

        entry = self._checkout(agent_type)
        released = False
        try:
            return callback(entry.session)
        except Exception:
            # Session is suspect — drop it from the pool and propagate.
            released = True  # _discard supersedes _release
            self._discard(agent_type, entry, reason="callback_exception")
            raise
        finally:
            if not released:
                self._release(entry)

    def evict_idle(self, *, now: Optional[float] = None) -> int:
        """Destroy entries idle longer than ``max_idle_seconds``.

        Returns the number of entries evicted. Safe to call from a
        background timer or inline after each checkout (the pool is
        small enough that the scan is cheap).
        """
        cutoff = (now or time.monotonic()) - self._config.max_idle_seconds
        evicted = 0
        with self._lock:
            stale = [
                k
                for k, e in self._entries.items()
                if not e.in_use and e.last_used_at < cutoff
            ]
            for k in stale:
                entry = self._entries.pop(k)
                evicted += 1
                # Best-effort destroy outside the lock.
                self._destroy_session_quiet(entry.session)
        if evicted:
            logger.info("Sandbox pool evicted %d idle sessions", evicted)
        return evicted

    def close_all(self) -> None:
        """Destroy every pooled session. Called from runtime shutdown."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._destroy_session_quiet(entry.session)

    # --- internals --------------------------------------------------------

    def _checkout(self, agent_type: str) -> _PoolEntry:
        """Return a healthy entry for ``agent_type``, creating if needed."""
        with self._lock:
            entry = self._entries.get(agent_type)
            if entry is not None and not entry.in_use:
                entry.in_use = True
                return entry

            # No reusable entry — make space then create a fresh one.
            self._evict_oldest_if_full_locked(skip_agent=agent_type)

            # Drop the lock for the (potentially slow) network calls below.
        # If checkout above hit "in_use" race (rare with single-thread per
        # event loop), fall through to creating a new session anyway. This
        # accepts a transient over-provision rather than blocking.
        new_session = self._create_with_spans()
        new_entry = _PoolEntry(
            agent_type=agent_type,
            session=new_session,
            last_used_at=time.monotonic(),
            in_use=True,
        )
        with self._lock:
            # If a concurrent checkout populated the slot meanwhile, drop
            # the loser to avoid leaking sessions.
            existing = self._entries.get(agent_type)
            if existing is not None and not existing.in_use:
                self._destroy_session_quiet(new_entry.session)
                existing.in_use = True
                return existing
            self._entries[agent_type] = new_entry
            return new_entry

    def _release(self, entry: _PoolEntry) -> None:
        with self._lock:
            entry.in_use = False
            entry.last_used_at = time.monotonic()

    def _discard(self, agent_type: str, entry: _PoolEntry, *, reason: str) -> None:
        with self._lock:
            current = self._entries.get(agent_type)
            if current is entry:
                self._entries.pop(agent_type, None)
        self._destroy_session_quiet(entry.session)
        logger.warning(
            "Sandbox pool dropped %s session (reason=%s)", agent_type, reason
        )

    def _evict_oldest_if_full_locked(self, *, skip_agent: str) -> None:
        """Caller holds the lock. Drop the oldest idle entry if at capacity."""
        if len(self._entries) < self._config.max_pool_size:
            return
        idle = [
            (k, e) for k, e in self._entries.items() if not e.in_use and k != skip_agent
        ]
        if not idle:
            # All slots in use — refuse to evict; new session will push us
            # one over capacity until something releases. Logged for ops.
            logger.warning(
                "Sandbox pool at capacity (%d) and all entries in use; "
                "creating an over-cap session",
                self._config.max_pool_size,
            )
            return
        # Drop oldest by last_used_at.
        idle.sort(key=lambda pair: pair[1].last_used_at)
        victim_key, victim_entry = idle[0]
        self._entries.pop(victim_key, None)
        self._destroy_session_quiet(victim_entry.session)

    @staticmethod
    def _destroy_session_quiet(session: Any) -> None:
        try:
            session.delete()
        except Exception as exc:
            logger.debug("Pool destroy session failed (non-fatal): %s", exc)

    def _create_with_spans(self) -> Any:
        """Create a session + wait for ready, emitting sandbox lifecycle spans."""
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("sandbox.create_session"):
            session = self._client.create_session()
        with tracer.start_as_current_span(
            "sandbox.wait_ready",
            attributes={"openshell.wait_timeout_s": self._wait_ready_timeout},
        ):
            self._client.wait_ready(
                session.sandbox.name,
                timeout_seconds=self._wait_ready_timeout,
            )
        return session

    def _destroy_with_span(self, session: Any) -> None:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("sandbox.delete"):
            self._destroy_session_quiet(session)
