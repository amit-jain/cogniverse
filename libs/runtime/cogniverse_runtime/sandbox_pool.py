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
    """One slot in the pool: a session plus its last-used timestamp.

    ``drain`` is set by ``close_all`` on sessions that are checked out at
    close time: the in-flight exec keeps the session until it returns, and
    ``_release`` then destroys it instead of re-pooling.
    """

    agent_type: str
    session: Any
    last_used_at: float
    in_use: bool = False
    drain: bool = False


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
        gateway_breaker: Any = None,
    ) -> None:
        self._client = client
        self._config = config or SandboxPoolConfig.from_environment()
        self._wait_ready_timeout = wait_ready_timeout_s
        # Shared circuit breaker for gateway dials; when open, session creation
        # raises CircuitOpenError immediately instead of hanging on wait_ready.
        self._gateway_breaker = gateway_breaker
        self._lock = threading.Lock()
        # agent_type -> _PoolEntry. One entry per agent at most; this keeps
        # the pool tiny and predictable. If finer-grained pooling is
        # needed later (e.g. multiple sessions per agent for parallelism),
        # extend the value to a list.
        self._entries: dict[str, _PoolEntry] = {}
        # Set by close_all and never cleared: a closed pool is being
        # discarded (shutdown or reconnect swap), so released sessions are
        # destroyed instead of re-pooled.
        self._draining = False

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
        with self._lock:
            stale = [
                k
                for k, e in self._entries.items()
                if not e.in_use and e.last_used_at < cutoff
            ]
            stale_entries = [self._entries.pop(k) for k in stale]
        # Destroy OUTSIDE the lock (like close_all): session.delete() is an
        # un-timed gateway RPC — holding self._lock across it would block every
        # checkout/release behind a hung gateway.
        for entry in stale_entries:
            self._destroy_session_quiet(entry.session)
        evicted = len(stale_entries)
        if evicted:
            logger.info("Sandbox pool evicted %d idle sessions", evicted)
        return evicted

    def close_all(self) -> None:
        """Destroy idle sessions now; drain checked-out ones on release.

        Called from runtime shutdown and from the reconnect path
        (``_drop_stale_pool``), which can run while a coding exec still
        holds a checked-out session — deleting that session here would tear
        it out from under the exec mid-call. Instead, in-use sessions are
        marked to drain and destroyed by ``_release`` when the exec
        returns. The pool stays draining afterwards, so nothing is ever
        re-pooled onto a closed pool.
        """
        with self._lock:
            self._draining = True
            idle = [e for e in self._entries.values() if not e.in_use]
            for entry in self._entries.values():
                if entry.in_use:
                    entry.drain = True
            for entry in idle:
                self._entries.pop(entry.agent_type, None)
        # Destroy OUTSIDE the lock: session.delete() is an un-timed gateway
        # RPC — holding self._lock across it would block every
        # checkout/release behind a hung gateway.
        for entry in idle:
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
            victim = self._evict_oldest_if_full_locked(skip_agent=agent_type)

            # Drop the lock for the (potentially slow) network calls below.
        # Destroy OUTSIDE the lock (like evict_idle/close_all): session.delete()
        # is an un-timed gateway RPC — holding self._lock across it would block
        # every checkout/release behind a hung gateway.
        if victim is not None:
            self._destroy_session_quiet(victim.session)
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
            # If a concurrent checkout populated the slot meanwhile, never
            # overwrite it — that orphans the live pooled session.
            existing = self._entries.get(agent_type)
            if existing is not None:
                if not existing.in_use:
                    # Reusable entry already pooled — take it, discard ours.
                    self._destroy_session_quiet(new_entry.session)
                    existing.in_use = True
                    return existing
                # Slot holds an in-use session; ours is an over-provisioned
                # transient that _release destroys instead of pooling.
                return new_entry
            self._entries[agent_type] = new_entry
            return new_entry

    def _release(self, entry: _PoolEntry) -> None:
        with self._lock:
            pooled = self._entries.get(entry.agent_type) is entry
            if pooled and not (self._draining or entry.drain):
                entry.in_use = False
                entry.last_used_at = time.monotonic()
                return
            if pooled:
                self._entries.pop(entry.agent_type, None)
        # Draining (close_all ran while this session was checked out), an
        # over-provisioned transient, or one replaced by a concurrent
        # checkout — destroy instead of re-pooling, outside the lock.
        self._destroy_session_quiet(entry.session)

    def _discard(self, agent_type: str, entry: _PoolEntry, *, reason: str) -> None:
        with self._lock:
            current = self._entries.get(agent_type)
            if current is entry:
                self._entries.pop(agent_type, None)
        self._destroy_session_quiet(entry.session)
        logger.warning(
            "Sandbox pool dropped %s session (reason=%s)", agent_type, reason
        )

    def _evict_oldest_if_full_locked(self, *, skip_agent: str) -> Optional[_PoolEntry]:
        """Caller holds the lock. Pop the oldest idle entry if at capacity.

        Returns the popped entry for the caller to destroy AFTER releasing
        the lock — session.delete() is an un-timed gateway RPC.
        """
        if len(self._entries) < self._config.max_pool_size:
            return None
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
            return None
        # Drop oldest by last_used_at.
        idle.sort(key=lambda pair: pair[1].last_used_at)
        victim_key, victim_entry = idle[0]
        self._entries.pop(victim_key, None)
        return victim_entry

    @staticmethod
    def _destroy_session_quiet(session: Any) -> None:
        try:
            session.delete()
        except Exception as exc:
            logger.debug("Pool destroy session failed (non-fatal): %s", exc)

    def _create_with_spans(self) -> Any:
        """Create a session + wait for ready, through the gateway breaker."""
        if self._gateway_breaker is not None:
            return self._gateway_breaker.call(self._do_create_session)
        return self._do_create_session()

    def _do_create_session(self) -> Any:
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
