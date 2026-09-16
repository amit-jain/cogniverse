"""SandboxSessionPool — capacity-bounded OpenShell sessions for tasks.

Every coding task leases its own session through ``task_session``: the
session is created on entry, owned exclusively for the duration of the task,
and destroyed on exit. ``max_pool_size`` caps how many sessions are live at
once, so container creation is bounded by the gateway's capacity rather than
by request concurrency; a task that arrives at the cap is refused with
``SandboxCapacityError``.

Operators tune the cap via ``COGNIVERSE_SANDBOX_POOL_SIZE`` (see
``SandboxPoolConfig``).

Concurrency: a single ``threading.Lock`` guards lease bookkeeping. The pool
is sync-only (matches the existing SDK call shape); callers wrap calls in
``asyncio.to_thread`` when used from async code.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxPoolConfig:
    """Capacity tuning for concurrent task sessions.

    Defaults are conservative. Operators bump these in production after
    measuring real container start latency on their gateway.
    """

    max_pool_size: int = 8

    @classmethod
    def from_environment(cls) -> "SandboxPoolConfig":
        """Honour COGNIVERSE_SANDBOX_POOL_* env vars; fall back to defaults."""
        return cls(
            max_pool_size=int(os.environ.get("COGNIVERSE_SANDBOX_POOL_SIZE", "8")),
        )


class SandboxCapacityError(RuntimeError):
    """Raised when every task-session slot is already taken."""


@dataclass
class _TaskLease:
    """One task's claim on a sandbox slot, holding its session once created."""

    session: Any = None


class SandboxSessionPool:
    """Capacity-bounded leases over per-task OpenShell sessions.

    Args:
        client: OpenShell ``SandboxClient`` (already connected). The pool
            calls ``client.create_session()`` and ``client.wait_ready()``
            when a task takes a lease.
        config: Capacity config. Defaults from env when omitted.
        wait_ready_timeout_s: Timeout passed to ``client.wait_ready``. Matches
            the openshell SDK's own wait_ready default. A cold sandbox start
            measures ~168s here, dominated by pulling the sandbox base image
            on first use, so a smaller budget turns a normal cold start into
            a timeout.
        gateway_breaker: Shared circuit breaker for gateway dials; when open,
            session creation raises ``CircuitOpenError`` immediately instead
            of hanging on wait_ready.
    """

    def __init__(
        self,
        client: Any,
        config: Optional[SandboxPoolConfig] = None,
        wait_ready_timeout_s: int = 300,
        gateway_breaker: Any = None,
    ) -> None:
        self._client = client
        self._config = config or SandboxPoolConfig.from_environment()
        self._wait_ready_timeout = wait_ready_timeout_s
        self._gateway_breaker = gateway_breaker
        self._lock = threading.Lock()
        # Leases handed out by ``task_session``. A task owns its sandbox
        # exclusively and the session is destroyed on release; the pool tracks
        # them here to hold them inside ``max_pool_size`` and to reclaim them
        # in ``close_all``.
        self._task_leases: list[_TaskLease] = []

    # --- public API -------------------------------------------------------

    @property
    def config(self) -> SandboxPoolConfig:
        return self._config

    def stats(self) -> dict:
        """Snapshot of pool occupancy. Used by tests + dashboards."""
        with self._lock:
            return {
                "max_pool_size": self._config.max_pool_size,
                "task_sessions": len(self._task_leases),
            }

    @contextmanager
    def task_session(self):
        """Own a fresh session until task completion, then destroy it.

        Raises ``SandboxCapacityError`` when ``max_pool_size`` task sessions
        are already live: container creation is bounded by the pool's
        capacity, not by request concurrency.
        """
        lease = _TaskLease()
        with self._lock:
            if len(self._task_leases) >= self._config.max_pool_size:
                raise SandboxCapacityError(
                    f"Sandbox task capacity reached: "
                    f"{self._config.max_pool_size} sessions in use"
                )
            self._task_leases.append(lease)
        try:
            lease.session = self._create_with_spans()
        except BaseException:
            self._release_task_lease(lease)
            raise
        try:
            yield lease.session
        finally:
            owed = self._release_task_lease(lease)
            if owed is not None:
                self._destroy_with_span(owed)

    def close_all(self) -> None:
        """Destroy every live task session now.

        Called from runtime shutdown and from the reconnect path
        (``_drop_stale_pool``), both of which close the client right after,
        so anything left alive here is a container the gateway keeps forever.
        A task session outlives any single call, so it is destroyed here and
        its owner's release skips the delete. A lease whose session is still
        being created stays with its owner, which destroys it on release.
        """
        with self._lock:
            leased = [lease for lease in self._task_leases if lease.session]
            self._task_leases = [
                lease for lease in self._task_leases if not lease.session
            ]
        # Destroy OUTSIDE the lock: session.delete() is an un-timed gateway
        # RPC — holding self._lock across it would block every lease behind a
        # hung gateway.
        for lease in leased:
            self._destroy_session_quiet(lease.session)

    # --- internals --------------------------------------------------------

    def _release_task_lease(self, lease: _TaskLease) -> Any:
        """Drop ``lease`` and return the session still owed a delete.

        ``None`` means ``close_all`` already took the lease and destroyed its
        session, so the task must not delete it a second time.
        """
        with self._lock:
            for index, held in enumerate(self._task_leases):
                if held is lease:
                    del self._task_leases[index]
                    return lease.session
        return None

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
        try:
            with tracer.start_as_current_span(
                "sandbox.wait_ready",
                attributes={"openshell.wait_timeout_s": self._wait_ready_timeout},
            ):
                self._client.wait_ready(
                    session.sandbox.name,
                    timeout_seconds=self._wait_ready_timeout,
                )
        except BaseException:
            self._destroy_with_span(session)
            raise
        return session

    def _destroy_with_span(self, session: Any) -> None:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("sandbox.delete"):
            self._destroy_session_quiet(session)
