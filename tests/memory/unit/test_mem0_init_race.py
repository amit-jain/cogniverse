"""Concurrent cold-init of the shared per-tenant Mem0MemoryManager builds once.

The manager is a per-tenant singleton shared by every agent for the tenant, and
``initialize()`` runs per dispatch on worker threads (via ``asyncio.to_thread``).
Two concurrent first requests for one tenant routed to *different* agents get
different dispatcher cache keys, so the per-(tenant,agent) stampede guard does
not serialize them — both reach ``initialize`` on the same shared manager. Cold
init must not run the Memory-stack build + tenant-schema redeploy twice. A
per-instance lock with a double-check serializes it; this pins that exactly one
build runs no matter how many threads race the cold path.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _fresh_manager() -> Mem0MemoryManager:
    # Unique tenant → a fresh singleton with ``memory is None`` (cold path).
    return Mem0MemoryManager(f"race:{uuid.uuid4().hex[:12]}")


def _slow_counting_build(calls: list):
    def _build(self, *, fingerprint, **kwargs):
        calls.append(1)
        # Widen the cold-build window so any un-serialized concurrent caller
        # also enters here before this one records ``memory``.
        time.sleep(0.15)
        self.memory = object()  # non-None: the double-check short-circuits
        self._init_fingerprint = fingerprint

    return _build


def _init_kwargs() -> dict:
    return dict(
        backend_host="localhost",
        backend_port=8080,
        llm_model="m",
        embedding_model="e",
        llm_base_url="http://lm/v1",
        embedder_base_url="http://emb/v1",
        config_manager=object(),
        schema_loader=object(),
        embedding_dims=768,
    )


def _race_initialize(mgr: Mem0MemoryManager, n: int) -> list:
    """Release ``n`` threads into ``initialize`` together; return any errors."""
    errors: list = []
    ready = threading.Barrier(n)

    def worker():
        try:
            ready.wait(timeout=5)
            mgr.initialize(**_init_kwargs())
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return errors


def test_concurrent_cold_init_builds_once(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        Mem0MemoryManager, "_build_and_store_memory", _slow_counting_build(calls)
    )
    mgr = _fresh_manager()

    errors = _race_initialize(mgr, 8)

    assert errors == []
    assert calls == [1]  # the lock serialized cold init to a single build
    assert mgr.memory is not None


def test_neutralized_lock_double_builds(monkeypatch):
    """Discrimination guard: with the per-instance lock replaced by a no-op,
    the same race double-builds — proving the single-build assertion above is
    not vacuous."""

    class _NoOpLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls: list = []
    monkeypatch.setattr(
        Mem0MemoryManager, "_build_and_store_memory", _slow_counting_build(calls)
    )
    mgr = _fresh_manager()
    mgr._init_lock = _NoOpLock()

    errors = _race_initialize(mgr, 8)

    assert errors == []
    assert sum(calls) > 1  # unserialized cold init runs the build more than once
