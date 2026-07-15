"""EntryPointRegistry.get() resolves concurrent cold-starts to ONE instance.

The base registry did a non-atomic get()-then-set(): N threads racing a cold
key each built an instance, and the LRU set() displaced (and CLOSED, via
on_evict) an instance another thread was already holding — a use-after-close on
a real store's HTTP/Vespa session. The BackendRegistry fix used set_if_absent;
this base class was missed. Every registry subclass (adapter/workflow/
telemetry/evaluation stores) inherits this get().
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

import pytest

from cogniverse_foundation.registry.entry_point_registry import EntryPointRegistry

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Plugin:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _make_registry():
    """A fresh registry class with a slow factory + a cache cap of 2."""

    class _Reg(EntryPointRegistry[Any]):
        _entry_point_group = "test.race.plugins"
        _label = "race-plugin"
        _tenant_scoped = False
        built: ClassVar[list] = []
        _lock_started: ClassVar[threading.Barrier]

        @classmethod
        def _create_instance(cls, klass, config, tenant_id):
            cls._lock_started.wait()  # force all threads into the build together
            inst = _Plugin()
            cls.built.append(inst)
            return inst

    _Reg.reset()
    _Reg._registry_classes = {"only": _Plugin}
    _Reg._entry_points_loaded = True
    return _Reg


def test_concurrent_cold_start_builds_one_live_instance():
    N = 16
    reg = _make_registry()
    reg._lock_started = threading.Barrier(N)

    results: list = []
    errors: list = []
    rlock = threading.Lock()

    def worker():
        try:
            inst = reg.get(name="only")
            # Touch it AFTER get returns — a displaced-then-closed instance
            # would be observably closed here (use-after-close).
            with rlock:
                results.append(inst)
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            with rlock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"get() raced: {errors}"
    # All callers must receive the SAME live instance...
    assert len({id(r) for r in results}) == 1, (
        f"{len({id(r) for r in results})} distinct instances handed out — "
        "get() double-built under concurrency"
    )
    # ...and no caller may be holding a closed instance.
    assert not any(r.closed for r in results), "a caller got a closed instance"
    # Losers built in the race must have been closed (resource released), and
    # the single survivor must be live.
    survivor = results[0]
    assert not survivor.closed
    losers = [b for b in reg.built if b is not survivor]
    assert all(loser.closed for loser in losers), (
        "loser instances built in the race were leaked (not closed)"
    )
