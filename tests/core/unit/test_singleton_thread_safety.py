"""Process-global get-or-create singletons must be thread-safe.

Both getters used a bare ``if _x is None: _x = X()`` — under concurrent
first-touch from multiple threads, several build an instance and all but the
last are orphaned; for the queue manager, events published to an orphan are
invisible to holders of the winner.
"""

from __future__ import annotations

import threading
import time

import pytest


def _race_constructions(getter, patch_target, monkeypatch, threads=24):
    """Run ``getter`` concurrently from many threads with a construction delay
    wide enough that a missing lock builds more than once. Returns
    (construction_count, set_of_returned_ids)."""
    count = {"n": 0}
    count_lock = threading.Lock()
    real = patch_target[1]

    def slow_ctor(*args, **kwargs):
        with count_lock:
            count["n"] += 1
        time.sleep(0.02)  # widen the race window
        return real(*args, **kwargs)

    monkeypatch.setattr(patch_target[0], patch_target[2], slow_ctor)

    results = []
    results_lock = threading.Lock()
    start = threading.Barrier(threads)

    def worker():
        start.wait()
        obj = getter()
        with results_lock:
            results.append(id(obj))

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)

    return count["n"], set(results)


@pytest.mark.unit
def test_get_queue_manager_builds_once_under_race(monkeypatch):
    import cogniverse_core.events.backends.memory as mod

    mod.reset_queue_manager()
    try:
        n, ids = _race_constructions(
            mod.get_queue_manager,
            (mod, mod.InMemoryQueueManager, "InMemoryQueueManager"),
            monkeypatch,
        )
        assert n == 1, f"queue manager constructed {n} times under concurrent race"
        assert len(ids) == 1, "threads received different queue-manager instances"
    finally:
        mod.reset_queue_manager()


@pytest.mark.unit
def test_get_registry_builds_once_under_race(monkeypatch):
    import cogniverse_core.registries.registry as mod

    mod._registry = None
    try:
        n, ids = _race_constructions(
            mod.get_registry,
            (mod, mod.StrategyRegistry, "StrategyRegistry"),
            monkeypatch,
        )
        assert n == 1, f"registry constructed {n} times under concurrent race"
        assert len(ids) == 1, "threads received different registry instances"
    finally:
        mod._registry = None
