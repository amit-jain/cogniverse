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
    (construction_count, set_of_returned_ids, exceptions)."""
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
    errors = []
    results_lock = threading.Lock()
    start = threading.Barrier(threads)

    def worker():
        try:
            start.wait()
            obj = getter()
            with results_lock:
                results.append(id(obj))
        except Exception as exc:
            with results_lock:
                errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)

    assert all(not thread.is_alive() for thread in ts)
    return count["n"], set(results), errors


@pytest.mark.unit
def test_get_queue_manager_builds_once_under_race(monkeypatch):
    import cogniverse_core.events.backends.memory as mod

    mod.reset_queue_manager()
    try:
        n, ids, errors = _race_constructions(
            mod.get_queue_manager,
            (mod, mod.InMemoryQueueManager, "InMemoryQueueManager"),
            monkeypatch,
        )
        assert errors == []
        assert n == 1, f"queue manager constructed {n} times under concurrent race"
        assert len(ids) == 1, "threads received different queue-manager instances"
    finally:
        mod.reset_queue_manager()


@pytest.mark.unit
def test_get_registry_builds_once_under_race(monkeypatch):
    import cogniverse_core.registries.registry as mod

    mod._registry = None
    mod.StrategyRegistry._instance = None
    try:
        monkeypatch.setattr(mod, "StrategyRegistry", lambda: object())
        n, ids, errors = _race_constructions(
            mod.get_registry,
            (mod, mod.StrategyRegistry, "StrategyRegistry"),
            monkeypatch,
        )
        assert errors == []
        assert n == 1, f"registry constructed {n} times under concurrent race"
        assert len(ids) == 1, "threads received different registry instances"
    finally:
        mod._registry = None


@pytest.mark.unit
def test_get_registry_recovers_after_constructor_failure(monkeypatch):
    import cogniverse_core.registries.registry as mod

    attempts = 0
    ready = object()

    def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("registry construction failed")
        return ready

    mod._registry = None
    monkeypatch.setattr(mod, "StrategyRegistry", fail_once)
    try:
        with pytest.raises(RuntimeError, match="registry construction failed"):
            mod.get_registry()
        assert mod._registry is None

        assert mod.get_registry() is ready
        assert mod.get_registry() is ready
        assert attempts == 2
    finally:
        mod._registry = None
