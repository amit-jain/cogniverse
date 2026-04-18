"""Unit tests for TenantLRUCache."""

import threading

import pytest

from cogniverse_foundation.caching import TenantLRUCache


def test_rejects_zero_capacity():
    with pytest.raises(ValueError):
        TenantLRUCache(capacity=0)


def test_get_or_set_calls_factory_once_per_key():
    cache: TenantLRUCache[str] = TenantLRUCache(capacity=4)
    calls = {"n": 0}

    def factory() -> str:
        calls["n"] += 1
        return "tenant-value"

    first = cache.get_or_set("acme", factory)
    second = cache.get_or_set("acme", factory)

    assert first == "tenant-value"
    assert second is first
    assert calls["n"] == 1


def test_eviction_drops_least_recently_used():
    cache: TenantLRUCache[int] = TenantLRUCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # promote "a"
    cache.set("c", 3)

    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache  # b was LRU
    assert len(cache) == 2


def test_get_promotes_to_most_recently_used():
    cache: TenantLRUCache[int] = TenantLRUCache(capacity=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # now a is MRU, b is LRU
    cache.set("c", 3)

    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache


def test_on_evict_called_with_key_and_value():
    evicted: list[tuple[str, int]] = []
    cache: TenantLRUCache[int] = TenantLRUCache(
        capacity=1, on_evict=lambda k, v: evicted.append((k, v))
    )
    cache.set("a", 1)
    cache.set("b", 2)
    assert evicted == [("a", 1)]


def test_on_evict_exception_does_not_block_eviction():
    def boom(key: str, value: int) -> None:
        raise RuntimeError("cleanup failed")

    cache: TenantLRUCache[int] = TenantLRUCache(capacity=1, on_evict=boom)
    cache.set("a", 1)
    cache.set("b", 2)  # must not raise
    assert "a" not in cache
    assert "b" in cache


def test_clear_invokes_on_evict_for_every_entry():
    evicted: list[str] = []
    cache: TenantLRUCache[int] = TenantLRUCache(
        capacity=4, on_evict=lambda k, v: evicted.append(k)
    )
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert sorted(evicted) == ["a", "b"]
    assert len(cache) == 0


def test_pop_removes_without_eviction_callback():
    evicted: list[str] = []
    cache: TenantLRUCache[int] = TenantLRUCache(
        capacity=4, on_evict=lambda k, v: evicted.append(k)
    )
    cache.set("a", 1)
    assert cache.pop("a") == 1
    assert cache.pop("a") is None
    assert cache.pop("missing", -1) == -1  # dict.pop-compatible default
    assert evicted == []  # pop is explicit removal, not eviction


def test_concurrent_get_or_set_builds_factory_once():
    """30 threads racing on the same key must all get the same instance."""
    cache: TenantLRUCache[object] = TenantLRUCache(capacity=4)
    calls = {"n": 0}
    lock = threading.Lock()

    def factory() -> object:
        with lock:
            calls["n"] += 1
        return object()

    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        value = cache.get_or_set("shared", factory)
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
    assert len({id(v) for v in results}) == 1


def test_copy_preserves_data_and_order_without_sharing_state():
    cache: TenantLRUCache[int] = TenantLRUCache(capacity=4)
    cache.set("a", 1)
    cache.set("b", 2)
    snapshot = cache.copy()

    assert isinstance(snapshot, TenantLRUCache)
    assert snapshot.keys() == cache.keys()
    assert snapshot.get("a") == 1
    assert snapshot.get("b") == 2

    # Mutating the snapshot must not touch the original.
    snapshot.set("c", 3)
    assert "c" not in cache
    assert "c" in snapshot


def test_capacity_holds_under_burst_of_unique_tenants():
    """The raison d'être test: 100 unique tenants hitting a cap-of-8 cache
    leaves exactly 8 entries."""
    evicted: list[str] = []
    cache: TenantLRUCache[int] = TenantLRUCache(
        capacity=8, on_evict=lambda k, v: evicted.append(k)
    )
    for i in range(100):
        cache.set(f"tenant-{i}", i)

    assert len(cache) == 8
    assert len(evicted) == 92
    # Final entries should be the 8 most-recently-written
    assert set(cache.keys()) == {f"tenant-{i}" for i in range(92, 100)}
