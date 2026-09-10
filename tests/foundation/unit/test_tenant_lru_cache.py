"""Unit tests for TenantLRUCache."""

import logging
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


def test_set_displacing_same_key_fires_on_evict():
    """Overwriting a key must release the displaced value — cached backend
    instances hold connection pools, and a silent overwrite leaks them."""
    evicted: list[tuple[str, object]] = []
    cache: TenantLRUCache[object] = TenantLRUCache(
        capacity=4, on_evict=lambda k, v: evicted.append((k, v))
    )
    first, second = object(), object()

    cache.set("k", first)
    cache.set("k", second)

    assert cache.get("k") is second
    assert evicted == [("k", first)]


def test_set_same_value_does_not_fire_on_evict():
    evicted: list[str] = []
    cache: TenantLRUCache[object] = TenantLRUCache(
        capacity=4, on_evict=lambda k, v: evicted.append(k)
    )
    value = object()

    cache.set("k", value)
    cache.set("k", value)

    assert evicted == []


def test_set_if_absent_inserts_and_returns_value():
    cache: TenantLRUCache[object] = TenantLRUCache(capacity=4)
    value = object()

    assert cache.set_if_absent("k", value) is value
    assert cache.get("k") is value


def test_set_if_absent_returns_existing_winner_and_fires_no_evict():
    """Two concurrent builders resolve to ONE cached instance: the loser gets
    the winner back and stays responsible for releasing its own duplicate —
    the cache must not evict either object."""
    evicted: list[str] = []
    cache: TenantLRUCache[object] = TenantLRUCache(
        capacity=4, on_evict=lambda k, v: evicted.append(k)
    )
    winner, loser = object(), object()

    assert cache.set_if_absent("k", winner) is winner
    assert cache.set_if_absent("k", loser) is winner
    assert cache.get("k") is winner
    assert evicted == []


def test_set_if_absent_evicts_over_capacity():
    evicted: list[str] = []
    cache: TenantLRUCache[int] = TenantLRUCache(
        capacity=1, on_evict=lambda k, v: evicted.append(k)
    )

    cache.set_if_absent("a", 1)
    cache.set_if_absent("b", 2)

    assert evicted == ["a"]
    assert "b" in cache and "a" not in cache


class TestTenantCacheRegistry:
    """register_tenant_cache + evict_tenant_from_registered_caches let tenant
    deletion drop a deleted tenant's entries from every per-tenant cache in
    the process, instead of each cache growing until its LRU bound."""

    def test_evict_drops_only_the_deleted_tenant_from_registered_caches(self):
        from cogniverse_foundation.caching import (
            evict_tenant_from_registered_caches,
            register_tenant_cache,
        )

        cache_a: TenantLRUCache[int] = register_tenant_cache(TenantLRUCache(capacity=4))
        cache_b: TenantLRUCache[int] = register_tenant_cache(TenantLRUCache(capacity=4))
        cache_a.set("lrureg1:one", 1)
        cache_a.set("lrureg1:two", 2)
        cache_b.set("lrureg1:one", 3)

        evicted = evict_tenant_from_registered_caches("lrureg1:one")

        assert evicted == 2
        assert "lrureg1:one" not in cache_a
        assert "lrureg1:one" not in cache_b
        assert cache_a.get("lrureg1:two") == 2

    def test_evict_canonicalizes_the_tenant_id(self):
        from cogniverse_foundation.caching import (
            evict_tenant_from_registered_caches,
            register_tenant_cache,
        )

        cache: TenantLRUCache[int] = register_tenant_cache(TenantLRUCache(capacity=4))
        cache.set("lrureg2:lrureg2", 1)

        assert evict_tenant_from_registered_caches("lrureg2") == 1
        assert "lrureg2:lrureg2" not in cache

    def test_unregistered_cache_is_untouched(self):
        from cogniverse_foundation.caching import evict_tenant_from_registered_caches

        cache: TenantLRUCache[int] = TenantLRUCache(capacity=4)
        cache.set("lrureg3:one", 1)

        evict_tenant_from_registered_caches("lrureg3:one")

        assert cache.get("lrureg3:one") == 1

    def test_registry_does_not_keep_caches_alive(self):
        import gc
        import weakref

        from cogniverse_foundation.caching import (
            evict_tenant_from_registered_caches,
            register_tenant_cache,
        )

        cache: TenantLRUCache[int] = register_tenant_cache(TenantLRUCache(capacity=4))
        cache.set("lrureg4:one", 1)
        ref = weakref.ref(cache)
        del cache
        gc.collect()

        assert ref() is None
        assert evict_tenant_from_registered_caches("lrureg4:one") == 0


class TestCheckoutLease:
    """A checked-out entry is never closed under it.

    Backend instances are handed out of this cache and then used for the
    length of a request; capacity eviction closing one mid-use kills the
    request that is holding it.
    """

    def test_capacity_eviction_skips_the_checked_out_entry(self):
        closed: list[tuple[str, int]] = []
        cache: TenantLRUCache[int] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: closed.append((k, v))
        )
        cache.set("held", 1)
        cache.set("a", 2)

        assert cache.acquire("held") == 1
        assert cache.lease_count("held") == 1

        cache.set("b", 3)
        cache.set("c", 4)

        assert closed == [("a", 2), ("b", 3)]
        assert cache.keys() == ["held", "c"]

        cache.release("held")

        assert cache.lease_count("held") == 0
        assert closed == [("a", 2), ("b", 3)]
        assert cache.keys() == ["held", "c"]

    def test_release_lets_the_formerly_held_entry_evict_again(self):
        closed: list[tuple[str, int]] = []
        cache: TenantLRUCache[int] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: closed.append((k, v))
        )
        cache.set("held", 1)
        cache.set("a", 2)
        cache.acquire("held")
        cache.set("b", 3)
        cache.release("held")

        cache.set("c", 4)

        assert closed == [("a", 2), ("held", 1)]
        assert cache.keys() == ["b", "c"]

    def test_concurrent_inserts_never_close_the_checked_out_entry(self):
        closed: list[str] = []
        closed_lock = threading.Lock()

        def record(key: str, value: int) -> None:
            with closed_lock:
                closed.append(key)

        cache: TenantLRUCache[int] = TenantLRUCache(capacity=2, on_evict=record)
        cache.set("held", 100)
        assert cache.acquire("held") == 100

        inserters = ["a", "b", "c"]
        start = threading.Barrier(len(inserters) + 1)
        inserted: list[str] = []
        inserted_lock = threading.Lock()

        def insert(key: str) -> None:
            start.wait(timeout=30)
            cache.set(key, ord(key))
            with inserted_lock:
                inserted.append(key)

        threads = [
            threading.Thread(target=insert, args=(key,), name=key) for key in inserters
        ]
        for thread in threads:
            thread.start()
        start.wait(timeout=30)
        for thread in threads:
            thread.join(timeout=30)

        assert [thread.is_alive() for thread in threads] == [False, False, False]
        assert sorted(inserted) == ["a", "b", "c"]
        assert cache.lease_count("held") == 1
        assert "held" not in closed
        assert len(closed) == 2
        assert sorted(closed) == sorted(set(closed))
        assert len(cache) == 2
        assert "held" in cache
        assert set(cache.keys()) | set(closed) == {"held", "a", "b", "c"}
        assert set(cache.keys()) & set(closed) == set()

        cache.release("held")

        assert cache.lease_count("held") == 0
        assert len(cache) == 2
        assert "held" in cache

    def test_all_entries_checked_out_overflows_capacity_and_warns_once(self, caplog):
        closed: list[tuple[str, int]] = []
        cache: TenantLRUCache[int] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: closed.append((k, v))
        )
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.acquire("a") == 1
        assert cache.acquire("b") == 2

        with caplog.at_level(
            logging.WARNING, logger="cogniverse_foundation.caching.tenant_lru"
        ):
            cache.set("c", 3)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert warnings == [
            "TenantLRUCache exceeds its capacity of 2 by 1: "
            "2 checked-out entries cannot be evicted"
        ]
        assert closed == []
        assert len(cache) == 3
        assert cache.keys() == ["a", "b", "c"]

        cache.release("a")

        assert closed == [("a", 1)]
        assert cache.keys() == ["b", "c"]
        assert len(cache) == 2

    def test_lease_releases_on_an_exception_and_the_entry_evicts_after(self):
        closed: list[tuple[str, int]] = []
        cache: TenantLRUCache[int] = TenantLRUCache(
            capacity=1, on_evict=lambda k, v: closed.append((k, v))
        )
        cache.set("held", 1)

        with pytest.raises(ValueError, match="request failed"):
            with cache.lease("held") as value:
                assert value == 1
                assert cache.lease_count("held") == 1
                raise ValueError("request failed")

        assert cache.lease_count("held") == 0
        assert closed == []

        cache.set("next", 2)

        assert closed == [("held", 1)]
        assert cache.keys() == ["next"]

    def test_lease_of_an_uncached_key_yields_none_and_takes_no_lease(self):
        cache: TenantLRUCache[int] = TenantLRUCache(capacity=2)

        with cache.lease("absent") as value:
            assert value is None
            assert cache.lease_count("absent") == 0

        assert cache.lease_count("absent") == 0
        assert cache.keys() == []

    def test_release_without_acquire_is_refused(self):
        cache: TenantLRUCache[int] = TenantLRUCache(capacity=2)
        cache.set("a", 1)

        with pytest.raises(RuntimeError) as excinfo:
            cache.release("a")

        assert str(excinfo.value) == (
            "TenantLRUCache.release('a') has no matching acquire()"
        )

    def test_lease_value_holds_the_entry_that_holds_that_object(self):
        closed: list[str] = []
        cache: TenantLRUCache[object] = TenantLRUCache(
            capacity=1, on_evict=lambda k, v: closed.append(k)
        )
        held = object()
        cache.set("held", held)

        with cache.lease_value(held):
            assert cache.lease_count("held") == 1
            cache.set("other", object())
            assert cache.keys() == ["held", "other"]
            assert closed == []

        assert cache.lease_count("held") == 0
        assert cache.keys() == ["other"]
        assert closed == ["held"]

    def test_lease_value_of_an_uncached_object_leases_nothing(self):
        cache: TenantLRUCache[object] = TenantLRUCache(capacity=2)
        cache.set("a", object())
        stranger = object()

        with cache.lease_value(stranger):
            assert cache.keys() == ["a"]

        assert cache.lease_count("a") == 0

    def test_overwriting_a_checked_out_key_defers_the_close_until_release(self):
        closed: list[tuple[str, int]] = []
        cache: TenantLRUCache[int] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: closed.append((k, v))
        )
        cache.set("a", 1)
        assert cache.acquire("a") == 1

        cache.set("a", 2)

        assert closed == []
        assert cache.get("a") == 2

        cache.release("a")

        assert closed == [("a", 1)]
        assert cache.get("a") == 2

    def test_clear_defers_the_close_of_a_checked_out_entry(self):
        closed: list[tuple[str, int]] = []
        cache: TenantLRUCache[int] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: closed.append((k, v))
        )
        cache.set("held", 1)
        cache.set("free", 2)
        assert cache.acquire("held") == 1

        cache.clear()

        assert closed == [("free", 2)]
        assert cache.keys() == []

        cache.release("held")

        assert closed == [("free", 2), ("held", 1)]


class TestEveryCloseIsAttributable:
    """A close releases native resources out from under whoever holds the
    value, so each one names the key, the reason and the value type. A
    runtime pod served 503 on every tenant-scoped request for 44 seconds
    before the first log line that could have explained it."""

    def _closes(self, caplog):
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
            and record.getMessage().startswith("TenantLRUCache closing ")
        ]

    def test_capacity_eviction_names_its_key_and_reason(self, caplog):
        cache: TenantLRUCache[str] = TenantLRUCache(
            capacity=1, on_evict=lambda k, v: None
        )
        cache.set("first", "alpha")
        with caplog.at_level(logging.WARNING):
            cache.set("second", "beta")

        assert self._closes(caplog) == [
            "TenantLRUCache closing first (capacity eviction): str"
        ]

    def test_an_overwriting_set_names_its_key_and_reason(self, caplog):
        cache: TenantLRUCache[str] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: None
        )
        cache.set("acme", "alpha")
        with caplog.at_level(logging.WARNING):
            cache.set("acme", "beta")

        assert self._closes(caplog) == [
            "TenantLRUCache closing acme (overwritten): str"
        ]

    def test_clear_names_every_key_and_its_reason(self, caplog):
        cache: TenantLRUCache[str] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: None
        )
        cache.set("acme", "alpha")
        cache.set("globex", "beta")
        with caplog.at_level(logging.WARNING):
            cache.clear()

        assert self._closes(caplog) == [
            "TenantLRUCache closing acme (cache cleared): str",
            "TenantLRUCache closing globex (cache cleared): str",
        ]

    def test_a_deferred_close_names_the_release_that_ran_it(self, caplog):
        cache: TenantLRUCache[str] = TenantLRUCache(
            capacity=2, on_evict=lambda k, v: None
        )
        cache.set("acme", "alpha")
        assert cache.acquire("acme") == "alpha"
        with caplog.at_level(logging.WARNING):
            cache.clear()
            assert self._closes(caplog) == []
            cache.release("acme")

        assert self._closes(caplog) == [
            "TenantLRUCache closing acme (deferred close on last release): str"
        ]
