"""The per-tenant router tier: storage, cache, invalidation, fault contract.

The tier is a tenant attribute in the configuration store, not a static map.
These pin what the store holds, what the cached reader answers, what a write
does to readers already holding the tenant, and what a store outage resolves
to on the request path.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.caching import refreshing_cache as refreshing_cache_module
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.tenant_tiers import (
    TENANT_TIER_KEY,
    TENANT_TIER_MAX_STALENESS_S,
    TENANT_TIER_REFRESH_S,
    TENANT_TIER_SERVICE,
    TENANT_TIER_VALUE_FIELD,
    TenantRouterTiers,
    read_tenant_tier,
    resolve_tenant_tier,
    set_tenant_tier,
    tenant_tier_reader,
    validate_router_tier,
)
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
)
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

REFRESH_THREAD = "router-tier-refresh"


class _CountingStore(InMemoryConfigStore):
    """Counts tier reads and the thread each ran on; optionally fails them or
    holds them on an event after the stored row was read."""

    def __init__(self):
        super().__init__()
        self.tier_reads = 0
        self.read_threads: list[str] = []
        self.fail_with: Exception | None = None
        self.gate: threading.Event | None = None
        self.read_started = threading.Event()

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        if service == TENANT_TIER_SERVICE and config_key == TENANT_TIER_KEY:
            self.tier_reads += 1
            self.read_threads.append(threading.current_thread().name)
            entry = super().get_config(tenant_id, scope, service, config_key, version)
            self.read_started.set()
            if self.gate is not None:
                self.gate.wait(timeout=10)
            if self.fail_with is not None:
                raise self.fail_with
            return entry
        return super().get_config(tenant_id, scope, service, config_key, version)


def _store_tier_elsewhere(store: _CountingStore, tenant_id: str, tier: str) -> None:
    """Write the tier the way another replica does: to the store, with no
    invalidation reaching this process's readers."""
    store.set_config(
        tenant_id=tenant_id,
        scope=ConfigScope.ROUTING,
        service=TENANT_TIER_SERVICE,
        config_key=TENANT_TIER_KEY,
        config_value={TENANT_TIER_VALUE_FIELD: tier},
    )


def _join_refreshes() -> None:
    for thread in threading.enumerate():
        if thread.name == REFRESH_THREAD:
            thread.join(timeout=10)
            assert thread.is_alive() is False


@pytest.fixture
def store() -> _CountingStore:
    s = _CountingStore()
    s.initialize()
    return s


@pytest.fixture
def config_manager(store) -> ConfigManager:
    return ConfigManager(store=store)


class TestStoredAttribute:
    def test_a_tenant_with_no_row_reads_as_the_default_tier(self, config_manager):
        assert read_tenant_tier(config_manager, "acme:prod") == DEFAULT_ROUTER_TIER

    def test_set_then_read_returns_exactly_what_was_set(self, config_manager):
        set_tenant_tier(config_manager, "acme:prod", "pro")
        assert read_tenant_tier(config_manager, "acme:prod") == "pro"

    def test_the_stored_row_is_exactly_this_entry(self, config_manager, store):
        set_tenant_tier(config_manager, "acme:prod", "free")
        entry = store.get_config(
            tenant_id="acme:prod",
            scope=ConfigScope.ROUTING,
            service=TENANT_TIER_SERVICE,
            config_key=TENANT_TIER_KEY,
        )
        assert entry.tenant_id == "acme:prod"
        assert entry.scope is ConfigScope.ROUTING
        assert entry.service == TENANT_TIER_SERVICE
        assert entry.config_key == TENANT_TIER_KEY
        assert entry.config_value == {TENANT_TIER_VALUE_FIELD: "free"}

    def test_simple_and_canonical_forms_address_one_row(self, config_manager):
        set_tenant_tier(config_manager, "acme", "pro")
        assert read_tenant_tier(config_manager, "acme:acme") == "pro"
        set_tenant_tier(config_manager, "acme:acme", "free")
        assert read_tenant_tier(config_manager, "acme") == "free"

    def test_every_shipped_tier_round_trips(self, config_manager):
        for tier in sorted(ROUTER_TIERS):
            set_tenant_tier(config_manager, "acme:prod", tier)
            assert read_tenant_tier(config_manager, "acme:prod") == tier

    def test_a_tier_outside_the_vocabulary_is_refused_before_the_write(
        self, config_manager, store
    ):
        with pytest.raises(ValueError) as exc:
            set_tenant_tier(config_manager, "acme:prod", "gold")
        assert str(exc.value) == (
            f"Unknown router tier 'gold'. Valid tiers: {sorted(ROUTER_TIERS)}"
        )
        assert (
            store.get_config(
                tenant_id="acme:prod",
                scope=ConfigScope.ROUTING,
                service=TENANT_TIER_SERVICE,
                config_key=TENANT_TIER_KEY,
            )
            is None
        )

    def test_validate_returns_the_tier_it_accepted(self):
        assert validate_router_tier("pro") == "pro"

    def test_a_stored_tier_outside_the_vocabulary_raises_on_read(
        self, config_manager, store
    ):
        store.set_config(
            tenant_id="acme:prod",
            scope=ConfigScope.ROUTING,
            service=TENANT_TIER_SERVICE,
            config_key=TENANT_TIER_KEY,
            config_value={TENANT_TIER_VALUE_FIELD: "platinum"},
        )
        with pytest.raises(ValueError) as exc:
            read_tenant_tier(config_manager, "acme:prod")
        assert "platinum" in str(exc.value)
        assert str(sorted(ROUTER_TIERS)) in str(exc.value)


class TestCachedReader:
    def test_a_repeat_call_reads_the_store_zero_times(self, config_manager, store):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=60, max_staleness_s=60
        )
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.tier_reads = 0
        assert [reader("acme:prod") for _ in range(10)] == ["pro"] * 10
        assert store.tier_reads == 1

    def test_a_tier_past_its_refresh_age_answers_while_one_background_read_runs(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=0.1, max_staleness_s=30
        )
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        _store_tier_elsewhere(store, "acme:prod", "pro")
        time.sleep(0.15)
        store.gate = threading.Event()
        store.read_started.clear()

        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        # The request returned while the refresh it started was held.
        assert store.read_started.wait(timeout=10)
        assert store.gate.is_set() is False
        caller = threading.current_thread().name
        assert store.read_threads == [caller, REFRESH_THREAD]
        store.gate.set()
        _join_refreshes()

        assert reader("acme:prod") == "pro"
        assert store.read_threads == [caller, REFRESH_THREAD]

    def test_a_tier_at_max_staleness_is_read_on_the_callers_thread(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=0.05, max_staleness_s=0.1
        )
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        _store_tier_elsewhere(store, "acme:prod", "free")
        time.sleep(0.15)

        assert reader("acme:prod") == "free"
        caller = threading.current_thread().name
        assert store.read_threads == [caller, caller]

    def test_a_write_here_is_seen_by_a_reader_already_holding_the_tenant(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=3600, max_staleness_s=3600
        )
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.tier_reads = 0
        assert reader("acme:prod") == "pro"
        assert store.tier_reads == 1

    def test_a_write_during_a_background_refresh_is_never_overwritten(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=0.1, max_staleness_s=3600
        )
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        time.sleep(0.15)
        store.gate = threading.Event()
        store.read_started.clear()
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        assert store.read_started.wait(timeout=10)

        # The refresh has read the pre-write row and is held there.
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.gate.set()
        _join_refreshes()

        assert [reader("acme:prod") for _ in range(3)] == ["pro"] * 3
        assert store.tier_reads == 3

    def test_a_write_for_another_tenant_keeps_this_tenants_entry(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=3600, max_staleness_s=3600
        )
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        set_tenant_tier(config_manager, "globex:prod", "pro")
        store.tier_reads = 0
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        assert store.tier_reads == 0

    def test_the_reader_is_built_once_per_config_manager(self, config_manager):
        assert tenant_tier_reader(config_manager) is tenant_tier_reader(config_manager)
        other = ConfigManager(store=InMemoryConfigStore())
        assert tenant_tier_reader(other) is not tenant_tier_reader(config_manager)

    def test_the_request_path_reader_holds_tiers_for_the_shipped_bounds(
        self, config_manager
    ):
        reader = tenant_tier_reader(config_manager)
        assert (reader.refresh_after_s, reader.max_staleness_s) == (
            TENANT_TIER_REFRESH_S,
            TENANT_TIER_MAX_STALENESS_S,
        )
        assert (TENANT_TIER_REFRESH_S, TENANT_TIER_MAX_STALENESS_S) == (15.0, 30.0)

    @pytest.mark.parametrize(
        ("refresh_after_s", "max_staleness_s", "message"),
        [
            (-1.0, 30.0, "refresh_after_s must be >= 0, got -1.0"),
            (15.0, 5.0, "max_staleness_s (5.0) must be >= refresh_after_s (15.0)"),
        ],
    )
    def test_inconsistent_bounds_raise(
        self, config_manager, refresh_after_s, max_staleness_s, message
    ):
        with pytest.raises(ValueError) as exc:
            TenantRouterTiers(
                config_manager,
                refresh_after_s=refresh_after_s,
                max_staleness_s=max_staleness_s,
            )
        assert str(exc.value) == message


class TestConcurrency:
    def test_eight_concurrent_first_touches_share_one_store_read(
        self, config_manager, store
    ):
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.tier_reads = 0
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=3600, max_staleness_s=3600
        )
        gate = threading.Event()
        store.gate = gate
        barrier = threading.Barrier(8)
        answers: list[str] = []
        answers_lock = threading.Lock()

        def ask():
            barrier.wait(timeout=10)
            tier = reader("acme:prod")
            with answers_lock:
                answers.append(tier)

        threads = [threading.Thread(target=ask) for _ in range(8)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        gate.set()
        for t in threads:
            t.join(timeout=20)

        assert answers == ["pro"] * 8
        assert store.tier_reads == 1

    def test_eight_tenants_each_get_their_own_tier(self, config_manager, store):
        tiers = sorted(ROUTER_TIERS)
        expected = {f"t{i}:prod": tiers[i % len(tiers)] for i in range(8)}
        for tenant, tier in expected.items():
            set_tenant_tier(config_manager, tenant, tier)
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=3600, max_staleness_s=3600
        )
        barrier = threading.Barrier(8)
        seen: dict[str, str] = {}
        seen_lock = threading.Lock()

        def ask(tenant: str):
            barrier.wait(timeout=10)
            tier = reader(tenant)
            with seen_lock:
                seen[tenant] = tier

        threads = [threading.Thread(target=ask, args=(t,)) for t in expected]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert seen == expected

    def test_eight_requests_at_refresh_age_share_one_background_read(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=0.1, max_staleness_s=30
        )
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        _store_tier_elsewhere(store, "acme:prod", "pro")
        time.sleep(0.15)
        store.gate = threading.Event()
        store.read_threads.clear()
        barrier = threading.Barrier(8)

        def ask(_: int) -> str:
            barrier.wait(timeout=10)
            return reader("acme:prod")

        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="request") as pool:
            answers = list(pool.map(ask, range(8)))

        # Every request answered while the one refresh was held at the store.
        assert store.gate.is_set() is False
        assert answers == [DEFAULT_ROUTER_TIER] * 8
        store.gate.set()
        _join_refreshes()
        assert store.read_threads == [REFRESH_THREAD]
        assert reader("acme:prod") == "pro"
        assert store.read_threads == [REFRESH_THREAD]


class TestFaultContract:
    def test_a_store_outage_raises_from_the_reader_and_caches_nothing(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=3600, max_staleness_s=3600
        )
        store.fail_with = RuntimeError("vespa down")
        with pytest.raises(RuntimeError) as exc:
            reader("acme:prod")
        assert str(exc.value) == "vespa down"
        assert reader._tiers.keys() == []

        # Written with no invalidation here: only a reader that cached nothing
        # for the failed read answers it.
        store.fail_with = None
        _store_tier_elsewhere(store, "acme:prod", "pro")
        assert reader("acme:prod") == "pro"

    def test_eight_concurrent_askers_all_receive_the_failure(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=3600, max_staleness_s=3600
        )
        store.fail_with = RuntimeError("vespa down")
        gate = threading.Event()
        store.gate = gate
        barrier = threading.Barrier(8)
        failures: list[str] = []
        failures_lock = threading.Lock()

        def ask():
            barrier.wait(timeout=10)
            try:
                reader("acme:prod")
            except RuntimeError as exc:
                with failures_lock:
                    failures.append(str(exc))

        threads = [threading.Thread(target=ask) for _ in range(8)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        gate.set()
        for t in threads:
            t.join(timeout=20)

        assert failures == ["vespa down"] * 8
        assert store.tier_reads == 1

    def test_a_failed_refresh_serves_the_held_tier_until_the_bound_then_raises(
        self, config_manager, store, caplog
    ):
        reader = TenantRouterTiers(
            config_manager, refresh_after_s=0.1, max_staleness_s=1.0
        )
        set_tenant_tier(config_manager, "acme:prod", "pro")
        read_at = time.monotonic()
        assert reader("acme:prod") == "pro"
        store.fail_with = RuntimeError("vespa down")
        time.sleep(0.15)

        with caplog.at_level(logging.ERROR, logger=refreshing_cache_module.__name__):
            assert reader("acme:prod") == "pro"
            _join_refreshes()
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == refreshing_cache_module.__name__
        ]
        assert len(messages) == 1
        assert re.fullmatch(
            r"router-tier: refreshing 'acme:prod' failed with RuntimeError: "
            r"vespa down; serving the value read 0\.\d+s ago until it is 1\.0s old",
            messages[0],
        )
        assert reader("acme:prod") == "pro"

        time.sleep(max(0.0, read_at + 1.05 - time.monotonic()))
        store.read_threads.clear()
        with pytest.raises(RuntimeError) as exc:
            reader("acme:prod")
        assert str(exc.value) == "vespa down"
        assert store.read_threads == [threading.current_thread().name]

    def test_the_request_path_degrades_to_the_default_tier_and_warns(
        self, config_manager, store, caplog
    ):
        class _Accessor:
            config_manager = None

        accessor = _Accessor()
        accessor.config_manager = config_manager
        store.fail_with = RuntimeError("vespa down")
        with caplog.at_level(logging.WARNING):
            assert resolve_tenant_tier(accessor, "acme:prod") == DEFAULT_ROUTER_TIER
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].getMessage() == (
            "Router tier read failed for tenant acme:prod (RuntimeError: "
            f"vespa down); routing as {DEFAULT_ROUTER_TIER}"
        )

    def test_an_accessor_without_a_config_manager_is_a_wiring_fault(self):
        class _NoManager:
            pass

        with pytest.raises(TypeError) as exc:
            resolve_tenant_tier(_NoManager(), "acme:prod")
        assert str(exc.value) == (
            "resolve_tenant_tier needs a config accessor exposing "
            "config_manager; got _NoManager"
        )
