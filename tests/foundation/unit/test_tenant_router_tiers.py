"""The per-tenant router tier: storage, cache, invalidation, fault contract.

The tier is a tenant attribute in the configuration store, not a static map.
These pin what the store holds, what the cached reader answers, what a write
does to readers already holding the tenant, and what a store outage resolves
to on the request path.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.tenant_tiers import (
    TENANT_TIER_KEY,
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


class _CountingStore(InMemoryConfigStore):
    """Counts tier reads; optionally fails them or blocks them on an event."""

    def __init__(self):
        super().__init__()
        self.tier_reads = 0
        self.fail_with: Exception | None = None
        self.gate: threading.Event | None = None

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        if service == TENANT_TIER_SERVICE and config_key == TENANT_TIER_KEY:
            self.tier_reads += 1
            if self.gate is not None:
                self.gate.wait(timeout=10)
            if self.fail_with is not None:
                raise self.fail_with
        return super().get_config(tenant_id, scope, service, config_key, version)


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
        reader = TenantRouterTiers(config_manager, ttl_s=60)
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.tier_reads = 0
        assert [reader("acme:prod") for _ in range(10)] == ["pro"] * 10
        assert store.tier_reads == 1

    def test_the_entry_expires_after_its_ttl(self, config_manager, store):
        reader = TenantRouterTiers(config_manager, ttl_s=0.05)
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        assert store.tier_reads == 1
        time.sleep(0.06)
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        assert store.tier_reads == 2

    def test_a_write_here_is_seen_by_a_reader_already_holding_the_tenant(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(config_manager, ttl_s=3600)
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.tier_reads = 0
        assert reader("acme:prod") == "pro"
        assert store.tier_reads == 1

    def test_a_write_for_another_tenant_keeps_this_tenants_entry(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(config_manager, ttl_s=3600)
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        set_tenant_tier(config_manager, "globex:prod", "pro")
        store.tier_reads = 0
        assert reader("acme:prod") == DEFAULT_ROUTER_TIER
        assert store.tier_reads == 0

    def test_the_reader_is_built_once_per_config_manager(self, config_manager):
        assert tenant_tier_reader(config_manager) is tenant_tier_reader(config_manager)
        other = ConfigManager(store=InMemoryConfigStore())
        assert tenant_tier_reader(other) is not tenant_tier_reader(config_manager)

    def test_ttl_must_be_positive(self, config_manager):
        with pytest.raises(ValueError) as exc:
            TenantRouterTiers(config_manager, ttl_s=0)
        assert str(exc.value) == "ttl_s must be positive, got 0"


class TestConcurrency:
    def test_eight_concurrent_first_touches_share_one_store_read(
        self, config_manager, store
    ):
        set_tenant_tier(config_manager, "acme:prod", "pro")
        store.tier_reads = 0
        reader = TenantRouterTiers(config_manager, ttl_s=3600)
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
        reader = TenantRouterTiers(config_manager, ttl_s=3600)
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


class TestFaultContract:
    def test_a_store_outage_raises_from_the_reader_and_caches_nothing(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(config_manager, ttl_s=3600)
        store.fail_with = RuntimeError("vespa down")
        with pytest.raises(RuntimeError) as exc:
            reader("acme:prod")
        assert str(exc.value) == "vespa down"
        assert reader._entries == {}

        store.fail_with = None
        set_tenant_tier(config_manager, "acme:prod", "pro")
        assert reader("acme:prod") == "pro"

    def test_eight_concurrent_askers_all_receive_the_failure(
        self, config_manager, store
    ):
        reader = TenantRouterTiers(config_manager, ttl_s=3600)
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
