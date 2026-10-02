"""ConfigManager scoped configs over a real Vespa config store.

Each worker process holds its own ConfigManager. These tests stand two managers
on one Vespa to pin what a worker serves after another one writes: the request
thread never runs the refresh read, every tenant gets exactly one refresh, a
store outage serves the last value read until the staleness bound and then
raises, and a profile read-modify-write never writes back over another
manager's change. The router tier and the system config are held the same
way.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from cogniverse_foundation.caching import refreshing_cache as refreshing_cache_module
from cogniverse_foundation.config import tenant_tiers as tenant_tiers_module
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.tenant_tiers import (
    TENANT_TIER_KEY,
    TENANT_TIER_SERVICE,
    TENANT_TIER_VALUE_FIELD,
    TenantRouterTiers,
    resolve_tenant_tier,
)
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    BackendProfileConfig,
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_sdk.interfaces.config_store import (
    ConfigScope,
    ConfigStoreUnavailableError,
)
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = [pytest.mark.integration, pytest.mark.requires_vespa]

REFRESH_THREAD = "scoped-config-refresh"
TIER_REFRESH_THREAD = "router-tier-refresh"
SYSTEM_REFRESH_THREAD = "system-config-refresh"


class _RecordingStore(VespaConfigStore):
    """A real Vespa config store that records which thread read which tenant,
    and can hold a read after Vespa has answered it."""

    def __init__(self, port: int) -> None:
        super().__init__(backend_url="http://localhost", backend_port=port)
        self.reads: list[tuple[str, str]] = []
        self.answered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self._reads_lock = threading.Lock()

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        entry = super().get_config(tenant_id, scope, service, config_key, version)
        with self._reads_lock:
            self.reads.append((tenant_id, threading.current_thread().name))
        self.answered.set()
        if not self.gate.wait(timeout=30):
            raise TimeoutError("the test never opened the read gate")
        return entry


class _SwitchableProxy:
    """Forwards reads to the real Vespa, or answers 503 while ``down``."""

    def __init__(self, upstream: str) -> None:
        self.down = False
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if proxy.down:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"config store unavailable")
                    return
                response = requests.get(upstream + self.path, timeout=30)
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.content)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_SwitchableProxy":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _tenant(label: str) -> str:
    name = f"refresh{label}{uuid.uuid4().hex[:8]}"
    return f"{name}:{name}"


def _routing(tenant: str, mode: str) -> RoutingConfigUnified:
    return RoutingConfigUnified(tenant_id=tenant, routing_mode=mode)


def _join_refreshes() -> None:
    for thread in threading.enumerate():
        if thread.name in (REFRESH_THREAD, TIER_REFRESH_THREAD, SYSTEM_REFRESH_THREAD):
            thread.join(timeout=30)
            assert thread.is_alive() is False


def _delete_routing(store: VespaConfigStore, *tenants: str) -> None:
    for tenant in tenants:
        store.delete_config(
            tenant_id=tenant,
            scope=ConfigScope.ROUTING,
            service="gateway_agent",
            config_key="routing_config",
        )


@pytest.fixture
def writer(vespa_instance):
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    yield ConfigManager(store=store)
    store.close()


def test_another_managers_write_is_served_after_one_off_thread_refresh(
    vespa_instance, writer
):
    tenant = _tenant("write")
    store = _RecordingStore(vespa_instance["http_port"])
    reader = ConfigManager(
        store=store, scoped_config_refresh_s=1.0, scoped_config_max_staleness_s=30.0
    )
    try:
        writer.set_routing_config(_routing(tenant, "tiered"))
        assert reader.get_routing_config(tenant).routing_mode == "tiered"
        writer.set_routing_config(_routing(tenant, "ensemble"))
        assert reader.get_routing_config(tenant).routing_mode == "tiered"
        caller = threading.current_thread().name
        assert store.reads == [(tenant, caller)]

        time.sleep(1.05)
        store.answered.clear()
        store.gate.clear()
        assert reader.get_routing_config(tenant).routing_mode == "tiered"
        # The request returned before the refresh it started was let go.
        assert store.answered.wait(timeout=30)
        assert store.reads == [(tenant, caller), (tenant, REFRESH_THREAD)]
        store.gate.set()
        _join_refreshes()

        assert reader.get_routing_config(tenant).routing_mode == "ensemble"
        assert len(store.reads) == 2
    finally:
        store.gate.set()
        _delete_routing(writer.store, tenant)
        store.close()


def test_concurrent_requests_at_refresh_share_one_vespa_read_per_tenant(
    vespa_instance, writer
):
    acme, globex = _tenant("acme"), _tenant("globex")
    store = _RecordingStore(vespa_instance["http_port"])
    reader = ConfigManager(
        store=store, scoped_config_refresh_s=1.0, scoped_config_max_staleness_s=30.0
    )
    try:
        writer.set_routing_config(_routing(acme, "tiered"))
        writer.set_routing_config(_routing(globex, "direct"))
        assert reader.get_routing_config(acme).routing_mode == "tiered"
        assert reader.get_routing_config(globex).routing_mode == "direct"
        writer.set_routing_config(_routing(acme, "ensemble"))
        writer.set_routing_config(_routing(globex, "hybrid"))
        time.sleep(1.05)
        store.gate.clear()
        store.reads.clear()
        tenants = [acme, globex] * 8
        ready = threading.Barrier(len(tenants))

        def request(tenant: str) -> tuple[str, str]:
            ready.wait(timeout=30)
            return tenant, reader.get_routing_config(tenant).routing_mode

        with ThreadPoolExecutor(
            max_workers=len(tenants), thread_name_prefix="request"
        ) as pool:
            answers = list(pool.map(request, tenants))

        assert Counter(answers) == Counter({(acme, "tiered"): 8, (globex, "direct"): 8})
        store.gate.set()
        _join_refreshes()
        assert sorted(store.reads) == sorted(
            [(acme, REFRESH_THREAD), (globex, REFRESH_THREAD)]
        )
        assert reader.get_routing_config(acme).routing_mode == "ensemble"
        assert reader.get_routing_config(globex).routing_mode == "hybrid"
        assert len(store.reads) == 2
    finally:
        store.gate.set()
        _delete_routing(writer.store, acme, globex)
        store.close()


def test_outage_serves_the_last_value_read_until_the_bound_then_raises(
    vespa_instance, writer, caplog
):
    tenant = _tenant("outage")
    upstream = f"http://localhost:{vespa_instance['http_port']}"
    with _SwitchableProxy(upstream) as proxy:
        store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=proxy.server.server_port
        )
        reader = ConfigManager(
            store=store, scoped_config_refresh_s=1.0, scoped_config_max_staleness_s=9.0
        )
        try:
            writer.set_routing_config(_routing(tenant, "tiered"))
            read_at = time.monotonic()
            assert reader.get_routing_config(tenant).routing_mode == "tiered"
            proxy.down = True
            writer.set_routing_config(_routing(tenant, "ensemble"))
            time.sleep(1.05)

            with caplog.at_level(
                logging.ERROR, logger=refreshing_cache_module.__name__
            ):
                assert reader.get_routing_config(tenant).routing_mode == "tiered"
                _join_refreshes()
            messages = [
                record.getMessage()
                for record in caplog.records
                if record.name == refreshing_cache_module.__name__
            ]
            assert len(messages) == 1
            assert re.fullmatch(
                r"scoped-config: refreshing \(<ConfigScope\.ROUTING: 'routing'>, "
                rf"'{tenant}', 'gateway_agent', 'routing_config'\) failed with "
                r"ConfigStoreUnavailableError: Failed to read Vespa config visit "
                r"after 5 attempts over \d+\.\d{3}s: HTTPError: 503 Server Error: "
                r"Service Unavailable for url: http://127\.0\.0\.1:\d+/document/v1/"
                r"\S+; serving the value read \d+\.\ds ago until it is 9\.0s old",
                messages[0],
            )
            assert reader.get_routing_config(tenant).routing_mode == "tiered"

            time.sleep(max(0.0, read_at + 9.05 - time.monotonic()))
            with pytest.raises(ConfigStoreUnavailableError) as caught:
                reader.get_routing_config(tenant)
            assert str(caught.value).startswith(
                "Failed to read Vespa config visit after 5 attempts over "
            )
            assert "HTTPError: 503 Server Error: Service Unavailable" in str(
                caught.value
            )

            proxy.down = False
            assert reader.get_routing_config(tenant).routing_mode == "ensemble"
        finally:
            _delete_routing(writer.store, tenant)
            store.close()


def _profile(name: str) -> BackendProfileConfig:
    return BackendProfileConfig.from_dict(
        name, {"type": "document", "schema_name": f"{name}_schema"}
    )


def test_a_held_backend_config_never_drops_another_managers_profile(
    vespa_instance, writer
):
    tenant = _tenant("profiles")
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    worker_b = ConfigManager(store=store)
    try:
        assert worker_b.list_backend_profiles(tenant) == {}
        writer.add_backend_profile(_profile("written_by_a"), tenant_id=tenant)
        worker_b.add_backend_profile(_profile("written_by_b"), tenant_id=tenant)

        fresh = ConfigManager(store=writer.store)
        assert sorted(fresh.list_backend_profiles(tenant)) == [
            "written_by_a",
            "written_by_b",
        ]
        assert sorted(worker_b.list_backend_profiles(tenant)) == [
            "written_by_a",
            "written_by_b",
        ]
    finally:
        writer.store.delete_config(
            tenant_id=tenant,
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="backend_config",
        )
        store.close()


def _store_tier(store: VespaConfigStore, tenant: str, tier: str) -> None:
    """Write the tier as another replica does: no reader here is invalidated."""
    store.set_config(
        tenant_id=tenant,
        scope=ConfigScope.ROUTING,
        service=TENANT_TIER_SERVICE,
        config_key=TENANT_TIER_KEY,
        config_value={TENANT_TIER_VALUE_FIELD: tier},
    )


def _delete_tier(store: VespaConfigStore, tenant: str) -> None:
    store.delete_config(
        tenant_id=tenant,
        scope=ConfigScope.ROUTING,
        service=TENANT_TIER_SERVICE,
        config_key=TENANT_TIER_KEY,
    )


class _Accessor:
    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager


def test_another_replicas_tier_write_is_served_after_one_off_thread_refresh(
    vespa_instance, writer
):
    tenant = _tenant("tier")
    store = _RecordingStore(vespa_instance["http_port"])
    reader = TenantRouterTiers(
        ConfigManager(store=store), refresh_after_s=1.0, max_staleness_s=30.0
    )
    try:
        _store_tier(writer.store, tenant, "free")
        assert reader(tenant) == "free"
        _store_tier(writer.store, tenant, "pro")
        assert reader(tenant) == "free"
        caller = threading.current_thread().name
        assert store.reads == [(tenant, caller)]

        time.sleep(1.05)
        store.answered.clear()
        store.gate.clear()
        assert reader(tenant) == "free"
        # The request returned before the refresh it started was let go.
        assert store.answered.wait(timeout=30)
        assert store.reads == [(tenant, caller), (tenant, TIER_REFRESH_THREAD)]
        store.gate.set()
        _join_refreshes()

        assert reader(tenant) == "pro"
        assert len(store.reads) == 2
    finally:
        store.gate.set()
        _delete_tier(writer.store, tenant)
        store.close()


def test_a_tier_outage_serves_the_held_tier_until_the_bound_then_routes_default(
    vespa_instance, writer, caplog
):
    tenant = _tenant("tieroutage")
    upstream = f"http://localhost:{vespa_instance['http_port']}"
    with _SwitchableProxy(upstream) as proxy:
        store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=proxy.server.server_port
        )
        manager = ConfigManager(store=store)
        reader = TenantRouterTiers(manager, refresh_after_s=1.0, max_staleness_s=9.0)
        try:
            _store_tier(writer.store, tenant, "pro")
            read_at = time.monotonic()
            assert reader(tenant) == "pro"
            proxy.down = True
            time.sleep(1.05)

            with caplog.at_level(
                logging.ERROR, logger=refreshing_cache_module.__name__
            ):
                assert reader(tenant) == "pro"
                _join_refreshes()
            messages = [
                record.getMessage()
                for record in caplog.records
                if record.name == refreshing_cache_module.__name__
            ]
            assert len(messages) == 1
            assert re.fullmatch(
                rf"router-tier: refreshing '{tenant}' failed with "
                r"ConfigStoreUnavailableError: Failed to read Vespa config visit "
                r"after 5 attempts over \d+\.\d{3}s: HTTPError: 503 Server Error: "
                r"Service Unavailable for url: http://127\.0\.0\.1:\d+/document/v1/"
                r"\S+; serving the value read \d+\.\ds ago until it is 9\.0s old",
                messages[0],
            )
            assert reader(tenant) == "pro"

            time.sleep(max(0.0, read_at + 9.05 - time.monotonic()))
            with pytest.raises(ConfigStoreUnavailableError) as caught:
                reader(tenant)
            assert str(caught.value).startswith(
                "Failed to read Vespa config visit after 5 attempts over "
            )

            # The request path's own reader for this manager holds nothing, so
            # it reads on the caller's thread and routes the outage as default.
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=tenant_tiers_module.__name__):
                assert resolve_tenant_tier(_Accessor(manager), tenant) == (
                    DEFAULT_ROUTER_TIER
                )
            warnings = [
                record.getMessage()
                for record in caplog.records
                if record.name == tenant_tiers_module.__name__
            ]
            assert len(warnings) == 1
            assert warnings[0].startswith(
                f"Router tier read failed for tenant {tenant} "
                "(ConfigStoreUnavailableError: Failed to read Vespa config visit "
            )
            assert warnings[0].endswith(f"routing as {DEFAULT_ROUTER_TIER}")

            proxy.down = False
            assert reader(tenant) == "pro"
        finally:
            _delete_tier(writer.store, tenant)
            store.close()


_SYSTEM_ROW = {
    "tenant_id": "_system",
    "scope": ConfigScope.SYSTEM,
    "service": "system",
    "config_key": "system_config",
}


@pytest.fixture
def system_config_row(writer):
    """Put the shared Vespa's system config row back as the test found it."""
    before = writer.store.get_config(**_SYSTEM_ROW)
    yield
    if before is None:
        writer.store.delete_config(**_SYSTEM_ROW)
    else:
        writer.store.set_config(**_SYSTEM_ROW, config_value=before.config_value)


def test_another_managers_system_config_write_is_served_after_one_off_thread_refresh(
    vespa_instance, writer, system_config_row
):
    store = _RecordingStore(vespa_instance["http_port"])
    reader = ConfigManager(
        store=store, system_config_refresh_s=1.0, system_config_max_staleness_s=30.0
    )
    try:
        writer.set_system_config(SystemConfig(llm_model="openai/first-model"))
        assert reader.get_system_config().llm_model == "openai/first-model"
        writer.set_system_config(SystemConfig(llm_model="openai/second-model"))
        assert reader.get_system_config().llm_model == "openai/first-model"
        caller = threading.current_thread().name
        assert store.reads == [("_system", caller)]

        time.sleep(1.05)
        store.answered.clear()
        store.gate.clear()
        assert reader.get_system_config().llm_model == "openai/first-model"
        # The call returned before the refresh it started was let go.
        assert store.answered.wait(timeout=30)
        assert store.reads == [("_system", caller), ("_system", SYSTEM_REFRESH_THREAD)]
        store.gate.set()
        _join_refreshes()

        assert reader.get_system_config().llm_model == "openai/second-model"
        assert len(store.reads) == 2
    finally:
        store.gate.set()
        store.close()


def test_a_system_config_outage_serves_the_held_value_until_the_bound_then_raises(
    vespa_instance, writer, system_config_row, caplog
):
    upstream = f"http://localhost:{vespa_instance['http_port']}"
    with _SwitchableProxy(upstream) as proxy:
        store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=proxy.server.server_port
        )
        reader = ConfigManager(
            store=store, system_config_refresh_s=1.0, system_config_max_staleness_s=9.0
        )
        try:
            writer.set_system_config(SystemConfig(llm_model="openai/held-model"))
            read_at = time.monotonic()
            assert reader.get_system_config().llm_model == "openai/held-model"
            proxy.down = True
            writer.set_system_config(SystemConfig(llm_model="openai/written-model"))
            time.sleep(1.05)

            with caplog.at_level(
                logging.ERROR, logger=refreshing_cache_module.__name__
            ):
                assert reader.get_system_config().llm_model == "openai/held-model"
                _join_refreshes()
            messages = [
                record.getMessage()
                for record in caplog.records
                if record.name == refreshing_cache_module.__name__
            ]
            assert len(messages) == 1
            assert re.fullmatch(
                r"system-config: refreshing 'system_config' failed with "
                r"ConfigStoreUnavailableError: Failed to read Vespa config visit "
                r"after 5 attempts over \d+\.\d{3}s: HTTPError: 503 Server Error: "
                r"Service Unavailable for url: http://127\.0\.0\.1:\d+/document/v1/"
                r"\S+; serving the value read \d+\.\ds ago until it is 9\.0s old",
                messages[0],
            )
            assert reader.get_system_config().llm_model == "openai/held-model"

            time.sleep(max(0.0, read_at + 9.05 - time.monotonic()))
            with pytest.raises(ConfigStoreUnavailableError) as caught:
                reader.get_system_config()
            assert str(caught.value).startswith(
                "Failed to read Vespa config visit after 5 attempts over "
            )
            assert "HTTPError: 503 Server Error: Service Unavailable" in str(
                caught.value
            )

            proxy.down = False
            assert reader.get_system_config().llm_model == "openai/written-model"
        finally:
            store.close()
