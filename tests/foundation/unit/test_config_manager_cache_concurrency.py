"""ConfigManager cache reads converge and never outlive a completed write."""

from __future__ import annotations

import inspect
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.caching import refreshing_cache as refreshing_cache_module
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _CoordinatedConfigStore(InMemoryConfigStore):
    def __init__(self):
        super().__init__()
        self.get_calls = 0
        self.delay_s = 0.0
        self.block_next_get = False
        self.fail_next_get = False
        self.fail_every_get = False
        self.read_captured = threading.Event()
        self.release_read = threading.Event()
        self.set_completed = threading.Event()
        # Cleared to hold every read after it has captured its value.
        self.gate = threading.Event()
        self.gate.set()
        self.reads: list[tuple[str, str]] = []
        self._count_lock = threading.Lock()

    def get_config(self, *args, **kwargs):
        tenant_id = kwargs.get("tenant_id", args[0] if args else None)
        with self._count_lock:
            self.get_calls += 1
            self.reads.append((tenant_id, threading.current_thread().name))
        if self.fail_next_get or self.fail_every_get:
            self.fail_next_get = False
            raise ConnectionError("configuration store unavailable")

        entry = super().get_config(*args, **kwargs)
        self.read_captured.set()
        if not self.gate.wait(timeout=10):
            raise TimeoutError("test did not open the read gate")
        if self.block_next_get:
            self.block_next_get = False
            self.read_captured.set()
            if not self.release_read.wait(timeout=5):
                raise TimeoutError("test did not release blocked config read")
        if self.delay_s:
            time.sleep(self.delay_s)
        return entry

    def set_config(self, *args, **kwargs):
        entry = super().set_config(*args, **kwargs)
        self.set_completed.set()
        return entry


def _seed_system(store: _CoordinatedConfigStore, model: str) -> None:
    store.set_config(
        tenant_id="_system",
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        config_value=SystemConfig(llm_model=model).to_dict(),
    )
    store.set_completed.clear()


def _seed_routing(
    store: _CoordinatedConfigStore, mode: str, tenant_id: str = "acme:acme"
) -> None:
    store.set_config(
        tenant_id=tenant_id,
        scope=ConfigScope.ROUTING,
        service="gateway_agent",
        config_key="routing_config",
        config_value=RoutingConfigUnified(
            tenant_id=tenant_id,
            routing_mode=mode,
        ).to_dict(),
    )
    store.set_completed.clear()


def _refreshing_manager(
    store, refresh_s: float, max_staleness_s: float = 30.0
) -> ConfigManager:
    return ConfigManager(
        store=store,
        scoped_config_refresh_s=refresh_s,
        scoped_config_max_staleness_s=max_staleness_s,
    )


def _join_scoped_refreshes() -> None:
    for thread in threading.enumerate():
        if thread.name == "scoped-config-refresh":
            thread.join(timeout=10)
            assert thread.is_alive() is False


def test_concurrent_system_cache_miss_reads_store_once():
    store = _CoordinatedConfigStore()
    _seed_system(store, "shared-model")
    store.delay_s = 0.03
    manager = ConfigManager(store=store)
    worker_count = 12
    ready = threading.Barrier(worker_count)

    def read_model():
        ready.wait()
        return manager.get_system_config().llm_model

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        models = list(pool.map(lambda _: read_model(), range(worker_count)))

    assert models == ["shared-model"] * worker_count
    assert store.get_calls == 1


def test_system_cache_cannot_refill_stale_value_after_write():
    store = _CoordinatedConfigStore()
    _seed_system(store, "old-model")
    store.block_next_get = True
    manager = ConfigManager(store=store)
    read_models = []
    errors = []

    reader = threading.Thread(
        target=lambda: read_models.append(manager.get_system_config().llm_model)
    )
    reader.start()
    assert store.read_captured.wait(timeout=5)

    def write_new():
        try:
            manager.set_system_config(SystemConfig(llm_model="new-model"))
        except Exception as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_new)
    writer.start()
    assert store.set_completed.wait(timeout=5)
    store.release_read.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert reader.is_alive() is False
    assert writer.is_alive() is False
    assert errors == []
    assert read_models == ["old-model"]
    assert manager.get_system_config().llm_model == "new-model"
    assert store.get_calls == 1


def test_system_store_failure_is_not_cached():
    store = _CoordinatedConfigStore()
    _seed_system(store, "recovered-model")
    store.fail_next_get = True
    manager = ConfigManager(store=store)

    with pytest.raises(ConnectionError, match="configuration store unavailable"):
        manager.get_system_config()

    assert manager.get_system_config().llm_model == "recovered-model"
    assert store.get_calls == 2


def test_concurrent_scoped_cache_miss_reads_store_once():
    store = _CoordinatedConfigStore()
    _seed_routing(store, "ensemble")
    store.delay_s = 0.03
    manager = ConfigManager(store=store)
    worker_count = 12
    ready = threading.Barrier(worker_count)

    def read_mode():
        ready.wait()
        return manager.get_routing_config("acme").routing_mode

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        modes = list(pool.map(lambda _: read_mode(), range(worker_count)))

    assert modes == ["ensemble"] * worker_count
    assert store.get_calls == 1


def test_scoped_cache_cannot_refill_stale_value_after_write():
    store = _CoordinatedConfigStore()
    _seed_routing(store, "tiered")
    store.block_next_get = True
    manager = ConfigManager(store=store)
    read_modes = []
    errors = []

    reader = threading.Thread(
        target=lambda: read_modes.append(
            manager.get_routing_config("acme").routing_mode
        )
    )
    reader.start()
    assert store.read_captured.wait(timeout=5)

    def write_new():
        try:
            manager.set_routing_config(
                RoutingConfigUnified(
                    tenant_id="acme",
                    routing_mode="ensemble",
                )
            )
        except Exception as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_new)
    writer.start()
    assert store.set_completed.wait(timeout=5)
    store.release_read.set()
    reader.join(timeout=5)
    writer.join(timeout=5)

    assert reader.is_alive() is False
    assert writer.is_alive() is False
    assert errors == []
    assert read_modes == ["tiered"]
    assert manager.get_routing_config("acme").routing_mode == "ensemble"
    assert store.get_calls == 2


def test_scoped_store_failure_is_not_cached():
    store = _CoordinatedConfigStore()
    _seed_routing(store, "direct")
    store.fail_next_get = True
    manager = ConfigManager(store=store)

    with pytest.raises(ConnectionError, match="configuration store unavailable"):
        manager.get_routing_config("acme")

    assert manager.get_routing_config("acme").routing_mode == "direct"
    assert store.get_calls == 2


def _seed_system_urls(store: _CoordinatedConfigStore, urls: dict) -> None:
    store.set_config(
        tenant_id="_system",
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        config_value=SystemConfig(
            llm_model="persisted-model", inference_service_urls=urls
        ).to_dict(),
    )
    store.set_completed.clear()


def test_pinned_inference_urls_are_served_and_never_persisted():
    store = _CoordinatedConfigStore()
    _seed_system_urls(store, {"vllm_colpali": "http://persisted-colpali:8000"})
    manager = ConfigManager(store=store)
    explicit = {"vllm_colpali": "http://127.0.0.1:59601", "gliner": "http://g:8080"}

    manager.pin_inference_service_urls(explicit)
    explicit["gliner"] = "http://mutated-after-pin:1"
    cold = manager.get_system_config()
    cold.inference_service_urls["vllm_colpali"] = "http://mutated-by-reader:1"
    warm = manager.get_system_config()

    expected = {"vllm_colpali": "http://127.0.0.1:59601", "gliner": "http://g:8080"}
    assert warm.inference_service_urls == expected
    assert (warm.llm_model, store.get_calls) == ("persisted-model", 1)
    assert ConfigManager(store=store).get_system_config().inference_service_urls == {
        "vllm_colpali": "http://persisted-colpali:8000"
    }


def test_concurrent_cold_reads_of_a_pinned_manager_all_serve_the_pin():
    store = _CoordinatedConfigStore()
    _seed_system_urls(store, {"denseon": "http://persisted-denseon:8000"})
    store.delay_s = 0.03
    manager = ConfigManager(store=store)
    manager.pin_inference_service_urls({"denseon": "http://explicit-denseon:8000"})
    worker_count = 12
    ready = threading.Barrier(worker_count)

    def read_urls():
        ready.wait()
        return manager.get_system_config().inference_service_urls

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        served = list(pool.map(lambda _: read_urls(), range(worker_count)))

    assert served == [{"denseon": "http://explicit-denseon:8000"}] * worker_count
    assert store.get_calls == 1


def test_a_pinned_manager_raises_when_the_store_cannot_answer():
    store = _CoordinatedConfigStore()
    _seed_system_urls(store, {"denseon": "http://persisted-denseon:8000"})
    store.fail_next_get = True
    manager = ConfigManager(store=store)
    manager.pin_inference_service_urls({"denseon": "http://explicit-denseon:8000"})

    with pytest.raises(ConnectionError, match="configuration store unavailable"):
        manager.get_system_config()

    recovered = manager.get_system_config()
    assert (recovered.llm_model, recovered.inference_service_urls) == (
        "persisted-model",
        {"denseon": "http://explicit-denseon:8000"},
    )


def test_stale_scoped_configs_are_served_while_one_refresh_per_tenant_runs():
    store = _CoordinatedConfigStore()
    _seed_routing(store, "tiered", tenant_id="acme:acme")
    _seed_routing(store, "direct", tenant_id="globex:globex")
    manager = _refreshing_manager(store, refresh_s=1.0)
    assert manager.get_routing_config("acme").routing_mode == "tiered"
    assert manager.get_routing_config("globex").routing_mode == "direct"
    _seed_routing(store, "ensemble", tenant_id="acme:acme")
    _seed_routing(store, "hybrid", tenant_id="globex:globex")
    time.sleep(1.05)
    store.gate.clear()
    store.reads.clear()
    tenants = ["acme", "globex"] * 8
    ready = threading.Barrier(len(tenants))

    def read_mode(tenant: str) -> tuple[str, str, str]:
        ready.wait(timeout=10)
        mode = manager.get_routing_config(tenant).routing_mode
        return tenant, mode, threading.current_thread().name

    with ThreadPoolExecutor(
        max_workers=len(tenants), thread_name_prefix="request"
    ) as pool:
        answers = list(pool.map(read_mode, tenants))

    # Every request returned while both refreshes were held at the store.
    assert store.gate.is_set() is False
    assert Counter((tenant, mode) for tenant, mode, _ in answers) == Counter(
        {("acme", "tiered"): 8, ("globex", "direct"): 8}
    )
    store.gate.set()
    _join_scoped_refreshes()
    assert sorted(store.reads) == [
        ("acme:acme", "scoped-config-refresh"),
        ("globex:globex", "scoped-config-refresh"),
    ]
    assert manager.get_routing_config("acme").routing_mode == "ensemble"
    assert manager.get_routing_config("globex").routing_mode == "hybrid"
    assert store.get_calls == 4


def test_scoped_refresh_failure_serves_last_known_good_then_raises(caplog):
    store = _CoordinatedConfigStore()
    _seed_routing(store, "tiered")
    manager = _refreshing_manager(store, refresh_s=0.2, max_staleness_s=1.5)
    filled_at = time.monotonic()
    assert manager.get_routing_config("acme").routing_mode == "tiered"
    store.fail_every_get = True
    time.sleep(0.25)

    with caplog.at_level(logging.ERROR, logger=refreshing_cache_module.__name__):
        assert manager.get_routing_config("acme").routing_mode == "tiered"
        _join_scoped_refreshes()
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == refreshing_cache_module.__name__
    ]
    assert len(messages) == 1
    assert re.fullmatch(
        r"scoped-config: refreshing \(<ConfigScope\.ROUTING: 'routing'>, "
        r"'acme:acme', 'gateway_agent', 'routing_config'\) failed with "
        r"ConnectionError: configuration store unavailable; serving the value "
        r"read 0\.\d+s ago until it is 1\.5s old",
        messages[0],
    )
    assert manager.get_routing_config("acme").routing_mode == "tiered"

    time.sleep(max(0.0, filled_at + 1.55 - time.monotonic()))
    store.reads.clear()
    for _ in range(2):
        with pytest.raises(ConnectionError) as caught:
            manager.get_routing_config("acme")
        assert str(caught.value) == "configuration store unavailable"
    caller = threading.current_thread().name
    assert store.reads == [("acme:acme", caller), ("acme:acme", caller)]

    store.fail_every_get = False
    _seed_routing(store, "ensemble")
    assert manager.get_routing_config("acme").routing_mode == "ensemble"


def test_scoped_write_during_a_background_refresh_is_never_overwritten():
    store = _CoordinatedConfigStore()
    _seed_routing(store, "tiered")
    manager = _refreshing_manager(store, refresh_s=0.2)
    assert manager.get_routing_config("acme").routing_mode == "tiered"
    time.sleep(0.25)
    store.read_captured.clear()
    store.gate.clear()

    assert manager.get_routing_config("acme").routing_mode == "tiered"
    assert store.read_captured.wait(timeout=5)
    manager.set_routing_config(
        RoutingConfigUnified(tenant_id="acme", routing_mode="ensemble")
    )
    store.gate.set()
    _join_scoped_refreshes()

    assert manager.get_routing_config("acme").routing_mode == "ensemble"
    assert manager.get_routing_config("acme").routing_mode == "ensemble"
    assert store.get_calls == 3


def _profile(name: str) -> BackendProfileConfig:
    return BackendProfileConfig.from_dict(
        name, {"type": "document", "schema_name": f"{name}_schema"}
    )


def test_profile_read_modify_write_never_drops_another_managers_write():
    store = InMemoryConfigStore()
    worker_a = ConfigManager(store=store)
    worker_b = ConfigManager(store=store)
    assert worker_b.list_backend_profiles("acme") == {}

    worker_a.add_backend_profile(_profile("written_by_a"), tenant_id="acme")
    worker_b.add_backend_profile(_profile("written_by_b"), tenant_id="acme")
    worker_a.add_backend_profile(_profile("second_by_a"), tenant_id="acme")
    assert worker_b.delete_backend_profile("written_by_b", tenant_id="acme") is True
    worker_a.add_backend_profile(_profile("base"), tenant_id="globex")
    worker_b.update_backend_profile(
        "base",
        {"embedding_model": "tenant-model"},
        base_tenant_id="globex",
        target_tenant_id="acme",
    )

    stored = ConfigManager(store=store).get_backend_config("acme")
    assert sorted(stored.profiles) == ["base", "second_by_a", "written_by_a"]
    assert stored.profiles["base"].embedding_model == "tenant-model"
    assert sorted(worker_b.list_backend_profiles("acme")) == [
        "base",
        "second_by_a",
        "written_by_a",
    ]


@pytest.mark.parametrize(
    ("refresh_s", "max_staleness_s", "message"),
    [
        (-1.0, 60.0, "refresh_after_s must be >= 0, got -1.0"),
        (5.0, 1.0, "max_staleness_s (1.0) must be >= refresh_after_s (5.0)"),
    ],
)
def test_inconsistent_scoped_config_bounds_raise(refresh_s, max_staleness_s, message):
    with pytest.raises(ValueError) as caught:
        _refreshing_manager(
            InMemoryConfigStore(),
            refresh_s=refresh_s,
            max_staleness_s=max_staleness_s,
        )
    assert str(caught.value) == message


SYSTEM_REFRESH_THREAD = "system-config-refresh"


def _system_refreshing_manager(
    store, refresh_s: float, max_staleness_s: float = 30.0
) -> ConfigManager:
    return ConfigManager(
        store=store,
        system_config_refresh_s=refresh_s,
        system_config_max_staleness_s=max_staleness_s,
    )


def _join_system_refreshes() -> None:
    for thread in threading.enumerate():
        if thread.name == SYSTEM_REFRESH_THREAD:
            thread.join(timeout=10)
            assert thread.is_alive() is False


def test_the_system_config_is_held_for_the_documented_bounds():
    parameters = inspect.signature(ConfigManager).parameters
    assert (
        parameters["system_config_refresh_s"].default,
        parameters["system_config_max_staleness_s"].default,
    ) == (5.0, 60.0)


def test_another_processes_system_config_write_is_served_after_one_background_read():
    store = _CoordinatedConfigStore()
    _seed_system(store, "old-model")
    manager = _system_refreshing_manager(store, refresh_s=0.2)
    assert manager.get_system_config().llm_model == "old-model"
    # Written straight to the store, as another process's manager does.
    _seed_system(store, "new-model")
    assert manager.get_system_config().llm_model == "old-model"
    time.sleep(0.25)
    store.read_captured.clear()
    store.gate.clear()

    assert manager.get_system_config().llm_model == "old-model"
    # The call returned while the refresh it started was held at the store.
    assert store.read_captured.wait(timeout=5)
    assert store.gate.is_set() is False
    caller = threading.current_thread().name
    assert store.reads == [("_system", caller), ("_system", SYSTEM_REFRESH_THREAD)]
    store.gate.set()
    _join_system_refreshes()

    assert manager.get_system_config().llm_model == "new-model"
    assert store.get_calls == 2


def test_concurrent_reads_at_refresh_age_share_one_background_system_read():
    store = _CoordinatedConfigStore()
    _seed_system(store, "old-model")
    manager = _system_refreshing_manager(store, refresh_s=0.2)
    assert manager.get_system_config().llm_model == "old-model"
    _seed_system(store, "new-model")
    time.sleep(0.25)
    store.gate.clear()
    store.reads.clear()
    worker_count = 12
    ready = threading.Barrier(worker_count)

    def read_model(_: int) -> str:
        ready.wait(timeout=10)
        return manager.get_system_config().llm_model

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="request"
    ) as pool:
        models = list(pool.map(read_model, range(worker_count)))

    # Every request returned while the one refresh was held at the store.
    assert store.gate.is_set() is False
    assert models == ["old-model"] * worker_count
    store.gate.set()
    _join_system_refreshes()
    assert store.reads == [("_system", SYSTEM_REFRESH_THREAD)]
    assert manager.get_system_config().llm_model == "new-model"
    assert store.get_calls == 2


def test_a_failed_system_refresh_serves_the_held_value_until_the_bound_then_raises(
    caplog,
):
    store = _CoordinatedConfigStore()
    _seed_system(store, "held-model")
    manager = _system_refreshing_manager(store, refresh_s=0.2, max_staleness_s=1.5)
    filled_at = time.monotonic()
    assert manager.get_system_config().llm_model == "held-model"
    store.fail_every_get = True
    time.sleep(0.25)

    with caplog.at_level(logging.ERROR, logger=refreshing_cache_module.__name__):
        assert manager.get_system_config().llm_model == "held-model"
        _join_system_refreshes()
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == refreshing_cache_module.__name__
    ]
    assert len(messages) == 1
    assert re.fullmatch(
        r"system-config: refreshing 'system_config' failed with ConnectionError: "
        r"configuration store unavailable; serving the value read 0\.\d+s ago "
        r"until it is 1\.5s old",
        messages[0],
    )
    assert manager.get_system_config().llm_model == "held-model"

    time.sleep(max(0.0, filled_at + 1.55 - time.monotonic()))
    store.reads.clear()
    for _ in range(2):
        with pytest.raises(ConnectionError) as caught:
            manager.get_system_config()
        assert str(caught.value) == "configuration store unavailable"
    caller = threading.current_thread().name
    assert store.reads == [("_system", caller), ("_system", caller)]

    store.fail_every_get = False
    _seed_system(store, "recovered-model")
    assert manager.get_system_config().llm_model == "recovered-model"


def test_a_system_write_during_a_background_refresh_is_never_overwritten():
    store = _CoordinatedConfigStore()
    _seed_system(store, "old-model")
    manager = _system_refreshing_manager(store, refresh_s=0.2)
    assert manager.get_system_config().llm_model == "old-model"
    time.sleep(0.25)
    store.read_captured.clear()
    store.gate.clear()

    assert manager.get_system_config().llm_model == "old-model"
    # The refresh has read the pre-write row and is held there.
    assert store.read_captured.wait(timeout=5)
    manager.set_system_config(SystemConfig(llm_model="new-model"))
    store.gate.set()
    _join_system_refreshes()

    assert [manager.get_system_config().llm_model for _ in range(3)] == [
        "new-model"
    ] * 3
    assert store.get_calls == 2


def test_a_refreshed_system_config_still_serves_the_pinned_inference_urls():
    store = _CoordinatedConfigStore()
    _seed_system_urls(store, {"denseon": "http://persisted-denseon:8000"})
    manager = _system_refreshing_manager(store, refresh_s=0.2)
    manager.pin_inference_service_urls({"denseon": "http://explicit-denseon:8000"})
    assert manager.get_system_config().inference_service_urls == {
        "denseon": "http://explicit-denseon:8000"
    }
    store.set_config(
        tenant_id="_system",
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        config_value=SystemConfig(
            llm_model="rewritten-model",
            inference_service_urls={"denseon": "http://rewritten-denseon:8000"},
        ).to_dict(),
    )
    time.sleep(0.25)
    manager.get_system_config()
    _join_system_refreshes()

    served = manager.get_system_config()
    assert (served.llm_model, served.inference_service_urls) == (
        "rewritten-model",
        {"denseon": "http://explicit-denseon:8000"},
    )
    assert store.get_calls == 2


@pytest.mark.parametrize(
    ("refresh_s", "max_staleness_s", "message"),
    [
        (-1.0, 60.0, "refresh_after_s must be >= 0, got -1.0"),
        (5.0, 1.0, "max_staleness_s (1.0) must be >= refresh_after_s (5.0)"),
    ],
)
def test_inconsistent_system_config_bounds_raise(refresh_s, max_staleness_s, message):
    with pytest.raises(ValueError) as caught:
        _system_refreshing_manager(
            InMemoryConfigStore(),
            refresh_s=refresh_s,
            max_staleness_s=max_staleness_s,
        )
    assert str(caught.value) == message
