"""ConfigManager cache reads converge and never outlive a completed write."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
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
        self.read_captured = threading.Event()
        self.release_read = threading.Event()
        self.set_completed = threading.Event()
        self._count_lock = threading.Lock()

    def get_config(self, *args, **kwargs):
        with self._count_lock:
            self.get_calls += 1
        if self.fail_next_get:
            self.fail_next_get = False
            raise ConnectionError("configuration store unavailable")

        entry = super().get_config(*args, **kwargs)
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


def _seed_routing(store: _CoordinatedConfigStore, mode: str) -> None:
    store.set_config(
        tenant_id="acme:acme",
        scope=ConfigScope.ROUTING,
        service="gateway_agent",
        config_key="routing_config",
        config_value=RoutingConfigUnified(
            tenant_id="acme:acme",
            routing_mode=mode,
        ).to_dict(),
    )
    store.set_completed.clear()


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
