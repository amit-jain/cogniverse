"""The e2e backend-env bridge, verified without loading the session fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

_MODULE_PATH = Path(__file__).resolve().parents[2] / "e2e" / "backend_env.py"
_spec = importlib.util.spec_from_file_location("e2e_backend_env", _MODULE_PATH)
backend_env = importlib.util.module_from_spec(_spec)
sys.modules["e2e_backend_env"] = backend_env
_spec.loader.exec_module(backend_env)

DEAD_SENTINEL = "29071"


def test_splits_an_explicit_port():
    assert backend_env.backend_env_from_vespa_url("http://localhost:33080") == (
        "http://localhost",
        "33080",
    )


def test_defaults_the_port_by_scheme_rather_than_leaving_it_empty():
    assert backend_env.backend_env_from_vespa_url("https://vespa.example") == (
        "https://vespa.example",
        "443",
    )
    assert backend_env.backend_env_from_vespa_url("http://vespa.example") == (
        "http://vespa.example",
        "80",
    )


def test_refuses_a_url_it_cannot_split():
    with pytest.raises(ValueError) as excinfo:
        backend_env.backend_env_from_vespa_url("localhost:33080")

    assert "VESPA_URL" in str(excinfo.value)


def test_export_publishes_the_live_endpoint_not_the_dead_sentinel(monkeypatch):
    monkeypatch.delenv("TEST_BACKEND_URL", raising=False)
    monkeypatch.delenv("TEST_BACKEND_PORT", raising=False)
    monkeypatch.setenv("VESPA_URL", "http://localhost:33080")

    assert backend_env.export_backend_env() == ("http://localhost", "33080")
    assert backend_env.os.environ["TEST_BACKEND_PORT"] == "33080"
    assert backend_env.os.environ["TEST_BACKEND_PORT"] != DEAD_SENTINEL


def test_export_does_not_override_an_explicit_value(monkeypatch):
    monkeypatch.setenv("TEST_BACKEND_URL", "http://explicit")
    monkeypatch.setenv("TEST_BACKEND_PORT", "44444")
    monkeypatch.setenv("VESPA_URL", "http://localhost:33080")

    backend_env.export_backend_env()

    assert backend_env.os.environ["TEST_BACKEND_URL"] == "http://explicit"
    assert backend_env.os.environ["TEST_BACKEND_PORT"] == "44444"


def _stored_system_config(store: InMemoryConfigStore, config: SystemConfig) -> None:
    store.set_config(
        tenant_id="_system",
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        config_value=config.to_dict(),
    )


def test_local_system_config_is_served_on_every_read_and_never_the_stored_one():
    cluster = InMemoryConfigStore()
    _stored_system_config(
        cluster,
        SystemConfig(
            backend_url="http://cogniverse-vespa",
            inference_service_urls={"denseon": "http://cogniverse-denseon:8000"},
        ),
    )
    local = SystemConfig(
        backend_url="http://localhost",
        backend_port=33080,
        inference_service_urls={"denseon": "http://localhost:33906"},
    )
    # Both bounds 0: every call reads the store, as a refresh does.
    manager = ConfigManager(
        store=backend_env.LocalSystemConfig(cluster, local),
        system_config_refresh_s=0,
        system_config_max_staleness_s=0,
    )

    served = [manager.get_system_config() for _ in range(2)]

    assert [
        (config.backend_url, config.backend_port, config.inference_service_urls)
        for config in served
    ] == [("http://localhost", 33080, {"denseon": "http://localhost:33906"})] * 2
    stored = cluster.get_config(
        "_system", ConfigScope.SYSTEM, "system", "system_config"
    )
    assert stored.config_value["backend_url"] == "http://cogniverse-vespa"


def test_local_system_config_reads_and_writes_every_other_config_in_the_store():
    cluster = InMemoryConfigStore()
    manager = ConfigManager(
        store=backend_env.LocalSystemConfig(cluster, SystemConfig()),
        scoped_config_refresh_s=0,
        scoped_config_max_staleness_s=0,
    )

    manager.set_routing_config(
        RoutingConfigUnified(tenant_id="acme:acme", routing_mode="ensemble")
    )

    assert manager.get_routing_config("acme:acme").routing_mode == "ensemble"
    stored = ConfigManager(store=cluster).get_routing_config("acme:acme")
    assert stored.routing_mode == "ensemble"
    assert manager.store.source == cluster.source
