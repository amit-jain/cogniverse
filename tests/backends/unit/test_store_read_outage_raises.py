"""Config and adapter store reads must raise on a backend outage.

A genuinely-absent config/adapter returns None, but a Vespa read FAILURE used
to return the same None — so a transient outage silently reverted a tenant to
default config or "no adapter". The two cases must be distinguishable: absent
-> None, backend error -> raise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_vespa.config.config_store import ConfigScope, VespaConfigStore
from cogniverse_vespa.registry.adapter_store import VespaAdapterStore


def _config_store(query_impl):
    store = object.__new__(VespaConfigStore)
    store.schema_name = "config_metadata"
    store.vespa_app = MagicMock()
    store.vespa_app.query = query_impl
    return store


def _adapter_store(query_impl):
    store = object.__new__(VespaAdapterStore)
    store.schema_name = "adapter_registry"
    store.vespa_app = MagicMock()
    store.vespa_app.query = query_impl
    return store


def _empty_response(*_args, **_kwargs):
    return SimpleNamespace(hits=[])


def _boom(*_args, **_kwargs):
    raise ConnectionError("vespa unreachable")


def test_config_absent_returns_none():
    store = _config_store(_empty_response)
    result = store.get_config(
        tenant_id="acme:acme",
        scope=ConfigScope.BACKEND,
        service="backend",
        config_key="k",
    )
    assert result is None


def test_config_backend_error_raises():
    store = _config_store(_boom)
    with pytest.raises(ConnectionError):
        store.get_config(
            tenant_id="acme:acme",
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="k",
        )


def test_adapter_absent_returns_none():
    store = _adapter_store(_empty_response)
    assert store.get_adapter("a1") is None


def test_adapter_backend_error_raises():
    store = _adapter_store(_boom)
    with pytest.raises(ConnectionError):
        store.get_adapter("a1")


def test_config_history_empty_returns_empty_list():
    store = _config_store(_empty_response)
    assert (
        store.get_config_history(
            tenant_id="acme:acme",
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="k",
        )
        == []
    )


def test_config_history_backend_error_raises():
    store = _config_store(_boom)
    with pytest.raises(ConnectionError):
        store.get_config_history(
            tenant_id="acme:acme",
            scope=ConfigScope.BACKEND,
            service="backend",
            config_key="k",
        )


def test_list_configs_empty_returns_empty_list():
    store = _config_store(_empty_response)
    assert store.list_configs(tenant_id="acme:acme") == []


def test_list_configs_backend_error_raises():
    store = _config_store(_boom)
    with pytest.raises(ConnectionError):
        store.list_configs(tenant_id="acme:acme")


def test_list_all_configs_empty_returns_empty_list():
    """list_all_configs reads the Document v1 visit path, not vespa_app.query."""
    from unittest.mock import patch

    store = _config_store(_empty_response)
    store.vespa_app = SimpleNamespace(url="http://localhost:8080")
    visit_response = MagicMock()
    visit_response.json.return_value = {"documents": [], "continuation": None}

    with patch("requests.get", return_value=visit_response):
        assert store.list_all_configs() == []


def test_list_all_configs_backend_error_raises():
    """The schema registry reads all schemas through this — an outage that
    returns [] loads zero schemas and wipes the in-memory cache, while a
    raise activates the registry's designed keep-existing-cache fallback."""
    import requests

    store = _config_store(_boom)
    # Document v1 visit path: a real connection-refused against a dead port.
    store.vespa_app = SimpleNamespace(url="http://127.0.0.1:9")

    with pytest.raises(requests.exceptions.ConnectionError):
        store.list_all_configs()


def test_list_adapters_empty_returns_empty_list():
    store = _adapter_store(_empty_response)
    assert store.list_adapters(tenant_id="acme:acme") == []


def test_list_adapters_backend_error_raises():
    """Finetuning resolves a tenant's LoRA through this — an outage that
    returns [] silently reverts the tenant to the base model."""
    store = _adapter_store(_boom)
    with pytest.raises(ConnectionError):
        store.list_adapters(tenant_id="acme:acme")
