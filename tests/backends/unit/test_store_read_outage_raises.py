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
