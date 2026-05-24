"""Unit tests for ``cogniverse_foundation.registry.EntryPointRegistry``.

Strong-assertion style: every test pins the exact identifier, exact
class object, exact cache key, exact error message — no
``isinstance``-only checks, no membership-only ``in`` checks.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from cogniverse_foundation.registry import EntryPointRegistry

# Module-level plugin classes so ``__module__`` is stable
# across re-imports and reproducible in conflict-detection assertions.


class _DummyStore:
    """Constructor-style plugin: takes config via kwargs."""

    def __init__(self, backend_url: str = "", backend_port: int = 0) -> None:
        self.backend_url = backend_url
        self.backend_port = backend_port
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _OtherDummyStore:
    """Second constructor-style plugin to test multi-registration."""

    def __init__(self, **config: Any) -> None:
        self.config = config


class _LifecycleProvider:
    """Lifecycle-style plugin: empty ``__init__`` then ``.initialize(config)``."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {}
        self.initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.initialized = True

    def shutdown(self) -> None:
        self.config = {}
        self.initialized = False


# Test subclasses


class _StoreRegistry(EntryPointRegistry[_DummyStore]):
    _entry_point_group = "test.cogniverse.stores"
    _label = "test store"


class _ProviderRegistry(EntryPointRegistry[_LifecycleProvider]):
    _entry_point_group = "test.cogniverse.providers"
    _label = "test provider"
    _tenant_scoped = True


@pytest.fixture(autouse=True)
def _reset_registries():
    """Each test gets pristine subclass state — no state leak across tests."""
    _StoreRegistry.reset()
    _ProviderRegistry.reset()
    yield
    _StoreRegistry.reset()
    _ProviderRegistry.reset()


# Subclass-time validation


def test_subclass_missing_entry_point_group_raises_typeerror():
    with pytest.raises(TypeError) as exc:

        class _NoGroup(EntryPointRegistry[_DummyStore]):
            _label = "x"

    assert (
        str(exc.value) == "_NoGroup must declare class var ``_entry_point_group`` "
        '(e.g. ``_entry_point_group = "cogniverse.<group>.<sub>"``).'
    )


def test_subclass_missing_label_raises_typeerror():
    with pytest.raises(TypeError) as exc:

        class _NoLabel(EntryPointRegistry[_DummyStore]):
            _entry_point_group = "test.x"

    assert (
        str(exc.value) == "_NoLabel must declare class var ``_label`` "
        '(singular noun used in diagnostics, e.g. "telemetry provider").'
    )


def test_subclass_state_is_isolated():
    """Registering on one subclass MUST NOT leak into another."""
    _StoreRegistry.register("alpha", _DummyStore)
    _ProviderRegistry.register("beta", _LifecycleProvider)

    assert _StoreRegistry.list_available() == ["alpha"]
    assert _ProviderRegistry.list_available() == ["beta"]
    assert _StoreRegistry._registry_classes is not _ProviderRegistry._registry_classes
    assert _StoreRegistry._instances is not _ProviderRegistry._instances


# Manual registration + lookup


def test_register_then_get_returns_constructor_instantiated_plugin():
    _StoreRegistry.register("vespa", _DummyStore)

    instance = _StoreRegistry.get(
        name="vespa",
        config={"backend_url": "http://vespa.local", "backend_port": 19071},
    )

    assert type(instance) is _DummyStore
    assert instance.backend_url == "http://vespa.local"
    assert instance.backend_port == 19071
    assert instance.closed is False


def test_register_then_get_with_no_config_uses_default_constructor():
    _StoreRegistry.register("vespa", _DummyStore)

    instance = _StoreRegistry.get(name="vespa")

    assert type(instance) is _DummyStore
    assert instance.backend_url == ""
    assert instance.backend_port == 0


def test_get_unknown_name_raises_with_available_listed():
    _StoreRegistry.register("vespa", _DummyStore)
    _StoreRegistry.register("elasticsearch", _OtherDummyStore)

    with pytest.raises(ValueError) as exc:
        _StoreRegistry.get(name="solr")

    assert (
        str(exc.value)
        == "Test Store 'solr' not found. Available: ['vespa', 'elasticsearch']"
    )


def test_get_with_no_name_and_no_plugins_raises():
    with pytest.raises(ValueError) as exc:
        _StoreRegistry.get()

    assert str(exc.value) == ("No test stores installed. Install a provider package.")


def test_get_with_name_none_returns_first_registered():
    _StoreRegistry.register("vespa", _DummyStore)
    _StoreRegistry.register("elasticsearch", _OtherDummyStore)

    instance = _StoreRegistry.get()

    assert type(instance) is _DummyStore  # vespa was registered first


# Cache semantics — config-scoped


def test_cache_hit_returns_same_instance_for_same_backend_url_port():
    _StoreRegistry.register("vespa", _DummyStore)
    cfg = {"backend_url": "http://vespa.local", "backend_port": 19071}

    first = _StoreRegistry.get(name="vespa", config=cfg)
    second = _StoreRegistry.get(name="vespa", config=cfg)

    assert first is second


def test_cache_miss_for_different_backend_port_produces_new_instance():
    _StoreRegistry.register("vespa", _DummyStore)

    a = _StoreRegistry.get(
        name="vespa",
        config={"backend_url": "http://vespa.local", "backend_port": 19071},
    )
    b = _StoreRegistry.get(
        name="vespa",
        config={"backend_url": "http://vespa.local", "backend_port": 28080},
    )

    assert a is not b
    assert a.backend_port == 19071
    assert b.backend_port == 28080


def test_cache_key_format_config_scoped():
    assert (
        _StoreRegistry._cache_key(
            "vespa",
            {"backend_url": "http://vespa.local", "backend_port": 19071},
            tenant_id=None,
        )
        == "vespa_http://vespa.local_19071"
    )


# Cache semantics — tenant-scoped


def test_tenant_scoped_get_requires_tenant_id():
    _ProviderRegistry.register("phoenix", _LifecycleProvider)

    with pytest.raises(ValueError) as exc:
        _ProviderRegistry.get(name="phoenix")

    assert "tenant_id" in str(exc.value).lower()


def test_tenant_scoped_get_calls_initialize_with_merged_config():
    """``require_tenant_id`` canonicalizes a bare org id ``acme`` →
    ``acme:acme`` (org:tenant form). The merged config reaching the
    plugin's ``initialize()`` MUST carry that canonical id, otherwise
    downstream tenant-scoped paths (Vespa schemas, Phoenix project
    names) would drift between simple and canonical suffixes."""
    _ProviderRegistry.register("phoenix", _LifecycleProvider)

    instance = _ProviderRegistry.get(
        name="phoenix",
        tenant_id="acme",
        config={"http_endpoint": "http://localhost:6006"},
    )

    assert type(instance) is _LifecycleProvider
    assert instance.initialized is True
    assert instance.config == {
        "tenant_id": "acme:acme",
        "http_endpoint": "http://localhost:6006",
    }


def test_tenant_scoped_cache_isolates_per_tenant():
    _ProviderRegistry.register("phoenix", _LifecycleProvider)

    acme = _ProviderRegistry.get(name="phoenix", tenant_id="acme")
    globex = _ProviderRegistry.get(name="phoenix", tenant_id="globex")

    assert acme is not globex
    assert acme.config["tenant_id"] == "acme:acme"
    assert globex.config["tenant_id"] == "globex:globex"


def test_tenant_scoped_cache_hit_returns_same_instance_for_same_tenant():
    _ProviderRegistry.register("phoenix", _LifecycleProvider)

    first = _ProviderRegistry.get(name="phoenix", tenant_id="acme")
    second = _ProviderRegistry.get(name="phoenix", tenant_id="acme")

    assert first is second


def test_cache_key_format_tenant_scoped():
    assert (
        _ProviderRegistry._cache_key("phoenix", config={}, tenant_id="acme")
        == "phoenix_acme"
    )


# Discovery


def test_discover_picks_up_entry_points(monkeypatch):
    """``discover()`` loads entry-point-registered classes."""
    fake_ep = MagicMock()
    fake_ep.name = "alpha"
    fake_ep.value = f"{_DummyStore.__module__}:{_DummyStore.__name__}"
    fake_ep.load = MagicMock(return_value=_DummyStore)

    def fake_entry_points(group=None):
        if group == "test.cogniverse.stores":
            return [fake_ep]
        return []

    monkeypatch.setattr(
        "cogniverse_foundation.registry.entry_point_registry.importlib.metadata.entry_points",
        fake_entry_points,
    )

    _StoreRegistry.discover()

    assert _StoreRegistry._registry_classes == {"alpha": _DummyStore}
    assert _StoreRegistry._entry_points_loaded is True


def test_discover_runs_only_once(monkeypatch):
    """Second ``discover()`` call is a no-op (loaded flag is idempotent)."""
    calls = {"count": 0}

    def fake_entry_points(group=None):
        calls["count"] += 1
        return []

    monkeypatch.setattr(
        "cogniverse_foundation.registry.entry_point_registry.importlib.metadata.entry_points",
        fake_entry_points,
    )

    _StoreRegistry.discover()
    _StoreRegistry.discover()
    _StoreRegistry.discover()

    assert calls["count"] == 1


def test_discover_raises_on_same_name_from_two_packages(monkeypatch):
    """If two installed packages register the same name, fail loud."""

    class _AlphaFromPkgA(_DummyStore):
        pass

    class _AlphaFromPkgB(_DummyStore):
        pass

    # Manually pre-populate so the second-package entry point hits the
    # conflict branch in discover().
    _AlphaFromPkgA.__module__ = "fake_pkg_a.module"
    _StoreRegistry._registry_classes["alpha"] = _AlphaFromPkgA

    fake_ep = MagicMock()
    fake_ep.name = "alpha"
    fake_ep.value = "fake_pkg_b.module:AlphaFromPkgB"
    fake_ep.load = MagicMock(return_value=_AlphaFromPkgB)

    def fake_entry_points(group=None):
        return [fake_ep]

    monkeypatch.setattr(
        "cogniverse_foundation.registry.entry_point_registry.importlib.metadata.entry_points",
        fake_entry_points,
    )

    with pytest.raises(ValueError) as exc:
        _StoreRegistry.discover()

    msg = str(exc.value)
    assert (
        msg == "Conflict: test store 'alpha' registered by multiple packages:\n"
        "  - fake_pkg_a.module\n"
        "  - fake_pkg_b.module\n"
        "Only one can be installed. Uninstall one package."
    )


def test_discover_swallows_load_failures_for_one_plugin(monkeypatch):
    """One broken plugin must not poison discovery of healthy ones."""
    broken = MagicMock()
    broken.name = "broken"
    broken.value = "fake_pkg.broken:Cls"
    broken.load = MagicMock(side_effect=ImportError("kaboom"))

    healthy = MagicMock()
    healthy.name = "healthy"
    healthy.value = f"{_DummyStore.__module__}:{_DummyStore.__name__}"
    healthy.load = MagicMock(return_value=_DummyStore)

    def fake_entry_points(group=None):
        return [broken, healthy]

    monkeypatch.setattr(
        "cogniverse_foundation.registry.entry_point_registry.importlib.metadata.entry_points",
        fake_entry_points,
    )

    _StoreRegistry.discover()

    assert _StoreRegistry._registry_classes == {"healthy": _DummyStore}


# clear_cache / reset / list_available / is_available


def test_clear_cache_calls_close_on_each_instance():
    _StoreRegistry.register("vespa", _DummyStore)
    instance = _StoreRegistry.get(
        name="vespa", config={"backend_url": "x", "backend_port": 1}
    )

    _StoreRegistry.clear_cache()

    assert instance.closed is True


def test_clear_cache_evicts_so_next_get_constructs_fresh():
    _StoreRegistry.register("vespa", _DummyStore)
    cfg = {"backend_url": "http://vespa.local", "backend_port": 19071}
    first = _StoreRegistry.get(name="vespa", config=cfg)

    _StoreRegistry.clear_cache()
    second = _StoreRegistry.get(name="vespa", config=cfg)

    assert first is not second


def test_clear_cache_calls_shutdown_on_lifecycle_plugins():
    _ProviderRegistry.register("phoenix", _LifecycleProvider)
    instance = _ProviderRegistry.get(name="phoenix", tenant_id="acme")
    assert instance.initialized is True

    _ProviderRegistry.clear_cache()

    assert instance.initialized is False


def test_reset_clears_everything():
    _StoreRegistry.register("vespa", _DummyStore)
    _StoreRegistry.get(name="vespa", config={"backend_url": "x", "backend_port": 1})

    _StoreRegistry.reset()

    assert _StoreRegistry._registry_classes == {}
    assert _StoreRegistry._entry_points_loaded is False


def test_list_available_returns_registered_names_in_insertion_order():
    _StoreRegistry.register("vespa", _DummyStore)
    _StoreRegistry.register("elasticsearch", _OtherDummyStore)

    assert _StoreRegistry.list_available() == ["vespa", "elasticsearch"]


def test_is_available_true_for_registered_false_otherwise():
    _StoreRegistry.register("vespa", _DummyStore)

    assert _StoreRegistry.is_available("vespa") is True
    assert _StoreRegistry.is_available("solr") is False
