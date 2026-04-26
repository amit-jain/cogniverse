"""Verifies the full profile-change propagation chain in-process.

Chain: ConfigManager.add_backend_profile → profile_change_listener →
BackendRegistry.add_profile_to_backends → VespaSearchBackend.add_profile

This is the wiring test the CLAUDE.md round-trip rule mandates: writing a
profile on one side makes it visible on the other. Previously persistence
worked but the in-memory search backend never learned about the profile,
which caused the "agent_memories not found" retry storm surfaced by the
e2e suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendProfileConfig


class _InMemoryStore:
    """Minimal ConfigStore stub that keeps configs in a dict."""

    def __init__(self) -> None:
        self._data: dict[tuple, dict] = {}
        self._version = 0

    def get_config(self, tenant_id, scope, service, config_key):
        entry = self._data.get((tenant_id, scope, service, config_key))
        if entry is None:
            return None
        # ConfigStore returns an object with `.config_value` and `.version`.
        # Mock one for our purposes.
        return MagicMock(config_value=entry["value"], version=entry["version"])

    def set_config(
        self, tenant_id, scope, service, config_key, config_value, **_kwargs
    ):
        self._version += 1
        self._data[(tenant_id, scope, service, config_key)] = {
            "value": config_value,
            "version": self._version,
        }
        return MagicMock(version=self._version)


@pytest.fixture
def clean_backend_registry():
    """Ensure BackendRegistry cache is isolated per test."""
    BackendRegistry._backend_instances.clear()
    yield
    BackendRegistry._backend_instances.clear()


def test_add_backend_profile_propagates_to_cached_search_backend(
    clean_backend_registry,
):
    # Install a fake search backend in the registry cache.
    fake_backend = MagicMock()
    fake_backend.add_profile = MagicMock()
    fake_backend.remove_profile = MagicMock()
    BackendRegistry._backend_instances.set("search_vespa", fake_backend)

    store = _InMemoryStore()

    # Wire the same listener main.py wires at startup.
    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    cm = ConfigManager(store=store, profile_change_listener=listener)

    profile = BackendProfileConfig(
        profile_name="agent_memories",
        type="memory",
        schema_name="agent_memories",
        embedding_model="lightonai/DenseOn",
        embedding_type="dense",
        schema_config={"embedding_dims": 768},
    )

    cm.add_backend_profile(profile, tenant_id="test:unit", service="backend")

    fake_backend.add_profile.assert_called_once()
    called_args = fake_backend.add_profile.call_args
    assert called_args.args[0] == "agent_memories"
    passed_cfg = called_args.args[1]
    assert passed_cfg["type"] == "memory"
    assert passed_cfg["schema_name"] == "agent_memories"


def test_delete_backend_profile_propagates_removal(clean_backend_registry):
    fake_backend = MagicMock()
    fake_backend.add_profile = MagicMock()
    fake_backend.remove_profile = MagicMock()
    BackendRegistry._backend_instances.set("search_vespa", fake_backend)

    store = _InMemoryStore()

    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    cm = ConfigManager(store=store, profile_change_listener=listener)

    # Add first so there's something to remove.
    profile = BackendProfileConfig(
        profile_name="to_delete",
        type="memory",
        schema_name="to_delete",
    )
    cm.add_backend_profile(profile, tenant_id="test:unit", service="backend")
    assert (
        cm.delete_backend_profile("to_delete", tenant_id="test:unit", service="backend")
        is True
    )

    fake_backend.remove_profile.assert_called_once_with("to_delete")


def test_listener_exception_does_not_break_add_backend_profile(
    clean_backend_registry,
):
    """A faulty listener must not corrupt ConfigManager behavior."""
    store = _InMemoryStore()

    def bad_listener(event, name, cfg):
        raise RuntimeError("intentional")

    cm = ConfigManager(store=store, profile_change_listener=bad_listener)
    profile = BackendProfileConfig(
        profile_name="safe", type="memory", schema_name="safe"
    )
    # Must not raise even though listener does.
    result = cm.add_backend_profile(profile, tenant_id="test:unit", service="backend")
    assert result.profile_name == "safe"


def test_backend_registry_add_profile_swallows_backend_exceptions(
    clean_backend_registry,
):
    """If one backend's add_profile raises, others must still be updated."""
    good = MagicMock()
    good.add_profile = MagicMock()
    bad = MagicMock()
    bad.add_profile = MagicMock(side_effect=RuntimeError("boom"))

    BackendRegistry._backend_instances.set("search_good", good)
    BackendRegistry._backend_instances.set("search_bad", bad)

    updated = BackendRegistry.add_profile_to_backends("p", {"type": "m"})

    # Bad backend raised so counts as not updated, good counts as 1.
    assert updated == 1
    good.add_profile.assert_called_once()
    bad.add_profile.assert_called_once()


def test_backend_registry_counts_only_search_and_backend_caches(
    clean_backend_registry,
):
    search_backend = MagicMock()
    search_backend.add_profile = MagicMock()
    ingestion_backend = MagicMock()
    ingestion_backend.add_profile = MagicMock()
    unrelated = MagicMock()
    unrelated.add_profile = MagicMock()

    BackendRegistry._backend_instances.set("search_vespa", search_backend)
    BackendRegistry._backend_instances.set("backend_vespa_t1", ingestion_backend)
    BackendRegistry._backend_instances.set("some_other_key", unrelated)

    updated = BackendRegistry.add_profile_to_backends("p", {"type": "m"})

    assert updated == 2
    search_backend.add_profile.assert_called_once()
    ingestion_backend.add_profile.assert_called_once()
    unrelated.add_profile.assert_not_called()
