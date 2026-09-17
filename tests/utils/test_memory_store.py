"""The shared test store preserves conditional writes and version history."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_KEY = {
    "tenant_id": "tenant",
    "scope": ConfigScope.SCHEMA,
    "service": "schema_deployment_intents",
    "config_key": "video_tenant",
}


def test_conditional_write_preserves_versions_and_rejects_stale_writers():
    store = InMemoryConfigStore()
    created = store.compare_and_set_config(
        **_KEY, config_value={"state": "pending"}, expected_version=0
    )
    assert (created.version, created.config_value) == (1, {"state": "pending"})

    updated = store.set_config(**_KEY, config_value={"state": "absent"})
    assert (updated.version, updated.config_value) == (2, {"state": "absent"})
    assert (
        store.compare_and_set_config(
            **_KEY, config_value={"state": "stale"}, expected_version=1
        )
        is None
    )

    completed = store.compare_and_set_config(
        **_KEY, config_value={"state": "complete"}, expected_version=2
    )
    assert (completed.version, completed.config_value) == (3, {"state": "complete"})
    assert store.get_config(**_KEY) == completed
    assert store.get_config(**_KEY, version=1) == created
    assert [
        (entry.version, entry.config_value)
        for entry in store.get_config_history(**_KEY)
    ] == [
        (3, {"state": "complete"}),
        (2, {"state": "absent"}),
        (1, {"state": "pending"}),
    ]


def test_conditional_write_rejects_negative_revision_without_writing():
    store = InMemoryConfigStore()
    with pytest.raises(ValueError, match="expected_version must be nonnegative"):
        store.compare_and_set_config(
            **_KEY, config_value={"state": "pending"}, expected_version=-1
        )
    assert store.list_all_configs() == []


def test_concurrent_conditional_creates_have_one_winner(monkeypatch):
    store = InMemoryConfigStore()
    get_config = store.get_config
    contenders = 8
    ready = threading.Barrier(contenders)

    def delayed_read(*args, **kwargs):
        entry = get_config(*args, **kwargs)
        time.sleep(0.01)
        return entry

    monkeypatch.setattr(store, "get_config", delayed_read)

    def claim(_):
        ready.wait(timeout=5)
        return store.compare_and_set_config(
            **_KEY, config_value={"state": "pending"}, expected_version=0
        )

    with ThreadPoolExecutor(max_workers=contenders) as pool:
        results = list(pool.map(claim, range(contenders)))

    assert results.count(None) == contenders - 1
    assert [
        (entry.version, entry.config_value) for entry in results if entry is not None
    ] == [(1, {"state": "pending"})]
    assert [
        (entry.version, entry.config_value)
        for entry in store.get_config_history(**_KEY)
    ] == [(1, {"state": "pending"})]


def test_import_files_every_config_under_the_requested_tenant():
    """The destination tenant is the caller's, never the file's.

    ``VespaConfigStore.import_configs`` writes every entry under the
    ``tenant_id`` it was given and never reads the payload's own id, so a
    double that stores under the exported tenant hides a cross-tenant write.
    """
    store = InMemoryConfigStore()
    store.set_config(
        "source:tenant", ConfigScope.AGENT, "search_agent", "settings", {"model": "m"}
    )
    exported = store.export_configs("source:tenant")

    assert store.import_configs(tenant_id="target:tenant", configs=exported) == 1
    assert store.get_config(
        "target:tenant", ConfigScope.AGENT, "search_agent", "settings"
    ).config_value == {"model": "m"}
    assert [
        entry["tenant_id"] for entry in store.export_configs("target:tenant")["configs"]
    ] == ["target:tenant"]
    assert [
        entry["tenant_id"] for entry in store.export_configs("source:tenant")["configs"]
    ] == ["source:tenant"]


def test_schema_rows_are_neither_exported_nor_imported():
    """Mirrors ``VespaConfigStore``: schema-scope rows record the source
    tenant's deployments, so an export omits them and an import that carries
    one is refused before anything is written."""
    store = InMemoryConfigStore()
    store.set_config(
        "source:tenant", ConfigScope.AGENT, "search_agent", "settings", {"model": "m"}
    )
    registration = {"tenant_id": "source:tenant", "base_schema_name": "document_text"}
    store.set_config(
        "source:tenant",
        ConfigScope.SCHEMA,
        "schema_registry",
        "schema_document_text",
        registration,
    )

    exported = store.export_configs("source:tenant")
    assert [
        (entry["scope"], entry["service"], entry["config_key"])
        for entry in exported["configs"]
    ] == [("agent", "search_agent", "settings")]

    exported["configs"].append(
        {
            "scope": "schema",
            "service": "schema_registry",
            "config_key": "schema_document_text",
            "config_value": registration,
        }
    )
    with pytest.raises(ValueError) as refused:
        store.import_configs(tenant_id="target:tenant", configs=exported)
    assert str(refused.value) == (
        "Configuration import for tenant target:tenant refused: schema rows record "
        "deployments made by the schema registry and are not importable: "
        "schema_registry/schema_document_text"
    )
    assert store.export_configs("target:tenant")["configs"] == []


def test_an_import_that_fails_partway_leaves_the_store_as_it_was():
    """Mirrors ``VespaConfigStore``: an import is all or nothing."""
    store = InMemoryConfigStore()
    store.set_config(
        "target:tenant", ConfigScope.AGENT, "search_agent", "settings", {"model": "m"}
    )
    payload = {
        "configs": [
            {
                "scope": "agent",
                "service": "search_agent",
                "config_key": "settings",
                "config_value": {"model": "imported"},
            },
            {
                "scope": "agent",
                "service": "search_agent",
                "config_key": "fresh",
                "config_value": {"n": 1},
            },
            {
                "scope": "no-such-scope",
                "service": "search_agent",
                "config_key": "broken",
                "config_value": {},
            },
        ]
    }

    with pytest.raises(RuntimeError) as raised:
        store.import_configs(tenant_id="target:tenant", configs=payload)

    assert str(raised.value) == (
        "Configuration import for tenant target:tenant failed at row 3 of 3 "
        "(search_agent/broken): 'no-such-scope' is not a valid ConfigScope; "
        "removed 2 of the 2 versions it had written"
    )
    assert [
        (entry["config_key"], entry["version"], entry["config_value"])
        for entry in store.export_configs("target:tenant", include_history=True)[
            "configs"
        ]
    ] == [("settings", 1, {"model": "m"})]
