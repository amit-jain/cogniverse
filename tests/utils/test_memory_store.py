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
