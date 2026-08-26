"""DatasetStore.replace_dataset: serialized last-write-wins with compensation.

``create_dataset`` appends a new version when the name already exists, so the
stable artefact names accumulated stale rows every save. ``replace_dataset``
serializes same-name writes, deletes then creates so the read returns only the
latest write, and restores the previous contents if the delete or create fails
after the old data is gone.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from cogniverse_foundation.telemetry.providers.base import (
    DatasetNotFoundError,
    DatasetStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _FakeStore(DatasetStore):
    """In-memory DatasetStore. ``fail_next_create`` fails exactly the next
    create (a transient torn write), then recovers — so the new write fails but
    the compensation restore succeeds."""

    def __init__(self):
        self.data: dict = {}
        self.fail_next_create = False
        self.creates: list = []

    async def create_dataset(self, name, data, metadata=None):
        self.creates.append(name)
        if self.fail_next_create:
            self.fail_next_create = False
            raise ConnectionError("create failed after delete")
        self.data[name] = data
        return name

    async def get_dataset(self, name):
        if name not in self.data:
            raise KeyError(name)
        return self.data[name]

    async def append_to_dataset(self, name, data, metadata=None):
        raise NotImplementedError

    async def delete_dataset(self, name):
        return self.data.pop(name, None) is not None


@pytest.mark.asyncio
async def test_replace_returns_only_latest_write():
    store = _FakeStore()
    await store.replace_dataset("d", pd.DataFrame([{"v": "first"}]))
    await store.replace_dataset("d", pd.DataFrame([{"v": "second"}]))
    got = await store.get_dataset("d")
    pd.testing.assert_frame_equal(got, pd.DataFrame([{"v": "second"}]))


@pytest.mark.asyncio
async def test_replace_restores_previous_on_torn_create():
    store = _FakeStore()
    await store.replace_dataset("d", pd.DataFrame([{"v": "original"}]))
    store.fail_next_create = True
    with pytest.raises(ConnectionError, match="create failed after delete"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))
    # The delete committed and the new create failed — the previous contents
    # must have been restored, not left destroyed.
    restored = await store.get_dataset("d")
    pd.testing.assert_frame_equal(restored, pd.DataFrame([{"v": "original"}]))


@pytest.mark.asyncio
async def test_replace_restores_existing_empty_dataset_on_torn_create():
    store = _FakeStore()
    empty = pd.DataFrame(columns=["v"])
    await store.replace_dataset("d", empty)
    store.fail_next_create = True

    with pytest.raises(ConnectionError, match="create failed after delete"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))

    restored = await store.get_dataset("d")
    pd.testing.assert_frame_equal(restored, empty)
    assert store.creates == ["d", "d", "d"]


@pytest.mark.asyncio
async def test_replace_on_absent_name_creates_fresh():
    store = _FakeStore()
    # No prior dataset — replace must simply create it (no restore attempted).
    await store.replace_dataset("d", pd.DataFrame([{"v": "x"}]))
    pd.testing.assert_frame_equal(
        await store.get_dataset("d"), pd.DataFrame([{"v": "x"}])
    )


class _OutageOnPreReadStore(DatasetStore):
    """``get_dataset`` raises a transient NON-KeyError outage (a backend blip,
    not a not-found). Records whether ``delete_dataset`` ran so a test can prove
    the destructive delete never fired when the pre-read could not confirm the
    prior contents."""

    def __init__(self):
        self.data: dict = {"d": pd.DataFrame([{"v": "PRECIOUS"}])}
        self.deleted: list = []

    async def create_dataset(self, name, data, metadata=None):
        self.data[name] = data
        return name

    async def get_dataset(self, name):
        raise ConnectionError("phoenix 503 during pre-read")

    async def append_to_dataset(self, name, data, metadata=None):
        raise NotImplementedError

    async def delete_dataset(self, name):
        self.deleted.append(name)
        return self.data.pop(name, None) is not None


@pytest.mark.asyncio
async def test_replace_pre_read_outage_propagates_before_delete():
    """A transient outage on the pre-read must propagate BEFORE the destructive
    delete, so a flapping backend can never destroy the prior dataset. Only a
    genuine not-found (KeyError/DatasetNotFoundError) may be treated as 'nothing to
    restore'."""
    store = _OutageOnPreReadStore()
    with pytest.raises(ConnectionError, match="503 during pre-read"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))
    assert store.deleted == [], (
        "destructive delete ran despite an unconfirmable pre-read"
    )
    assert "d" in store.data and list(store.data["d"]["v"]) == ["PRECIOUS"]


class _InvalidResponseOnPreReadStore(_OutageOnPreReadStore):
    async def get_dataset(self, name):
        raise ValueError("malformed dataset response")


@pytest.mark.asyncio
async def test_replace_invalid_pre_read_propagates_before_delete():
    store = _InvalidResponseOnPreReadStore()

    with pytest.raises(ValueError, match="malformed dataset response"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))

    assert store.deleted == []
    assert list(store.data["d"]["v"]) == ["PRECIOUS"]


class _AbsentDatasetStore(_FakeStore):
    async def get_dataset(self, name):
        if name not in self.data:
            raise DatasetNotFoundError(name)
        return self.data[name]


class _ControlledReplaceStore(DatasetStore):
    """In-memory DatasetStore with fault and concurrency hooks."""

    def __init__(self):
        self.data: dict[str, pd.DataFrame] = {}
        self.create_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.fail_next_create = False
        self.fail_delete_after_commit = False
        self.block_first_create = False
        self.first_create_entered = asyncio.Event()
        self.release_first_create = asyncio.Event()
        self.active_creates = 0
        self.max_active_creates = 0

    async def create_dataset(self, name, data, metadata=None):
        self.create_calls.append(name)
        self.active_creates += 1
        if self.active_creates > self.max_active_creates:
            self.max_active_creates = self.active_creates
        try:
            if self.block_first_create and not self.first_create_entered.is_set():
                self.first_create_entered.set()
                await self.release_first_create.wait()
            if self.fail_next_create:
                self.fail_next_create = False
                raise ConnectionError("create failed inside create")
            self.data[name] = data.copy()
            return name
        finally:
            self.active_creates -= 1

    async def get_dataset(self, name):
        if name not in self.data:
            raise KeyError(name)
        return self.data[name]

    async def append_to_dataset(self, name, data, metadata=None):
        raise NotImplementedError

    async def delete_dataset(self, name):
        self.delete_calls.append(name)
        existed = self.data.pop(name, None) is not None
        if self.fail_delete_after_commit:
            self.fail_delete_after_commit = False
            raise ConnectionError("delete failed after commit")
        return existed


@pytest.mark.asyncio
async def test_replace_typed_not_found_creates_fresh():
    store = _AbsentDatasetStore()

    result = await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))

    assert result == "d"
    pd.testing.assert_frame_equal(
        await store.get_dataset("d"), pd.DataFrame([{"v": "new"}])
    )


@pytest.mark.asyncio
async def test_replace_restores_previous_when_delete_commits_then_raises():
    store = _ControlledReplaceStore()
    original = pd.DataFrame([{"v": "original"}])
    store.data["d"] = original.copy()
    store.fail_delete_after_commit = True

    with pytest.raises(ConnectionError, match="delete failed after commit"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))

    restored = await store.get_dataset("d")
    pd.testing.assert_frame_equal(restored, original)
    assert store.delete_calls == ["d"]
    assert store.create_calls == ["d"]


@pytest.mark.asyncio
async def test_replace_restores_previous_when_create_raises():
    store = _ControlledReplaceStore()
    original = pd.DataFrame([{"v": "original"}])
    store.data["d"] = original.copy()
    store.fail_next_create = True

    with pytest.raises(ConnectionError, match="create failed inside create"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))

    restored = await store.get_dataset("d")
    pd.testing.assert_frame_equal(restored, original)
    assert store.delete_calls == ["d"]
    assert store.create_calls == ["d", "d"]


@pytest.mark.asyncio
async def test_replace_serializes_concurrent_writes():
    store = _ControlledReplaceStore()
    first = pd.DataFrame([{"v": "first"}])
    second = pd.DataFrame([{"v": "second"}])
    store.data["d"] = pd.DataFrame([{"v": "seed"}])
    store.block_first_create = True

    first_task = asyncio.create_task(store.replace_dataset("d", first))
    await asyncio.wait_for(store.first_create_entered.wait(), timeout=5)

    second_task = asyncio.create_task(store.replace_dataset("d", second))
    await asyncio.sleep(0)

    assert store.active_creates == 1
    assert store.max_active_creates == 1
    assert store.delete_calls == ["d"]
    assert store.create_calls == ["d"]

    store.release_first_create.set()
    await asyncio.gather(first_task, second_task)

    final = await store.get_dataset("d")
    pd.testing.assert_frame_equal(final, second)
    assert store.delete_calls == ["d", "d"]
    assert store.create_calls == ["d", "d"]
    assert store.max_active_creates == 1
