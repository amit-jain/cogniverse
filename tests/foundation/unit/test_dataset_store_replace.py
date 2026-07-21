"""DatasetStore.replace_dataset: last-write-wins with torn-create compensation.

``create_dataset`` appends a new version when the name already exists, so the
stable artefact names accumulated stale rows every save. ``replace_dataset``
deletes then creates so the read returns only the latest write — and restores
the previous contents if the create fails after the delete committed, so a torn
replace never destroys the prior dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cogniverse_foundation.telemetry.providers.base import DatasetStore

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
    assert list(got["v"]) == ["second"], "replace must not accumulate prior rows"


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
    assert list(restored["v"]) == ["original"], "torn replace destroyed prior data"


@pytest.mark.asyncio
async def test_replace_on_absent_name_creates_fresh():
    store = _FakeStore()
    # No prior dataset — replace must simply create it (no restore attempted).
    await store.replace_dataset("d", pd.DataFrame([{"v": "x"}]))
    assert list((await store.get_dataset("d"))["v"]) == ["x"]


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
    genuine not-found (KeyError/ValueError) may be treated as 'nothing to
    restore'."""
    store = _OutageOnPreReadStore()
    with pytest.raises(ConnectionError, match="503 during pre-read"):
        await store.replace_dataset("d", pd.DataFrame([{"v": "new"}]))
    assert store.deleted == [], (
        "destructive delete ran despite an unconfirmable pre-read"
    )
    assert "d" in store.data and list(store.data["d"]["v"]) == ["PRECIOUS"]
