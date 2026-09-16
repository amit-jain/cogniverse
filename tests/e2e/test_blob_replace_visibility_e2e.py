"""Replacing a serving blob must never expose its absence to a reader.

A replacement used to delete the blob's dataset and recreate it, so every
reader in that window saw no blob at all and fell back to its default. The
shipped form publishes each revision as its own immutable row in a ring of
slots and only prunes a slot two revisions behind, so the committed revision
stays readable throughout.

These drive the shipped artifact manager against the cluster's Phoenix over
its host port, which is how ``test_canary_state_machine_e2e`` reaches the same
store.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import (
    _BLOB_RING_SLOTS,
    ArtifactManager,
)
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.e2e.conftest import PHOENIX_URL, run_async, unique_id

pytestmark = pytest.mark.e2e

PHOENIX_GRPC = "localhost:33317"

BLOB_KIND = "config"


def _manager(tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": PHOENIX_URL,
            "grpc_endpoint": PHOENIX_GRPC,
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


def _ring_state(manager: ArtifactManager, key: str) -> dict[int, int]:
    """Slot index → the revision that slot holds, for every populated slot."""
    state: dict[int, int] = {}
    for slot in range(_BLOB_RING_SLOTS):
        record = run_async(manager._read_blob_slot(BLOB_KIND, key, slot))
        if record is not None:
            state[slot] = record["revision"]
    return state


class _BlobReader:
    """Reads the blob in a loop and records every value it observes."""

    def __init__(self, manager: ArtifactManager, key: str) -> None:
        self._manager = manager
        self._key = key
        self._stop = threading.Event()
        self.observations: list[tuple[float, str | None]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        async def loop() -> None:
            while not self._stop.is_set():
                value = await self._manager.load_blob(BLOB_KIND, self._key)
                self.observations.append((time.monotonic(), value))
                await asyncio.sleep(0.05)

        asyncio.run(loop())

    def __enter__(self) -> "_BlobReader":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=60)


@pytest.fixture
def owned_blob(request):
    """A blob key under a tenant this test owns, removed afterwards."""
    tenant_id = unique_id("prode2epipe") + ":t1"
    key = f"replace_probe_{uuid.uuid4().hex}"
    manager = _manager(tenant_id)

    def release() -> None:
        for slot in range(_BLOB_RING_SLOTS):
            name = manager._blob_slot_name(BLOB_KIND, key, slot)
            run_async(manager._provider.datasets.delete_dataset(name))

    request.addfinalizer(release)
    return tenant_id, key


def test_a_replacement_never_shows_a_reader_an_absent_blob(owned_blob):
    """Concurrent readers observe only the committed values, in order.

    Before the ring, a replacement deleted the blob's dataset before writing
    the new one, so a reader in that window observed ``None`` — which every
    caller renders as "no override", silently serving defaults.
    """
    tenant_id, key = owned_blob
    writer = _manager(tenant_id)
    reader = _manager(tenant_id)
    first = f"first-{uuid.uuid4().hex}"
    second = f"second-{uuid.uuid4().hex}"

    run_async(writer.save_blob(BLOB_KIND, key, first))
    assert run_async(writer.load_blob(BLOB_KIND, key)) == first
    assert _ring_state(writer, key) == {1 % _BLOB_RING_SLOTS: 1}

    with _BlobReader(reader, key) as watcher:
        # Let the reader take observations of the committed value first, so a
        # replacement that hid it has something to hide.
        time.sleep(2)
        before_replace = len(watcher.observations)
        replace_started = time.monotonic()
        run_async(writer.save_blob(BLOB_KIND, key, second))
        time.sleep(2)

    observations = list(watcher.observations)
    values = [value for _, value in observations]
    assert values[:before_replace] == [first] * before_replace
    assert set(values) == {first, second}
    assert values.count(None) == 0
    first_second_at = next(stamp for stamp, value in observations if value == second)
    assert first_second_at >= replace_started
    tail = values[values.index(second) :]
    assert tail == [second] * len(tail)

    # The predecessor stays readable in its own slot; nothing is pruned until
    # a further revision supersedes it.
    assert _ring_state(writer, key) == {
        1 % _BLOB_RING_SLOTS: 1,
        2 % _BLOB_RING_SLOTS: 2,
    }
    assert run_async(writer.load_blob(BLOB_KIND, key)) == second


def test_the_ring_prunes_only_the_revision_two_behind(owned_blob):
    """A third publication drops the first revision's slot and nothing else.

    This is what bounds the store: without it every revision would accumulate;
    with a shallower ring the publication would target a slot a reader is
    still being served from.
    """
    tenant_id, key = owned_blob
    manager = _manager(tenant_id)
    contents = [f"r{index}-{uuid.uuid4().hex}" for index in range(1, 4)]
    for content in contents:
        run_async(manager.save_blob(BLOB_KIND, key, content))

    assert _ring_state(manager, key) == {
        2 % _BLOB_RING_SLOTS: 2,
        3 % _BLOB_RING_SLOTS: 3,
    }
    assert run_async(manager.load_blob(BLOB_KIND, key)) == contents[-1]
