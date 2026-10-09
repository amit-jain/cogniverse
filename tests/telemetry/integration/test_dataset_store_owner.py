"""PhoenixDatasetStore records the tenant that owns a dataset, against a real
Phoenix.

A dataset created with a ``tenant_id`` carries it in its metadata from the
moment it exists; ``describe_datasets`` reports it. Another tenant cannot
append to it, a dataset whose rows fail to upload is removed again, and a
listing that Phoenix refuses raises instead of reading as no datasets.
"""

from __future__ import annotations

import asyncio
import uuid

import pandas as pd
import pytest

from cogniverse_foundation.telemetry.providers.base import (
    DatasetStoreUnavailableError,
)
from cogniverse_telemetry_phoenix.provider import (
    DatasetOwnedByAnotherTenantError,
    PhoenixDatasetStore,
)
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.requires_docker,
]


def _keys(tenant=None):
    keys = {"input_keys": ["question"], "output_keys": ["answer"]}
    return {**keys, "tenant_id": tenant} if tenant else keys


def _qa(*pairs):
    return pd.DataFrame(
        {"question": [q for q, _ in pairs], "answer": [a for _, a in pairs]}
    )


def _rows(frame):
    return [(row["input"], row["output"]) for _, row in frame.iterrows()]


@pytest.fixture
def store(phoenix_container) -> PhoenixDatasetStore:
    return PhoenixDatasetStore(phoenix_container["http_endpoint"])


async def _summary(store, name):
    return [d for d in await store.describe_datasets() if d.name == name]


@pytest.mark.asyncio
async def test_a_created_dataset_is_described_with_its_owner(store):
    owned, unowned = (f"owned-{uuid.uuid4().hex[:8]}" for _ in range(2))
    owned_id = await store.create_dataset(
        owned, _qa(("q1", "a1"), ("q2", "a2")), _keys("acme:owner")
    )
    await store.create_dataset(unowned, _qa(("q", "a")), _keys())
    await store.create_dataset(owned, _qa(("q3", "a3")), _keys("acme:owner"))

    (summary,) = await _summary(store, owned)
    assert (
        summary.id,
        summary.example_count,
        summary.tenant_id,
        summary.metadata,
    ) == (owned_id, 3, "acme:owner", {"tenant_id": "acme:owner"})
    assert [(s.tenant_id, s.metadata) for s in await _summary(store, unowned)] == [
        (None, {})
    ]
    assert _rows(await store.get_dataset(owned)) == [
        ({"question": "q1"}, {"answer": "a1"}),
        ({"question": "q2"}, {"answer": "a2"}),
        ({"question": "q3"}, {"answer": "a3"}),
    ]


@pytest.mark.asyncio
async def test_listing_is_newest_first(store):
    names = [f"order-{i}-{uuid.uuid4().hex[:8]}" for i in range(3)]
    for name in names:
        await store.create_dataset(name, _qa(("q", "a")), _keys("acme:order"))
        await asyncio.sleep(1.1)
    listed = [d.name for d in await store.describe_datasets() if d.name in names]
    assert listed == list(reversed(names))


@pytest.mark.asyncio
async def test_another_tenant_cannot_append_to_an_owned_dataset(store):
    name = f"guarded-{uuid.uuid4().hex[:8]}"
    await store.create_dataset(name, _qa(("q1", "a1")), _keys("acme:owner"))
    with pytest.raises(DatasetOwnedByAnotherTenantError) as refused:
        await store.create_dataset(name, _qa(("q2", "a2")), _keys("acme:intruder"))
    assert str(refused.value) == (
        f"Dataset {name!r} exists and is not owned by tenant 'acme:intruder'"
    )
    assert _rows(await store.get_dataset(name)) == [
        ({"question": "q1"}, {"answer": "a1"})
    ]


@pytest.mark.asyncio
async def test_concurrent_creates_of_one_name_leave_it_to_one_owner(store):
    """Two tenants create the same fresh name at once, twice each: the
    tenant whose create lands first owns the dataset and both its writes
    land in it; the other tenant's writes are all refused."""
    name = f"race-{uuid.uuid4().hex[:8]}"
    tenants = ["acme:first", "acme:second"] * 2

    async def create(index, tenant):
        try:
            await store.create_dataset(name, _qa((f"q{index}", tenant)), _keys(tenant))
        except DatasetOwnedByAnotherTenantError:
            return tenant, False
        return tenant, True

    outcomes = await asyncio.gather(*(create(i, t) for i, t in enumerate(tenants)))
    (summary,) = await _summary(store, name)
    winners = {tenant for tenant, landed in outcomes if landed}
    assert winners == {summary.tenant_id}
    assert (
        sorted(o for o in outcomes if o[0] == summary.tenant_id)
        == [(summary.tenant_id, True)] * 2
    )
    assert (
        sorted(o for o in outcomes if o[0] != summary.tenant_id)
        == [(next(iter({"acme:first", "acme:second"} - winners)), False)] * 2
    )
    assert (
        sorted(a["answer"] for _, a in _rows(await store.get_dataset(name)))
        == [summary.tenant_id] * 2
    )


@pytest.mark.asyncio
async def test_a_dataset_whose_rows_fail_to_upload_is_removed(phoenix_container):
    name = f"torn-{uuid.uuid4().hex[:8]}"
    with InterceptFaultProxy(
        phoenix_container["http_endpoint"],
        lambda method, path, body: (
            (503, {"detail": "down"})
            if path.startswith("/v1/datasets/upload")
            else None
        ),
    ) as proxy:
        failing = PhoenixDatasetStore(proxy.url)
        with pytest.raises(Exception) as failed:
            await failing.create_dataset(name, _qa(("q", "a")), _keys("acme:torn"))
    assert type(failed.value).__name__ == "HTTPStatusError"
    store = PhoenixDatasetStore(phoenix_container["http_endpoint"])
    assert await _summary(store, name) == []


@pytest.mark.asyncio
async def test_a_refused_listing_raises_instead_of_listing_nothing(phoenix_container):
    with InterceptFaultProxy(
        phoenix_container["http_endpoint"],
        lambda method, path, body: (503, {"detail": "down"}),
    ) as proxy:
        with pytest.raises(DatasetStoreUnavailableError) as refused:
            await PhoenixDatasetStore(proxy.url).describe_datasets()
    assert (refused.value.endpoint, refused.value.dataset) == (proxy.url, "*")
