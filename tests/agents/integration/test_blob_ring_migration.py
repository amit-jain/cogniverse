"""One-time migration of pre-ring base-name blobs into ring slot 0.

Seeds base-name blob datasets the way the pre-ring writer did, runs the
migration, and asserts every reader resolves the exact pre-migration payload
while the base-name datasets stay as they were.
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest
from phoenix.client import AsyncClient

from cogniverse_agents.optimizer.artifact_manager import (
    _BLOB_RING_SLOTS,
    ArtifactManager,
)
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from scripts.migrate_blob_ring_slots import MIGRATION_REVISION, migrate

pytestmark = pytest.mark.integration

# What each tenant held before the ring: the kinds load_blob reads.
SEEDED_BLOBS = {
    ("config", "artefact_state_summarizer"): '{"active": {"version": 2}}',
    ("config", "pin_quotas"): '{"user": 7, "tenant_admin": 9, "org_admin": -1}',
    ("config", "signature_variants"): '{"summarizer": "terse"}',
    ("config", "blob_state_config_entity_extraction_ground_truth"): '{"version": 3}',
    ("model", "profile_selection"): '{"incumbent": "keep"}',
    ("xgboost", "training_decision_model"): '{"trees": []}',
}


def _manager(endpoint, grpc_endpoint, tenant_id) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return ArtifactManager(provider, tenant_id=tenant_id)


async def _seed_pre_ring_blob(manager: ArtifactManager, kind, key, content) -> str:
    """Write the dataset shape the pre-ring ``save_blob`` wrote."""
    name = manager._blob_dataset_name(kind, key)
    await manager._provider.datasets.create_dataset(
        name=name,
        data=pd.DataFrame([{"content": content}]),
        metadata={
            "artifact_type": f"blob_{kind}",
            "key": key,
            "tenant_id": manager._tenant_id,
            "input_keys": ["content"],
            "output_keys": [],
        },
    )
    return name


@pytest.fixture
async def seeded_tenants(phoenix_container):
    endpoint = phoenix_container["http_endpoint"]
    grpc_endpoint = phoenix_container["otlp_endpoint"]
    tenants = [f"mig{uuid.uuid4().hex[:8]}:t1", f"mig{uuid.uuid4().hex[:8]}:t2"]
    managers = {}
    for tenant_id in tenants:
        manager = _manager(endpoint, grpc_endpoint, tenant_id)
        for (kind, key), content in SEEDED_BLOBS.items():
            await _seed_pre_ring_blob(manager, kind, key, f"{content}|{tenant_id}")
        managers[tenant_id] = manager
    return endpoint, managers


@pytest.mark.asyncio
async def test_migration_makes_every_base_name_blob_readable_through_the_ring(
    seeded_tenants,
):
    endpoint, managers = seeded_tenants
    tenants = list(managers)

    for manager in managers.values():
        for kind, key in SEEDED_BLOBS:
            assert await manager.load_blob(kind, key) is None

    summary, unattributed = await migrate(endpoint, tenants)

    assert unattributed == []
    assert summary == {
        tenant_id: {
            "migrated": sorted(
                managers[tenant_id]._blob_dataset_name(kind, key)
                for kind, key in SEEDED_BLOBS
            )
        }
        for tenant_id in tenants
    }

    client = AsyncClient(base_url=endpoint)
    names = {row["name"] for row in await client.datasets.list()}
    for tenant_id, manager in managers.items():
        for (kind, key), content in SEEDED_BLOBS.items():
            expected = f"{content}|{tenant_id}"
            assert await manager.load_blob(kind, key) == expected
            base_name = manager._blob_dataset_name(kind, key)
            base_rows = (
                await manager._provider.datasets.get_dataset(name=base_name)
            ).to_dict("records")
            assert base_rows == [
                {"input": {"content": expected}, "output": {}, "metadata": {}}
            ]
            assert manager._blob_slot_name(kind, key, MIGRATION_REVISION) in names
            assert [
                slot
                for slot in range(_BLOB_RING_SLOTS)
                if manager._blob_slot_name(kind, key, slot) in names
            ] == [MIGRATION_REVISION]


@pytest.mark.asyncio
async def test_second_run_changes_nothing_and_a_new_publication_follows_slot_zero(
    seeded_tenants,
):
    endpoint, managers = seeded_tenants
    tenants = list(managers)
    await migrate(endpoint, tenants)

    repeat, unattributed = await migrate(endpoint, tenants)

    assert unattributed == []
    assert repeat == {
        tenant_id: {
            "already_in_ring": sorted(
                managers[tenant_id]._blob_dataset_name(kind, key)
                for kind, key in SEEDED_BLOBS
            )
        }
        for tenant_id in tenants
    }

    tenant_id = tenants[0]
    manager = managers[tenant_id]
    seeded = f"{SEEDED_BLOBS[('config', 'pin_quotas')]}|{tenant_id}"
    assert await manager.load_blob("config", "pin_quotas") == seeded

    await manager.save_blob("config", "pin_quotas", '{"user": 11}')
    assert await manager.load_blob("config", "pin_quotas") == '{"user": 11}'
    published = await manager._read_blob_slot(
        "config", "pin_quotas", MIGRATION_REVISION + 1
    )
    assert published == {"revision": 1, "content": '{"user": 11}'}


@pytest.mark.asyncio
async def test_a_tenant_left_out_is_reported_and_fails_the_run(seeded_tenants):
    endpoint, managers = seeded_tenants
    covered, omitted = list(managers)

    summary, unattributed = await migrate(endpoint, [covered])

    assert list(summary) == [covered]
    assert unattributed == sorted(
        managers[omitted]._blob_dataset_name(kind, key) for kind, key in SEEDED_BLOBS
    )
