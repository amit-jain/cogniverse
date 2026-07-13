"""save_blob must overwrite, not append — one row per blob dataset.

The old save_blob called create_dataset on the same name each save, which
Phoenix versions (append), so the dataset grew a full payload copy per save and
load_blob downloaded the whole history. save_blob now deletes before create so
the dataset holds exactly one row (last-write-wins).
"""

from __future__ import annotations

import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def manager(phoenix_container) -> ArtifactManager:
    tenant_id = f"blob_{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.mark.asyncio
async def test_save_blob_overwrites_instead_of_appending(manager):
    await manager.save_blob("model", "k1", "v1")
    await manager.save_blob("model", "k1", "v2")

    assert await manager.load_blob("model", "k1") == "v2"

    name = manager._blob_dataset_name("model", "k1")
    df = await manager._provider.datasets.get_dataset(name=name)
    assert len(df) == 1, f"blob dataset must hold exactly one row, got {len(df)}"
