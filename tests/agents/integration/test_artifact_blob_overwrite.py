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


@pytest.mark.asyncio
async def test_gateway_reload_serves_rewritten_thresholds(manager):
    """Re-running _load_artifact on an ALREADY-LOADED gateway agent must apply
    a rewritten gateway_thresholds blob — the contract behind the dispatcher's
    periodic re-load, which is how a warm pod starts serving a recalibration
    without a restart."""
    import asyncio
    import json
    from types import SimpleNamespace

    from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps

    await manager.save_blob(
        "config",
        "gateway_thresholds",
        json.dumps({"fast_path_confidence_threshold": 0.5, "gliner_threshold": 0.49}),
    )

    agent = GatewayAgent(deps=GatewayDeps())
    agent.telemetry_manager = SimpleNamespace(
        get_provider=lambda tenant_id: manager._provider
    )
    agent._artifact_tenant_id = manager._tenant_id

    # Off-loop, exactly as the dispatcher runs it.
    await asyncio.to_thread(agent._load_artifact)
    assert agent.artifact_load_status == "loaded"
    assert agent.deps.fast_path_confidence_threshold == 0.5
    assert agent.deps.gliner_threshold == 0.49

    await manager.save_blob(
        "config",
        "gateway_thresholds",
        json.dumps({"fast_path_confidence_threshold": 0.35, "gliner_threshold": 0.2}),
    )

    await asyncio.to_thread(agent._load_artifact)
    assert agent.artifact_load_status == "loaded"
    assert agent.deps.fast_path_confidence_threshold == 0.35
    assert agent.deps.gliner_threshold == 0.2
