"""A reset to the base module leaves the ledger and the served blob agreeing.

The e2e batch module reseeds its tenant by publishing the agent's base state
through ``save_blob_versioned`` + ``activate_version`` — the pair a rollback
uses. Writing the served blob alone moves what the pod loads without moving
the ledger's active version, and a later test then reads an artifact "flip"
that never happened.
"""

from __future__ import annotations

import json
import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration

KEY = "entity_extraction"
BASE = json.dumps({"extract.predict": {"signature": {"instructions": "BASE"}}})
PROMOTED = json.dumps({"extract.predict": {"signature": {"instructions": "PROMOTED"}}})


@pytest.fixture
def manager(phoenix_container) -> ArtifactManager:
    tenant = f"reset{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant)


async def _promote(manager: ArtifactManager, content: str, decision: str) -> int:
    _, version = await manager.save_blob_versioned(
        kind="model",
        key=KEY,
        content=content,
        consumed_example_ids=[f"span:{uuid.uuid4().hex[:8]}"],
        decision=decision,
        scored=False,
        base_score=None,
        candidate_score=None,
    )
    await manager.activate_version("model", KEY, version)
    return version


@pytest.mark.asyncio
async def test_reset_through_the_rollback_seam_agrees_with_the_ledger(manager):
    promoted_version = await _promote(manager, PROMOTED, "promote")

    reset_version = await _promote(manager, BASE, "rollback")

    assert reset_version == promoted_version + 1
    assert await manager.active_blob_version("model", KEY) == reset_version
    assert await manager.load_blob("model", KEY) == BASE
    content, ledger = await manager.load_blob_version("model", KEY, reset_version)
    assert content == BASE
    assert ledger["decision"] == "rollback"
    assert ledger["version"] == reset_version


@pytest.mark.asyncio
async def test_overwriting_the_served_blob_alone_leaves_the_ledger_stale(manager):
    promoted_version = await _promote(manager, PROMOTED, "promote")

    await manager.save_blob("model", KEY, BASE)

    assert await manager.load_blob("model", KEY) == BASE
    assert await manager.active_blob_version("model", KEY) == promoted_version
    content, _ = await manager.load_blob_version("model", KEY, promoted_version)
    assert content == PROMOTED
