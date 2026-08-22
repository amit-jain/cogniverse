"""Versioned optimizer artifacts on real Phoenix.

Every optimization run persists its candidate as a new version whose ledger
row records exactly which examples it consumed and how it scored. Activation
is a separate step: promotion moves the active pointer and copies the
version's content into the last-write-wins blob the served pod loads; keep /
rollback / reject leave the pointer where it is.
"""

from __future__ import annotations

import uuid

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def manager(phoenix_container) -> ArtifactManager:
    tenant_id = f"blobver_{uuid.uuid4().hex[:8]}"
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
async def test_save_blob_versioned_records_consumed_ids(manager, phoenix_container):
    """A saved version's ledger names its exact inputs; the active blob is untouched."""
    dataset_id, version = await manager.save_blob_versioned(
        "model",
        "simba_query_enhancement",
        '{"enhancer.predict": {"demos": ["v1"]}}',
        consumed_example_ids=["span:s-1", "span:s-2", "approved:batch-a_0"],
        decision="reject",
        scored=True,
        score=0.55,
        base_score=0.6,
        candidate_score=0.55,
    )

    assert version == 1
    from phoenix.client import Client

    dataset_name = manager._versioned_dataset_name(
        "model", "simba_query_enhancement", version
    )
    raw = Client(base_url=phoenix_container["http_endpoint"]).datasets.get_dataset(
        dataset=dataset_name
    )
    assert dataset_id == raw.id

    lineage = await manager.get_version_lineage("model", "simba_query_enhancement")
    assert [entry["version"] for entry in lineage] == [1]
    assert lineage[0]["name"] == manager._versioned_dataset_name(
        "model", "simba_query_enhancement", 1
    )
    assert lineage[0]["row_count"] == 1
    assert lineage[0]["consumed_example_ids"] == [
        "span:s-1",
        "span:s-2",
        "approved:batch-a_0",
    ]
    assert lineage[0]["decision"] == "reject"
    assert lineage[0]["scored"] is True
    assert lineage[0]["score"] == 0.55
    assert lineage[0]["base_score"] == 0.6
    assert lineage[0]["candidate_score"] == 0.55
    assert lineage[0]["created_at"].endswith("+00:00")

    content, ledger = await manager.load_blob_version(
        "model", "simba_query_enhancement", 1
    )
    assert content == '{"enhancer.predict": {"demos": ["v1"]}}'
    assert ledger["consumed_example_ids"] == [
        "span:s-1",
        "span:s-2",
        "approved:batch-a_0",
    ]
    assert ledger["score"] == 0.55
    assert ledger["version"] == 1

    # A rejected version never touches what the pod serves.
    assert await manager.load_blob("model", "simba_query_enhancement") is None
    assert await manager.get_blob_state("model", "simba_query_enhancement") == {
        "active": None
    }


@pytest.mark.asyncio
async def test_save_blob_versioned_accepts_insufficient_population(
    manager, phoenix_container
):
    key = "simba_query_enhancement"
    dataset_id, version = await manager.save_blob_versioned(
        "model",
        key,
        "{}",
        consumed_example_ids=["span:a", "span:b"],
        decision="insufficient_population",
        scored=False,
        score=None,
        base_score=None,
        candidate_score=None,
    )

    assert version == 1
    from phoenix.client import Client

    dataset_name = manager._versioned_dataset_name("model", key, version)
    raw = Client(base_url=phoenix_container["http_endpoint"]).datasets.get_dataset(
        dataset=dataset_name
    )
    assert dataset_id == raw.id

    lineage = await manager.get_version_lineage("model", key)
    assert [entry["version"] for entry in lineage] == [1]
    assert lineage[0]["name"] == dataset_name
    assert lineage[0]["row_count"] == 1
    assert lineage[0]["consumed_example_ids"] == ["span:a", "span:b"]
    assert lineage[0]["decision"] == "insufficient_population"
    assert lineage[0]["scored"] is False
    assert lineage[0]["score"] is None
    assert lineage[0]["base_score"] is None
    assert lineage[0]["candidate_score"] is None
    assert lineage[0]["created_at"].endswith("+00:00")

    content, ledger = await manager.load_blob_version("model", key, version)
    assert content == "{}"
    assert ledger["decision"] == "insufficient_population"
    assert ledger["consumed_example_ids"] == ["span:a", "span:b"]
    assert ledger["score"] is None
    assert ledger["version"] == 1
    assert await manager.load_blob("model", key) is None
    assert await manager.get_blob_state("model", key) == {"active": None}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "score", "base_score", "content"),
    [
        ("profile_selection", 0.66, 0.56, "profile-version"),
        ("entity_extraction", 0.77, 0.67, "entity-version"),
    ],
)
async def test_save_blob_versioned_round_trips_score_for_other_optimizers(
    manager, phoenix_container, key, score, base_score, content
):
    dataset_id, version = await manager.save_blob_versioned(
        "model",
        key,
        content,
        consumed_example_ids=["span:s-1"],
        decision="promote",
        scored=True,
        score=score,
        base_score=base_score,
        candidate_score=score,
    )

    assert version == 1
    from phoenix.client import Client

    dataset_name = manager._versioned_dataset_name("model", key, version)
    raw = Client(base_url=phoenix_container["http_endpoint"]).datasets.get_dataset(
        dataset=dataset_name
    )
    assert dataset_id == raw.id

    lineage = await manager.get_version_lineage("model", key)
    assert [entry["version"] for entry in lineage] == [1]
    assert lineage[0]["name"] == dataset_name
    assert lineage[0]["row_count"] == 1
    assert lineage[0]["consumed_example_ids"] == ["span:s-1"]
    assert lineage[0]["decision"] == "promote"
    assert lineage[0]["scored"] is True
    assert lineage[0]["score"] == score
    assert lineage[0]["base_score"] == base_score
    assert lineage[0]["candidate_score"] == score
    assert lineage[0]["created_at"].endswith("+00:00")

    content_value, ledger = await manager.load_blob_version("model", key, 1)
    assert content_value == content
    assert ledger["score"] == score
    assert ledger["consumed_example_ids"] == ["span:s-1"]
    assert ledger["version"] == 1


@pytest.mark.asyncio
async def test_activate_version_switches_active_blob(manager):
    """Promotion = activation: load_blob serves exactly the activated version."""
    key = "simba_query_enhancement"
    _, v1 = await manager.save_blob_versioned(
        "model",
        key,
        "content-v1",
        consumed_example_ids=["span:a"],
        decision="promote",
        scored=True,
        base_score=0.5,
        candidate_score=0.7,
    )
    _, v2 = await manager.save_blob_versioned(
        "model",
        key,
        "content-v2",
        consumed_example_ids=["span:a", "span:b"],
        decision="promote",
        scored=True,
        base_score=0.5,
        candidate_score=0.8,
    )
    assert (v1, v2) == (1, 2)

    state = await manager.activate_version("model", key, 1)
    assert state["active"]["version"] == 1
    assert state["active"]["activated_at"].endswith("+00:00")
    assert await manager.load_blob("model", key) == "content-v1"

    state = await manager.activate_version("model", key, 2)
    assert state["active"]["version"] == 2
    assert await manager.load_blob("model", key) == "content-v2"

    # A rejected v3 is persisted as a version and leaves the pointer at v2.
    _, v3 = await manager.save_blob_versioned(
        "model",
        key,
        "content-v3",
        consumed_example_ids=["span:c"],
        decision="reject",
        scored=True,
        base_score=0.5,
        candidate_score=0.4,
    )
    assert v3 == 3
    assert [e["version"] for e in await manager.get_version_lineage("model", key)] == [
        1,
        2,
        3,
    ]
    assert (await manager.get_blob_state("model", key))["active"]["version"] == 2
    assert await manager.load_blob("model", key) == "content-v2"

    # Re-activating an older version is a rollback: the pointer and the
    # served content both move back.
    state = await manager.activate_version("model", key, 1)
    assert state["active"]["version"] == 1
    assert await manager.load_blob("model", key) == "content-v1"


@pytest.mark.asyncio
async def test_activate_missing_version_raises_and_leaves_active_untouched(manager):
    key = "simba_query_enhancement"
    await manager.save_blob_versioned(
        "model",
        key,
        "content-v1",
        consumed_example_ids=["span:a"],
        decision="promote",
        scored=True,
        base_score=0.5,
        candidate_score=0.7,
    )
    await manager.activate_version("model", key, 1)

    with pytest.raises(ValueError) as err:
        await manager.activate_version("model", key, 7)
    assert str(err.value) == (
        f"No version 7 of blob model/{key} exists for tenant {manager._tenant_id}"
    )
    assert (await manager.get_blob_state("model", key))["active"]["version"] == 1
    assert await manager.load_blob("model", key) == "content-v1"


@pytest.mark.asyncio
async def test_save_blob_versioned_rejects_unattributable_input(manager):
    """A version that cannot say what it consumed is not a ledger entry."""
    for bad_ids, message in [
        ([], "consumed_example_ids must name at least one example"),
        (["span:a", "span:a"], "consumed_example_ids contains duplicates: ['span:a']"),
        (["span:a", ""], "consumed_example_ids must be non-empty strings"),
    ]:
        with pytest.raises(ValueError) as err:
            await manager.save_blob_versioned(
                "model",
                "simba_query_enhancement",
                "content",
                consumed_example_ids=bad_ids,
                decision="promote",
                scored=True,
                base_score=0.5,
                candidate_score=0.7,
            )
        assert str(err.value) == message

    with pytest.raises(ValueError) as err:
        await manager.save_blob_versioned(
            "model",
            "simba_query_enhancement",
            "content",
            consumed_example_ids=["span:a"],
            decision="shipped",
            scored=True,
            base_score=0.5,
            candidate_score=0.7,
        )
    assert str(err.value) == (
        "decision must be one of ['insufficient_population', 'keep', 'promote', "
        "'reject', 'rollback'], got 'shipped'"
    )
    assert await manager.get_version_lineage("model", "simba_query_enhancement") == []
