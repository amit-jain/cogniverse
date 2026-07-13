"""Pin the approved_synthetic_data dataset read/reconstruct contract.

Writes an approved synthetic item via ApprovalStorageImpl.append_to_training_dataset
and reads it back via provider.datasets.get_dataset, so the orchestrator's
loader knows the exact column shape (and whether list fields survive as lists or
JSON strings) before reconstructing example dicts.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def telemetry_manager(phoenix_container):
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_approved_synthetic_dataset_roundtrip(
    phoenix_container, telemetry_manager
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewItem

    tenant_id = "ds_rt"
    project_name = "finetuning"
    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )

    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
    )

    items = [
        ReviewItem(
            item_id="syn_0",
            data={
                "query": "PyTorch was released by Meta AI",
                "entities": [{"text": "PyTorch", "type": "ORG"}],
                "entity_types": "ORG",
                "relationships": [],
            },
            confidence=0.9,
            metadata={"agent_type": "entity_extraction", "synthetic": True},
            status=ApprovalStatus.APPROVED,
        )
    ]

    dataset_name = "approved_synthetic_data"
    ok = await storage.append_to_training_dataset(dataset_name, items)
    assert ok is True

    df = await storage.provider.datasets.get_dataset(name=dataset_name)

    from cogniverse_finetuning.dataset.synthetic_reader import (
        format_synthetic_sft,
        load_approved_synthetic_examples,
    )

    loaded = load_approved_synthetic_examples(df, "entity_extraction")
    assert len(loaded) == 1
    example = loaded[0]
    assert example["query"] == "PyTorch was released by Meta AI"
    # List field survived the Phoenix string round-trip and parsed back to a list.
    assert example["entities"] == [{"text": "PyTorch", "type": "ORG"}]
    assert example["relationships"] == []
    # Bookkeeping columns are not part of the reconstructed example.
    assert "item_id" not in example and "status" not in example

    # And it folds into an SFT record via the reader.
    records = format_synthetic_sft(loaded, "entity_extraction")
    assert len(records) == 1
    assert "### Input:\nPyTorch was released by Meta AI" in records[0]["text"]
