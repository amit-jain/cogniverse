"""Real-Phoenix: a resumed run trains on approved synthetic data.

Simulates the post-approval state — approved synthetic examples persisted in the
approved_synthetic_data dataset (what the dashboard's apply_decision writes) —
then runs the orchestrator and asserts it loads them, counts them so the method
moves to SFT, folds them into the training set, and reports used_synthetic. The
LoRA train step is stubbed (the loop closure, not training, is under test).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

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
async def test_resumed_run_trains_on_approved_synthetic(
    phoenix_container, telemetry_manager
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewItem
    from cogniverse_finetuning.orchestrator import (
        FinetuningOrchestrator,
        OrchestrationConfig,
    )
    from cogniverse_finetuning.training.backend import LocalTrainingBackend

    tenant_id = "orch_resume"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )
    with telemetry_manager.span(
        name="entity_extraction_agent",
        tenant_id=tenant_id,
        project_name=project_name,
        attributes={"input.query": "seed"},
    ):
        pass
    telemetry_manager.force_flush(timeout_millis=10000)

    # Prior run's approved synthetic, persisted to the training dataset.
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
    )
    items = [
        ReviewItem(
            item_id=f"syn_{i}",
            data={
                "query": f"Company{i} was founded by Person{i}",
                "entities": [
                    {"text": f"Company{i}", "type": "ORG"},
                    {"text": f"Person{i}", "type": "PERSON"},
                ],
                "entity_types": "ORG,PERSON",
                "relationships": [],
            },
            confidence=0.9,
            metadata={"agent_type": "entity_extraction"},
            status=ApprovalStatus.APPROVED,
        )
        for i in range(50)
    ]
    assert await storage.append_to_training_dataset("approved_synthetic_data", items)

    orchestrator = FinetuningOrchestrator(telemetry_provider=storage.provider)
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=full_project,
        model_type="llm",
        agent_type="entity_extraction",
        min_sft_examples=50,
        generate_synthetic=False,
        backend="local",
        enable_registry=False,
        evaluate_after_training=False,
    )

    captured = {}

    async def _fake_train_sft(self, dataset, base_model, output_dir, config):
        captured["dataset"] = dataset
        return SimpleNamespace(adapter_path="/tmp/stub_adapter", metrics={"loss": 0.0})

    with patch.object(LocalTrainingBackend, "train_sft", _fake_train_sft):
        result = await orchestrator.run(config)

    # The loop closed: approved synthetic counted -> SFT, folded into the set.
    assert result.training_method == "sft"
    assert result.used_synthetic is True
    assert len(captured["dataset"]) == 50
    text = captured["dataset"][0]["text"]
    assert "### Instruction:\nExtract entities and relationships" in text
    assert captured["dataset"][0]["metadata"]["synthetic"] is True
