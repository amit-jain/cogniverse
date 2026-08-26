"""Real-Phoenix: a resumed run trains on approved synthetic data.

Simulates the post-approval state — approved synthetic examples persisted in a
tenant-qualified dataset — then runs the orchestrator and asserts it loads them,
counts them so the method moves to SFT, folds them into the training set, and
reports used_synthetic. The LoRA train step is stubbed because the approval loop
closure, rather than model training, is under test.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def approval_redis_url():
    port = _free_port()
    container_name = f"cogniverse-finetuning-resume-redis-{os.getpid()}-{port}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start Redis: {result.stderr}")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", container_name, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        pytest.fail("Redis did not become ready within 30 seconds")

    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


@pytest.fixture
def telemetry_manager(phoenix_container):
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    TelemetryManager.reset()
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
    try:
        yield manager
    finally:
        TelemetryManager.reset()


@pytest.mark.asyncio
async def test_resumed_run_trains_on_approved_synthetic(
    phoenix_container, telemetry_manager, approval_redis_url
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_core.approval.interfaces import (
        ApprovalStatus,
        ReviewItem,
        approved_synthetic_dataset_name,
    )
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
        redis_url=approval_redis_url,
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
                "relationships": [],
            },
            confidence=0.9,
            metadata={"agent_type": "entity_extraction"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        )
        for i in range(50)
    ]
    assert await storage.append_to_training_dataset(
        approved_synthetic_dataset_name(tenant_id), items
    )

    orchestrator = FinetuningOrchestrator(
        telemetry_provider=storage.provider,
        telemetry_manager=telemetry_manager,
    )
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

    assert result.training_method == "sft"
    assert result.used_synthetic is True
    assert result.synthetic_approval_count == 50
    assert result.adapter_path == "/tmp/stub_adapter"
    assert result.metrics == {"loss": 0.0}
    assert captured["dataset"] == [
        {
            "text": (
                "### Instruction:\nExtract entities and relationships from the "
                "following text.\n\n"
                f"### Input:\nCompany{i} was founded by Person{i}\n\n"
                "### Response:\n"
                + json.dumps(
                    {
                        "entities": [
                            {"text": f"Company{i}", "type": "ORG"},
                            {"text": f"Person{i}", "type": "PERSON"},
                        ],
                        "relationships": [],
                    },
                    separators=(",", ":"),
                )
            ),
            "metadata": {
                "synthetic": True,
                "agent_type": "entity_extraction",
            },
        }
        for i in range(50)
    ]

    experiments_project = telemetry_manager.config.get_project_name(
        tenant_id, "experiments"
    )
    experiments_provider = telemetry_manager.get_provider(
        tenant_id=tenant_id,
        project_name="experiments",
    )
    experiment = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        spans = await experiments_provider.traces.get_spans(
            project=experiments_project,
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=100,
        )
        if spans is not None and not spans.empty and "name" in spans.columns:
            matches = spans[spans["name"] == "experiment.entity_extraction.sft"]
            if len(matches) == 1:
                experiment = matches.iloc[0]
                break
        await asyncio.sleep(1)

    assert experiment is not None
    assert experiment["attributes.params"]["method"] == "sft"
    assert experiment["attributes.data"]["dataset_size"] == 50
    assert experiment["attributes.data"]["synthetic_approved_count"] == 50
    assert experiment["attributes.output"]["adapter_path"] == "/tmp/stub_adapter"
    assert experiment["attributes.tenant"]["id"] == tenant_id


@pytest.mark.asyncio
async def test_auto_approved_routing_trains_from_persisted_canonical_dataset(
    phoenix_container, telemetry_manager, approval_redis_url
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
    from cogniverse_core.approval.interfaces import (
        ApprovalStatus,
        approved_synthetic_dataset_name,
    )
    from cogniverse_finetuning.dataset.method_selector import TrainingMethodSelector
    from cogniverse_finetuning.orchestrator import (
        FinetuningOrchestrator,
        OrchestrationConfig,
    )
    from cogniverse_finetuning.training.backend import LocalTrainingBackend
    from cogniverse_synthetic.approval.confidence_extractor import (
        SyntheticDataConfidenceExtractor,
    )

    tenant_id = "orch_auto_route"
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
        name="gateway_agent",
        tenant_id=tenant_id,
        project_name=project_name,
        attributes={"input.query": "seed"},
    ):
        pass
    telemetry_manager.force_flush(timeout_millis=10000)

    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    generated_at = datetime(2026, 8, 5, 5, 30, tzinfo=timezone.utc)
    generated = {
        "query": "Find the launch keynote video",
        "entities": [{"text": "launch keynote", "type": "EVENT"}],
        "relationships": [],
        "enhanced_query": "Find the launch keynote(EVENT) video",
        "chosen_agent": "video_search_agent",
        "routing_confidence": 0.93,
        "search_quality": 0.0,
        "agent_success": False,
        "user_satisfaction": None,
        "processing_time": 0.0,
        "reward": None,
        "timestamp": generated_at,
        "metadata": {
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": {
                    "routing_confidence": "observed_gateway_confidence",
                    "search_quality": "unobserved_zero_sentinel",
                    "agent_success": "unobserved_false_sentinel",
                    "processing_time": "unobserved_zero_sentinel",
                },
            }
        },
    }

    class _SyntheticService:
        def __init__(self):
            self.requests = []

        async def generate(self, request):
            self.requests.append(request)
            return SimpleNamespace(count=1, data=[generated])

    synthetic_service = _SyntheticService()
    approval_agent = HumanApprovalAgent(
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        confidence_threshold=0.85,
        storage=storage,
    )
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=storage.provider,
        telemetry_manager=telemetry_manager,
        synthetic_service=synthetic_service,
        approval_agent=approval_agent,
    )
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=full_project,
        model_type="llm",
        agent_type="routing",
        min_sft_examples=1,
        min_dpo_pairs=1,
        generate_synthetic=True,
        backend="local",
        enable_registry=False,
        evaluate_after_training=False,
    )
    captured = {}

    async def _fake_train_sft(self, dataset, base_model, output_dir, config):
        captured["dataset"] = dataset
        return SimpleNamespace(
            adapter_path="/tmp/routing_auto_adapter",
            metrics={"loss": 0.125},
        )

    with patch.object(LocalTrainingBackend, "train_sft", _fake_train_sft):
        result = await orchestrator.run(config)

    assert len(synthetic_service.requests) == 1
    request = synthetic_service.requests[0]
    assert request.optimizer == "routing"
    assert request.count == 1
    assert request.tenant_id == "orch_auto_route:orch_auto_route"
    assert result.training_method == "sft"
    assert result.used_synthetic is True
    assert result.synthetic_approval_count == 1
    assert result.adapter_path == "/tmp/routing_auto_adapter"
    assert result.metrics == {"loss": 0.125}
    assert captured["dataset"] == [
        {
            "text": (
                "### Instruction:\nRoute the following query to the appropriate "
                "modality agent.\n\n### Input:\nFind the launch keynote video\n\n"
                '### Response:\n{"recommended_agent":"video_search_agent"}'
            ),
            "metadata": {"synthetic": True, "agent_type": "routing"},
        }
    ]

    retry_batch = await TrainingMethodSelector(
        synthetic_service=synthetic_service,
        approval_agent=approval_agent,
    )._generate_and_approve_synthetic(
        agent_type="routing",
        num_needed=1,
        tenant_id=tenant_id,
    )
    assert len(synthetic_service.requests) == 2
    retry_request = synthetic_service.requests[1]
    assert retry_request.optimizer == "routing"
    assert retry_request.count == 1
    assert retry_request.tenant_id == "orch_auto_route:orch_auto_route"
    assert retry_batch.approved_count == 1
    assert retry_batch.pending_review == []

    canonical_batch = await storage.get_batch(retry_batch.batch_id)
    assert canonical_batch is not None
    assert canonical_batch.batch_id == retry_batch.batch_id
    assert canonical_batch.created_at == retry_batch.created_at
    assert len(canonical_batch.items) == 1
    assert canonical_batch.items[0].item_id == retry_batch.items[0].item_id
    assert canonical_batch.items[0].created_at == retry_batch.items[0].created_at
    assert canonical_batch.items[0].status is ApprovalStatus.AUTO_APPROVED
    assert canonical_batch.items[0].data["query"] == generated["query"]
    assert canonical_batch.items[0].data["chosen_agent"] == generated["chosen_agent"]

    persisted = await orchestrator._load_approved_synthetic(config)
    assert persisted == [
        {
            "query": "Find the launch keynote video",
            "entities": [{"text": "launch keynote", "type": "EVENT"}],
            "relationships": [],
            "enhanced_query": "Find the launch keynote(EVENT) video",
            "chosen_agent": "video_search_agent",
            "routing_confidence": 0.93,
            "search_quality": 0.0,
            "agent_success": False,
            "processing_time": 0.0,
            "timestamp": generated_at.isoformat(),
            "metadata": generated["metadata"],
        }
    ]
    dataset = await storage.provider.datasets.get_dataset(
        name=approved_synthetic_dataset_name(tenant_id)
    )
    assert len(dataset) == 1
    assert dataset.iloc[0]["input"]["status"] == "approved"
    assert dataset.iloc[0]["input"]["metadata.agent_type"] == "routing"


def test_telemetry_manager_fixture_leaves_no_global_state(phoenix_container):
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    TelemetryManager.reset()
    fixture = telemetry_manager.__wrapped__(phoenix_container)
    manager = next(fixture)
    assert TelemetryManager._instance is manager
    assert telemetry_manager_module._telemetry_manager is manager

    with pytest.raises(StopIteration):
        next(fixture)

    assert TelemetryManager._instance is None
    assert telemetry_manager_module._telemetry_manager is None
