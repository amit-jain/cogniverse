"""Pin tenant-qualified approved synthetic reads against real Phoenix.

Two tenants write distinct datasets, load concurrently through the production
finetuning consumer, survive a real Phoenix outage, and recover without reading
the obsolete shared dataset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

pytestmark = pytest.mark.integration


def test_reader_rejects_approved_record_without_agent_type():
    from cogniverse_finetuning.dataset.synthetic_reader import (
        load_approved_synthetic_examples,
    )

    frame = pd.DataFrame(
        [
            {
                "input": {
                    "status": "approved",
                    "query": "find the launch video",
                    "chosen_agent": "video_search",
                }
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "approved dataset record at position 0 requires a non-empty "
            "metadata.agent_type string"
        ),
    ):
        load_approved_synthetic_examples(frame, "routing")


@pytest.mark.parametrize(
    ("record", "agent_type", "message"),
    [
        (
            "not-a-dictionary",
            "routing",
            "approved dataset row at position 0 must contain an input dictionary",
        ),
        (
            {
                "status": "pending_review",
                "metadata.agent_type": "routing",
                "query": "find the launch video",
                "chosen_agent": "video_search",
            },
            "routing",
            "approved dataset record at position 0 requires status 'approved'",
        ),
        (
            {
                "status": "approved",
                "metadata.agent_type": "unsupported",
                "query": "find the launch video",
            },
            "routing",
            "approved dataset record at position 0 has unsupported agent_type 'unsupported'",
        ),
        (
            {
                "status": "approved",
                "metadata.agent_type": "routing",
                "query": " ",
                "chosen_agent": "video_search",
            },
            "routing",
            "approved routing record at position 0 requires a non-empty query string",
        ),
        (
            {
                "status": "approved",
                "metadata.agent_type": "routing",
                "query": "find the launch video",
            },
            "routing",
            "approved routing record at position 0 requires a non-empty chosen_agent string",
        ),
        (
            {
                "status": "approved",
                "metadata.agent_type": "profile_selection",
                "query": "find the launch video",
                "available_profiles": "video_search,document_search",
                "selected_profile": "",
                "reasoning": "The query asks for a video.",
                "query_intent": "video_retrieval",
                "modality": "video",
                "complexity": "simple",
            },
            "profile_selection",
            (
                "approved profile_selection record at position 0 requires a "
                "non-empty selected_profile string"
            ),
        ),
        (
            {
                "status": "approved",
                "metadata.agent_type": "entity_extraction",
                "query": "PyTorch was released by Meta AI",
                "entities": "not-a-list",
                "relationships": "[]",
            },
            "entity_extraction",
            (
                "approved entity_extraction record at position 0 entities must be "
                "a non-empty list"
            ),
        ),
    ],
)
def test_reader_rejects_records_outside_canonical_agent_schema(
    record, agent_type, message
):
    from cogniverse_finetuning.dataset.synthetic_reader import (
        load_approved_synthetic_examples,
    )

    with pytest.raises(ValueError, match=message):
        load_approved_synthetic_examples(
            pd.DataFrame([{"input": record}]),
            agent_type,
        )


@pytest.mark.parametrize(
    ("example", "agent_type", "message"),
    [
        (
            {"query": "", "chosen_agent": "video_search"},
            "routing",
            "synthetic routing example at position 0 requires a non-empty query string",
        ),
        (
            {"query": "find the launch video"},
            "routing",
            (
                "synthetic routing example at position 0 requires a non-empty "
                "chosen_agent string"
            ),
        ),
        (
            {
                "query": "find the launch video",
                "available_profiles": "video_search,document_search",
                "selected_profile": "",
                "reasoning": "The query asks for a video.",
                "query_intent": "video_retrieval",
                "modality": "video",
                "complexity": "simple",
            },
            "profile_selection",
            (
                "synthetic profile_selection example at position 0 requires a "
                "non-empty selected_profile string"
            ),
        ),
    ],
)
def test_formatter_rejects_examples_that_cannot_produce_training_records(
    example, agent_type, message
):
    from cogniverse_finetuning.dataset.synthetic_reader import format_synthetic_sft

    with pytest.raises(ValueError, match=message):
        format_synthetic_sft([example], agent_type)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _signed_approved_record(item_id: str, record: dict) -> dict:
    reviewed_at = "2026-08-05T00:00:00+00:00"
    signed = {
        "item_id": item_id,
        "confidence": 0.9,
        "created_at": "2026-08-04T00:00:00+00:00",
        "reviewed_at": reviewed_at,
        **record,
    }
    decision = signed.get("metadata.decision")
    decision_intent = dict(decision) if isinstance(decision, dict) else decision
    if isinstance(decision_intent, dict):
        decision_intent.pop("timestamp", None)
    identity = {
        "item_id": item_id,
        "status": signed.get("status"),
        "decision": decision_intent,
    }
    identity_json = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    signed["metadata.approval_decision_sha256"] = hashlib.sha256(
        identity_json.encode("utf-8")
    ).hexdigest()
    signed["metadata.approval_decision_timestamp"] = reviewed_at
    canonical_json = json.dumps(
        signed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    signed["metadata.approval_record_json"] = canonical_json
    signed["metadata.approval_record_sha256"] = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return signed


@pytest.fixture(scope="module")
def approval_redis_url():
    port = _free_port()
    container_name = f"cogniverse-finetuning-approved-redis-{os.getpid()}-{port}"
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


@pytest.mark.parametrize(
    ("item_id", "metadata", "data", "cause_message"),
    [
        (
            "missing_agent",
            {},
            {"query": "find the launch video", "chosen_agent": "video_search"},
            (
                "Training dataset item 'missing_agent' requires a non-empty "
                "metadata.agent_type string"
            ),
        ),
        (
            "blank_agent",
            {"agent_type": " "},
            {"query": "find the launch video", "chosen_agent": "video_search"},
            (
                "Training dataset item 'blank_agent' requires a non-empty "
                "metadata.agent_type string"
            ),
        ),
        (
            "unsupported_agent",
            {"agent_type": "unsupported"},
            {"query": "find the launch video"},
            (
                "Training dataset item 'unsupported_agent' has unsupported "
                "metadata.agent_type 'unsupported'"
            ),
        ),
        (
            "routing_without_output",
            {"agent_type": "routing"},
            {"query": "find the launch video"},
            (
                "Training dataset item 'routing_without_output' requires a "
                "non-empty chosen_agent string"
            ),
        ),
        (
            "profile_without_output",
            {"agent_type": "profile_selection"},
            {
                "query": "find the launch video",
                "available_profiles": "video_search,document_search",
                "selected_profile": "",
                "reasoning": "The query asks for a video.",
                "query_intent": "video_retrieval",
                "modality": "video",
                "complexity": "simple",
            },
            (
                "Training dataset item 'profile_without_output' requires a "
                "non-empty selected_profile string"
            ),
        ),
        (
            "entity_without_lists",
            {"agent_type": "entity_extraction"},
            {
                "query": "PyTorch was released by Meta AI",
                "entities": "[]",
                "relationships": [],
            },
            (
                "Training dataset item 'entity_without_lists' entities must be a "
                "non-empty list"
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_producer_rejects_records_outside_canonical_agent_schema(
    phoenix_container,
    telemetry_manager,
    approval_redis_url,
    item_id,
    metadata,
    data,
    cause_message,
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewItem
    from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

    tenant_id = f"invalid:{item_id}"
    dataset_name = f"approved_synthetic_data-{tenant_id}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    item = ReviewItem(
        item_id=item_id,
        data=data,
        confidence=0.95,
        metadata=metadata,
        status=ApprovalStatus.APPROVED,
        reviewed_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )

    with pytest.raises(RuntimeError) as error:
        await storage.append_to_training_dataset(dataset_name, [item])

    assert str(error.value) == (
        "Failed to append items to training dataset: "
        f"tenant={tenant_id} dataset={dataset_name}"
    )
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == cause_message
    with pytest.raises(DatasetNotFoundError):
        await storage.provider.datasets.get_dataset(name=dataset_name)


@pytest.mark.asyncio
async def test_real_approved_dataset_rejects_repeated_query_across_batches(
    phoenix_container,
    telemetry_manager,
    approval_redis_url,
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewItem
    from cogniverse_finetuning.dataset.synthetic_reader import (
        load_approved_synthetic_examples,
    )

    tenant_id = "duplicate:query"
    dataset_name = f"approved_synthetic_data-{tenant_id}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    reviewed_at = datetime(2026, 8, 5, tzinfo=timezone.utc)
    items = [
        ReviewItem(
            item_id="batch-one-route",
            data={
                "query": "find sunset videos",
                "chosen_agent": "search_agent",
            },
            confidence=0.96,
            metadata={"agent_type": "routing", "synthetic": True},
            status=ApprovalStatus.APPROVED,
            reviewed_at=reviewed_at,
        ),
        ReviewItem(
            item_id="batch-two-route",
            data={
                "query": "find sunset videos",
                "chosen_agent": "document_agent",
            },
            confidence=0.91,
            metadata={"agent_type": "routing", "synthetic": True},
            status=ApprovalStatus.APPROVED,
            reviewed_at=reviewed_at,
        ),
    ]

    assert await storage.append_to_training_dataset(dataset_name, items) is True
    frame = await storage.provider.datasets.get_dataset(name=dataset_name)
    assert [row["input"]["item_id"] for _, row in frame.iterrows()] == [
        "batch-one-route",
        "batch-two-route",
    ]

    with pytest.raises(ValueError) as error:
        load_approved_synthetic_examples(frame, "routing")

    assert str(error.value) == (
        "approved routing dataset contains duplicate canonical query "
        "'find sunset videos' at positions 0 and 1"
    )


@pytest.mark.parametrize(
    ("tenant_id", "record", "cause_message"),
    [
        (
            "corrupt:status",
            {
                "status": "pending_review",
                "metadata.agent_type": "routing",
                "query": "find the launch video",
                "chosen_agent": "video_search",
            },
            "approved dataset record at position 0 requires status 'approved'",
        ),
        (
            "corrupt:profile",
            {
                "status": "approved",
                "metadata.agent_type": "profile_selection",
                "query": "find the launch video",
                "available_profiles": "video_search,document_search",
                "selected_profile": "",
                "reasoning": "The query asks for a video.",
                "query_intent": "video_retrieval",
                "modality": "video",
                "complexity": "simple",
            },
            (
                "approved profile_selection record at position 0 requires a "
                "non-empty selected_profile string"
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_orchestrator_rejects_any_malformed_row_with_dataset_context(
    phoenix_container,
    telemetry_manager,
    approval_redis_url,
    tenant_id,
    record,
    cause_message,
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_finetuning.orchestrator import (
        FinetuningOrchestrator,
        OrchestrationConfig,
    )

    dataset_name = f"approved_synthetic_data-{tenant_id}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    await storage.provider.datasets.create_dataset(
        name=dataset_name,
        data=pd.DataFrame([_signed_approved_record("malformed-schema", record)]),
    )
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=storage.provider,
        telemetry_manager=telemetry_manager,
    )
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=f"cogniverse-{tenant_id}-finetuning",
        model_type="llm",
        agent_type="routing",
    )

    with pytest.raises(RuntimeError) as error:
        await orchestrator._load_approved_synthetic(config)

    assert str(error.value) == (
        "Malformed approved synthetic dataset for "
        f"tenant={tenant_id} agent_type=routing dataset={dataset_name}"
    )
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == cause_message


@pytest.mark.asyncio
async def test_real_phoenix_malformed_integrity_is_rejected_before_consumption(
    phoenix_container,
    telemetry_manager,
    approval_redis_url,
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_finetuning.orchestrator import (
        FinetuningOrchestrator,
        OrchestrationConfig,
    )

    tenant_id = f"integrity:missing-{time.time_ns()}"
    dataset_name = f"approved_synthetic_data-{tenant_id}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    dataset_store = storage.provider.datasets
    assert hasattr(dataset_store, "_delegate")
    await dataset_store._delegate.create_dataset(
        name=dataset_name,
        data=pd.DataFrame(
            [
                {
                    "item_id": "malformed-item",
                    "status": "approved",
                    "metadata.agent_type": "routing",
                    "query": "find the launch video",
                    "chosen_agent": "video_search",
                }
            ]
        ),
    )
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=storage.provider,
        telemetry_manager=telemetry_manager,
    )
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=f"cogniverse-{tenant_id}-finetuning",
        model_type="llm",
        agent_type="routing",
    )

    with pytest.raises(RuntimeError) as error:
        await orchestrator._load_approved_synthetic(config)

    assert str(error.value) == (
        "Failed to load approved synthetic dataset for "
        f"tenant={tenant_id} agent_type=routing dataset={dataset_name}"
    )
    integrity_error = error.value.__cause__
    assert isinstance(integrity_error, RuntimeError)
    assert str(integrity_error) == (
        "Approved dataset item has invalid metadata.approval_record_json: "
        f"tenant={tenant_id} dataset={dataset_name} row=0 item=malformed-item"
    )


@pytest.mark.asyncio
async def test_fresh_provider_rejects_tampered_approved_record_before_consumption(
    phoenix_container,
    telemetry_manager,
    approval_redis_url,
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_finetuning.orchestrator import (
        FinetuningOrchestrator,
        OrchestrationConfig,
    )

    tenant_id = f"integrity:fresh-{time.time_ns()}"
    dataset_name = f"approved_synthetic_data-{tenant_id}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    tampered = _signed_approved_record(
        "tampered-item",
        {
            "status": "approved",
            "metadata.agent_type": "routing",
            "query": "find the launch video",
            "chosen_agent": "video_search",
        },
    )
    tampered["chosen_agent"] = "wrong_agent"
    await storage.provider.datasets._delegate.create_dataset(
        name=dataset_name,
        data=pd.DataFrame([tampered]),
    )

    fresh_provider = telemetry_manager.get_provider(
        tenant_id=tenant_id,
        project_name="finetuning",
    )
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=fresh_provider,
        telemetry_manager=telemetry_manager,
    )
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=f"cogniverse-{tenant_id}-finetuning",
        model_type="llm",
        agent_type="routing",
    )

    with pytest.raises(RuntimeError) as error:
        await orchestrator._load_approved_synthetic(config)

    assert str(error.value) == (
        "Malformed approved synthetic dataset for "
        f"tenant={tenant_id} agent_type=routing dataset={dataset_name}"
    )
    integrity_error = error.value.__cause__
    assert isinstance(integrity_error, RuntimeError)
    assert str(integrity_error) == (
        "Approved dataset item content differs from canonical content: "
        f"tenant={tenant_id} dataset={dataset_name} row=0 item=tampered-item "
        "missing=[] unexpected=[] mismatched=['chosen_agent']"
    )


@pytest.mark.asyncio
async def test_approved_synthetic_dataset_roundtrip(
    phoenix_container, telemetry_manager, approval_redis_url
):
    from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
    from cogniverse_core.approval.interfaces import (
        ApprovalStatus,
        ReviewItem,
        approved_synthetic_dataset_name,
    )
    from cogniverse_finetuning.dataset.synthetic_reader import format_synthetic_sft
    from cogniverse_finetuning.orchestrator import (
        FinetuningOrchestrator,
        OrchestrationConfig,
        analyze_dataset_status,
    )

    tenants = ("acme:alpha", "acme:beta")
    examples = {
        "acme:alpha": [
            {
                "query": "PyTorch was released by Meta AI",
                "entities": [
                    {"text": "PyTorch", "type": "TECHNOLOGY"},
                    {"text": "Meta AI", "type": "ORG"},
                ],
                "relationships": [
                    {
                        "source": "PyTorch",
                        "target": "Meta AI",
                        "type": "RELEASED_BY",
                    }
                ],
            },
            {
                "query": "JAX was developed by Google",
                "entities": [
                    {"text": "JAX", "type": "TECHNOLOGY"},
                    {"text": "Google", "type": "ORG"},
                ],
                "relationships": [
                    {
                        "source": "JAX",
                        "target": "Google",
                        "type": "DEVELOPED_BY",
                    }
                ],
            },
        ],
        "acme:beta": [
            {
                "query": "Vespa was created by Yahoo",
                "entities": [
                    {"text": "Vespa", "type": "TECHNOLOGY"},
                    {"text": "Yahoo", "type": "ORG"},
                ],
                "relationships": [
                    {
                        "source": "Vespa",
                        "target": "Yahoo",
                        "type": "CREATED_BY",
                    }
                ],
            },
            {
                "query": "Phoenix is maintained by Arize",
                "entities": [
                    {"text": "Phoenix", "type": "TECHNOLOGY"},
                    {"text": "Arize", "type": "ORG"},
                ],
                "relationships": [
                    {
                        "source": "Phoenix",
                        "target": "Arize",
                        "type": "MAINTAINED_BY",
                    }
                ],
            },
        ],
    }
    storages = {
        tenant: ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id=tenant,
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        for tenant in tenants
    }

    async def write_tenant(tenant):
        items = [
            ReviewItem(
                item_id=f"{tenant.replace(':', '_')}_{index}",
                data=example,
                confidence=0.9 + index / 100,
                metadata={"agent_type": "entity_extraction", "synthetic": True},
                status=ApprovalStatus.APPROVED,
                reviewed_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
            )
            for index, example in enumerate(examples[tenant])
        ]
        return await storages[tenant].append_to_training_dataset(
            approved_synthetic_dataset_name(tenant),
            items,
        )

    assert await asyncio.gather(*(write_tenant(tenant) for tenant in tenants)) == [
        True,
        True,
    ]

    legacy_record = _signed_approved_record(
        "legacy_shared_0",
        {
            "status": "approved",
            "metadata.agent_type": "entity_extraction",
            "query": "This obsolete shared row must never be consumed",
            "entities": [{"text": "obsolete", "type": "CONCEPT"}],
            "relationships": [],
        },
    )
    await storages["acme:alpha"].provider.datasets._delegate.create_dataset(
        name="approved_synthetic_data",
        data=pd.DataFrame([legacy_record]),
    )

    schema_tenant = "acme:schema"
    schema_storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=schema_tenant,
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )
    schema_created_at = datetime(2026, 8, 4, tzinfo=timezone.utc)
    schema_reviewed_at = datetime(2026, 8, 5, tzinfo=timezone.utc)
    assert await schema_storage.append_to_training_dataset(
        approved_synthetic_dataset_name(schema_tenant),
        [
            ReviewItem(
                item_id="schema_routing",
                data={
                    "query": "find the product launch video",
                    "chosen_agent": "video_search",
                },
                confidence=0.97,
                metadata={"agent_type": "routing", "synthetic": True},
                status=ApprovalStatus.APPROVED,
                created_at=schema_created_at,
                reviewed_at=schema_reviewed_at,
            ),
            ReviewItem(
                item_id="schema_profile",
                data={
                    "query": "find exact text in presentation slides",
                    "available_profiles": (
                        "video_colpali_smol500_mv_frame,video_search"
                    ),
                    "selected_profile": "video_colpali_smol500_mv_frame",
                    "reasoning": "Exact slide text requires frame retrieval.",
                    "query_intent": "exact_text_retrieval",
                    "modality": "video",
                    "complexity": "medium",
                },
                confidence=0.96,
                metadata={"agent_type": "profile_selection", "synthetic": True},
                status=ApprovalStatus.APPROVED,
                created_at=schema_created_at,
                reviewed_at=schema_reviewed_at,
            ),
        ],
    )

    def config(tenant, agent_type="entity_extraction"):
        return OrchestrationConfig(
            tenant_id=tenant,
            project=f"cogniverse-{tenant}-finetuning",
            model_type="llm",
            agent_type=agent_type,
        )

    class OverlapDatasetStore:
        def __init__(self, delegate):
            self._delegate = delegate
            self._barrier = asyncio.Barrier(2)
            self.calls = Counter()
            self.active = 0
            self.peak_active = 0

        async def get_dataset(self, name):
            self.calls[name] += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                await asyncio.wait_for(self._barrier.wait(), timeout=5)
                return await self._delegate.get_dataset(name=name)
            finally:
                self.active -= 1

    overlap_store = OverlapDatasetStore(storages["acme:alpha"].provider.datasets)
    concurrent_orchestrators = {
        tenant: FinetuningOrchestrator(
            telemetry_provider=SimpleNamespace(datasets=overlap_store),
            telemetry_manager=telemetry_manager,
        )
        for tenant in tenants
    }
    alpha, beta = await asyncio.gather(
        concurrent_orchestrators["acme:alpha"]._load_approved_synthetic(
            config("acme:alpha")
        ),
        concurrent_orchestrators["acme:beta"]._load_approved_synthetic(
            config("acme:beta")
        ),
    )
    assert alpha == examples["acme:alpha"]
    assert beta == examples["acme:beta"]
    assert overlap_store.calls == Counter(
        {
            "approved_synthetic_data-acme:alpha": 1,
            "approved_synthetic_data-acme:beta": 1,
        }
    )
    assert overlap_store.peak_active == 2
    assert overlap_store.active == 0

    same_dataset_store = OverlapDatasetStore(storages["acme:alpha"].provider.datasets)
    same_dataset_readers = [
        FinetuningOrchestrator(
            telemetry_provider=SimpleNamespace(datasets=same_dataset_store),
            telemetry_manager=telemetry_manager,
        )
        for _ in range(2)
    ]
    alpha_first, alpha_second = await asyncio.gather(
        *(
            reader._load_approved_synthetic(config("acme:alpha"))
            for reader in same_dataset_readers
        )
    )
    assert alpha_first == examples["acme:alpha"]
    assert alpha_second == examples["acme:alpha"]
    assert same_dataset_store.calls == Counter(
        {"approved_synthetic_data-acme:alpha": 2}
    )
    assert same_dataset_store.peak_active == 2
    assert same_dataset_store.active == 0
    alpha_first[0]["entities"][0]["text"] = "mutated reader copy"
    assert alpha_second == examples["acme:alpha"]

    orchestrators = {
        tenant: FinetuningOrchestrator(
            telemetry_provider=storages[tenant].provider,
            telemetry_manager=telemetry_manager,
        )
        for tenant in tenants
    }
    schema_orchestrator = FinetuningOrchestrator(
        telemetry_provider=schema_storage.provider,
        telemetry_manager=telemetry_manager,
    )
    routing = await schema_orchestrator._load_approved_synthetic(
        config(schema_tenant, "routing")
    )
    profile_selection = await schema_orchestrator._load_approved_synthetic(
        config(schema_tenant, "profile_selection")
    )
    assert routing == [
        {
            "query": "find the product launch video",
            "chosen_agent": "video_search",
        }
    ]
    assert profile_selection == [
        {
            "query": "find exact text in presentation slides",
            "available_profiles": "video_colpali_smol500_mv_frame,video_search",
            "selected_profile": "video_colpali_smol500_mv_frame",
            "reasoning": "Exact slide text requires frame retrieval.",
            "query_intent": "exact_text_retrieval",
            "modality": "video",
            "complexity": "medium",
        }
    ]
    status = await analyze_dataset_status(
        schema_storage.provider,
        project=f"cogniverse-{schema_tenant}-finetuning",
        tenant_id=schema_tenant,
        agent_type="routing",
        min_sft_examples=1,
        min_dpo_pairs=20,
    )
    assert {
        "total_spans": status["total_spans"],
        "approved_count": status["approved_count"],
        "preference_pairs": status["preference_pairs"],
        "sft_ready": status["sft_ready"],
        "dpo_ready": status["dpo_ready"],
        "recommended_method": status["recommended_method"],
        "needs_synthetic": status["needs_synthetic"],
    } == {
        "total_spans": 0,
        "approved_count": 1,
        "preference_pairs": 0,
        "sft_ready": True,
        "dpo_ready": False,
        "recommended_method": "sft",
        "needs_synthetic": False,
    }
    assert format_synthetic_sft(routing, "routing") == [
        {
            "text": (
                "### Instruction:\nRoute the following query to the appropriate "
                "modality agent.\n\n### Input:\nfind the product launch video\n\n"
                '### Response:\n{"recommended_agent":"video_search"}'
            ),
            "metadata": {"synthetic": True, "agent_type": "routing"},
        }
    ]
    assert format_synthetic_sft(profile_selection, "profile_selection") == [
        {
            "text": (
                "### Instruction:\nSelect the optimal backend profile(s) for the "
                "following query.\n\n### Input:\nfind exact text in presentation "
                "slides\n\n### Response:\n"
                '{"selected_profile":"video_colpali_smol500_mv_frame"}'
            ),
            "metadata": {
                "synthetic": True,
                "agent_type": "profile_selection",
            },
        }
    ]

    schema_frame = await schema_storage.provider.datasets.get_dataset(
        name="approved_synthetic_data-acme:schema"
    )
    serialized_records = [row["input"] for _, row in schema_frame.iterrows()]
    assert serialized_records == [
        {
            "item_id": "schema_routing",
            "confidence": 0.97,
            "status": "approved",
            "created_at": "2026-08-04T00:00:00+00:00",
            "reviewed_at": "2026-08-05T00:00:00+00:00",
            "query": "find the product launch video",
            "chosen_agent": "video_search",
            "metadata.agent_type": "routing",
            "metadata.synthetic": True,
            "metadata.approval_decision_sha256": (
                "a6a737faa8e098bc8221ec7d8a067b1a5d852ce50eb002a387c3f050e016806d"
            ),
            "metadata.approval_decision_timestamp": "2026-08-05T00:00:00+00:00",
            "metadata.approval_record_json": (
                '{"chosen_agent":"video_search","confidence":0.97,'
                '"created_at":"2026-08-04T00:00:00+00:00",'
                '"item_id":"schema_routing","metadata.agent_type":"routing",'
                '"metadata.approval_decision_sha256":'
                '"a6a737faa8e098bc8221ec7d8a067b1a5d852ce50eb002a387c3f050e016806d",'
                '"metadata.approval_decision_timestamp":'
                '"2026-08-05T00:00:00+00:00","metadata.synthetic":true,'
                '"query":"find the product launch video",'
                '"reviewed_at":"2026-08-05T00:00:00+00:00",'
                '"status":"approved"}'
            ),
            "metadata.approval_record_sha256": (
                "22eaffddcd48072a2628b450982366f93940f1a3540cb0c8a47861dd0bc13fff"
            ),
        },
        {
            "item_id": "schema_profile",
            "confidence": 0.96,
            "status": "approved",
            "created_at": "2026-08-04T00:00:00+00:00",
            "reviewed_at": "2026-08-05T00:00:00+00:00",
            "query": "find exact text in presentation slides",
            "available_profiles": "video_colpali_smol500_mv_frame,video_search",
            "selected_profile": "video_colpali_smol500_mv_frame",
            "reasoning": "Exact slide text requires frame retrieval.",
            "query_intent": "exact_text_retrieval",
            "modality": "video",
            "complexity": "medium",
            "metadata.agent_type": "profile_selection",
            "metadata.synthetic": True,
            "metadata.approval_decision_sha256": (
                "0c8aef7b0ab765903d56d091f2fa8cec58929b89418b7d4ec2f51f05b26d3a13"
            ),
            "metadata.approval_decision_timestamp": "2026-08-05T00:00:00+00:00",
            "metadata.approval_record_json": (
                '{"available_profiles":'
                '"video_colpali_smol500_mv_frame,video_search",'
                '"complexity":"medium","confidence":0.96,'
                '"created_at":"2026-08-04T00:00:00+00:00",'
                '"item_id":"schema_profile",'
                '"metadata.agent_type":"profile_selection",'
                '"metadata.approval_decision_sha256":'
                '"0c8aef7b0ab765903d56d091f2fa8cec58929b89418b7d4ec2f51f05b26d3a13",'
                '"metadata.approval_decision_timestamp":'
                '"2026-08-05T00:00:00+00:00","metadata.synthetic":true,'
                '"modality":"video",'
                '"query":"find exact text in presentation slides",'
                '"query_intent":"exact_text_retrieval",'
                '"reasoning":"Exact slide text requires frame retrieval.",'
                '"reviewed_at":"2026-08-05T00:00:00+00:00",'
                '"selected_profile":"video_colpali_smol500_mv_frame",'
                '"status":"approved"}'
            ),
            "metadata.approval_record_sha256": (
                "52724689c6ccd6fee1085555378f133da0305aab65645f415759ff365ce30d72"
            ),
        },
    ]

    dataset_store = storages["acme:alpha"].provider.datasets
    alpha_frame = await dataset_store.get_dataset(
        name="approved_synthetic_data-acme:alpha"
    )
    beta_frame = await dataset_store.get_dataset(
        name="approved_synthetic_data-acme:beta"
    )
    assert [row["input"]["query"] for _, row in alpha_frame.iterrows()] == [
        "PyTorch was released by Meta AI",
        "JAX was developed by Google",
    ]
    assert [row["input"]["query"] for _, row in beta_frame.iterrows()] == [
        "Vespa was created by Yahoo",
        "Phoenix is maintained by Arize",
    ]

    missing = await orchestrators["acme:alpha"]._load_approved_synthetic(
        config("acme:missing")
    )
    assert missing == []

    container_name = phoenix_container["container_name"]
    await asyncio.to_thread(
        subprocess.run,
        ["docker", "stop", container_name],
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to load approved synthetic dataset for "
                "tenant=acme:alpha agent_type=entity_extraction "
                "dataset=approved_synthetic_data-acme:alpha"
            ),
        ):
            await orchestrators["acme:alpha"]._load_approved_synthetic(
                config("acme:alpha")
            )
    finally:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "start", container_name],
            check=True,
            capture_output=True,
            timeout=30,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    phoenix_container["http_endpoint"],
                    timeout=2,
                )
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            await asyncio.sleep(1)
        else:
            pytest.fail("Phoenix did not recover after finetuning dataset outage")

    recovered = await orchestrators["acme:alpha"]._load_approved_synthetic(
        config("acme:alpha")
    )
    assert recovered == examples["acme:alpha"]
    assert format_synthetic_sft(recovered, "entity_extraction") == [
        {
            "text": (
                "### Instruction:\nExtract entities and relationships from the "
                f"following text.\n\n### Input:\n{example['query']}\n\n"
                "### Response:\n"
                + json.dumps(
                    {
                        "entities": example["entities"],
                        "relationships": example["relationships"],
                    },
                    separators=(",", ":"),
                )
            ),
            "metadata": {
                "synthetic": True,
                "agent_type": "entity_extraction",
            },
        }
        for example in examples["acme:alpha"]
    ]


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
