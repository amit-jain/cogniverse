"""
End-to-end integration tests for synthetic data generation with human-in-the-loop approval

Tests the complete flow:
1. Generate synthetic data with DSPy
2. Extract confidence scores
3. Auto-approve high confidence items
4. Queue low confidence items for review
5. Store in Phoenix
6. Process human decisions
7. Regenerate rejected items
8. Verify in Phoenix traces
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import redis.asyncio as aioredis
import requests
from phoenix.client import AsyncClient as PhoenixAsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.replacement_store import RedisReplacementRecordStore
from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)
from cogniverse_synthetic.approval.feedback_handler import SyntheticDataFeedbackHandler
from cogniverse_synthetic.dspy_modules import ValidatedSyntheticExampleRegenerator
from cogniverse_synthetic.schemas import (
    RoutingExperienceSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
from tests.utils.async_polling import wait_for_phoenix_processing
from tests.utils.vespa_test_helpers import make_config_manager

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.requires_lm, pytest.mark.local_only]


class _BoundTestRegenerator(ValidatedSyntheticExampleRegenerator):
    def __init__(self, forward):
        super().__init__(max_retries=3)
        self.lm = SimpleNamespace(model="test-lm")
        self._test_forward = forward

    def forward(self, **kwargs):
        return self._test_forward(**kwargs)


def _profile_selection_record(query: str) -> dict:
    return {
        "query": query,
        "available_profiles": "video_colpali,video_colqwen",
        "selected_profile": "video_colpali",
        "reasoning": "Patch retrieval preserves exact text.",
        "query_intent": "video_search",
        "modality": "video",
        "complexity": "medium",
    }


def _query_enhancement_record(query: str, framework: str) -> dict:
    return {
        "query": query,
        "enhanced_query": f"{query} {framework} framework",
        "expansion_terms": [framework, "framework"],
        "synonyms": ["tutorial", "guide"],
        "context": "machine learning documentation",
        "reasoning": f"The reviewed query names the {framework} framework exactly.",
    }


def _routing_record(query: str) -> dict:
    return {
        "query": query,
        "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
        "relationships": [],
        "enhanced_query": query,
        "chosen_agent": "search_agent",
        "routing_confidence": 0.84,
        "search_quality": 0.0,
        "agent_success": False,
        "user_satisfaction": None,
        "processing_time": 0.0,
        "reward": None,
        "timestamp": datetime(2026, 8, 5, tzinfo=timezone.utc),
        "metadata": {
            "source": "real feedback regeneration",
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": {
                    "routing_confidence": "observed_gateway_confidence",
                    "search_quality": "unobserved_zero_sentinel",
                    "agent_success": "unobserved_false_sentinel",
                    "processing_time": "unobserved_zero_sentinel",
                },
            },
        },
    }


def is_teacher_api_available() -> bool:
    """Check if router optimizer teacher API key is available."""
    return bool(os.getenv("ROUTER_OPTIMIZER_TEACHER_KEY"))


skip_if_no_teacher_api = pytest.mark.skipif(
    not is_teacher_api_available(),
    reason="ROUTER_OPTIMIZER_TEACHER_KEY environment variable not set",
)


@pytest.fixture
def telemetry_provider(telemetry_manager):
    """Telemetry provider for querying approval data via abstraction"""
    return telemetry_manager.get_provider(tenant_id="test-tenant1")


@pytest.fixture
def dspy_lm(dspy_test_lm):
    """Reuse the shared provider-agnostic LM fixture."""
    yield dspy_test_lm


@pytest.fixture
def telemetry_manager(phoenix_container):
    """TelemetryManager configured for approval tests"""
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    yield manager
    manager.shutdown()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def approval_redis_url():
    port = _free_port()
    container_name = f"cogniverse-approval-redis-{os.getpid()}"
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
def approval_storage(phoenix_container, telemetry_manager, approval_redis_url):
    """Approval storage with proper TelemetryManager integration"""
    return ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id="test-tenant1",
        telemetry_manager=telemetry_manager,
        redis_url=approval_redis_url,
    )


@pytest.fixture
def confidence_extractor():
    """Confidence extractor for synthetic data"""
    return SyntheticDataConfidenceExtractor()


@pytest.fixture
def feedback_handler(dspy_lm):
    """Feedback handler for synthetic data"""
    generator = ValidatedSyntheticExampleRegenerator(max_retries=3)
    generator.lm = dspy_lm
    return SyntheticDataFeedbackHandler(
        generator=generator,
        generation_timeout_seconds=120.0,
    )


@pytest.fixture
def approval_agent(approval_storage, confidence_extractor, feedback_handler):
    """Human approval agent"""
    return HumanApprovalAgent(
        storage=approval_storage,
        confidence_extractor=confidence_extractor,
        feedback_handler=feedback_handler,
        confidence_threshold=0.8,
    )


@pytest.fixture(scope="module")
def synthetic_service(shared_vespa):
    """Synthetic service grounded in one isolated real Vespa document."""
    tenant_id = f"syn{uuid.uuid4().hex[:8]}:review"
    base_schema = "video_colpali_smol500_mv_frame"
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        profiles={
            base_schema: BackendProfileConfig(
                profile_name=base_schema,
                type="video",
                description=(
                    "Frame-based ColPali profile with transcript and description "
                    "content for entity-rich sampling."
                ),
                schema_name=base_schema,
                embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
                pipeline_config={
                    "extract_keyframes": True,
                    "transcribe_audio": True,
                    "generate_descriptions": True,
                    "generate_embeddings": True,
                    "keyframe_strategy": "fps",
                    "keyframe_fps": 0.5,
                },
                embedding_type="multi_vector",
                model_loader="colpali",
                schema_config={
                    "schema_name": "video_colpali",
                    "model_name": "TomoroAI/tomoro-colqwen3-embed-4b",
                    "num_patches": 1024,
                    "embedding_dim": 320,
                    "binary_dim": 40,
                },
            )
        },
        tenant_id=tenant_id,
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant_id})
    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    schema = registry.deploy_schema(
        tenant_id=tenant_id,
        base_schema_name=base_schema,
    )
    documents = [
        {
            "data_id": "curie-radium-segment",
            "fields": {
                "video_id": "curie-radium",
                "video_title": "Marie Curie discovered radium",
                "source_url": "http://example.test/curie-radium",
                "segment_id": 0,
                "segment_description": (
                    "Marie Curie and Pierre Curie isolated radium in a Paris "
                    "laboratory."
                ),
                "audio_transcript": (
                    "Marie Curie and Pierre Curie isolated radium in a Paris "
                    "laboratory."
                ),
                "start_time": 0.0,
                "end_time": 12.0,
            },
        },
        {
            "data_id": "shoreline-segment",
            "fields": {
                "video_id": "shoreline",
                "video_title": "Quiet shoreline at dusk",
                "source_url": "http://example.test/shoreline",
                "segment_id": 1,
                "segment_description": (
                    "The frame shows a quiet shoreline with waves and sky."
                ),
                "audio_transcript": (
                    "The frame shows a quiet shoreline with waves and sky."
                ),
                "start_time": 12.0,
                "end_time": 24.0,
            },
        },
    ]
    app = make_vespa_app(url="http://localhost", port=shared_vespa["http_port"])
    for document in documents:
        feed = app.feed_data_point(
            schema=schema,
            data_id=document["data_id"],
            fields=document["fields"],
        )
        assert feed.is_successful(), feed.json

    expected_titles = {document["fields"]["video_title"] for document in documents}
    for _ in range(20):
        indexed = backend.query_metadata_documents(
            schema=schema,
            yql=f"select * from sources {schema} where true limit 2",
            hits=2,
        )
        indexed_titles = {item["video_title"] for item in indexed}
        if indexed_titles == expected_titles:
            break
        time.sleep(0.5)
    else:
        pytest.fail("Approval source documents were not indexed by Vespa")

    config_manager.set_backend_config(backend_config)

    raw_config = json.loads(Path("configs/config.json").read_text())
    synthetic_config = dict(raw_config["synthetic"])
    synthetic_config["tenant_id"] = tenant_id
    generator_config = SyntheticGeneratorConfig.from_dict(synthetic_config)
    entity_agent = EntityExtractionAgent(deps=EntityExtractionDeps())
    entity_agent.telemetry_manager = RecordingTelemetryManager()
    expected_entities = [
        ("Marie Curie", "PERSON"),
        ("Pierre Curie", "PERSON"),
    ]

    async def extract_entities(text: str, request_tenant_id: str):
        assert text == "Marie Curie and Pierre Curie isolated"
        assert request_tenant_id == tenant_id
        result = await entity_agent.process(
            EntityExtractionInput(query=text, tenant_id=request_tenant_id)
        )
        assert result.path_used == "dspy"
        assert [(entity.text, entity.type) for entity in result.entities] == (
            expected_entities
        )
        return result

    async def decide_route(query: str, request_tenant_id: str):
        assert request_tenant_id == tenant_id
        assert all(entity_text in query for entity_text, _ in expected_entities)
        return {
            "routed_to": "search_agent",
            "confidence": 0.94,
        }

    service = SyntheticDataService(
        backend=backend,
        generator_config=generator_config,
        backend_config=backend_config,
        agents_config=raw_config["agents"],
        entity_extractor=extract_entities,
        routing_decider=decide_route,
        config_manager=config_manager,
    )
    try:
        yield SimpleNamespace(
            service=service,
            tenant_id=tenant_id,
            base_schema=base_schema,
        )
    finally:
        backend.close()


@pytest.mark.integration
@pytest.mark.ci_fast
class TestSyntheticApprovalIntegration:
    """End-to-end integration tests for synthetic data approval workflow"""

    @pytest.mark.asyncio
    @pytest.mark.requires_lm
    async def test_end_to_end_approval_workflow(
        self, synthetic_service, approval_agent, approval_storage, dspy_lm
    ):
        """Generate, review, reject, regenerate, and reload one grounded item."""

        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=approval_agent.confidence_extractor,
            feedback_handler=approval_agent.feedback_handler,
            confidence_threshold=1.0,
        )

        request = SyntheticDataRequest(
            tenant_id=synthetic_service.tenant_id,
            optimizer="routing",
            count=1,
            vespa_sample_size=2,
            max_profiles=1,
        )
        response = await synthetic_service.service.generate(request)

        assert len(response.data) == 1
        assert response.optimizer == "routing"
        assert response.selected_profiles == [synthetic_service.base_schema]
        assert response.metadata["sampled_content_count"] == 2

        items = [dict(example) for example in response.data]

        batch_id = f"test_batch_{int(time.time())}"
        context = {
            "tenant_id": approval_storage.tenant_id,
            "agent_type": "routing",
            "optimizer": "routing",
            "purpose": "integration_test",
        }

        batch = await approval_agent.process_batch(items, batch_id, context)
        assert batch.batch_id == batch_id

        retrieved_batch = await approval_storage.get_batch(batch_id)
        assert retrieved_batch is not None
        original_ids = [item.item_id for item in retrieved_batch.items]
        assert original_ids == [f"{batch_id}_0"]

        decisions = [
            ReviewDecision(
                item_id=retrieved_batch.items[0].item_id,
                approved=False,
                feedback=(
                    "Use the exact corrected query and Marie Curie entity. Treat "
                    "radioactivity research as prompt guidance only."
                ),
                corrections={
                    "query": "find Marie Curie radioactivity research",
                    "entities": [{"text": "Marie Curie", "type": "PERSON"}],
                    "relationships": [],
                    "topics": ["radioactivity research"],
                },
                reviewer="test_user",
            ),
        ]

        results = [
            await approval_agent.apply_decision(batch_id, decision)
            for decision in decisions
        ]
        assert [result.status for result in results] == [ApprovalStatus.REGENERATED]
        replacement_id = f"{original_ids[0]}_regen_0"
        assert results[0].item_id == replacement_id
        assert results[0].data["entities"] == [
            {"text": "Marie Curie", "type": "PERSON"}
        ]
        assert results[0].data["query"] == "find Marie Curie radioactivity research"
        assert results[0].data["enhanced_query"] == (
            "find Marie Curie(PERSON) radioactivity research"
        )
        assert "topics" not in results[0].data

        wait_for_phoenix_processing(delay=2.0, description="annotation indexing")

        final_batch = await approval_storage.get_batch(batch_id)
        assert final_batch is not None
        final_by_id = {item.item_id: item for item in final_batch.items}
        assert set(final_by_id) == {*original_ids, replacement_id}
        assert {item_id: final_by_id[item_id].status for item_id in original_ids} == {
            original_ids[0]: ApprovalStatus.REJECTED,
        }
        assert final_by_id[replacement_id].status is ApprovalStatus.REGENERATED
        assert final_by_id[replacement_id].metadata["decision"] == {
            "reviewer": "test_user",
            "feedback": decisions[0].feedback,
            "corrections": {
                "query": "find Marie Curie radioactivity research",
                "entities": [{"text": "Marie Curie", "type": "PERSON"}],
                "relationships": [],
                "topics": ["radioactivity research"],
            },
            "timestamp": decisions[0].timestamp.isoformat(),
        }

    @pytest.mark.asyncio
    async def test_approved_replacement_is_loaded_for_optimization_exactly(
        self,
        approval_storage,
        confidence_extractor,
        approval_redis_url,
    ):
        """Redis-selected replacements reach optimization through Phoenix."""
        from cogniverse_runtime.optimization_cli import (
            _load_approved_synthetic_data,
        )

        suffix = time.time_ns()
        optimizer_type = "profile"
        batch_id = f"approved_optimizer_input_{suffix}"
        original = ReviewItem(
            item_id=f"{batch_id}_original",
            data=_profile_selection_record("Find a framework tutorial"),
            confidence=0.4,
            metadata={"agent_type": "profile_selection"},
        )
        second = ReviewItem(
            item_id=f"{batch_id}_second",
            data=_profile_selection_record("Find exact JAX tutorials"),
            confidence=0.88,
            metadata={"agent_type": "profile_selection"},
        )
        await approval_storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=[original, second],
                context={
                    "tenant_id": approval_storage.tenant_id,
                    "optimizer": optimizer_type,
                    "purpose": "optimizer training contract",
                },
            )
        )

        replacement = ReviewItem(
            item_id=f"{original.item_id}_regen_0",
            data=_profile_selection_record("Find exact PyTorch tutorials"),
            confidence=0.91,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "agent_type": "profile_selection",
                "decision": {
                    "reviewer": "optimizer-reviewer",
                    "feedback": "Name the framework exactly.",
                    "corrections": {"query": "Find exact PyTorch tutorials"},
                    "timestamp": "2026-08-05T01:00:00+00:00",
                },
            },
        )
        await approval_storage.replace_item(batch_id, original, replacement)

        redis = aioredis.from_url(approval_redis_url, decode_responses=True)
        try:
            selected_json = await redis.get(
                RedisReplacementRecordStore._key(
                    approval_storage.tenant_id,
                    batch_id,
                    original.item_id,
                )
            )
        finally:
            await redis.aclose()
        assert selected_json is not None
        selected_payload = json.loads(selected_json)
        expected_replacement_payload = {
            "item_id": replacement.item_id,
            "data": replacement.data,
            "confidence": replacement.confidence,
            "status": ApprovalStatus.REGENERATED.value,
            "metadata": replacement.metadata,
            "created_at": replacement.created_at.isoformat(),
            "reviewed_at": None,
        }
        assert selected_payload == expected_replacement_payload
        assert selected_json == json.dumps(
            expected_replacement_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        replacement_spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        matching_spans = replacement_spans[
            (replacement_spans["attributes.batch_id"] == batch_id)
            & (replacement_spans["attributes.original_item_id"] == original.item_id)
        ]
        assert len(matching_spans) == 1
        replacement_span = matching_spans.iloc[0]
        assert replacement_span["attributes.replacement_item_id"] == (
            replacement.item_id
        )
        assert replacement_span["attributes.replacement_record_json"] == selected_json
        assert replacement_span["attributes.replacement_record_sha256"] == (
            hashlib.sha256(selected_json.encode("utf-8")).hexdigest()
        )

        agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
        )
        pending = await agent.get_pending_items(
            {
                "optimizer": optimizer_type,
                "purpose": "optimizer training contract",
            }
        )
        assert [
            (item.item_id, item.data, item.status, item.metadata["approval_batch_id"])
            for item in pending
        ] == [
            (
                second.item_id,
                second.data,
                ApprovalStatus.PENDING_REVIEW,
                batch_id,
            ),
            (
                replacement.item_id,
                replacement.data,
                ApprovalStatus.REGENERATED,
                batch_id,
            ),
        ]

        approved_replacement = await agent.apply_decision(
            batch_id,
            ReviewDecision(
                item_id=replacement.item_id,
                approved=True,
                reviewer="optimizer-reviewer",
            ),
        )
        assert (
            approved_replacement.item_id,
            approved_replacement.data,
            approved_replacement.status,
        ) == (
            replacement.item_id,
            replacement.data,
            ApprovalStatus.APPROVED,
        )

        pending_after_replacement = await agent.get_pending_items(
            {
                "optimizer": optimizer_type,
                "purpose": "optimizer training contract",
            }
        )
        assert [
            (item.item_id, item.data, item.status) for item in pending_after_replacement
        ] == [(second.item_id, second.data, ApprovalStatus.PENDING_REVIEW)]

        approved_second = await agent.apply_decision(
            batch_id,
            ReviewDecision(
                item_id=second.item_id,
                approved=True,
                reviewer="optimizer-reviewer",
            ),
        )
        assert (
            approved_second.item_id,
            approved_second.data,
            approved_second.status,
        ) == (second.item_id, second.data, ApprovalStatus.APPROVED)

        expected = [
            {**replacement.data, "example_id": f"approved:{replacement.item_id}"},
            {**second.data, "example_id": f"approved:{second.item_id}"},
        ]
        first, second_load = await asyncio.gather(
            _load_approved_synthetic_data(
                approval_storage.provider,
                approval_storage.tenant_id,
                optimizer_type,
            ),
            _load_approved_synthetic_data(
                approval_storage.provider,
                approval_storage.tenant_id,
                optimizer_type,
            ),
        )
        assert first == expected
        assert second_load == expected

    @pytest.mark.asyncio
    async def test_approved_datasets_are_tenant_qualified_and_isolated(
        self,
        phoenix_container,
        telemetry_manager,
        confidence_extractor,
        approval_redis_url,
    ):
        from cogniverse_foundation.telemetry.providers.base import (
            DatasetNotFoundError,
        )
        from cogniverse_runtime.optimization_cli import (
            _load_approved_synthetic_data,
        )

        suffix = time.time_ns()
        optimizer_type = "query_enhancement"
        tenants = ("acme:alpha", "acme:beta")
        examples = {
            "acme:alpha": [
                _query_enhancement_record("Find exact tutorials", "PyTorch"),
                _query_enhancement_record("Find compiler tutorials", "JAX"),
            ],
            "acme:beta": [
                _query_enhancement_record("Find search tutorials", "Vespa"),
                _query_enhancement_record("Find tracing tutorials", "Phoenix"),
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

        missing_tenant_batch_id = f"missing_tenant_{suffix}"
        missing_tenant_item = ReviewItem(
            item_id=f"{missing_tenant_batch_id}_0",
            data={"query": "Never approve through a dataset-name hint"},
            confidence=0.9,
        )
        with pytest.raises(ValueError) as missing_tenant:
            await storages["acme:alpha"].save_batch(
                ApprovalBatch(
                    batch_id=missing_tenant_batch_id,
                    items=[missing_tenant_item],
                    context={
                        "optimizer": optimizer_type,
                        "dataset_name": "poisoned_context_dataset",
                    },
                )
            )
        assert str(missing_tenant.value) == (
            f"tenant_id is required on approval batch {missing_tenant_batch_id} "
            "context. The runtime no longer falls back to a bootstrap tenant for "
            "user requests — pass tenant_id explicitly in the request body or A2A "
            "metadata."
        )
        with pytest.raises(DatasetNotFoundError):
            await storages["acme:alpha"].provider.datasets.get_dataset(
                name="poisoned_context_dataset"
            )

        producer_barrier = asyncio.Barrier(len(tenants))

        async def approve_tenant_examples(tenant):
            batch_id = f"tenant_approved_{tenant.replace(':', '_')}_{suffix}"
            items = [
                ReviewItem(
                    item_id=f"{batch_id}_{index}",
                    data=example,
                    confidence=0.9 + index / 100,
                    metadata={"agent_type": "query_enhancement"},
                )
                for index, example in enumerate(examples[tenant])
            ]
            await storages[tenant].save_batch(
                ApprovalBatch(
                    batch_id=batch_id,
                    items=items,
                    context={
                        "tenant_id": tenant,
                        "optimizer": optimizer_type,
                        "purpose": "tenant isolation contract",
                        "dataset_name": "poisoned_context_dataset",
                    },
                )
            )
            agent = HumanApprovalAgent(
                storage=storages[tenant],
                confidence_extractor=confidence_extractor,
            )
            await producer_barrier.wait()
            approved = [
                await agent.apply_decision(
                    batch_id,
                    ReviewDecision(
                        item_id=item.item_id,
                        approved=True,
                        reviewer=f"reviewer-{tenant}",
                    ),
                )
                for item in items
            ]
            assert [(item.item_id, item.data, item.status) for item in approved] == [
                (item.item_id, example, ApprovalStatus.APPROVED)
                for item, example in zip(items, examples[tenant], strict=True)
            ]
            return approved

        approved_by_tenant = await asyncio.gather(
            *(approve_tenant_examples(tenant) for tenant in tenants)
        )
        assert [
            [approved_item.data for approved_item in approved]
            for approved in approved_by_tenant
        ] == [examples[tenant] for tenant in tenants]

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
                    "Failed to load approved synthetic data for "
                    f"tenant=acme:alpha optimizer={optimizer_type} "
                    "dataset=approved_synthetic_data-acme:alpha"
                ),
            ) as outage:
                await _load_approved_synthetic_data(
                    storages["acme:alpha"].provider,
                    "acme:alpha",
                    optimizer_type,
                )
            assert not isinstance(outage.value.__cause__, DatasetNotFoundError)
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
                pytest.fail("Phoenix did not recover after approved dataset outage")

        alpha, beta = await asyncio.gather(
            _load_approved_synthetic_data(
                storages["acme:alpha"].provider,
                "acme:alpha",
                optimizer_type,
            ),
            _load_approved_synthetic_data(
                storages["acme:beta"].provider,
                "acme:beta",
                optimizer_type,
            ),
        )
        assert alpha == [
            {**item.data, "example_id": f"approved:{item.item_id}"}
            for item in approved_by_tenant[0]
        ]
        assert beta == [
            {**item.data, "example_id": f"approved:{item.item_id}"}
            for item in approved_by_tenant[1]
        ]

        dataset_store = storages["acme:alpha"].provider.datasets
        alpha_frame = await dataset_store.get_dataset(
            name="approved_synthetic_data-acme:alpha"
        )
        beta_frame = await dataset_store.get_dataset(
            name="approved_synthetic_data-acme:beta"
        )
        assert [row["input"]["query"] for _, row in alpha_frame.iterrows()] == [
            example["query"] for example in examples["acme:alpha"]
        ]
        assert [row["input"]["query"] for _, row in beta_frame.iterrows()] == [
            example["query"] for example in examples["acme:beta"]
        ]
        with pytest.raises(DatasetNotFoundError):
            await dataset_store.get_dataset(name="approved_synthetic_data")
        with pytest.raises(DatasetNotFoundError):
            await dataset_store.get_dataset(name="poisoned_context_dataset")

    @pytest.mark.asyncio
    async def test_concurrent_approval_is_one_exact_dataset_record(
        self,
        phoenix_container,
        telemetry_manager,
        confidence_extractor,
        approval_redis_url,
    ):
        suffix = uuid.uuid4().hex[:10]
        tenant_id = f"approval:concurrent-{suffix}"
        batch_id = f"concurrent-approval-{suffix}"
        dataset_name = approved_synthetic_dataset_name(tenant_id)
        storages = [
            ApprovalStorageImpl(
                grpc_endpoint=phoenix_container["grpc_endpoint"],
                http_endpoint=phoenix_container["http_endpoint"],
                tenant_id=tenant_id,
                telemetry_manager=telemetry_manager,
                redis_url=approval_redis_url,
            )
            for _ in range(2)
        ]
        item = ReviewItem(
            item_id=f"approval-item-{suffix}",
            data={
                "query": "find Marie Curie discovering radium",
                "chosen_agent": "video_search_agent",
            },
            confidence=0.875,
            metadata={"agent_type": "routing"},
        )
        context = {
            "tenant_id": tenant_id,
            "optimizer": "routing",
        }
        await storages[0].save_batch(
            ApprovalBatch(batch_id=batch_id, items=[item], context=context)
        )
        reviewed_at = datetime(2026, 8, 5, 5, 6, 7, tzinfo=timezone.utc)
        decisions = [
            ReviewDecision(
                item_id=item.item_id,
                approved=True,
                feedback="The routing decision is exact.",
                corrections={"chosen_agent": "video_search_agent"},
                reviewer="reviewer@example.com",
                timestamp=reviewed_at,
            )
            for _ in storages
        ]
        barrier = asyncio.Barrier(2)
        attempt_count = 0

        async def approve(storage, decision):
            nonlocal attempt_count
            persist = storage.persist_approved_item

            async def synchronized_persist(**kwargs):
                nonlocal attempt_count
                attempt_count += 1
                await barrier.wait()
                return await persist(**kwargs)

            storage.persist_approved_item = synchronized_persist
            agent = HumanApprovalAgent(
                storage=storage,
                confidence_extractor=confidence_extractor,
            )
            return await agent.apply_decision(batch_id, decision)

        approved = await asyncio.gather(
            *(
                approve(storage, decision)
                for storage, decision in zip(storages, decisions)
            )
        )

        expected_decision = {
            "reviewer": "reviewer@example.com",
            "feedback": "The routing decision is exact.",
            "corrections": {"chosen_agent": "video_search_agent"},
            "timestamp": reviewed_at.isoformat(),
        }
        assert attempt_count == 2
        assert [
            (
                result.item_id,
                result.data,
                result.status,
                result.reviewed_at,
                result.metadata["decision"],
            )
            for result in approved
        ] == [
            (
                item.item_id,
                item.data,
                ApprovalStatus.APPROVED,
                reviewed_at,
                expected_decision,
            ),
            (
                item.item_id,
                item.data,
                ApprovalStatus.APPROVED,
                reviewed_at,
                expected_decision,
            ),
        ]

        dataset = await storages[0].provider.datasets.get_dataset(name=dataset_name)
        assert len(dataset) == 1
        record = dataset.iloc[0]["input"]
        canonical_expected_record = {
            "item_id": item.item_id,
            "confidence": 0.875,
            "status": ApprovalStatus.APPROVED.value,
            "created_at": item.created_at.isoformat(),
            "reviewed_at": reviewed_at.isoformat(),
            "query": "find Marie Curie discovering radium",
            "chosen_agent": "video_search_agent",
            "metadata.agent_type": "routing",
            "metadata.approval_batch_id": batch_id,
            "metadata.decision": expected_decision,
            "context.tenant_id": tenant_id,
            "context.optimizer": "routing",
        }
        decision_identity = {
            "item_id": item.item_id,
            "status": ApprovalStatus.APPROVED.value,
            "decision": {
                "reviewer": "reviewer@example.com",
                "feedback": "The routing decision is exact.",
                "corrections": {"chosen_agent": "video_search_agent"},
            },
        }
        identity_json = json.dumps(
            decision_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        canonical_expected_record["metadata.approval_decision_sha256"] = hashlib.sha256(
            identity_json.encode("utf-8")
        ).hexdigest()
        canonical_expected_record["metadata.approval_decision_timestamp"] = (
            reviewed_at.isoformat()
        )
        canonical_record = json.dumps(
            canonical_expected_record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        expected_record = canonical_expected_record | {
            "metadata.approval_record_json": canonical_record,
            "metadata.approval_record_sha256": hashlib.sha256(
                canonical_record.encode("utf-8")
            ).hexdigest(),
        }
        assert record == expected_record

        wait_for_phoenix_processing(delay=1.0, description="approval annotations")
        final_batch = await storages[0].get_batch(batch_id)
        assert final_batch is not None
        assert [
            (candidate.item_id, candidate.status, candidate.reviewed_at)
            for candidate in final_batch.items
        ] == [(item.item_id, ApprovalStatus.APPROVED, reviewed_at)]

    @pytest.mark.asyncio
    async def test_fresh_approval_retry_after_decision_annotation_converges(
        self,
        phoenix_container,
        telemetry_manager,
        confidence_extractor,
        approval_redis_url,
    ):
        suffix = uuid.uuid4().hex[:10]
        tenant_id = f"approval:retry-{suffix}"
        batch_id = f"retry-approval-{suffix}"
        dataset_name = approved_synthetic_dataset_name(tenant_id)
        storage = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id=tenant_id,
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        item = ReviewItem(
            item_id=f"retry-item-{suffix}",
            data={
                "query": "find the exact Vespa ranking tutorial",
                "chosen_agent": "video_search_agent",
            },
            confidence=0.82,
            metadata={"agent_type": "routing"},
        )
        context = {"tenant_id": tenant_id, "optimizer": "routing"}
        await storage.save_batch(
            ApprovalBatch(batch_id=batch_id, items=[item], context=context)
        )
        ready_batch = await storage.get_batch(batch_id)
        assert ready_batch.batch_id == batch_id
        assert [
            (candidate.item_id, candidate.status, candidate.reviewed_at)
            for candidate in ready_batch.items
        ] == [(item.item_id, ApprovalStatus.PENDING_REVIEW, None)]
        first_timestamp = datetime(2026, 8, 5, 6, 7, 8, tzinfo=timezone.utc)
        retry_timestamp = first_timestamp + timedelta(minutes=5)
        first_decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            feedback="The routing decision is exact.",
            corrections={"chosen_agent": "video_search_agent"},
            reviewer="reviewer@example.com",
            timestamp=first_timestamp,
        )
        agent = HumanApprovalAgent(
            storage=storage,
            confidence_extractor=confidence_extractor,
        )
        persisted_log = storage.log_approval_decision
        container_name = phoenix_container["container_name"]
        phoenix_stopped = False
        boundary_events = []

        async def wait_for_annotations(active_storage, name, count):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                spans = await active_storage.provider.traces.get_spans(
                    project=active_storage.full_project_name,
                    filters={"name": "approval_item"},
                )
                if not spans.empty and "attributes.item_id" in spans.columns:
                    item_spans = spans[spans["attributes.item_id"] == item.item_id]
                    annotations = (
                        await active_storage.provider.annotations.get_annotations(
                            spans_df=item_spans,
                            project=active_storage.full_project_name,
                            annotation_names=[name],
                        )
                    )
                    if not annotations.empty:
                        matching = annotations[
                            (annotations["annotation_name"] == name)
                            & annotations["metadata"].apply(
                                lambda metadata: (
                                    isinstance(metadata, dict)
                                    and metadata.get("item_id") == item.item_id
                                )
                            )
                        ]
                        if len(matching) >= count:
                            return matching
                await asyncio.sleep(0.5)
            pytest.fail(f"Expected {count} {name} annotations for item {item.item_id}")

        async def commit_history_then_interrupt_phoenix(**kwargs):
            nonlocal phoenix_stopped
            result = await persisted_log(**kwargs)
            history = await wait_for_annotations(storage, "human_approval", 1)
            assert history["result.label"].tolist() == ["approved"]
            assert history.iloc[0]["metadata"]["reviewed_at"] == (
                first_timestamp.isoformat()
            )
            boundary_events.append("human_approval_visible")
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "stop", container_name],
                check=True,
                capture_output=True,
                timeout=30,
            )
            phoenix_stopped = True
            boundary_events.append("phoenix_stopped")
            return result

        storage.log_approval_decision = commit_history_then_interrupt_phoenix
        try:
            with pytest.raises(RuntimeError) as error:
                await agent.apply_decision(batch_id, first_decision)
        finally:
            storage.log_approval_decision = persisted_log
            if phoenix_stopped:
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
                    pytest.fail("Phoenix did not recover after approval interruption")

        assert str(error.value) == (
            "Failed to persist approved item: "
            f"tenant={tenant_id} dataset={dataset_name} "
            f"batch={batch_id} item={item.item_id}"
        )
        assert str(error.value.__cause__) == (
            "Failed to query every span from Phoenix project "
            f"{storage.full_project_name}"
        )
        assert boundary_events == ["human_approval_visible", "phoenix_stopped"]

        retry_storage = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id=tenant_id,
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        pending = await retry_storage.get_batch(batch_id)
        assert pending.batch_id == batch_id
        assert [
            (candidate.item_id, candidate.status, candidate.reviewed_at)
            for candidate in pending.items
        ] == [(item.item_id, ApprovalStatus.PENDING_REVIEW, None)]
        first_history = await wait_for_annotations(
            retry_storage,
            "human_approval",
            1,
        )
        assert len(first_history) == 1

        retry_agent = HumanApprovalAgent(
            storage=retry_storage,
            confidence_extractor=confidence_extractor,
        )
        approved = await retry_agent.apply_decision(
            batch_id,
            ReviewDecision(
                item_id=item.item_id,
                approved=True,
                feedback="The routing decision is exact.",
                corrections={"chosen_agent": "video_search_agent"},
                reviewer="reviewer@example.com",
                timestamp=retry_timestamp,
            ),
        )
        assert (
            approved.item_id,
            approved.data,
            approved.status,
            approved.reviewed_at,
            approved.metadata["decision"]["timestamp"],
        ) == (
            item.item_id,
            item.data,
            ApprovalStatus.APPROVED,
            first_timestamp,
            first_timestamp.isoformat(),
        )
        dataset = await retry_storage.provider.datasets.get_dataset(name=dataset_name)
        assert len(dataset) == 1
        record = dataset.iloc[0]["input"]
        assert record["item_id"] == item.item_id
        assert record["reviewed_at"] == first_timestamp.isoformat()
        assert record["metadata.approval_decision_timestamp"] == (
            first_timestamp.isoformat()
        )
        assert record["metadata.decision"] == {
            "reviewer": "reviewer@example.com",
            "feedback": "The routing decision is exact.",
            "corrections": {"chosen_agent": "video_search_agent"},
            "timestamp": first_timestamp.isoformat(),
        }

        histories = await wait_for_annotations(
            retry_storage,
            "human_approval",
            1,
        )
        assert len(histories) == 1
        assert histories["result.label"].tolist() == ["approved"]
        assert {
            metadata["reviewed_at"] for metadata in histories["metadata"].tolist()
        } == {first_timestamp.isoformat()}
        statuses = await wait_for_annotations(
            retry_storage,
            "item_status_update",
            1,
        )
        assert len(statuses) == 1
        assert statuses.iloc[0]["result.label"] == ApprovalStatus.APPROVED.value
        assert statuses.iloc[0]["metadata"]["reviewed_at"] == (
            first_timestamp.isoformat()
        )

        final_batch = await retry_storage.get_batch(batch_id)
        assert final_batch is not None
        assert [
            (candidate.item_id, candidate.status, candidate.reviewed_at)
            for candidate in final_batch.items
        ] == [(item.item_id, ApprovalStatus.APPROVED, first_timestamp)]

    @pytest.mark.asyncio
    async def test_approval_boundary_failures_never_report_success(
        self,
        phoenix_container,
        telemetry_manager,
        confidence_extractor,
        approval_redis_url,
        monkeypatch,
    ):
        from cogniverse_foundation.telemetry.providers.base import (
            DatasetNotFoundError,
        )

        suffix = uuid.uuid4().hex[:10]

        async def save_pending(storage, tenant_id, boundary):
            batch_id = f"{boundary}-failure-{suffix}"
            item = ReviewItem(
                item_id=f"{boundary}-item-{suffix}",
                data={
                    "query": f"find the exact {boundary} failure example",
                    "chosen_agent": "search_agent",
                },
                confidence=0.84,
                metadata={"agent_type": "routing"},
            )
            await storage.save_batch(
                ApprovalBatch(
                    batch_id=batch_id,
                    items=[item],
                    context={"tenant_id": tenant_id, "optimizer": "routing"},
                )
            )
            return batch_id, item

        redis_tenant = f"approval:redis-failure-{suffix}"
        healthy_redis_storage = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id=redis_tenant,
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        redis_batch_id, redis_item = await save_pending(
            healthy_redis_storage,
            redis_tenant,
            "redis",
        )
        unavailable_redis_port = _free_port()
        unavailable_redis_storage = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id=redis_tenant,
            telemetry_manager=telemetry_manager,
            redis_url=f"redis://127.0.0.1:{unavailable_redis_port}/0",
        )
        redis_agent = HumanApprovalAgent(
            storage=unavailable_redis_storage,
            confidence_extractor=confidence_extractor,
        )
        with pytest.raises(RuntimeError) as redis_error:
            await redis_agent.apply_decision(
                redis_batch_id,
                ReviewDecision(
                    item_id=redis_item.item_id,
                    approved=True,
                    reviewer="boundary-reviewer@example.com",
                ),
            )

        assert str(redis_error.value) == (
            "Failed to select canonical review decision for "
            f"tenant={redis_tenant} batch={redis_batch_id} "
            f"original={redis_item.item_id}"
        )
        assert isinstance(redis_error.value.__cause__, RedisConnectionError)
        redis_pending = await healthy_redis_storage.get_batch(redis_batch_id)
        assert redis_pending.batch_id == redis_batch_id
        assert [
            (candidate.item_id, candidate.status, candidate.reviewed_at)
            for candidate in redis_pending.items
        ] == [(redis_item.item_id, ApprovalStatus.PENDING_REVIEW, None)]
        with pytest.raises(DatasetNotFoundError):
            await healthy_redis_storage.provider.datasets.get_dataset(
                name=approved_synthetic_dataset_name(redis_tenant)
            )

        phoenix_tenant = f"approval:phoenix-failure-{suffix}"
        phoenix_storage = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id=phoenix_tenant,
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        phoenix_batch_id, phoenix_item = await save_pending(
            phoenix_storage,
            phoenix_tenant,
            "phoenix",
        )
        phoenix_agent = HumanApprovalAgent(
            storage=phoenix_storage,
            confidence_extractor=confidence_extractor,
        )
        phoenix_dataset = approved_synthetic_dataset_name(phoenix_tenant)
        get_dataset = phoenix_storage.provider.datasets.get_dataset
        phoenix_failure = ConnectionError("Phoenix dataset endpoint unavailable")

        async def fail_dataset_read(name):
            assert name == phoenix_dataset
            raise phoenix_failure

        monkeypatch.setattr(
            phoenix_storage.provider.datasets,
            "get_dataset",
            fail_dataset_read,
        )
        with pytest.raises(RuntimeError) as phoenix_error:
            await phoenix_agent.apply_decision(
                phoenix_batch_id,
                ReviewDecision(
                    item_id=phoenix_item.item_id,
                    approved=True,
                    reviewer="boundary-reviewer@example.com",
                ),
            )

        assert str(phoenix_error.value) == (
            "Failed to persist approved item: "
            f"tenant={phoenix_tenant} dataset={phoenix_dataset} "
            f"batch={phoenix_batch_id} item={phoenix_item.item_id}"
        )
        assert phoenix_error.value.__cause__ is phoenix_failure
        monkeypatch.setattr(
            phoenix_storage.provider.datasets,
            "get_dataset",
            get_dataset,
        )
        phoenix_pending = await phoenix_storage.get_batch(phoenix_batch_id)
        assert phoenix_pending.batch_id == phoenix_batch_id
        assert [
            (candidate.item_id, candidate.status, candidate.reviewed_at)
            for candidate in phoenix_pending.items
        ] == [(phoenix_item.item_id, ApprovalStatus.PENDING_REVIEW, None)]
        with pytest.raises(DatasetNotFoundError):
            await phoenix_storage.provider.datasets.get_dataset(name=phoenix_dataset)

    @pytest.mark.asyncio
    async def test_unobserved_profile_examples_require_review(
        self, approval_agent, approval_storage
    ):

        items = [_profile_selection_record(f"test query {index}") for index in range(5)]

        batch_id = "confidence_test"
        context = {
            "tenant_id": approval_storage.tenant_id,
            "agent_type": "profile_selection",
            "purpose": "threshold_test",
        }

        await approval_agent.process_batch(items, batch_id, context)

        retrieved = await approval_storage.get_batch(batch_id)
        assert retrieved is not None

        assert [item.item_id for item in retrieved.items] == [
            f"{batch_id}_{index}" for index in range(5)
        ]
        assert [item.confidence for item in retrieved.items] == [0.0] * 5
        assert [item.status for item in retrieved.items] == [
            ApprovalStatus.PENDING_REVIEW
        ] * 5

    @pytest.mark.asyncio
    async def test_invalid_entity_relationship_correction_selects_no_replacement(
        self,
        approval_storage,
        confidence_extractor,
        approval_redis_url,
    ):
        batch_id = f"invalid_relationship_{time.time_ns()}"
        original = ReviewItem(
            item_id=f"{batch_id}_0",
            data={
                "query": "TensorFlow was created by Google Brain",
                "entities": [
                    {"text": "TensorFlow", "type": "TECHNOLOGY"},
                    {"text": "Google Brain", "type": "ORG"},
                ],
                "relationships": [
                    {
                        "source": "TensorFlow",
                        "target": "Google Brain",
                        "type": "CREATED_BY",
                    }
                ],
            },
            confidence=0.4,
        )
        await approval_storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=[original],
                context={
                    "tenant_id": approval_storage.tenant_id,
                    "optimizer": "entity_extraction",
                },
            )
        )

        class _RelationshipPreservingGenerator:
            def __init__(self):
                self.calls = 0

            def forward(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(
                    updates={"entities": kwargs["corrections"]["entities"]},
                    reasoning="Applied the requested entity replacement.",
                    _retry_count=0,
                    _max_retries=3,
                )

        generator = _RelationshipPreservingGenerator()
        agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
            feedback_handler=SyntheticDataFeedbackHandler(
                generator=_BoundTestRegenerator(generator.forward),
                generation_timeout_seconds=5.0,
            ),
        )
        decision = ReviewDecision(
            item_id=original.item_id,
            approved=False,
            feedback="Replace both entities.",
            corrections={
                "entities": [
                    {"text": "PyTorch", "type": "TECHNOLOGY"},
                    {"text": "Meta AI", "type": "ORG"},
                ]
            },
            reviewer="relationship-reviewer",
        )

        with pytest.raises(RuntimeError) as error:
            await agent.apply_decision(batch_id, decision)

        assert str(error.value) == (
            f"Failed to regenerate {original.item_id} after 2 regeneration attempts"
        )
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            f"item={original.item_id} schema=EntityExtractionExampleSchema "
            "relationship_index=0 endpoint=source value='TensorFlow' is not one "
            "of the regenerated entity texts ['PyTorch', 'Meta AI']"
        )
        assert generator.calls == 2

        persisted = await approval_storage.get_batch(batch_id)
        assert persisted.batch_id == batch_id
        assert [(item.item_id, item.status) for item in persisted.items] == [
            (original.item_id, ApprovalStatus.PENDING_REVIEW)
        ]

        replacement_spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        if replacement_spans.empty:
            matching_spans = replacement_spans
        else:
            matching_spans = replacement_spans[
                (replacement_spans["attributes.batch_id"] == batch_id)
                & (replacement_spans["attributes.original_item_id"] == original.item_id)
            ]
        assert matching_spans.empty

        redis = aioredis.from_url(approval_redis_url, decode_responses=True)
        try:
            selected_payload = await redis.get(
                RedisReplacementRecordStore._key(
                    approval_storage.tenant_id,
                    batch_id,
                    original.item_id,
                )
            )
        finally:
            await redis.aclose()
        assert selected_payload is None

    @pytest.mark.asyncio
    async def test_non_finite_replacement_is_rejected_before_boundary_writes(
        self,
        approval_storage,
        approval_redis_url,
    ):
        batch_id = f"non_finite_replacement_{time.time_ns()}"
        original = ReviewItem(
            item_id=f"{batch_id}_0",
            data={"query": "original query"},
            confidence=0.4,
        )
        replacement = ReviewItem(
            item_id=f"{original.item_id}_regen_0",
            data={"query": "replacement query"},
            confidence=float("nan"),
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "finite-reviewer",
                    "feedback": "Use a finite score.",
                    "corrections": {"query": "replacement query"},
                    "timestamp": "2026-08-05T02:30:00+00:00",
                },
            },
        )

        with pytest.raises(
            ValueError,
            match="Replacement confidence must be a finite number from 0 to 1",
        ):
            await approval_storage.replace_item(batch_id, original, replacement)

        redis = aioredis.from_url(approval_redis_url, decode_responses=True)
        try:
            assert (
                await redis.exists(
                    RedisReplacementRecordStore._key(
                        approval_storage.tenant_id,
                        batch_id,
                        original.item_id,
                    )
                )
                == 0
            )
        finally:
            await redis.aclose()
        spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        if not spans.empty:
            matching = spans[
                (spans["attributes.batch_id"] == batch_id)
                & (spans["attributes.original_item_id"] == original.item_id)
            ]
            assert matching.empty

    @pytest.mark.asyncio
    async def test_malformed_redis_replacement_never_reaches_phoenix(
        self,
        approval_storage,
        approval_redis_url,
    ):
        batch_id = f"malformed_redis_replacement_{time.time_ns()}"
        original = ReviewItem(
            item_id=f"{batch_id}_0",
            data={"query": "original query"},
            confidence=0.4,
        )
        replacement = ReviewItem(
            item_id=f"{original.item_id}_regen_0",
            data={"query": "replacement query"},
            confidence=0.8,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "schema-reviewer",
                    "feedback": "Use the reviewed query.",
                    "corrections": {"query": "replacement query"},
                    "timestamp": "2026-08-05T02:45:00+00:00",
                },
            },
        )
        key = RedisReplacementRecordStore._key(
            approval_storage.tenant_id,
            batch_id,
            original.item_id,
        )
        redis = aioredis.from_url(approval_redis_url, decode_responses=True)
        try:
            malformed_payload = {
                "item_id": replacement.item_id,
                "data": replacement.data,
                "confidence": replacement.confidence,
                "status": replacement.status.value,
                "metadata": replacement.metadata,
                "created_at": "2026-08-05T02:45:00",
                "reviewed_at": None,
            }
            await redis.set(
                key,
                json.dumps(
                    malformed_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        finally:
            await redis.aclose()

        with pytest.raises(
            RuntimeError,
            match=(
                "Canonical replacement record is invalid: "
                f"batch={batch_id} original={original.item_id} "
                f"replacement={replacement.item_id}"
            ),
        ):
            await approval_storage.replace_item(batch_id, original, replacement)

        spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        if not spans.empty:
            matching = spans[
                (spans["attributes.batch_id"] == batch_id)
                & (spans["attributes.original_item_id"] == original.item_id)
            ]
            assert matching.empty

    @pytest.mark.asyncio
    @pytest.mark.requires_lm
    async def test_feedback_driven_regeneration(self, approval_agent, approval_storage):
        """A rejected query is regenerated from exact reviewer corrections."""

        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=approval_agent.confidence_extractor,
            feedback_handler=approval_agent.feedback_handler,
            confidence_threshold=1.0,
        )
        original_data = _routing_record("find Curie laboratory footage")
        original_data["entities"] = [{"text": "Curie", "type": "PERSON"}]
        original_data["enhanced_query"] = "find Curie(PERSON) laboratory footage"
        batch_id = f"regen_batch_{time.time_ns()}"
        context = {
            "tenant_id": approval_storage.tenant_id,
            "agent_type": "routing",
            "purpose": "regen_test",
            "optimizer": "routing",
        }

        batch = await approval_agent.process_batch([original_data], batch_id, context)
        item_id = batch.items[0].item_id

        decision = ReviewDecision(
            item_id=item_id,
            approved=False,
            feedback=(
                "Replace the abbreviated scientist name with Marie Curie and use "
                "the exact corrected query. Preserve every other training value."
            ),
            corrections={
                "query": "find Marie Curie radium laboratory footage",
                "entities": [{"text": "Marie Curie", "type": "PERSON"}],
            },
            reviewer="test_user",
        )

        result = await approval_agent.apply_decision(batch_id, decision)

        assert result.status is ApprovalStatus.REGENERATED
        assert result.item_id == f"{item_id}_regen_0"
        assert result.data["query"] == "find Marie Curie radium laboratory footage"
        assert result.data["enhanced_query"] == (
            "find Marie Curie(PERSON) radium laboratory footage"
        )
        assert result.data["entities"] == [{"text": "Marie Curie", "type": "PERSON"}]
        assert result.data["relationships"] == []
        assert result.data["chosen_agent"] == "search_agent"
        assert "topics" not in result.data
        RoutingExperienceSchema.model_validate(result.data)
        assert result.data["metadata"]["source"] == "real feedback regeneration"
        generation_metadata = result.data["metadata"]["_generation_metadata"]
        assert generation_metadata == {
            "retry_count": generation_metadata["retry_count"],
            "max_retries": 3,
            "regeneration_attempt": 1,
            "max_regeneration_attempts": 2,
            "regeneration": True,
            "original_query": "find Curie laboratory footage",
            "human_feedback": decision.feedback,
            "corrections_applied": decision.corrections,
            "reasoning": generation_metadata["reasoning"],
        }
        assert generation_metadata["retry_count"] in {0, 1, 2}
        assert result.metadata["generation"] == {
            "retry_count": generation_metadata["retry_count"],
            "max_retries": 3,
            "reasoning": generation_metadata["reasoning"],
        }

        persisted = await approval_storage.get_batch(batch_id)
        assert persisted.batch_id == batch_id
        by_id = {item.item_id: item for item in persisted.items}
        assert set(by_id) == {item_id, result.item_id}
        assert by_id[item_id].status is ApprovalStatus.REJECTED
        assert by_id[result.item_id].status is ApprovalStatus.REGENERATED
        assert by_id[result.item_id].data == result.data
        assert by_id[result.item_id].metadata == {
            "agent_type": "routing",
            "original_item_id": item_id,
            "regeneration_attempt": 1,
            "feedback": decision.feedback,
            "generation": result.metadata["generation"],
            "regeneration_feedback": decision.feedback,
            "decision": {
                "reviewer": "test_user",
                "feedback": decision.feedback,
                "corrections": decision.corrections,
                "timestamp": decision.timestamp.isoformat(),
            },
        }

    @pytest.mark.asyncio
    async def test_regeneration_persists_internal_retry_and_enhanced_query(
        self,
        approval_storage,
        confidence_extractor,
        dspy_lm,
    ):
        def regenerate(**_kwargs):
            return SimpleNamespace(
                updates={
                    "query": "find TensorFlow tutorials",
                    "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
                },
                reasoning="Used TensorFlow after the validation retry.",
                _retry_count=1,
                _max_retries=3,
            )

        generator = _BoundTestRegenerator(regenerate)
        generator.lm = dspy_lm
        feedback_handler = SyntheticDataFeedbackHandler(
            generator=generator,
            generation_timeout_seconds=5.0,
            max_regeneration_attempts=2,
        )
        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
            feedback_handler=feedback_handler,
            confidence_threshold=1.0,
        )
        batch_id = f"internal_retry_{time.time_ns()}"
        original_data = _routing_record("find a framework tutorial")
        original_data["metadata"]["source"] = "internal retry contract"
        batch = await approval_agent.process_batch(
            [original_data],
            batch_id,
            {
                "tenant_id": approval_storage.tenant_id,
                "agent_type": "routing",
                "purpose": "internal retry contract",
                "optimizer": "routing",
            },
        )
        original_item = batch.items[0]
        decision = ReviewDecision(
            item_id=original_item.item_id,
            approved=False,
            feedback="Include the exact framework name.",
            corrections={
                "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
                "topics": ["neural network training"],
            },
            reviewer="retry-reviewer",
        )

        regenerated = await approval_agent.apply_decision(batch_id, decision)

        assert regenerated.data["query"] == "find TensorFlow tutorials"
        assert regenerated.data["enhanced_query"] == (
            "find TensorFlow(TECHNOLOGY) tutorials"
        )
        assert regenerated.data["metadata"] == {
            "source": "internal retry contract",
            "_outcome_metadata": {
                "observed": False,
                "required_field_semantics": {
                    "routing_confidence": "unobserved_zero_sentinel",
                    "search_quality": "unobserved_zero_sentinel",
                    "agent_success": "unobserved_false_sentinel",
                    "processing_time": "unobserved_zero_sentinel",
                },
            },
            "_generation_metadata": {
                "retry_count": 1,
                "max_retries": 3,
                "regeneration_attempt": 1,
                "max_regeneration_attempts": 2,
                "regeneration": True,
                "original_query": "find a framework tutorial",
                "human_feedback": "Include the exact framework name.",
                "corrections_applied": decision.corrections,
                "reasoning": "Used TensorFlow after the validation retry.",
            },
        }
        assert regenerated.data["routing_confidence"] == 0.0
        assert regenerated.data["search_quality"] == 0.0
        assert regenerated.data["agent_success"] is False
        assert regenerated.data["processing_time"] == 0.0
        assert regenerated.data["user_satisfaction"] is None
        assert regenerated.data["reward"] is None
        assert confidence_extractor.extract(regenerated.data) == 0.0

        persisted = await approval_storage.get_batch(batch_id)
        assert persisted.batch_id == batch_id
        persisted_by_id = {item.item_id: item for item in persisted.items}
        assert set(persisted_by_id) == {
            original_item.item_id,
            regenerated.item_id,
        }
        assert persisted_by_id[original_item.item_id].status is ApprovalStatus.REJECTED
        assert persisted_by_id[regenerated.item_id].data == regenerated.data

    @pytest.mark.asyncio
    async def test_pending_batches_retrieval(
        self, approval_storage, confidence_extractor
    ):
        """Test retrieving batches with pending reviews"""

        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
            confidence_threshold=1.0,
        )

        items1 = [_profile_selection_record("test1")]
        items2 = [_profile_selection_record("test2")]

        suffix = time.time_ns()
        batch_ids = [f"pending_routing_{suffix}", f"pending_modality_{suffix}"]
        purpose = f"pending-filter-{suffix}"
        await approval_agent.process_batch(
            items1,
            batch_ids[0],
            {
                "tenant_id": approval_storage.tenant_id,
                "agent_type": "profile_selection",
                "optimizer": "routing",
                "purpose": purpose,
            },
        )
        await approval_agent.process_batch(
            items2,
            batch_ids[1],
            {
                "tenant_id": approval_storage.tenant_id,
                "agent_type": "profile_selection",
                "optimizer": "modality",
                "purpose": purpose,
            },
        )

        wait_for_phoenix_processing(delay=2.0, description="batch indexing")

        pending = await approval_storage.get_pending_batches(
            context_filter={"purpose": purpose}
        )
        assert {batch.batch_id for batch in pending} == set(batch_ids)
        assert {
            batch.batch_id: [item.status for item in batch.items] for batch in pending
        } == {
            batch_ids[0]: [ApprovalStatus.PENDING_REVIEW],
            batch_ids[1]: [ApprovalStatus.PENDING_REVIEW],
        }

        routing_batches = await approval_storage.get_pending_batches(
            context_filter={"optimizer": "routing", "purpose": purpose}
        )
        assert [batch.batch_id for batch in routing_batches] == [batch_ids[0]]

    @pytest.mark.asyncio
    async def test_oldest_pending_batch_survives_phoenix_cursor_pagination(
        self,
        approval_storage,
        confidence_extractor,
        phoenix_container,
    ):
        suffix = time.time_ns()
        batch_id = f"oldest_pending_{suffix}"
        purpose = f"cursor-pagination-{suffix}"
        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
            confidence_threshold=1.0,
        )
        await approval_agent.process_batch(
            [_profile_selection_record("find the oldest exact pending batch")],
            batch_id,
            {
                "agent_type": "profile_selection",
                "purpose": purpose,
                "tenant_id": approval_storage.tenant_id,
            },
        )

        now = datetime.now(timezone.utc)
        noise = [
            {
                "name": "approval_item_replacement",
                "context": {
                    "trace_id": uuid.uuid4().hex,
                    "span_id": uuid.uuid4().hex[:16],
                },
                "span_kind": "CHAIN",
                "start_time": (now + timedelta(microseconds=index)).isoformat(),
                "end_time": (now + timedelta(microseconds=index + 1)).isoformat(),
                "status_code": "OK",
                "attributes": {"pagination_noise_index": index},
            }
            for index in range(1_001)
        ]
        async with httpx.AsyncClient(
            base_url=phoenix_container["http_endpoint"], timeout=120
        ) as http_client:
            client = PhoenixAsyncClient(
                base_url=phoenix_container["http_endpoint"],
                http_client=http_client,
            )
            result = await client.spans.log_spans(
                project_identifier=approval_storage.full_project_name,
                spans=noise,
                timeout=120,
            )
        assert result == {"total_received": 1_001, "total_queued": 1_001}

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                all_spans = await approval_storage.provider.traces.get_all_spans(
                    project=approval_storage.full_project_name,
                    filters={
                        "name": [
                            "approval_batch",
                            "approval_item",
                            "approval_item_replacement",
                        ]
                    },
                )
            except RuntimeError:
                await asyncio.sleep(0.5)
                continue
            if len(all_spans) >= 1_003:
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Phoenix did not index the complete paginated span set")

        pending = await approval_storage.get_pending_batches(
            context_filter={"purpose": purpose}
        )

        assert [batch.batch_id for batch in pending] == [batch_id]
        assert [
            (item.item_id, item.data["query"], item.status) for item in pending[0].items
        ] == [
            (
                f"{batch_id}_0",
                "find the oldest exact pending batch",
                ApprovalStatus.PENDING_REVIEW,
            )
        ]

    @pytest.mark.asyncio
    async def test_batch_approval_rate_calculation(
        self, approval_storage, confidence_extractor
    ):
        """Test approval rate calculation for batch"""

        approval_agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
            confidence_threshold=0.8,
        )
        batch_id = "rate_test"
        context = {
            "tenant_id": approval_storage.tenant_id,
            "purpose": "rate_calculation",
        }
        confidences = [0.9, 0.85, 0.6, 0.5]
        queries = [
            "find Python tutorials",
            "find Vespa documentation",
            "short",
            "tiny",
        ]
        batch = ApprovalBatch(
            batch_id=batch_id,
            context=context,
            items=[
                ReviewItem(
                    item_id=f"{batch_id}_{index}",
                    data=_profile_selection_record(query),
                    confidence=confidence,
                    metadata={"agent_type": "profile_selection"},
                )
                for index, (query, confidence) in enumerate(
                    zip(queries, confidences, strict=True)
                )
            ],
        )
        await approval_agent.submit_for_review(batch)

        assert [item.status for item in batch.items] == [
            ApprovalStatus.AUTO_APPROVED,
            ApprovalStatus.AUTO_APPROVED,
            ApprovalStatus.PENDING_REVIEW,
            ApprovalStatus.PENDING_REVIEW,
        ]
        pending_item = batch.pending_review[0]
        decision = ReviewDecision(
            item_id=pending_item.item_id, approved=True, reviewer="test_user"
        )
        await approval_agent.apply_decision(batch_id, decision)
        wait_for_phoenix_processing(
            delay=1.0,
            description="approval-rate terminal status",
        )

        final_batch = await approval_storage.get_batch(batch_id)
        assert final_batch is not None
        assert [item.status for item in final_batch.items] == [
            ApprovalStatus.AUTO_APPROVED,
            ApprovalStatus.AUTO_APPROVED,
            ApprovalStatus.APPROVED,
            ApprovalStatus.PENDING_REVIEW,
        ]
        assert final_batch.approval_rate == 0.75

    @pytest.mark.asyncio
    async def test_phoenix_storage_integration(
        self, approval_storage, telemetry_provider
    ):
        """Test that approval data is correctly stored and retrievable from Phoenix"""

        item = ReviewItem(
            item_id="phoenix_test_item",
            data={"query": "test query", "entities": ["TestEntity"]},
            confidence=0.7,
            metadata={"source": "phoenix_integration_test"},
        )

        batch = ApprovalBatch(
            batch_id="phoenix_test_batch",
            items=[item],
            context={
                "tenant_id": approval_storage.tenant_id,
                "purpose": "phoenix_integration",
            },
        )

        batch_id = await approval_storage.save_batch(batch)
        assert batch_id == "phoenix_test_batch"

        wait_for_phoenix_processing(delay=3.0, description="span indexing")

        retrieved = await approval_storage.get_batch(batch_id)
        assert retrieved is not None
        assert len(retrieved.items) == 1
        assert retrieved.items[0].item_id == "phoenix_test_item"

        retrieved.items[0].status = ApprovalStatus.APPROVED
        retrieved.items[0].reviewed_at = datetime.now(timezone.utc)
        await approval_storage.update_item(retrieved.items[0], batch_id=batch_id)

        wait_for_phoenix_processing(delay=2.0, description="annotation update")

        final = await approval_storage.get_batch(batch_id)
        assert final.items[0].status == ApprovalStatus.APPROVED
        assert final.items[0].reviewed_at is not None

    @pytest.mark.asyncio
    async def test_rejection_preserves_exact_decision_in_real_phoenix(
        self,
        approval_storage,
        confidence_extractor,
    ):
        suffix = time.time_ns()
        batch_id = f"reject_exact_{suffix}"
        item = ReviewItem(
            item_id=f"reject_item_{suffix}",
            data=_profile_selection_record("find the launch video"),
            confidence=0.4,
            metadata={"agent_type": "profile_selection"},
        )
        await approval_storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=[item],
                context={
                    "tenant_id": approval_storage.tenant_id,
                    "agent_type": "profile_selection",
                },
            )
        )
        wait_for_phoenix_processing(delay=1.0, description="rejection batch indexing")

        decision_time = datetime(
            2026,
            8,
            5,
            11,
            22,
            33,
            tzinfo=timezone(timedelta(hours=5, minutes=30)),
        )
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=False,
            feedback="The selected profile cannot retrieve slide text.",
            corrections={"selected_profile": "video_colpali_smol500_mv_frame"},
            reviewer="reviewer@example.com",
            timestamp=decision_time,
        )
        agent = HumanApprovalAgent(
            storage=approval_storage,
            confidence_extractor=confidence_extractor,
        )
        rejected = await agent.apply_decision(batch_id, decision)

        expected_decision = {
            "reviewer": "reviewer@example.com",
            "feedback": "The selected profile cannot retrieve slide text.",
            "corrections": {"selected_profile": "video_colpali_smol500_mv_frame"},
            "timestamp": decision_time.isoformat(),
        }
        assert rejected.status is ApprovalStatus.REJECTED
        assert rejected.reviewed_at == decision_time
        assert rejected.metadata["decision"] == expected_decision

        wait_for_phoenix_processing(delay=1.0, description="rejection annotations")
        persisted = await approval_storage.get_batch(batch_id)
        assert persisted.batch_id == batch_id
        persisted_item = persisted.items[0]
        assert persisted_item.status is ApprovalStatus.REJECTED
        assert persisted_item.reviewed_at == decision_time
        assert persisted_item.metadata["decision"] == expected_decision

        spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item"},
        )
        item_spans = spans[spans["attributes.item_id"] == item.item_id]
        annotations = await approval_storage.provider.annotations.get_annotations(
            spans_df=item_spans,
            project=approval_storage.full_project_name,
            annotation_names=["human_approval"],
        )
        matching = annotations[
            (annotations["annotation_name"] == "human_approval")
            & annotations["metadata"].apply(
                lambda metadata: metadata.get("item_id") == item.item_id
            )
        ]
        assert len(matching) == 1
        annotation = matching.iloc[0]
        assert annotation["result.label"] == "rejected"
        assert annotation["result.score"] == 0.0
        assert annotation["metadata"] == {
            "item_id": item.item_id,
            "timestamp": decision_time.isoformat(),
            "reviewed_at": decision_time.isoformat(),
            "reviewer": "reviewer@example.com",
            "feedback": "The selected profile cannot retrieve slide text.",
        }

    @pytest.mark.asyncio
    async def test_item_span_lookup_is_scoped_to_its_phoenix_batch(
        self, approval_storage
    ):
        suffix = time.time_ns()
        item_id = f"shared_item_{suffix}"
        batch_ids = [f"span_lookup_a_{suffix}", f"span_lookup_b_{suffix}"]
        items = []
        for index, batch_id in enumerate(batch_ids):
            item = ReviewItem(
                item_id=item_id,
                data={"query": f"query for batch {index}"},
                confidence=0.6 + index / 10,
            )
            items.append(item)
            await approval_storage.save_batch(
                ApprovalBatch(
                    batch_id=batch_id,
                    items=[item],
                    context={
                        "tenant_id": approval_storage.tenant_id,
                        "purpose": "batch-scoped span lookup",
                    },
                )
            )

        items[0].status = ApprovalStatus.APPROVED
        items[0].reviewed_at = datetime.now(timezone.utc)
        await approval_storage.update_item(items[0], batch_id=batch_ids[0])
        await asyncio.sleep(2.0)

        first_batch, second_batch = await asyncio.gather(
            approval_storage.get_batch(batch_ids[0]),
            approval_storage.get_batch(batch_ids[1]),
        )

        assert first_batch.batch_id == batch_ids[0]
        assert second_batch.batch_id == batch_ids[1]
        assert [(item.data, item.status) for item in first_batch.items] == [
            ({"query": "query for batch 0"}, ApprovalStatus.APPROVED)
        ]
        assert first_batch.items[0].metadata == {"approval_batch_id": batch_ids[0]}
        assert [(item.data, item.status) for item in second_batch.items] == [
            ({"query": "query for batch 1"}, ApprovalStatus.PENDING_REVIEW)
        ]
        assert second_batch.items[0].metadata == {"approval_batch_id": batch_ids[1]}

    @pytest.mark.asyncio
    async def test_record_decision_persists_approval_span(self, approval_storage):
        """record_decision must emit an ``approval_decision`` span to Phoenix.

        This is what the dashboard approval handlers now call to persist a
        human decision; previously they discarded the ReviewDecision. Verify
        the span lands with the decision's attributes against real Phoenix.
        """
        item_id = f"decision_item_{int(time.time() * 1000)}"
        item = ReviewItem(
            item_id=item_id,
            data={"query": "is this relevant?"},
            confidence=0.6,
        )
        decision = ReviewDecision(
            item_id=item_id,
            approved=True,
            feedback="clear and specific",
            reviewer="reviewer@example.com",
        )

        await approval_storage.record_decision(decision, item)
        approval_storage.telemetry_manager.force_flush(timeout_millis=10000)

        project = approval_storage.full_project_name
        provider = approval_storage.provider

        decision_span = None
        deadline = time.time() + 60
        while time.time() < deadline:
            end_time = datetime.now(timezone.utc)
            spans_df = await provider.traces.get_spans(
                project=project,
                start_time=end_time - timedelta(hours=1),
                end_time=end_time,
                limit=1000,
            )
            if (
                spans_df is not None
                and not spans_df.empty
                and "name" in spans_df.columns
            ):
                matches = spans_df[
                    (spans_df["name"] == "approval_decision")
                    & (spans_df.get("attributes.item_id") == item_id)
                ]
                if not matches.empty:
                    decision_span = matches.iloc[0]
                    break
            await asyncio.sleep(2.0)

        assert decision_span is not None, (
            f"approval_decision span for {item_id} not found in {project}"
        )
        assert str(decision_span["attributes.approved"]).lower() == "true"
        assert decision_span["attributes.reviewer"] == "reviewer@example.com"
        assert decision_span["attributes.feedback"] == "clear and specific"

    @pytest.mark.asyncio
    async def test_concurrent_replacements_persist_exact_pairs(self, approval_storage):
        batch_id = f"concurrent_replacements_{time.time_ns()}"
        originals = [
            ReviewItem(
                item_id=f"{batch_id}_{index}",
                data={"query": f"original query {index}"},
                confidence=0.4 + index / 10,
            )
            for index in range(3)
        ]
        await approval_storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=originals,
                context={
                    "tenant_id": approval_storage.tenant_id,
                    "purpose": "concurrent replacement contract",
                },
            )
        )
        replacements = [
            ReviewItem(
                item_id=f"{original.item_id}_regen_0",
                data={"query": f"corrected query {index}"},
                confidence=0.8,
                status=ApprovalStatus.REGENERATED,
                metadata={
                    "original_item_id": original.item_id,
                    "decision": {
                        "reviewer": f"reviewer-{index}",
                        "feedback": f"feedback-{index}",
                        "corrections": {"query": f"corrected query {index}"},
                        "timestamp": f"2026-08-02T00:00:0{index}+00:00",
                    },
                },
            )
            for index, original in enumerate(originals)
        ]

        await asyncio.gather(
            *(
                approval_storage.replace_item(batch_id, original, replacement)
                for original, replacement in zip(originals, replacements, strict=True)
            )
        )

        persisted = await approval_storage.get_batch(batch_id)
        assert persisted.batch_id == batch_id
        by_id = {item.item_id: item for item in persisted.items}
        assert set(by_id) == {
            *(original.item_id for original in originals),
            *(replacement.item_id for replacement in replacements),
        }
        assert {
            original.item_id: by_id[original.item_id].status for original in originals
        } == {original.item_id: ApprovalStatus.REJECTED for original in originals}
        assert {
            replacement.item_id: (
                by_id[replacement.item_id].status,
                by_id[replacement.item_id].data,
                by_id[replacement.item_id].metadata,
            )
            for replacement in replacements
        } == {
            replacement.item_id: (
                ApprovalStatus.REGENERATED,
                replacement.data,
                replacement.metadata,
            )
            for replacement in replacements
        }

    @pytest.mark.asyncio
    async def test_concurrent_agents_accept_one_decision_and_reject_the_conflict(
        self,
        phoenix_container,
        telemetry_manager,
        confidence_extractor,
        approval_redis_url,
    ):
        batch_id = f"canonical_replacement_{time.time_ns()}"
        original = ReviewItem(
            item_id=f"{batch_id}_0",
            data={"query": "find the original TensorFlow tutorial"},
            confidence=0.4,
            metadata={"agent_type": "routing"},
        )
        storage_one = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id="test-tenant1",
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        storage_two = ApprovalStorageImpl(
            grpc_endpoint=phoenix_container["grpc_endpoint"],
            http_endpoint=phoenix_container["http_endpoint"],
            tenant_id="test-tenant1",
            telemetry_manager=telemetry_manager,
            redis_url=approval_redis_url,
        )
        await storage_one.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=[original],
                context={
                    "tenant_id": storage_one.tenant_id,
                    "agent_type": "routing",
                    "purpose": "canonical replacement contract",
                },
            )
        )

        start = asyncio.Barrier(2)

        class CompetingFeedbackHandler:
            def __init__(self, query: str, confidence: float):
                self.query = query
                self.confidence = confidence

            async def process_rejection(self, item, decision):
                return ReviewItem(
                    item_id=f"{item.item_id}_regen_0",
                    data={"query": self.query},
                    confidence=self.confidence,
                    status=ApprovalStatus.REGENERATED,
                )

        decisions = [
            ReviewDecision(
                item_id=original.item_id,
                approved=False,
                feedback="Use the exact PyTorch query.",
                corrections={"query": "find the exact PyTorch tutorial"},
                reviewer="reviewer-pytorch",
            ),
            ReviewDecision(
                item_id=original.item_id,
                approved=False,
                feedback="Use the exact JAX query.",
                corrections={"query": "find the exact JAX tutorial"},
                reviewer="reviewer-jax",
            ),
        ]
        agents = [
            HumanApprovalAgent(
                storage=storage_one,
                confidence_extractor=confidence_extractor,
                feedback_handler=CompetingFeedbackHandler(
                    "find the exact PyTorch tutorial", 0.8125
                ),
            ),
            HumanApprovalAgent(
                storage=storage_two,
                confidence_extractor=confidence_extractor,
                feedback_handler=CompetingFeedbackHandler(
                    "find the exact JAX tutorial", 0.875
                ),
            ),
        ]

        async def apply_concurrently(agent, decision):
            await start.wait()
            return await agent.apply_decision(batch_id, decision)

        outcomes = await asyncio.gather(
            *(
                apply_concurrently(agent, decision)
                for agent, decision in zip(agents, decisions, strict=True)
            ),
            return_exceptions=True,
        )
        selected = [outcome for outcome in outcomes if isinstance(outcome, ReviewItem)]
        conflicts = [
            outcome for outcome in outcomes if isinstance(outcome, RuntimeError)
        ]
        assert len(selected) == 1
        assert len(conflicts) == 1
        assert str(conflicts[0]) == (
            "Review decision conflicts with canonical review decision for "
            f"tenant={storage_one.tenant_id} batch={batch_id} "
            f"original={original.item_id}"
        )
        selected_snapshot = {
            "item_id": selected[0].item_id,
            "data": selected[0].data,
            "confidence": selected[0].confidence,
            "status": selected[0].status,
            "metadata": selected[0].metadata,
        }
        unexpected = [
            outcome
            for outcome in outcomes
            if not isinstance(outcome, (ReviewItem, RuntimeError))
        ]
        assert unexpected == []
        expected_snapshots = [
            {
                "item_id": f"{original.item_id}_regen_0",
                "data": {"query": query},
                "confidence": confidence,
                "status": ApprovalStatus.REGENERATED,
                "metadata": {
                    "agent_type": "routing",
                    "original_item_id": original.item_id,
                    "regeneration_feedback": decision.feedback,
                    "decision": {
                        "reviewer": decision.reviewer,
                        "feedback": decision.feedback,
                        "corrections": dict(decision.corrections),
                        "timestamp": decision.timestamp.isoformat(),
                    },
                },
            }
            for query, confidence, decision in zip(
                [
                    "find the exact PyTorch tutorial",
                    "find the exact JAX tutorial",
                ],
                [0.8125, 0.875],
                decisions,
                strict=True,
            )
        ]

        assert selected_snapshot in expected_snapshots

        persisted = await storage_one.get_batch(batch_id)
        assert persisted.batch_id == batch_id
        persisted_by_id = {item.item_id: item for item in persisted.items}
        assert set(persisted_by_id) == {
            original.item_id,
            selected_snapshot["item_id"],
        }
        assert persisted_by_id[original.item_id].status is ApprovalStatus.REJECTED
        persisted_replacement = persisted_by_id[selected_snapshot["item_id"]]
        assert {
            "data": persisted_replacement.data,
            "confidence": persisted_replacement.confidence,
            "status": persisted_replacement.status,
            "metadata": persisted_replacement.metadata,
        } == {
            key: selected_snapshot[key]
            for key in ("data", "confidence", "status", "metadata")
        }

        replacement_spans = await storage_one.provider.traces.get_spans(
            project=storage_one.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        matching_spans = replacement_spans[
            (replacement_spans["attributes.batch_id"] == batch_id)
            & (replacement_spans["attributes.original_item_id"] == original.item_id)
        ]
        assert len(matching_spans) == 1
        canonical_json = matching_spans.iloc[0]["attributes.replacement_record_json"]
        assert json.loads(canonical_json) == {
            "item_id": persisted_replacement.item_id,
            "data": persisted_replacement.data,
            "confidence": persisted_replacement.confidence,
            "status": ApprovalStatus.REGENERATED.value,
            "metadata": persisted_replacement.metadata,
            "created_at": persisted_replacement.created_at.isoformat(),
            "reviewed_at": None,
        }
        assert (
            matching_spans.iloc[0]["attributes.replacement_record_sha256"]
            == hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        )

    @pytest.mark.parametrize("event_kind", ["duplicate", "conflicting"])
    @pytest.mark.asyncio
    async def test_batch_handles_repeated_replacement_events(
        self,
        approval_storage,
        event_kind,
    ):
        batch_id = f"{event_kind}_replacement_event_{time.time_ns()}"
        original = ReviewItem(
            item_id=f"{batch_id}_0",
            data={"query": "original exact query"},
            confidence=0.4,
        )
        replacement = ReviewItem(
            item_id=f"{original.item_id}_regen_0",
            data={"query": "replacement exact query"},
            confidence=0.8,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "replacement-reviewer",
                    "feedback": "Use the reviewed query.",
                    "corrections": {"query": "replacement exact query"},
                    "timestamp": "2026-08-05T02:00:00+00:00",
                },
            },
        )
        await approval_storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=[original],
                context={
                    "tenant_id": approval_storage.tenant_id,
                    "purpose": f"{event_kind} replacement event contract",
                },
            )
        )
        await approval_storage.replace_item(batch_id, original, replacement)

        spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        matching = spans[
            (spans["attributes.batch_id"] == batch_id)
            & (spans["attributes.original_item_id"] == original.item_id)
        ]
        assert len(matching) == 1
        first = matching.iloc[0]
        record_json = first["attributes.replacement_record_json"]
        if event_kind == "conflicting":
            conflicting_payload = json.loads(record_json)
            conflicting_payload["data"]["query"] = "conflicting exact query"
            record_json = json.dumps(
                conflicting_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        attributes = {
            "batch_id": batch_id,
            "original_item_id": original.item_id,
            "replacement_item_id": replacement.item_id,
            "replacement_record_json": record_json,
            "replacement_record_sha256": hashlib.sha256(
                record_json.encode("utf-8")
            ).hexdigest(),
        }
        with approval_storage.telemetry_manager.span(
            name="approval_item_replacement",
            tenant_id=approval_storage.tenant_id,
            project_name=approval_storage.project_name,
            attributes=attributes,
        ):
            pass
        assert approval_storage.telemetry_manager.force_flush(timeout_millis=10000)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            spans = await approval_storage.provider.traces.get_spans(
                project=approval_storage.full_project_name,
                filters={"name": "approval_item_replacement"},
            )
            matching = spans[
                (spans["attributes.batch_id"] == batch_id)
                & (spans["attributes.original_item_id"] == original.item_id)
            ]
            if len(matching) == 2:
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Second replacement event was not indexed by Phoenix")

        if event_kind == "duplicate":
            batch = await approval_storage.get_batch(batch_id)
            assert batch.batch_id == batch_id
            assert [(item.item_id, item.status, item.data) for item in batch.items] == [
                (original.item_id, ApprovalStatus.REJECTED, original.data),
                (
                    replacement.item_id,
                    ApprovalStatus.REGENERATED,
                    replacement.data,
                ),
            ]
        else:
            with pytest.raises(RuntimeError) as conflict:
                await approval_storage.get_batch(batch_id)
            assert str(conflict.value) == (
                f"Approval batch {batch_id!r} contains conflicting replacement "
                f"events for original item {original.item_id!r}"
            )

    @pytest.mark.asyncio
    async def test_replacement_outage_recovers_the_redis_selected_payload_once(
        self, approval_storage, phoenix_container
    ):
        batch_id = f"replacement_outage_{time.time_ns()}"
        original = ReviewItem(
            item_id=f"{batch_id}_0",
            data={"query": "original outage query"},
            confidence=0.4,
        )
        replacement = ReviewItem(
            item_id=f"{original.item_id}_regen_0",
            data={"query": "corrected outage query"},
            confidence=0.8,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "outage-reviewer",
                    "feedback": "make the query exact",
                    "corrections": {"query": "corrected outage query"},
                    "timestamp": "2026-08-02T00:01:00+00:00",
                },
            },
        )
        await approval_storage.save_batch(
            ApprovalBatch(
                batch_id=batch_id,
                items=[original],
                context={
                    "tenant_id": approval_storage.tenant_id,
                    "purpose": "replacement outage contract",
                },
            )
        )
        before = await approval_storage.get_batch(batch_id)
        assert before.batch_id == batch_id
        assert [(item.item_id, item.status) for item in before.items] == [
            (original.item_id, ApprovalStatus.PENDING_REVIEW)
        ]

        container_name = phoenix_container["container_name"]
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "stop", container_name],
            check=True,
            capture_output=True,
            timeout=30,
        )
        try:
            expected_message = (
                f"batch={batch_id} original={original.item_id} "
                f"replacement={replacement.item_id}"
            )
            with pytest.raises(RuntimeError, match=expected_message):
                await approval_storage.replace_item(batch_id, original, replacement)
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
                pytest.fail("Phoenix did not recover after replacement outage")

        after = await approval_storage.get_batch(batch_id)
        assert after.batch_id == batch_id
        assert [(item.item_id, item.status) for item in after.items] == [
            (original.item_id, ApprovalStatus.PENDING_REVIEW)
        ]

        competing = ReviewItem(
            item_id=f"{original.item_id}_regen_1",
            data={"query": "competing outage query"},
            confidence=0.95,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "outage-reviewer",
                    "feedback": "make the query exact",
                    "corrections": {"query": "corrected outage query"},
                    "timestamp": "2026-08-02T00:01:00+00:00",
                },
            },
        )
        await approval_storage.replace_item(batch_id, original, competing)

        recovered = await approval_storage.get_batch(batch_id)
        assert recovered.batch_id == batch_id
        assert [(item.item_id, item.status) for item in recovered.items] == [
            (original.item_id, ApprovalStatus.REJECTED),
            (replacement.item_id, ApprovalStatus.REGENERATED),
        ]
        recovered_replacement = recovered.items[1]
        assert {
            "data": recovered_replacement.data,
            "confidence": recovered_replacement.confidence,
            "metadata": recovered_replacement.metadata,
        } == {
            "data": replacement.data,
            "confidence": replacement.confidence,
            "metadata": replacement.metadata,
        }

        replacement_spans = await approval_storage.provider.traces.get_spans(
            project=approval_storage.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        matching_spans = replacement_spans[
            (replacement_spans["attributes.batch_id"] == batch_id)
            & (replacement_spans["attributes.original_item_id"] == original.item_id)
        ]
        assert len(matching_spans) == 1
        assert matching_spans.iloc[0]["attributes.replacement_item_id"] == (
            replacement.item_id
        )


@pytest.mark.integration
@pytest.mark.requires_lm
class TestSyntheticServiceIntegration:
    """Synthetic service integration with the configured real LM."""

    @pytest.mark.asyncio
    async def test_routing_generation_matches_response_contract(
        self, synthetic_service, dspy_lm
    ):
        """Routing generation returns validated, entity-grounded examples."""

        request = SyntheticDataRequest(
            tenant_id=synthetic_service.tenant_id,
            optimizer="routing",
            count=1,
            vespa_sample_size=2,
            max_profiles=1,
        )
        response = await synthetic_service.service.generate(request)

        assert response.optimizer == "routing"
        assert response.schema_name == "RoutingExperienceSchema"
        assert response.count == 1
        assert response.selected_profiles == [synthetic_service.base_schema]
        assert response.profile_selection_reasoning == (
            "Rule-based selection for routing. Selected 1 profiles: "
            "video_colpali_smol500_mv_frame (score: 4.50): rich descriptions, "
            "text content, ColPali model"
        )
        assert response.metadata["backend_query_strategy"] == "entity_rich"
        assert response.metadata["sampled_content_count"] == 2
        assert response.metadata["target_count"] == 1
        assert response.metadata["vespa_sample_size"] == 2
        assert len(response.metadata["sampled_content"]) == 2
        assert [set(record) for record in response.metadata["sampled_content"]] == [
            {
                "profile_name",
                "schema_name",
                "source_id",
                "segment_id",
                "description",
            }
        ] * 2
        # The trace exists to say WHICH content was grounded on, so two
        # sampled records must not be the same document.
        assert (
            len(
                {
                    record["description"]
                    for record in response.metadata["sampled_content"]
                }
            )
            == 2
        )
        assert response.metadata["generation"] == {
            "requested_count": 1,
            "returned_count": 1,
            "shortfall_count": 0,
            "floor_count": 1,
            "surplus_exhausted": False,
            "dropped_count": 0,
            "dropped_examples": [],
        }
        assert len(response.data) == 1
        expected_entities = [
            {"text": "Marie Curie", "type": "PERSON"},
            {"text": "Pierre Curie", "type": "PERSON"},
        ]
        for example in response.data:
            validated = RoutingExperienceSchema.model_validate(example)
            assert validated.model_dump() == example
            assert example["entities"] == expected_entities
            for entity in expected_entities:
                assert re.search(
                    rf"(?<!\w){re.escape(entity['text'])}(?!\w)",
                    example["query"],
                    flags=re.IGNORECASE,
                )
            expected_enhanced = example["query"]
            for entity in example["entities"]:
                pattern = re.compile(
                    rf"(?<!\w){re.escape(entity['text'])}(?!\w)",
                    flags=re.IGNORECASE,
                )
                expected_enhanced = pattern.sub(
                    lambda match, entity_type=entity["type"]: (
                        f"{match.group(0)}({entity_type})"
                    ),
                    expected_enhanced,
                )
            assert example["enhanced_query"] == expected_enhanced
            assert example["chosen_agent"] == "search_agent"
            assert example["routing_confidence"] == 0.94


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
