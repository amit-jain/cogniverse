"""BackendQuerier grounds synthetic data in real Vespa content.

Real Vespa (managed by the shared_vespa fixture): deploy a video schema,
feed a metadata doc, and assert _query_profile returns the fed content.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import dspy
import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
)
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    BackendConfig,
    BackendProfileConfig,
    FieldMappingConfig,
)
from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_synthetic.generators import WorkflowGenerator
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_synthetic.service import SyntheticDataService
from cogniverse_synthetic.utils import AgentInferrer
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.utils.synthetic_config import video_synthetic_generator_config
from tests.utils.vespa_test_helpers import deploy_tenant_schema

pytestmark = pytest.mark.integration


def _backend_config(
    shared_vespa,
    tenant_id: str,
    base_schema: str,
    modality: str,
    embedding_type: str,
    pipeline_config: dict | None = None,
) -> BackendConfig:
    return BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        tenant_id=tenant_id,
        profiles={
            base_schema: BackendProfileConfig(
                profile_name=base_schema,
                type=modality,
                schema_name=base_schema,
                embedding_type=embedding_type,
                pipeline_config=pipeline_config or {},
            )
        },
    )


@pytest.mark.asyncio
async def test_query_profile_returns_real_vespa_content(
    shared_vespa, config_manager, schema_loader
):
    base_schema = "video_colpali_smol500_mv_frame"
    tenant = f"bq{uuid.uuid4().hex[:6]}"
    decoy_tenant = f"bq{uuid.uuid4().hex[:6]}"
    schemas = {
        current_tenant: deploy_tenant_schema(
            shared_vespa,
            tenant_id=current_tenant,
            base_schema_name=base_schema,
            config_manager=config_manager,
        )
        for current_tenant in (tenant, decoy_tenant)
    }

    http_port = shared_vespa["http_port"]
    vespa_app = make_vespa_app(url="http://localhost", port=http_port)
    documents = {
        tenant: {
            "video_id": "vidA",
            "video_title": "Robots playing soccer",
            "source_url": "http://example.test/vidA",
            "segment_id": 0,
            "segment_description": "two robots play soccer on a field",
            "start_time": 0.0,
            "end_time": 5.0,
        },
        decoy_tenant: {
            "video_id": "vidB",
            "video_title": "Divers exploring a reef",
            "source_url": "http://example.test/vidB",
            "segment_id": 0,
            "segment_description": "two divers swim above coral",
            "start_time": 10.0,
            "end_time": 15.0,
        },
    }
    for current_tenant, fields in documents.items():
        feed = vespa_app.feed_data_point(
            schema=schemas[current_tenant],
            data_id=f"{fields['video_id']}_seg_0",
            fields=fields,
        )
        assert feed.is_successful(), feed.json

    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        "video",
        "multi_vector",
        {"extract_keyframes": True},
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})

    querier = BackendQuerier(
        backend=backend,
        backend_config=backend_config,
        field_mappings=FieldMappingConfig(),
    )

    samples = []
    for _ in range(20):
        samples = await querier._query_profile(
            {
                "profile_name": base_schema,
                **backend_config.profiles[base_schema].to_dict(),
            },
            sample_size=5,
            strategy="diverse",
            tenant_id=tenant,
        )
        if samples:
            break
        await asyncio.sleep(0.5)

    assert len(samples) == 1
    assert samples[0]["video_id"] == "vidA"
    assert samples[0]["source_id"] == "vidA"
    assert samples[0]["segment_id"] == 0
    assert samples[0]["topic"] == "Robots playing soccer"
    assert samples[0]["description"] == "two robots play soccer on a field"
    assert samples[0]["schema_name"] == base_schema


@pytest.mark.asyncio
async def test_diverse_sampling_with_overfetch_past_default_limit(
    shared_vespa, config_manager, schema_loader
):
    base_schema = "video_colpali_smol500_mv_frame"
    tenant = f"bqv{uuid.uuid4().hex[:6]}"
    corpus_count = 90
    sample_size = 90
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name=base_schema,
        config_manager=config_manager,
    )

    vespa_app = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    )
    for index in range(corpus_count):
        video_id = f"vid-{index:03d}"
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id=f"{video_id}_seg_0",
            fields={
                "video_id": video_id,
                "video_title": f"Robots scene {index:03d}",
                "source_url": f"http://example.test/{video_id}",
                "segment_id": 0,
                "segment_description": f"robots in scene {index:03d}",
                "start_time": float(index),
                "end_time": float(index) + 5.0,
            },
        )
        assert feed.is_successful(), feed.json

    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        "video",
        "multi_vector",
        {"extract_keyframes": True},
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})

    visible_documents = []
    for _ in range(20):
        visible_documents = await asyncio.to_thread(
            backend.query_metadata_documents,
            schema=schema,
            yql=f"select * from sources {schema} where true",
            hits=corpus_count,
        )
        if len(visible_documents) == corpus_count:
            break
        await asyncio.sleep(0.5)

    assert len(visible_documents) == corpus_count

    querier = BackendQuerier(
        backend=backend,
        backend_config=backend_config,
        field_mappings=FieldMappingConfig(),
    )

    samples = await querier._query_profile(
        {
            "profile_name": base_schema,
            **backend_config.profiles[base_schema].to_dict(),
        },
        sample_size=sample_size,
        strategy="diverse",
        tenant_id=tenant,
    )

    assert len(samples) == min(sample_size, corpus_count)


@pytest.mark.asyncio
async def test_service_samples_deployed_configured_profile_from_real_vespa(
    shared_vespa, config_manager, schema_loader, real_telemetry
):
    class _RecordingVespaBackend(VespaBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.schema_checks = []
            self.query_calls = []

        def schema_exists(self, schema_name, tenant_id=None):
            exists = super().schema_exists(schema_name, tenant_id=tenant_id)
            self.schema_checks.append((schema_name, tenant_id, exists))
            return exists

        def query_metadata_documents(
            self,
            schema,
            query=None,
            yql=None,
            **kwargs,
        ):
            self.query_calls.append((schema, yql, kwargs.get("tenant_id")))
            return super().query_metadata_documents(
                schema=schema,
                query=query,
                yql=yql,
                **kwargs,
            )

    base_schema = "video_colpali_smol500_mv_frame"
    tenant = f"bqs{uuid.uuid4().hex[:6]}:media"
    http_port = shared_vespa["http_port"]
    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        "video",
        "multi_vector",
        {"extract_keyframes": True},
    )
    backend = _RecordingVespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})
    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    schema = registry.deploy_schema(
        tenant_id=tenant,
        base_schema_name=base_schema,
    )
    feed = make_vespa_app(
        url="http://localhost",
        port=http_port,
    ).feed_data_point(
        schema=schema,
        data_id="vidDefault_seg_0",
        fields={
            "video_id": "vidDefault",
            "video_title": "Robots playing soccer",
            "source_url": "http://example.test/vidDefault",
            "segment_id": 0,
            "segment_description": "Robots playing soccer",
            "start_time": 0.0,
            "end_time": 5.0,
        },
    )
    assert feed.is_successful(), feed.json

    duplicate_feed = make_vespa_app(
        url="http://localhost",
        port=http_port,
    ).feed_data_point(
        schema=schema,
        data_id="vidDefault_seg_1",
        fields={
            "video_id": "vidDefault",
            "video_title": "Cats playing chess",
            "source_url": "http://example.test/vidDefault",
            "segment_id": 1,
            "segment_description": "Cats playing chess",
            "start_time": 5.0,
            "end_time": 10.0,
        },
    )
    assert duplicate_feed.is_successful(), duplicate_feed.json

    indexed = []
    for _ in range(20):
        indexed = backend.query_metadata_documents(
            schema=schema,
            yql=f"select * from sources {schema} where true limit 2",
            hits=2,
        )
        if len(indexed) == 2:
            break
        await asyncio.sleep(0.5)
    assert len(indexed) == 2
    assert {item["video_id"] for item in indexed} == {"vidDefault"}
    assert {item["video_title"] for item in indexed} == {
        "Robots playing soccer",
        "Cats playing chess",
    }
    assert {item["segment_description"] for item in indexed} == {
        "Robots playing soccer",
        "Cats playing chess",
    }
    backend.query_calls.clear()

    profile_agent = ProfileSelectionAgent(
        deps=ProfileSelectionDeps(available_profiles=[base_schema])
    )
    profile_agent.set_telemetry_manager(real_telemetry)
    profile_agent.dspy_module.selector = lambda **_: dspy.Prediction(
        selected_profile=base_schema,
        confidence="0.95",
        reasoning="The production selector chose the indexed video profile.",
        query_intent="video_search",
        modality="video",
        complexity="medium",
    )

    async def label_profile(
        query: str, available_profiles: list[str], request_tenant_id: str
    ):
        return await profile_agent.process(
            ProfileSelectionInput(
                query=query,
                available_profiles=available_profiles,
                tenant_id=request_tenant_id,
            )
        )

    service = SyntheticDataService(
        backend=backend,
        backend_config=backend_config,
        generator_config=video_synthetic_generator_config(tenant),
        agents_config=json.loads(Path("configs/config.json").read_text())["agents"],
        config_manager=config_manager,
        profile_labeler=label_profile,
    )

    config_manager.add_backend_profile(
        backend_config.profiles[base_schema],
        tenant_id=tenant,
    )

    response = await service.generate(
        SyntheticDataRequest(
            tenant_id=tenant,
            optimizer="profile",
            count=1,
            vespa_sample_size=2,
            strategy="diverse",
            max_profiles=1,
        )
    )

    assert response.selected_profiles == [base_schema]
    assert response.metadata["sampled_content_count"] == 2
    assert len(response.data) == 1
    assert response.data[0]["query"] in {
        "find a video frame showing Robots playing soccer",
        "find a video frame showing Cats playing chess",
    }
    expected_query = response.data[0]["query"]
    assert response.data[0] == {
        "query": expected_query,
        "available_profiles": base_schema,
        "selected_profile": base_schema,
        "modality": "video",
        "complexity": "medium",
        "query_intent": "video_search",
        "reasoning": "The production selector chose the indexed video profile.",
    }
    assert backend.schema_checks == [(base_schema, tenant, True)]
    assert backend.query_calls == [
        (
            base_schema,
            f"select * from sources {base_schema} where true limit 10",
            tenant,
        )
    ]


@pytest.mark.requires_lm
@pytest.mark.asyncio
async def test_routing_service_keeps_real_vespa_sources_and_agents_aligned(
    shared_vespa,
    config_manager,
    schema_loader,
    dspy_test_lm,
    real_telemetry,
):
    _ = (dspy_test_lm, real_telemetry)
    tenant = f"bqr{uuid.uuid4().hex[:6]}:media"
    profile_specs = {
        "document_text": BackendProfileConfig(
            profile_name="document_text",
            type="document",
            description="Document text",
            schema_name="document_text",
            embedding_type="single_vector",
            pipeline_config={"generate_descriptions": True},
        ),
        "video_colpali_smol500_mv_frame": BackendProfileConfig(
            profile_name="video_colpali_smol500_mv_frame",
            type="video",
            description="Video frames",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_type="multi_vector",
            pipeline_config={},
        ),
    }
    schemas = {
        profile_name: deploy_tenant_schema(
            shared_vespa,
            tenant_id=tenant,
            base_schema_name=profile.schema_name,
            config_manager=config_manager,
        )
        for profile_name, profile in profile_specs.items()
    }
    vespa_app = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    )
    source_documents = {
        "document_text": (
            "tensorflow-guide",
            {
                "document_id": "tensorflow-guide",
                "document_title": "guide to TensorFlow",
                "creation_timestamp": int(time.time() * 1000),
                "document_type": "guide",
                "document_path": "/documents/tensorflow.pdf",
                "page_count": 1,
                "full_text": "reference material for TensorFlow",
                "section_headings": "TensorFlow",
            },
        ),
        "video_colpali_smol500_mv_frame": (
            "pytorch-video-segment-0",
            {
                "video_id": "pytorch-video",
                "video_title": "tutorial on PyTorch",
                "source_url": "http://example.test/pytorch-video",
                "segment_id": 0,
                "segment_description": "practical lesson for PyTorch",
                "start_time": 0.0,
                "end_time": 5.0,
            },
        ),
    }
    for profile_name, (document_id, fields) in source_documents.items():
        feed = vespa_app.feed_data_point(
            schema=schemas[profile_name],
            data_id=document_id,
            fields=fields,
        )
        assert feed.is_successful(), feed.json
        duplicate_fields = dict(fields)
        duplicate_document_id = f"{document_id}-copy"
        if "document_id" in duplicate_fields:
            duplicate_fields["document_id"] = duplicate_document_id
        elif "video_id" in duplicate_fields:
            duplicate_fields["video_id"] = duplicate_document_id
        elif "image_id" in duplicate_fields:
            duplicate_fields["image_id"] = duplicate_document_id
        elif "code_id" in duplicate_fields:
            duplicate_fields["code_id"] = duplicate_document_id
        elif "doc_id" in duplicate_fields:
            duplicate_fields["doc_id"] = duplicate_document_id
        feed = vespa_app.feed_data_point(
            schema=schemas[profile_name],
            data_id=duplicate_document_id,
            fields=duplicate_fields,
        )
        assert feed.is_successful(), feed.json

    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        tenant_id=tenant,
        profiles=profile_specs,
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})
    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    indexed_sources = {}
    for profile_name, schema in schemas.items():
        id_field = "document_id" if profile_name == "document_text" else "video_id"
        for _ in range(20):
            indexed = await asyncio.to_thread(
                backend.query_metadata_documents,
                schema=schema,
                yql=f"select * from sources {schema} where true limit 1",
                hits=1,
            )
            if indexed:
                indexed_sources[profile_name] = indexed[0][id_field]
                break
            await asyncio.sleep(0.5)
        assert indexed_sources[profile_name] == (
            "tensorflow-guide" if profile_name == "document_text" else "pytorch-video"
        )

    generator_config = video_synthetic_generator_config(tenant)
    generator_config.optimizer_configs["modality"].agent_mappings = [
        AgentMappingRule(modality="DOCUMENT", agent_name="document_agent"),
        AgentMappingRule(modality="VIDEO", agent_name="video_search_agent"),
    ]
    generator_config.optimizer_configs["routing"].dspy_modules[
        "query_generator"
    ].metadata = {"max_retries": 3}
    entity_agent = EntityExtractionAgent(deps=EntityExtractionDeps())
    entity_agent.set_telemetry_manager(real_telemetry)

    async def extract_entities(text: str, tenant_id: str):
        return await entity_agent.process(
            EntityExtractionInput(query=text, tenant_id=tenant_id)
        )

    async def decide_route(query: str, request_tenant_id: str):
        assert request_tenant_id == tenant
        if "TensorFlow" in query:
            return {
                "routed_to": "document_agent",
                "confidence": 0.97,
            }
        if "PyTorch" in query:
            return {
                "routed_to": "video_search_agent",
                "confidence": 0.96,
            }
        raise AssertionError(f"route query lost its source entity: {query!r}")

    service = SyntheticDataService(
        backend=backend,
        backend_config=backend_config,
        generator_config=generator_config,
        agents_config={
            "gateway_agent": {
                "enabled": True,
                "modalities": [],
                "capabilities": ["gateway", "classification"],
                "timeout": 10,
            },
            "entity_extraction_agent": {
                "enabled": True,
                "modalities": [],
                "capabilities": ["entity_extraction", "ner"],
                "timeout": 15,
            },
            "document_agent": {
                "enabled": True,
                "modalities": ["DOCUMENT"],
                "capabilities": ["document_analysis"],
            },
            "video_search_agent": {
                "enabled": True,
                "modalities": ["VIDEO"],
                "capabilities": ["video_search"],
            },
        },
        entity_extractor=extract_entities,
        routing_decider=decide_route,
    )
    routing_generator = service._get_generator("routing")
    assert routing_generator.production_label_timeout_seconds == 300.0
    assert routing_generator.entity_labeler.extraction_timeout_seconds == 300.0

    response = await service.generate(
        SyntheticDataRequest(
            tenant_id=tenant,
            optimizer="routing",
            count=2,
            vespa_sample_size=2,
            strategy="diverse",
            max_profiles=2,
        )
    )

    assert response.selected_profiles == [
        "video_colpali_smol500_mv_frame",
        "document_text",
    ]
    assert response.metadata["sampled_content_count"] == 2
    assert [
        {
            "entities": item["entities"],
            "chosen_agent": item["chosen_agent"],
            "routing_confidence": item["routing_confidence"],
        }
        for item in response.data
    ] == [
        {
            "entities": [
                {"text": "practical lesson", "type": "EVENT"},
                {"text": "PyTorch", "type": "TECHNOLOGY"},
            ],
            "chosen_agent": "video_search_agent",
            "routing_confidence": 0.96,
        },
        {
            "entities": [
                {"text": "reference material", "type": "CONCEPT"},
                {"text": "TensorFlow", "type": "TECHNOLOGY"},
            ],
            "chosen_agent": "document_agent",
            "routing_confidence": 0.97,
        },
    ]
    for item, own_entity, other_entity in zip(
        response.data,
        ("PyTorch", "TensorFlow"),
        ("TensorFlow", "PyTorch"),
        strict=True,
    ):
        assert own_entity.casefold() in item["query"].casefold()
        assert other_entity.casefold() not in item["query"].casefold()
        for entity in item["entities"]:
            assert entity["text"].casefold() in item["query"].casefold()
            assert (
                f"{entity['text']}({entity['type']})".casefold()
                in item["enhanced_query"].casefold()
            )
        assert item["metadata"]["_generation_metadata"]["max_retries"] == 3
        assert item["metadata"]["_generation_metadata"]["retry_count"] in {
            0,
            1,
            2,
        }


@pytest.mark.parametrize(
    (
        "base_schema",
        "document_id",
        "fields",
        "modality",
        "embedding_type",
        "expected_topic",
        "expected_description",
        "expected_modality",
        "expected_agent",
    ),
    [
        (
            "image_colpali_mv",
            "apollo-launch",
            {
                "image_id": "apollo-launch",
                "image_title": "Saturn V launch",
                "source_url": "http://example.test/apollo-launch",
                "creation_timestamp": 1_724_000_000_000,
                "image_description": "The rocket clears the launch tower.",
                "image_path": "/images/apollo-launch.jpg",
                "image_width": 2048,
                "image_height": 1365,
            },
            "image",
            "multi_vector",
            "Saturn V launch",
            "The rocket clears the launch tower.",
            "IMAGE",
            "image_source_agent",
        ),
        (
            "document_text",
            "apollo-flight-plan",
            {
                "document_id": "apollo-flight-plan",
                "document_title": "Apollo 11 flight plan",
                "creation_timestamp": 1_724_000_000_000,
                "document_type": "flight_plan",
                "document_path": "/documents/apollo-11.pdf",
                "page_count": 117,
                "full_text": "The plan specifies the lunar landing sequence.",
                "section_headings": "Lunar landing sequence",
            },
            "document",
            "single_vector",
            "Apollo 11 flight plan",
            "The plan specifies the lunar landing sequence.",
            "DOCUMENT",
            "document_source_agent",
        ),
        (
            "code_lateon_mv",
            "config-parser",
            {
                "code_id": "config-parser",
                "file_path": "cogniverse/config.py",
                "chunk_name": "parse_runtime_config",
                "chunk_type": "function",
                "language": "python",
                "signature": "parse_runtime_config(raw: dict) -> RuntimeConfig",
                "line_start": 40,
                "line_end": 52,
                "source_code": "def parse_runtime_config(raw):\n    return RuntimeConfig(**raw)",
            },
            "code",
            "multi_vector",
            "parse_runtime_config",
            "def parse_runtime_config(raw):\n    return RuntimeConfig(**raw)",
            "CODE",
            "code_source_agent",
        ),
        (
            "wiki_pages",
            "redis-lease",
            {
                "doc_id": "redis-lease",
                "page_type": "architecture",
                "title": "Redis lease coordination",
                "content": "A renewable tenant lease serializes shared writes.",
                "slug": "redis-lease-coordination",
            },
            "wiki",
            "single_vector",
            "Redis lease coordination",
            "A renewable tenant lease serializes shared writes.",
            "WIKI",
            "wiki_source_agent",
        ),
    ],
    ids=["image", "document", "code", "wiki"],
)
@pytest.mark.asyncio
async def test_default_field_mappings_read_real_non_video_content(
    shared_vespa,
    config_manager,
    schema_loader,
    base_schema,
    document_id,
    fields,
    modality,
    embedding_type,
    expected_topic,
    expected_description,
    expected_modality,
    expected_agent,
):
    tenant = f"bqf{uuid.uuid4().hex[:6]}"
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name=base_schema,
        config_manager=config_manager,
    )
    feed = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    ).feed_data_point(schema=schema, data_id=document_id, fields=fields)
    assert feed.is_successful(), feed.json

    duplicate_fields = dict(fields)
    duplicate_document_id = f"{document_id}-copy"
    if "document_id" in duplicate_fields:
        duplicate_fields["document_id"] = duplicate_document_id
    elif "video_id" in duplicate_fields:
        duplicate_fields["video_id"] = duplicate_document_id
    elif "image_id" in duplicate_fields:
        duplicate_fields["image_id"] = duplicate_document_id
    elif "code_id" in duplicate_fields:
        duplicate_fields["code_id"] = duplicate_document_id
    elif "doc_id" in duplicate_fields:
        duplicate_fields["doc_id"] = duplicate_document_id
    if base_schema == "image_colpali_mv":
        duplicate_fields["image_title"] = "Falcon V landing"
    elif base_schema == "document_text":
        duplicate_fields["document_title"] = "Gemini 11 flight report"
    elif base_schema == "code_lateon_mv":
        duplicate_fields["chunk_name"] = "load_runtime_metadata"
    elif base_schema == "wiki_pages":
        duplicate_fields["title"] = "Postgres lease management"
    feed = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    ).feed_data_point(
        schema=schema,
        data_id=duplicate_document_id,
        fields=duplicate_fields,
    )
    assert feed.is_successful(), feed.json

    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        modality,
        embedding_type,
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})
    querier = BackendQuerier(
        backend=backend,
        backend_config=backend_config,
        field_mappings=FieldMappingConfig(),
    )

    samples = []
    for _ in range(20):
        samples = await querier._query_profile(
            {
                "profile_name": base_schema,
                **backend_config.profiles[base_schema].to_dict(),
            },
            sample_size=2,
            strategy="diverse",
            tenant_id=tenant,
        )
        if len(samples) == 2:
            break
        await asyncio.sleep(0.5)

    assert len(samples) == 2
    topics = [sample["topic"] for sample in samples]
    assert expected_topic in topics
    assert len({topic for topic in topics if topic}) == 2
    assert {sample["description"] for sample in samples} == {expected_description}
    primary_sample = next(
        sample for sample in samples if sample["topic"] == expected_topic
    )
    assert primary_sample["schema_name"] == base_schema
    assert primary_sample["profile_type"] == modality
    assert primary_sample["modality"] == expected_modality

    required_capability = {
        "IMAGE": "image_search",
        "DOCUMENT": "document_analysis",
        "CODE": "coding",
        "WIKI": "document_analysis",
    }[expected_modality]
    inferrer = AgentInferrer(
        agents_config={
            expected_agent: {
                "enabled": True,
                "modalities": [expected_modality],
                "capabilities": [required_capability],
            }
        },
        agent_mappings=[
            AgentMappingRule(
                modality=expected_modality,
                agent_name=expected_agent,
            )
        ],
    )
    ordered_samples = [
        primary_sample,
        *[sample for sample in samples if sample is not primary_sample],
    ]
    workflow_samples = [{**sample, "description": ""} for sample in ordered_samples]
    workflow_expected_topic = (
        expected_topic.replace("_", " ")
        if base_schema == "code_lateon_mv"
        else expected_topic
    )
    example = (
        await WorkflowGenerator(agent_inferrer=inferrer).generate(
            workflow_samples,
            target_count=1,
        )
    )[0]

    assert example.query == f"find {workflow_expected_topic}"
    assert example.query_type == expected_modality
    assert example.agent_sequence == [expected_agent]
    assert example.task_count == 1


@pytest.mark.asyncio
async def test_entity_rich_audio_query_uses_real_schema_fields(
    shared_vespa, config_manager, schema_loader
):
    class _RecordingVespaBackend(VespaBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.query_calls = []

        def query_metadata_documents(
            self,
            schema,
            query=None,
            yql=None,
            **kwargs,
        ):
            self.query_calls.append({"yql": yql, "kwargs": kwargs})
            return super().query_metadata_documents(
                schema=schema,
                query=query,
                yql=yql,
                **kwargs,
            )

    base_schema = "audio_content"
    tenant = f"bqa{uuid.uuid4().hex[:6]}"
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name=base_schema,
        config_manager=config_manager,
    )
    http_port = shared_vespa["http_port"]
    vespa_app = make_vespa_app(url="http://localhost", port=http_port)
    now_ms = int(time.time() * 1000)
    documents = [
        {
            "audio_id": f"blank-{index}",
            "audio_title": f"Blank transcript {index}",
            "creation_timestamp": now_ms - index,
            "audio_transcript": "",
        }
        for index in range(10)
    ] + [
        {
            "audio_id": "audio-rich",
            "audio_title": "The discovery of radium",
            "creation_timestamp": now_ms - 10_000,
            "audio_transcript": "Marie Curie and Pierre Curie discovered radium.",
        }
    ]
    for document in documents:
        audio_id = document["audio_id"]
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id=audio_id,
            fields={
                **document,
                "source_url": f"http://example.test/{audio_id}",
                "audio_path": f"/recordings/{audio_id}.wav",
                "audio_duration": 12.5,
                "audio_language": "en",
            },
        )
        assert feed.is_successful(), feed.json

    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        "audio",
        "single_vector",
        {"transcribe_audio": True},
    )
    backend = _RecordingVespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})
    visible_ids = set()
    for _ in range(20):
        visible = await asyncio.to_thread(
            backend.query_metadata_documents,
            schema=schema,
            yql=(
                f"select * from sources {schema} where true "
                "order by creation_timestamp desc limit 11"
            ),
            hits=11,
        )
        visible_ids = {document.get("audio_id") for document in visible}
        if visible_ids == {document["audio_id"] for document in documents}:
            break
        await asyncio.sleep(0.5)
    assert visible_ids == {document["audio_id"] for document in documents}
    backend.query_calls.clear()

    querier = BackendQuerier(
        backend=backend,
        backend_config=backend_config,
        field_mappings=FieldMappingConfig(),
    )

    samples = []
    for _ in range(20):
        samples = await querier._query_profile(
            {
                "profile_name": base_schema,
                **backend_config.profiles[base_schema].to_dict(),
                "pipeline_config": {
                    "generate_descriptions": False,
                    "transcribe_audio": True,
                },
            },
            sample_size=1,
            strategy="entity_rich",
            tenant_id=tenant,
        )
        if samples:
            break
        await asyncio.sleep(0.5)

    assert len(samples) == 1
    assert samples[0]["topic"] == "The discovery of radium"
    assert samples[0]["transcript"] == "Marie Curie and Pierre Curie discovered radium."
    assert samples[0]["schema_name"] == base_schema
    assert backend.query_calls == [
        {
            "yql": (
                f"select * from sources {base_schema} where true "
                "order by creation_timestamp desc limit 10"
            ),
            "kwargs": {"hits": 10, "tenant_id": tenant},
        },
        {
            "yql": (
                f"select * from sources {base_schema} where true "
                "order by creation_timestamp desc limit 20 offset 10"
            ),
            "kwargs": {"hits": 10, "tenant_id": tenant, "offset": 10},
        },
    ]


@pytest.mark.asyncio
async def test_entity_rich_rejects_profile_without_text_pipeline_before_real_query(
    shared_vespa, config_manager, schema_loader
):
    class _RecordingVespaBackend(VespaBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.query_calls = []

        def query_metadata_documents(
            self,
            schema,
            query=None,
            yql=None,
            **kwargs,
        ):
            self.query_calls.append((schema, yql))
            return super().query_metadata_documents(
                schema=schema,
                query=query,
                yql=yql,
                **kwargs,
            )

    base_schema = "audio_content"
    tenant = f"bqe{uuid.uuid4().hex[:6]}"
    deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name=base_schema,
        config_manager=config_manager,
    )
    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        "audio",
        "single_vector",
        {
            "generate_descriptions": False,
            "transcribe_audio": False,
        },
    )
    backend = _RecordingVespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})
    querier = BackendQuerier(
        backend=backend,
        backend_config=backend_config,
        field_mappings=FieldMappingConfig(
            topic_fields=["audio_title"],
            transcript_fields=["audio_transcript"],
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "^entity_rich requires the profile pipeline to generate descriptions "
            "or transcribe audio$"
        ),
    ):
        await querier._query_profile(
            {
                **backend_config.profiles[base_schema].to_dict(),
                "pipeline_config": {
                    "generate_descriptions": False,
                    "transcribe_audio": False,
                },
            },
            sample_size=1,
            strategy="entity_rich",
            tenant_id=tenant,
        )

    assert backend.query_calls == []


@pytest.mark.asyncio
async def test_temporal_recent_returns_newest_real_vespa_documents_first(
    shared_vespa, config_manager, schema_loader
):
    base_schema = "audio_content"
    tenant = f"bqt{uuid.uuid4().hex[:6]}"
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name=base_schema,
        config_manager=config_manager,
    )
    vespa_app = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    )
    now_ms = int(time.time() * 1000)
    temporal_window_ms = 90 * 24 * 60 * 60 * 1000
    timestamps = {
        "audio-outside-window": now_ms - temporal_window_ms - 3_600_000,
        "audio-inside-window": now_ms - temporal_window_ms + 3_600_000,
        "audio-middle": now_ms - 200_000,
        "audio-newest": now_ms - 100_000,
    }
    for audio_id, created_at in timestamps.items():
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id=audio_id,
            fields={
                "audio_id": audio_id,
                "audio_title": audio_id,
                "source_url": f"http://example.test/{audio_id}",
                "creation_timestamp": created_at,
                "audio_transcript": f"Transcript for {audio_id}",
                "audio_path": f"/recordings/{audio_id}.wav",
                "audio_duration": 10.0,
                "audio_language": "en",
            },
        )
        assert feed.is_successful(), feed.json

    backend_config = _backend_config(
        shared_vespa,
        tenant,
        base_schema,
        "audio",
        "single_vector",
        {"transcribe_audio": True},
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant})

    expected_documents = timestamps
    visible_documents = {}
    unfiltered_yql = f"select * from sources {schema} where true limit 4"
    for _ in range(20):
        indexed_documents = await asyncio.to_thread(
            backend.query_metadata_documents,
            schema=schema,
            yql=unfiltered_yql,
            hits=4,
        )
        visible_documents = {
            document.get("audio_id"): document.get("creation_timestamp")
            for document in indexed_documents
        }
        if visible_documents == expected_documents:
            break
        await asyncio.sleep(0.5)

    assert visible_documents == expected_documents

    querier = BackendQuerier(
        backend=backend,
        backend_config=backend_config,
        field_mappings=FieldMappingConfig(
            topic_fields=["audio_title"],
            transcript_fields=["audio_transcript"],
            metadata_fields={
                "audio_id": "audio_id",
                "creation_timestamp": "creation_timestamp",
            },
        ),
    )

    samples = await querier._query_profile(
        {
            "profile_name": base_schema,
            **backend_config.profiles[base_schema].to_dict(),
        },
        sample_size=4,
        strategy="temporal_recent",
        tenant_id=tenant,
    )

    assert [sample["audio_id"] for sample in samples] == [
        "audio-newest",
        "audio-middle",
        "audio-inside-window",
    ]
    assert [sample["creation_timestamp"] for sample in samples] == [
        timestamps["audio-newest"],
        timestamps["audio-middle"],
        timestamps["audio-inside-window"],
    ]
    assert {sample["audio_id"] for sample in samples}.isdisjoint(
        {"audio-outside-window"}
    )
