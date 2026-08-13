"""
Integration tests for SyntheticDataService

Tests the main service orchestrator end-to-end.
"""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    BackendConfig,
    BackendProfileConfig,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    ProfileScoringRule,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.registry import OPTIMIZER_REGISTRY, get_optimizer_config
from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    SyntheticDataRequest,
    WorkflowExecutionSchema,
)
from cogniverse_synthetic.service import SyntheticDataService

pytestmark = [pytest.mark.unit]


def create_test_agent_mappings() -> list[AgentMappingRule]:
    return [
        AgentMappingRule(modality="VIDEO", agent_name="video_search"),
        AgentMappingRule(modality="DOCUMENT", agent_name="document_agent"),
        AgentMappingRule(modality="IMAGE", agent_name="image_search"),
        AgentMappingRule(modality="AUDIO", agent_name="audio_analysis"),
    ]


def create_test_generator_config(*, synthetic_generation_timeout_seconds: float = 23.0):
    """Create test generator configuration with all required optimizer configs"""
    scoring_configs = {
        optimizer_name: OptimizerGenerationConfig(
            optimizer_type=optimizer_name,
            profile_scoring_rules=[
                ProfileScoringRule(
                    condition={"field": "type", "equals": "video"},
                    score_adjustment=1.0,
                    reason=f"video source for {optimizer_name}",
                )
            ],
        )
        for optimizer_name in OPTIMIZER_REGISTRY
    }
    scoring_configs["routing"].dspy_modules = {
        "query_generator": DSPyModuleConfig(
            signature_class=(
                "cogniverse_synthetic.dspy_signatures.GenerateEntityQuery"
            ),
            module_type="Predict",
        )
    }
    scoring_configs["modality"] = OptimizerGenerationConfig(
        optimizer_type="modality",
        agent_mappings=create_test_agent_mappings(),
    )
    return SyntheticGeneratorConfig(
        tenant_id="test:unit",
        synthetic_generation_timeout_seconds=synthetic_generation_timeout_seconds,
        optimizer_configs=scoring_configs,
    )


def create_test_agents_config():
    return {
        "gateway_agent": {
            "enabled": True,
            "modalities": [],
            "capabilities": ["routing"],
            "timeout": 11,
        },
        "entity_extraction_agent": {
            "enabled": True,
            "modalities": [],
            "capabilities": ["entity_extraction"],
            "timeout": 13,
        },
        "query_enhancement_agent": {
            "enabled": True,
            "modalities": [],
            "capabilities": ["query_enhancement"],
            "timeout": 17,
        },
        "profile_selection_agent": {
            "enabled": True,
            "modalities": [],
            "capabilities": ["profile_selection"],
            "timeout": 19,
        },
        "text_analysis_agent": {
            "enabled": True,
            "modalities": ["DOCUMENT"],
            "capabilities": ["text_analysis"],
        },
        "video_search": {
            "enabled": True,
            "modalities": ["VIDEO"],
            "capabilities": ["video_search"],
        },
        "document_agent": {
            "enabled": True,
            "modalities": ["DOCUMENT"],
            "capabilities": ["document_analysis"],
        },
        "image_search": {
            "enabled": True,
            "modalities": ["IMAGE"],
            "capabilities": ["image_search"],
        },
        "audio_analysis": {
            "enabled": True,
            "modalities": ["AUDIO"],
            "capabilities": ["audio_analysis"],
        },
        "summarizer": {
            "enabled": True,
            "modalities": [],
            "capabilities": ["summarization"],
        },
        "reporter": {
            "enabled": True,
            "modalities": [],
            "capabilities": ["detailed_report"],
        },
    }


def create_test_backend_config() -> BackendConfig:
    return BackendConfig(
        tenant_id="test:unit",
        profiles={
            "video_frames": BackendProfileConfig(
                profile_name="video_frames",
                type="video",
                schema_name="video_segments",
                embedding_type="multi_vector",
                pipeline_config={
                    "extract_keyframes": True,
                    "generate_descriptions": True,
                },
            ),
            "audio_semantic": BackendProfileConfig(
                profile_name="audio_semantic",
                type="audio",
                schema_name="audio_segments",
                embedding_type="multi_vector",
                pipeline_config={"transcribe_audio": True},
            ),
        },
    )


class _GroundedBackend:
    def schema_exists(self, schema_name, tenant_id=None):
        return True

    def get_tenant_schema_name(self, tenant_id, base_schema_name):
        return f"{base_schema_name}_{tenant_id.replace(':', '_')}"

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        return [
            {
                "video_title": "Saturn V launch",
                "segment_description": "The Saturn V rocket clears the launch tower.",
                "audio_transcript": "Saturn V mission control confirms ignition.",
                "schema": schema,
            }
        ]


class _BackendAccessRecorder(_GroundedBackend):
    def __init__(self) -> None:
        self.calls = []

    def schema_exists(self, schema_name, tenant_id=None):
        self.calls.append(("schema_exists", schema_name, tenant_id))
        return True

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        self.calls.append(("query_metadata_documents", schema, kwargs))
        return super().query_metadata_documents(schema, query=query, yql=yql, **kwargs)


def _generator_config_with_mappings(
    agent_mappings: list[AgentMappingRule],
) -> SyntheticGeneratorConfig:
    config = create_test_generator_config()
    config.optimizer_configs["modality"].agent_mappings = agent_mappings
    return config


async def _extract_grounded_entities(text: str, tenant_id: str) -> dict:
    assert tenant_id == "test:unit"
    assert "Saturn V" in text
    return {
        "query": text,
        "entities": [{"text": "Saturn V", "type": "TECHNOLOGY"}],
        "relationships": [],
    }


async def _route_grounded_query(query: str, tenant_id: str) -> dict:
    assert tenant_id == "test:unit"
    return {"query": query, "routed_to": "video_search", "confidence": 0.79}


async def _label_grounded_profile(
    query: str, available_profiles: list[str], tenant_id: str
) -> dict:
    selected_profile = available_profiles[0]
    if "audio" in selected_profile or selected_profile == "profile_a":
        modality = "audio"
    elif "document" in selected_profile or selected_profile == "profile_b":
        modality = "document"
    elif "image" in selected_profile:
        modality = "image"
    else:
        modality = "video"
    return {
        "query": query,
        "selected_profile": selected_profile,
        "reasoning": f"Production selector chose {selected_profile}.",
        "query_intent": f"{modality}_search",
        "modality": modality,
        "complexity": "medium",
    }


async def _enhance_grounded_query(query: str, tenant_id: str) -> dict:
    assert tenant_id == "test:unit"
    return {
        "original_query": query,
        "enhanced_query": f"{query} mission",
        "expansion_terms": ["mission"],
        "synonyms": ["flight"],
        "reasoning": "The production enhancer added a source-grounded term.",
    }


def create_test_service(
    *,
    agents_config: dict | None = None,
    generator_config: SyntheticGeneratorConfig | None = None,
) -> SyntheticDataService:
    return SyntheticDataService(
        backend=_GroundedBackend(),
        backend_config=create_test_backend_config(),
        generator_config=(
            create_test_generator_config()
            if generator_config is None
            else generator_config
        ),
        agents_config=(
            create_test_agents_config() if agents_config is None else agents_config
        ),
        entity_extractor=_extract_grounded_entities,
        routing_decider=_route_grounded_query,
        query_enhancer=_enhance_grounded_query,
        profile_labeler=_label_grounded_profile,
    )


class TestSyntheticDataService:
    """Integration tests for SyntheticDataService"""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service can be initialized"""
        service = create_test_service()
        assert service.profile_selector is not None
        assert service.backend_querier is not None
        assert service.pattern_extractor is not None
        assert service.agent_inferrer is not None
        # Generators are initialized lazily, so check starts empty
        assert isinstance(service.generators, dict)

    def test_callback_generators_use_synthetic_generation_timeout(self):
        agents_config = create_test_agents_config()
        service = create_test_service(agents_config=agents_config)

        routing = service._get_generator("routing")
        entity = service._get_generator("entity_extraction")
        query_enhancement = service._get_generator("query_enhancement")
        profile = service._get_generator("profile")
        cross_modal = service._get_generator("cross_modal")

        assert service.agents_config is agents_config
        assert service.generator_config.synthetic_generation_timeout_seconds == 23.0
        assert routing.production_label_timeout_seconds == 23.0
        assert routing.entity_labeler.extraction_timeout_seconds == 23.0
        assert entity.extraction_timeout_seconds == 23.0
        assert query_enhancement.production_label_timeout_seconds == 23.0
        assert profile.production_label_timeout_seconds == 23.0
        assert cross_modal is profile

    @pytest.mark.parametrize(
        ("agent_name", "optimizer_name"),
        [
            ("gateway_agent", "routing"),
            ("entity_extraction_agent", "routing"),
            ("entity_extraction_agent", "entity_extraction"),
            ("query_enhancement_agent", "query_enhancement"),
            ("profile_selection_agent", "profile"),
            ("profile_selection_agent", "cross_modal"),
        ],
    )
    def test_callback_generators_ignore_missing_agent_timeout(
        self,
        agent_name,
        optimizer_name,
    ):
        agents_config = create_test_agents_config()
        del agents_config[agent_name]["timeout"]
        service = create_test_service(agents_config=agents_config)

        generator = service._get_generator(optimizer_name)

        assert service.generator_config.synthetic_generation_timeout_seconds == 23.0
        if optimizer_name == "routing":
            assert generator.production_label_timeout_seconds == 23.0
            assert generator.entity_labeler.extraction_timeout_seconds == 23.0
        elif optimizer_name == "entity_extraction":
            assert generator.extraction_timeout_seconds == 23.0
        else:
            assert generator.production_label_timeout_seconds == 23.0
        assert service.generators
        assert service.generators == {generator.__class__.__name__: generator}

    @pytest.mark.parametrize(
        "timeout",
        [True, "15", 0, -1, float("nan"), float("inf")],
    )
    def test_synthetic_generation_timeout_rejects_invalid_values(self, timeout):
        with pytest.raises(ValueError) as raised:
            create_test_generator_config(synthetic_generation_timeout_seconds=timeout)

        assert str(raised.value) == (
            "synthetic_generation_timeout_seconds must be finite and positive"
        )

    @pytest.mark.asyncio
    async def test_service_gateway_timeout_bounds_hung_routing_callback(self):
        never_released = asyncio.Event()

        async def hung_routing_decider(query, tenant_id):
            await never_released.wait()

        class _GroundedQueryGenerator:
            max_retries = 3

            def __call__(self, *, topics, entities, entity_types):
                assert entities == ["Saturn V"]
                assert entity_types == ["TECHNOLOGY"]
                result = type("QueryResult", (), {})()
                result.query = "find Saturn V"
                result.reasoning = f"Use Saturn V from {topics}."
                result._retry_count = 0
                result._max_retries = self.max_retries
                return result

        class _GroundedPatternExtractor:
            def extract(self, records):
                assert records == [{"title": "Saturn V launch"}]
                return {"topics": ["Saturn V launch"]}

        generator_config = create_test_generator_config(
            synthetic_generation_timeout_seconds=0.02
        )
        agents_config = create_test_agents_config()
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=create_test_backend_config(),
            generator_config=generator_config,
            agents_config=agents_config,
            entity_extractor=_extract_grounded_entities,
            routing_decider=hung_routing_decider,
        )
        generator = service._get_generator("routing")
        generator.pattern_extractor = _GroundedPatternExtractor()
        generator.query_generator = _GroundedQueryGenerator()

        with pytest.raises(TimeoutError) as raised:
            await asyncio.wait_for(
                generator.generate(
                    [{"title": "Saturn V launch"}],
                    target_count=1,
                    tenant_id="test:unit",
                ),
                timeout=1.0,
            )

        assert str(raised.value) == (
            "routing optimizer callback routing_decider timed out after 0.02 "
            "seconds for tenant='test:unit' query='find Saturn V'"
        )
        assert isinstance(raised.value.__cause__, TimeoutError)

    def test_service_requires_backend(self):
        with pytest.raises(ValueError, match="^backend is required$"):
            SyntheticDataService(
                backend=None,
                backend_config=create_test_backend_config(),
                generator_config=create_test_generator_config(),
                agents_config=create_test_agents_config(),
            )

    def test_service_requires_profile_configuration(self):
        with pytest.raises(
            ValueError,
            match="^backend_config with at least one profile is required$",
        ):
            SyntheticDataService(
                backend=_GroundedBackend(),
                backend_config=BackendConfig(profiles={}, tenant_id="test:unit"),
                generator_config=create_test_generator_config(),
                agents_config=create_test_agents_config(),
            )

    def test_service_requires_generator_configuration(self):
        with pytest.raises(ValueError) as exc_info:
            SyntheticDataService(
                backend=_GroundedBackend(),
                backend_config=create_test_backend_config(),
                generator_config=None,
                agents_config=create_test_agents_config(),
            )

        assert str(exc_info.value) == "generator_config is required"

    def test_service_requires_agent_configuration(self):
        with pytest.raises(ValueError) as exc_info:
            SyntheticDataService(
                backend=_GroundedBackend(),
                backend_config=create_test_backend_config(),
                generator_config=create_test_generator_config(),
                agents_config=None,
            )

        assert str(exc_info.value) == "agents_config is required"

    def test_current_config_routes_every_modality_to_the_explicit_agent(self):
        config_data = json.loads(Path("configs/config.json").read_text())
        synthetic_data = {**config_data["synthetic"], "tenant_id": "test:unit"}
        generator_config = SyntheticGeneratorConfig.from_dict(synthetic_data)
        backend = _BackendAccessRecorder()
        service = SyntheticDataService(
            backend=backend,
            backend_config=BackendConfig(
                tenant_id="test:unit",
                profiles={
                    "document_text": BackendProfileConfig(
                        profile_name="document_text",
                        type="document",
                        schema_name="document_text",
                        embedding_type="single_vector",
                    )
                },
            ),
            generator_config=generator_config,
            agents_config=config_data["agents"],
        )

        assert {
            modality: service.agent_inferrer.infer_from_modality(modality)
            for modality in ("VIDEO", "DOCUMENT", "IMAGE", "AUDIO")
        } == {
            "VIDEO": "search_agent",
            "DOCUMENT": "document_agent",
            "IMAGE": "image_search_agent",
            "AUDIO": "audio_analysis_agent",
        }
        assert backend.calls == []

    @pytest.mark.parametrize(
        ("agent_mappings", "agents_config", "message"),
        [
            (
                [AgentMappingRule(modality="video", agent_name="video_search")],
                create_test_agents_config(),
                (
                    "mapping modality must be one of: "
                    "AUDIO, CODE, DOCUMENT, IMAGE, VIDEO, WIKI"
                ),
            ),
            (
                [AgentMappingRule(modality="VIDEO", agent_name="video_search")],
                {
                    "video_search": {
                        "enabled": False,
                        "modalities": ["VIDEO"],
                        "capabilities": ["video_search"],
                    },
                    "summarizer": {
                        "enabled": True,
                        "modalities": [],
                        "capabilities": ["summarization"],
                    },
                },
                "mapping for modality 'VIDEO' targets disabled agent 'video_search'",
            ),
            (
                [AgentMappingRule(modality="VIDEO", agent_name="missing_agent")],
                create_test_agents_config(),
                "mapping for modality 'VIDEO' targets unknown agent 'missing_agent'",
            ),
            (
                [AgentMappingRule(modality="VIDEO", agent_name="video_search")],
                {
                    "video_search": {
                        "enabled": True,
                        "modalities": ["DOCUMENT"],
                        "capabilities": ["video_search"],
                    }
                },
                "agent 'video_search' does not declare mapped modality 'VIDEO'",
            ),
            (
                [AgentMappingRule(modality="VIDEO", agent_name="video_search")],
                {
                    "video_search": {
                        "enabled": True,
                        "modalities": ["VIDEO"],
                        "capabilities": ["search"],
                    }
                },
                (
                    "agent 'video_search' does not declare required capability "
                    "'video_search' for modality 'VIDEO'"
                ),
            ),
        ],
    )
    def test_invalid_mapping_fails_before_backend_access(
        self,
        agent_mappings,
        agents_config,
        message,
    ):
        backend = _BackendAccessRecorder()

        with pytest.raises(ValueError, match=message):
            SyntheticDataService(
                backend=backend,
                backend_config=BackendConfig(
                    tenant_id="test:unit",
                    profiles={
                        "video_frames": BackendProfileConfig(
                            profile_name="video_frames",
                            type="video",
                            schema_name="video_segments",
                            embedding_type="multi_vector",
                        )
                    },
                ),
                generator_config=_generator_config_with_mappings(agent_mappings),
                agents_config=agents_config,
            )

        assert backend.calls == []

    def test_profile_modality_requires_an_explicit_mapping_before_backend_access(
        self,
    ):
        backend = _BackendAccessRecorder()

        with pytest.raises(
            ValueError,
            match="agent_mappings missing required modalities: AUDIO",
        ):
            SyntheticDataService(
                backend=backend,
                backend_config=BackendConfig(
                    tenant_id="test:unit",
                    profiles={
                        "audio_semantic": BackendProfileConfig(
                            profile_name="audio_semantic",
                            type="audio",
                            schema_name="audio_segments",
                            embedding_type="multi_vector",
                        )
                    },
                ),
                generator_config=_generator_config_with_mappings(
                    [AgentMappingRule(modality="VIDEO", agent_name="video_search")]
                ),
                agents_config=create_test_agents_config(),
            )

        assert backend.calls == []

    def test_concurrent_services_do_not_bleed_modality_mappings(self):
        worker_count = 12
        start = threading.Barrier(worker_count)

        def create_service(agent_name: str) -> SyntheticDataService:
            return SyntheticDataService(
                backend=_BackendAccessRecorder(),
                backend_config=BackendConfig(
                    tenant_id="test:unit",
                    profiles={
                        "video_frames": BackendProfileConfig(
                            profile_name="video_frames",
                            type="video",
                            schema_name="video_segments",
                            embedding_type="multi_vector",
                        )
                    },
                ),
                generator_config=_generator_config_with_mappings(
                    [AgentMappingRule(modality="VIDEO", agent_name=agent_name)]
                ),
                agents_config={
                    agent_name: {
                        "enabled": True,
                        "modalities": ["VIDEO"],
                        "capabilities": ["video_search"],
                    }
                },
            )

        services = {
            name: create_service(name) for name in ("tenant_a_video", "tenant_b_video")
        }
        expected = [
            "tenant_a_video" if index % 2 == 0 else "tenant_b_video"
            for index in range(worker_count)
        ]

        def resolve(agent_name: str) -> str:
            start.wait()
            return services[agent_name].agent_inferrer.infer_from_modality("VIDEO")

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            actual = list(pool.map(resolve, expected))

        assert actual == expected

    @pytest.mark.asyncio
    async def test_service_with_backend(self):
        """Test service can be initialized with Backend interface"""
        mock_backend = type(
            "MockBackend", (), {"query_metadata_documents": lambda *args, **kwargs: []}
        )()
        service = SyntheticDataService(
            backend=mock_backend,
            backend_config=create_test_backend_config(),
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
        )
        assert service.backend == mock_backend

    @pytest.mark.asyncio
    async def test_service_with_backend_config(self):
        """Test service with backend configuration"""
        config = create_test_backend_config()
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=config,
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
        )
        assert service.backend_config == config

    @pytest.mark.asyncio
    async def test_generate_profile_examples(self):
        """Test generating profile selection examples"""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=1
        )

        response = await service.generate(request)

        assert response.optimizer == "profile"
        assert response.count == 1
        assert response.schema_name == "ProfileSelectionExampleSchema"
        assert len(response.data) == 1
        assert isinstance(response.selected_profiles, list)
        assert len(response.selected_profiles) > 0
        assert isinstance(response.metadata, dict)
        assert isinstance(response.profile_selection_reasoning, str)

    @pytest.mark.asyncio
    async def test_generate_rejects_duplicate_query_identity_after_generator(
        self, monkeypatch
    ):
        service = create_test_service()
        duplicate_query = "Saturn V launch"
        examples = [
            ProfileSelectionExampleSchema(
                query=duplicate_query,
                available_profiles="video_frames,audio_semantic",
                selected_profile="video_frames",
                reasoning=reasoning,
                query_intent="video_search",
                modality="video",
                complexity="medium",
            )
            for reasoning in (
                "Production selector chose the video profile.",
                "A conflicting output used the same training input.",
            )
        ]
        monkeypatch.setattr(
            service,
            "_generate_examples",
            AsyncMock(return_value=examples),
        )

        with pytest.raises(
            ValueError,
            match="SyntheticDataService generated duplicate query 'Saturn V launch'",
        ):
            await service.generate(
                SyntheticDataRequest(
                    tenant_id="test:unit", optimizer="profile", count=2
                )
            )

    @pytest.mark.parametrize(
        ("examples", "count", "message"),
        [
            (
                [
                    ProfileSelectionExampleSchema(
                        query="Saturn V launch",
                        available_profiles="video_frames",
                        selected_profile="video_frames",
                        reasoning="Production selector chose the video profile.",
                        query_intent="video_search",
                        modality="video",
                        complexity="medium",
                    )
                ],
                2,
                "SyntheticDataService generated 1 examples but request count is 2",
            ),
            (
                [
                    WorkflowExecutionSchema(
                        workflow_id="wrong-schema",
                        query="Saturn V launch",
                        query_type="VIDEO",
                        execution_time=0.0,
                        success=False,
                        agent_sequence=["video_search"],
                        task_count=1,
                        parallel_efficiency=0.0,
                        confidence_score=0.0,
                    )
                ],
                1,
                "generated example 0 must be ProfileSelectionExampleSchema",
            ),
            (
                [
                    ProfileSelectionExampleSchema(
                        query=" Saturn V launch ",
                        available_profiles="video_frames",
                        selected_profile="video_frames",
                        reasoning="Production selector chose the video profile.",
                        query_intent="video_search",
                        modality="video",
                        complexity="medium",
                    )
                ],
                1,
                "generated example 0 requires a canonical non-empty query",
            ),
        ],
        ids=["short-count", "wrong-schema", "noncanonical-query"],
    )
    def test_generated_example_contract_is_central(self, examples, count, message):
        service = create_test_service()
        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=count
        )

        with pytest.raises(ValueError, match=message):
            service._validate_generated_examples(
                examples,
                request,
                get_optimizer_config("profile"),
            )

    @pytest.mark.asyncio
    async def test_generate_routing_examples(self):
        """Test generating routing experience examples"""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="routing", count=1
        )

        response = await service.generate(request)

        assert response.optimizer == "routing"
        assert response.count == 1
        assert response.schema_name == "RoutingExperienceSchema"
        assert len(response.data) == 1
        assert {tuple(item["entities"][0].values()) for item in response.data} == {
            ("Saturn V", "TECHNOLOGY")
        }

    @pytest.mark.asyncio
    async def test_generate_workflow_examples(self):
        """Test generating workflow execution examples"""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="workflow", count=3
        )

        response = await service.generate(request)

        assert response.optimizer == "workflow"
        assert response.count == 3
        assert response.schema_name == "WorkflowExecutionSchema"
        assert len(response.data) == 3

    @pytest.mark.asyncio
    async def test_generate_with_custom_sample_size(self):
        """Test generation with custom sample size"""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=1, vespa_sample_size=50
        )

        response = await service.generate(request)

        assert response.count == 1
        assert response.metadata["vespa_sample_size"] == 50

    @pytest.mark.asyncio
    async def test_generate_with_max_profiles(self):
        """Profile labels receive every deployed candidate, not the sample subset."""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=1, max_profiles=1
        )

        response = await service.generate(request)

        assert response.count == 1
        assert response.selected_profiles == ["video_frames"]
        assert {item["available_profiles"] for item in response.data} == {
            "video_frames,audio_semantic"
        }
        assert {item["selected_profile"] for item in response.data} == {"video_frames"}
        assert {item["modality"] for item in response.data} == {"video"}

    @pytest.mark.asyncio
    async def test_generate_reports_the_requested_sampling_strategy(self):
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit",
            optimizer="profile",
            count=1,
            strategy="temporal_recent",
        )

        response = await service.generate(request)

        assert response.count == 1
        assert response.metadata["backend_query_strategy"] == "temporal_recent"

    @pytest.mark.asyncio
    async def test_generate_invalid_optimizer(self):
        """Test generation with invalid optimizer name"""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="nonexistent_optimizer", count=10
        )

        with pytest.raises(ValueError, match="Unknown optimizer"):
            await service.generate(request)

    @pytest.mark.asyncio
    async def test_get_optimizer_info(self):
        """Test getting optimizer information"""
        service = create_test_service()

        info = service.get_optimizer_info("profile")

        assert info["name"] == "profile"
        assert "description" in info
        assert info["schema"] == "ProfileSelectionExampleSchema"
        assert info["generator"] == "ProfileGenerator"
        assert info["backend_strategy"] == "diverse"
        assert info["requires_agent_mapping"] is False
        assert "defaults" in info
        # Note: "generator_info" is only present if generator has been initialized (lazy init)

    @pytest.mark.asyncio
    async def test_get_optimizer_info_all_optimizers(self):
        """Test getting info for all optimizers"""
        service = create_test_service()

        for optimizer_name in ["routing", "workflow", "profile", "unified"]:
            info = service.get_optimizer_info(optimizer_name)
            assert info["name"] == optimizer_name
            assert "description" in info
            assert "schema" in info
            assert "generator" in info

    def test_shared_generator_registry_names_use_one_cached_instance(self):
        service = create_test_service()

        profile_generator = service._get_generator("profile")
        cross_modal_generator = service._get_generator("cross_modal")
        workflow_generator = service._get_generator("workflow")
        unified_generator = service._get_generator("unified")

        assert cross_modal_generator is profile_generator
        assert unified_generator is workflow_generator
        assert set(service.generators) == {"ProfileGenerator", "WorkflowGenerator"}
        assert service.get_optimizer_info("profile")["generator_info"] == (
            profile_generator.get_generator_info()
        )
        assert service.get_optimizer_info("cross_modal")["generator_info"] == (
            profile_generator.get_generator_info()
        )
        assert service.get_optimizer_info("workflow")["generator_info"] == (
            workflow_generator.get_generator_info()
        )
        assert service.get_optimizer_info("unified")["generator_info"] == (
            workflow_generator.get_generator_info()
        )

    @pytest.mark.asyncio
    async def test_list_all_optimizers(self):
        """Test listing all available optimizers"""
        service = create_test_service()

        all_optimizers = service.list_all_optimizers()

        assert len(all_optimizers) >= 3
        assert "routing" in all_optimizers
        assert "workflow" in all_optimizers
        assert "profile" in all_optimizers

        for name, info in all_optimizers.items():
            assert "name" in info
            assert "description" in info
            assert "schema" in info

    @pytest.mark.asyncio
    async def test_service_orchestration_flow(self):
        """Test complete service orchestration flow"""
        # This test validates the entire pipeline:
        # Request -> Profile Selection -> Backend Query -> Generation -> Response

        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="routing", count=1, vespa_sample_size=20
        )

        response = await service.generate(request)

        # Validate response structure
        assert response.optimizer == "routing"
        assert response.count == 1
        assert len(response.data) == 1
        assert len(response.selected_profiles) > 0

        # Validate metadata
        assert "sampled_content_count" in response.metadata
        assert response.metadata["target_count"] == 1

        # Validate examples are proper dicts
        for example in response.data:
            assert isinstance(example, dict)
            assert "query" in example
            assert example["entities"] == [{"text": "Saturn V", "type": "TECHNOLOGY"}]
            assert "enhanced_query" in example


class TestServiceErrorHandling:
    """Test error handling in SyntheticDataService"""

    @pytest.mark.asyncio
    async def test_invalid_optimizer_in_generate(self):
        """Test error handling for invalid optimizer"""
        service = create_test_service()

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="invalid_name", count=10
        )

        with pytest.raises(ValueError) as exc_info:
            await service.generate(request)

        assert "Unknown optimizer" in str(exc_info.value)
        assert "invalid_name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_optimizer_in_get_info(self):
        """Test error handling for get_optimizer_info"""
        service = create_test_service()

        with pytest.raises(ValueError) as exc_info:
            service.get_optimizer_info("nonexistent")

        assert "Unknown optimizer" in str(exc_info.value)


class TestServiceWithBackendConfig:
    """Test service with various backend configurations"""

    @pytest.mark.asyncio
    async def test_service_rejects_empty_backend_profile_config(self):
        with pytest.raises(
            ValueError,
            match="^backend_config with at least one profile is required$",
        ):
            SyntheticDataService(
                backend=_GroundedBackend(),
                backend_config=BackendConfig(profiles={}, tenant_id="test:unit"),
                generator_config=create_test_generator_config(),
                agents_config=create_test_agents_config(),
            )

    @pytest.mark.asyncio
    async def test_service_rejects_missing_backend_config(self):
        with pytest.raises(
            ValueError,
            match="^backend_config with at least one profile is required$",
        ):
            SyntheticDataService(
                backend=_GroundedBackend(),
                backend_config=None,
                generator_config=create_test_generator_config(),
                agents_config=create_test_agents_config(),
            )

    @pytest.mark.asyncio
    async def test_profile_examples_use_the_configured_profile_universe(self):
        configured_profiles = {
            "custom_image": BackendProfileConfig(
                profile_name="custom_image",
                type="image",
                schema_name="image_segments",
                embedding_type="multi_vector",
            ),
            "custom_audio": BackendProfileConfig(
                profile_name="custom_audio",
                type="audio",
                schema_name="audio_segments",
                embedding_type="multi_vector",
                pipeline_config={"transcribe_audio": True},
            ),
        }
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=BackendConfig(
                profiles=configured_profiles,
                tenant_id="test:unit",
            ),
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
            profile_labeler=_label_grounded_profile,
        )

        response = await service.generate(
            SyntheticDataRequest(
                tenant_id="test:unit",
                optimizer="profile",
                count=1,
                max_profiles=2,
            )
        )

        assert response.count == 1
        assert set(response.selected_profiles) == set(configured_profiles)
        assert {item["available_profiles"] for item in response.data} == {
            "custom_image,custom_audio"
        }
        assert {item["selected_profile"] for item in response.data} == {"custom_image"}
        assert {item["modality"] for item in response.data} == {"image"}
        assert {item["query_intent"] for item in response.data} == {"image_search"}

    @pytest.mark.asyncio
    async def test_service_passes_full_profile_configs_to_profile_generation(self):
        audio_profile = BackendProfileConfig(
            profile_name="audio_semantic",
            type="audio",
            schema_name="audio_segments",
            embedding_type="multi_vector",
            pipeline_config={"transcribe_audio": True},
        )
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=BackendConfig(
                profiles={"audio_semantic": audio_profile},
                tenant_id="test:unit",
            ),
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
            profile_labeler=_label_grounded_profile,
        )
        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=1
        )

        examples = await service._generate_examples(
            request,
            get_optimizer_config("profile"),
            [{"topic": "Curie lecture", "schema_name": "audio_segments"}],
            {"audio_semantic": audio_profile.to_dict()},
        )

        assert examples[0].model_dump() == {
            "query": "find Curie lecture in an audio transcript",
            "available_profiles": "audio_semantic",
            "selected_profile": "audio_semantic",
            "reasoning": "Production selector chose audio_semantic.",
            "query_intent": "audio_search",
            "modality": "audio",
            "complexity": "medium",
        }

    @pytest.mark.asyncio
    async def test_service_cross_modal_generation_spans_two_source_modalities(self):
        profiles = {
            "audio_semantic": BackendProfileConfig(
                profile_name="audio_semantic",
                type="audio",
                schema_name="audio_segments",
                embedding_type="multi_vector",
                pipeline_config={"transcribe_audio": True},
            ),
            "document_semantic": BackendProfileConfig(
                profile_name="document_semantic",
                type="document",
                schema_name="document_pages",
                embedding_type="multi_vector",
            ),
        }
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=BackendConfig(profiles=profiles, tenant_id="test:unit"),
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
            profile_labeler=_label_grounded_profile,
        )
        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="cross_modal", count=1
        )

        examples = await service._generate_examples(
            request,
            get_optimizer_config("cross_modal"),
            [
                {"topic": "Curie lecture", "schema_name": "audio_segments"},
                {"topic": "Radium notes", "schema_name": "document_pages"},
            ],
            {name: profile.to_dict() for name, profile in profiles.items()},
        )

        assert [example.model_dump() for example in examples] == [
            {
                "query": (
                    "find Curie lecture in audio content together with "
                    "Radium notes in document content"
                ),
                "available_profiles": "audio_semantic,document_semantic",
                "selected_profile": "audio_semantic",
                "reasoning": "Production selector chose audio_semantic.",
                "query_intent": "audio_search",
                "modality": "audio",
                "complexity": "medium",
            }
        ]

    @pytest.mark.asyncio
    async def test_service_rejects_cross_modal_target_above_grounded_combinations(
        self,
    ):
        profiles = {
            "audio_semantic": BackendProfileConfig(
                profile_name="audio_semantic",
                type="audio",
                schema_name="audio_segments",
                embedding_type="multi_vector",
            ),
            "document_semantic": BackendProfileConfig(
                profile_name="document_semantic",
                type="document",
                schema_name="document_pages",
                embedding_type="single_vector",
            ),
        }
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=BackendConfig(profiles=profiles, tenant_id="test:unit"),
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
            profile_labeler=_label_grounded_profile,
        )
        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="cross_modal", count=3
        )

        with pytest.raises(ValueError) as error:
            await service._generate_examples(
                request,
                get_optimizer_config("cross_modal"),
                [
                    {"topic": "Curie lecture", "schema_name": "audio_segments"},
                    {"topic": "Radium notes", "schema_name": "document_pages"},
                ],
                {name: profile.to_dict() for name, profile in profiles.items()},
            )

        assert str(error.value) == (
            "ProfileGenerator generated 2 unique grounded examples but "
            "target_count=3; source_context=2 unique cross-modal query combinations"
        )

    @pytest.mark.asyncio
    async def test_cross_modal_selection_uses_distinct_configured_modalities(self):
        profiles = {
            "video_colpali": BackendProfileConfig(
                profile_name="video_colpali",
                type="video",
                schema_name="video_frames",
                embedding_type="multi_vector",
                pipeline_config={"transcribe_audio": True},
            ),
            "video_colqwen": BackendProfileConfig(
                profile_name="video_colqwen",
                type="video",
                schema_name="video_chunks",
                embedding_type="multi_vector",
                pipeline_config={"transcribe_audio": True},
            ),
            "audio_semantic": BackendProfileConfig(
                profile_name="audio_semantic",
                type="audio",
                schema_name="audio_segments",
                embedding_type="multi_vector",
            ),
        }
        service = SyntheticDataService(
            backend=_GroundedBackend(),
            backend_config=BackendConfig(profiles=profiles, tenant_id="test:unit"),
            generator_config=create_test_generator_config(),
            agents_config=create_test_agents_config(),
            profile_labeler=_label_grounded_profile,
        )

        response = await service.generate(
            SyntheticDataRequest(
                tenant_id="test:unit",
                optimizer="cross_modal",
                count=2,
                vespa_sample_size=4,
                strategy="multi_modal_sequences",
                max_profiles=2,
            )
        )

        assert response.selected_profiles == ["video_colpali", "audio_semantic"]
        assert response.count == 2
        assert [item["selected_profile"] for item in response.data] == [
            "video_colpali",
            "video_colpali",
        ]
        assert [item["modality"] for item in response.data] == ["video", "video"]
        assert all(
            item["available_profiles"] == "video_colpali,video_colqwen,audio_semantic"
            for item in response.data
        )


class _TenantProfileBackend:
    def __init__(self, deployed_schemas: dict[str, set[str]]) -> None:
        self.deployed_schemas = deployed_schemas
        self.schema_checks: list[tuple[str, str]] = []
        self.query_calls: list[dict] = []

    def schema_exists(self, schema_name, tenant_id=None):
        self.schema_checks.append((tenant_id, schema_name))
        return schema_name in self.deployed_schemas[tenant_id]

    def get_tenant_schema_name(self, tenant_id, base_schema_name):
        return f"{base_schema_name}_{tenant_id.replace(':', '_')}"

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        self.query_calls.append(
            {"schema": schema, "yql": yql, "tenant_id": kwargs["tenant_id"]}
        )
        return [{"title": f"grounded content for {kwargs['tenant_id']}"}]


def _tenant_profile_service(backend) -> SyntheticDataService:
    profiles = {
        "profile_a": BackendProfileConfig(
            profile_name="profile_a",
            type="audio",
            schema_name="schema_a",
            embedding_type="multi_vector",
            pipeline_config={"transcribe_audio": True},
        ),
        "profile_b": BackendProfileConfig(
            profile_name="profile_b",
            type="document",
            schema_name="schema_b",
            embedding_type="single_vector",
        ),
    }
    return SyntheticDataService(
        backend=backend,
        backend_config=BackendConfig(profiles=profiles, tenant_id="test:unit"),
        generator_config=create_test_generator_config(),
        agents_config=create_test_agents_config(),
        profile_labeler=_label_grounded_profile,
    )


@pytest.mark.asyncio
async def test_live_backend_samples_a_deployed_configured_profile_schema():
    tenant_id = "acme:media"
    profile_name = "video_frames"
    schema_name = "video_segments"
    backend = _TenantProfileBackend({tenant_id: {schema_name}})
    service = SyntheticDataService(
        backend=backend,
        backend_config=BackendConfig(
            tenant_id="test:unit",
            profiles={
                profile_name: BackendProfileConfig(
                    profile_name=profile_name,
                    type="video",
                    schema_name=schema_name,
                    embedding_type="multi_vector",
                    pipeline_config={"extract_keyframes": True},
                )
            },
        ),
        generator_config=create_test_generator_config(),
        agents_config=create_test_agents_config(),
        profile_labeler=_label_grounded_profile,
    )

    response = await service.generate(
        SyntheticDataRequest(
            tenant_id=tenant_id,
            optimizer="profile",
            count=1,
            vespa_sample_size=1,
            max_profiles=1,
        )
    )

    assert response.selected_profiles == [profile_name]
    assert response.metadata["sampled_content_count"] == 1
    assert response.data[0]["modality"] == "video"
    assert backend.schema_checks == [(tenant_id, schema_name)]
    assert backend.query_calls == [
        {
            "schema": schema_name,
            "yql": f"select * from sources {schema_name} where true limit 5",
            "tenant_id": tenant_id,
        }
    ]


@pytest.mark.asyncio
async def test_generation_uses_only_each_tenants_deployed_profiles_concurrently():
    tenants = [f"tenant_{index}:media" for index in range(8)]
    deployed = {
        tenant: {"schema_a" if index % 2 == 0 else "schema_b"}
        for index, tenant in enumerate(tenants)
    }
    backend = _TenantProfileBackend(deployed)
    service = _tenant_profile_service(backend)

    responses = await asyncio.gather(
        *[
            service.generate(
                SyntheticDataRequest(
                    tenant_id=tenant,
                    optimizer="profile",
                    count=1,
                    vespa_sample_size=1,
                    max_profiles=2,
                )
            )
            for tenant in tenants
        ]
    )

    for tenant, response in zip(tenants, responses, strict=True):
        expected_profile = (
            "profile_a" if deployed[tenant] == {"schema_a"} else "profile_b"
        )
        expected_modality = "audio" if expected_profile == "profile_a" else "document"
        expected_schema = next(iter(deployed[tenant]))
        assert response.selected_profiles == [expected_profile]
        assert response.data[0]["available_profiles"] == expected_profile
        assert response.data[0]["selected_profile"] == expected_profile
        assert response.data[0]["modality"] == expected_modality
        assert response.data[0]["query_intent"] == f"{expected_modality}_search"
        expected_query = (
            f"find grounded content for {tenant} in an audio transcript"
            if expected_profile == "profile_a"
            else f"find grounded content for {tenant} in document content"
        )
        assert response.data[0]["query"] == expected_query
    expected_query_calls = []
    for tenant in tenants:
        expected_schema = next(iter(deployed[tenant]))
        expected_query_calls.append(
            {
                "schema": expected_schema,
                "yql": f"select * from sources {expected_schema} where true limit 5",
                "tenant_id": tenant,
            }
        )
    assert sorted(backend.query_calls, key=lambda call: call["tenant_id"]) == sorted(
        expected_query_calls, key=lambda call: call["tenant_id"]
    )

    assert sorted(backend.schema_checks) == sorted(
        (tenant, schema) for tenant in tenants for schema in ("schema_a", "schema_b")
    )


@pytest.mark.asyncio
async def test_profile_schema_checks_do_not_block_the_event_loop():
    release = threading.Event()

    class _BlockingSchemaBackend(_TenantProfileBackend):
        def __init__(self):
            super().__init__({"acme:media": {"schema_a"}})
            self.released_by_event_loop = False

        def schema_exists(self, schema_name, tenant_id=None):
            self.released_by_event_loop = release.wait(timeout=0.2)
            return super().schema_exists(schema_name, tenant_id=tenant_id)

    async def release_schema_check() -> None:
        await asyncio.sleep(0)
        release.set()

    backend = _BlockingSchemaBackend()
    response, _ = await asyncio.gather(
        _tenant_profile_service(backend).generate(
            SyntheticDataRequest(
                tenant_id="acme:media",
                optimizer="profile",
                count=1,
                vespa_sample_size=1,
            )
        ),
        release_schema_check(),
    )

    assert backend.released_by_event_loop is True
    assert response.selected_profiles == ["profile_a"]


@pytest.mark.asyncio
async def test_profile_schema_lookup_failure_propagates_without_querying():
    class _BrokenSchemaBackend(_TenantProfileBackend):
        def schema_exists(self, schema_name, tenant_id=None):
            raise RuntimeError("schema registry unavailable")

    backend = _BrokenSchemaBackend({"acme:media": {"schema_a"}})

    with pytest.raises(RuntimeError, match="schema registry unavailable"):
        await _tenant_profile_service(backend).generate(
            SyntheticDataRequest(
                tenant_id="acme:media",
                optimizer="profile",
                count=1,
                vespa_sample_size=1,
            )
        )

    assert backend.query_calls == []


@pytest.mark.parametrize(
    "profile_data",
    [{}, {"schema_name": " "}, {"schema_name": 7}],
)
@pytest.mark.asyncio
async def test_profile_requires_canonical_schema_name_before_backend_access(
    profile_data,
):
    class _ProfileConfig:
        def to_dict(self):
            return dict(profile_data)

    tenant_id = "acme:media"
    backend = _TenantProfileBackend({tenant_id: {"legacy_profile", " ", 7}})
    service = SyntheticDataService(
        backend=backend,
        backend_config=BackendConfig(
            profiles={"legacy_profile": _ProfileConfig()},
            tenant_id="test:unit",
        ),
        generator_config=create_test_generator_config(),
        agents_config=create_test_agents_config(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Backend profile 'legacy_profile' requires a non-empty string schema_name"
        ),
    ):
        await service.generate(
            SyntheticDataRequest(
                tenant_id=tenant_id,
                optimizer="profile",
                count=1,
                vespa_sample_size=1,
            )
        )

    assert backend.schema_checks == []
    assert backend.query_calls == []


def test_generator_cache_constructs_once_under_concurrent_cold_start(
    monkeypatch,
) -> None:
    worker_count = 8
    start = threading.Barrier(worker_count)
    count_lock = threading.Lock()
    constructor_calls = 0
    observed_timeouts = []

    class _CountingProfileGenerator:
        def __init__(
            self,
            profile_labeler=None,
            production_label_timeout_seconds=None,
        ) -> None:
            nonlocal constructor_calls
            with count_lock:
                constructor_calls += 1
                observed_timeouts.append(production_label_timeout_seconds)
            time.sleep(0.05)

    monkeypatch.setattr(
        "cogniverse_synthetic.service.ProfileGenerator",
        _CountingProfileGenerator,
    )
    service = create_test_service()

    def get_generator():
        start.wait()
        return service._get_generator("profile")

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        generators = list(pool.map(lambda _: get_generator(), range(worker_count)))

    assert constructor_calls == 1
    assert observed_timeouts == [23.0]
    assert len({id(generator) for generator in generators}) == 1


def test_generator_cache_recovers_after_constructor_failure(monkeypatch) -> None:
    worker_count = 8
    start = threading.Barrier(worker_count)
    count_lock = threading.Lock()
    constructor_calls = 0

    class _FailsFirstProfileGenerator:
        def __init__(
            self,
            profile_labeler=None,
            production_label_timeout_seconds=None,
        ) -> None:
            nonlocal constructor_calls
            assert production_label_timeout_seconds == 23.0
            with count_lock:
                constructor_calls += 1
                call_number = constructor_calls
            if call_number == 1:
                raise RuntimeError("profile generator unavailable")

    monkeypatch.setattr(
        "cogniverse_synthetic.service.ProfileGenerator",
        _FailsFirstProfileGenerator,
    )
    service = create_test_service()

    def get_generator():
        start.wait()
        try:
            return service._get_generator("profile")
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        results = list(pool.map(lambda _: get_generator(), range(worker_count)))

    errors = [result for result in results if isinstance(result, RuntimeError)]
    generators = [result for result in results if not isinstance(result, RuntimeError)]
    assert [str(error) for error in errors] == ["profile generator unavailable"]
    assert constructor_calls == 2
    assert len({id(generator) for generator in generators}) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
