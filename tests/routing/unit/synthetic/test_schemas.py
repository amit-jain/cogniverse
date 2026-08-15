"""
Unit tests for synthetic data schemas
"""

import asyncio
from types import SimpleNamespace
from typing import Literal, get_args, get_origin

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cogniverse_synthetic import api as synthetic_api
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    RoutingExperienceSchema,
    SyntheticDataRequest,
    SyntheticDataResponse,
    WorkflowExecutionSchema,
)
from cogniverse_synthetic.service import SyntheticDataService
from tests.utils.synthetic_config import video_synthetic_generator_config

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (
            EntityExtractionExampleSchema,
            {
                "query": "Marie Curie discovered radium",
                "entities": [
                    {"text": "Marie Curie", "type": "PERSON"},
                    {"text": "radium", "type": "SUBSTANCE"},
                ],
                "entity_types": "PERSON,SUBSTANCE",
                "relationships": [
                    {
                        "source": "Marie Curie",
                        "target": "radium",
                        "type": "discovered",
                    }
                ],
            },
        ),
        (
            RoutingExperienceSchema,
            {
                "query": "find the product launch recording",
                "entities": [],
                "relationships": [],
                "enhanced_query": "find the exact product launch recording",
                "chosen_agent": "search_agent",
                "routing_confidence": 0.9,
                "search_quality": 0.0,
                "agent_success": False,
            },
        ),
        (
            WorkflowExecutionSchema,
            {
                "workflow_id": "workflow-1",
                "query": "summarize the product launch recording",
                "query_type": "VIDEO",
                "execution_time": 0.0,
                "success": False,
                "agent_sequence": ["search_agent", "summarizer_agent"],
                "task_count": 2,
                "parallel_efficiency": 0.0,
                "confidence_score": 0.0,
            },
        ),
        (
            SyntheticDataResponse,
            {
                "optimizer": "routing",
                "schema_name": "RoutingExperienceSchema",
                "count": 1,
                "selected_profiles": ["video_colpali_smol500_mv_frame"],
                "profile_selection_reasoning": "The profile serves video content.",
                "data": [],
                "metadata": {"backend_type": "vespa"},
            },
        ),
    ],
    ids=["entity-extraction", "routing", "workflow", "response"],
)
def test_public_synthetic_schemas_reject_unknown_fields(schema, payload) -> None:
    with pytest.raises(ValidationError) as captured:
        schema.model_validate({**payload, "legacy_payload": "discard me"})

    assert captured.value.errors(
        include_url=False,
        include_context=False,
    ) == [
        {
            "type": "extra_forbidden",
            "loc": ("legacy_payload",),
            "msg": "Extra inputs are not permitted",
            "input": "discard me",
        }
    ]


class _ProfileConfig:
    def to_dict(self) -> dict:
        return {
            "profile_name": "source_profile",
            "schema_name": "source_schema",
            "type": "video",
        }


class _StrategyRecorder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    async def query_profiles(
        self,
        profile_configs,
        sample_size,
        strategy,
        *,
        tenant_id,
    ):
        await asyncio.sleep(0)
        self.calls.append((tenant_id, strategy))
        if self.fail:
            raise ConnectionError("backend unavailable")
        return [{"topic": f"source for {tenant_id}"}]


class _StrategyProbeService(SyntheticDataService):
    def __init__(self, recorder: _StrategyRecorder) -> None:
        self.backend_config = SimpleNamespace(
            profiles={"source_profile": _ProfileConfig()}
        )
        self.generator_config = video_synthetic_generator_config("test:unit")
        self.backend_querier = recorder

    async def _get_available_profiles(self, tenant_id):
        return {"source_profile": _ProfileConfig().to_dict()}

    async def _select_profiles(self, request, config, available_profiles):
        return ["source_profile"], "selected the only deployed profile"

    async def _generate_examples(
        self,
        request,
        config,
        sampled_content,
        selected_profile_configs,
        *,
        generation_tracker=None,
        available_profile_configs,
    ):
        assert selected_profile_configs == {
            "source_profile": _ProfileConfig().to_dict()
        }
        assert available_profile_configs == selected_profile_configs
        return [
            config.schema_class.model_construct(
                query=f"source query for {request.tenant_id}"
            )
        ]


class TestProfileSelectionExampleSchema:
    """Test ProfileSelectionExampleSchema validation and serialization"""

    def test_valid_profile_selection_example(self):
        example = ProfileSelectionExampleSchema(
            query="find a clip about machine learning",
            available_profiles="video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s",
            selected_profile="video_colqwen_omni_mv_chunk_30s",
            reasoning="Chunk-based profile fits clip-style queries",
            query_intent="video_search",
            modality="video",
            complexity="medium",
        )

        assert example.query == "find a clip about machine learning"
        assert example.selected_profile == "video_colqwen_omni_mv_chunk_30s"
        assert example.modality == "video"
        assert "confidence" not in example.model_dump()

    def test_complexity_contract_is_literal(self):
        annotation = ProfileSelectionExampleSchema.model_fields["complexity"].annotation
        assert get_origin(annotation) is Literal
        assert get_args(annotation) == ("simple", "medium", "complex")

    @pytest.mark.parametrize("confidence", [0.0, 0.85, 1.0])
    def test_rejects_unobserved_confidence_target(self, confidence):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ProfileSelectionExampleSchema(
                query="q",
                available_profiles="a,b",
                selected_profile="a",
                confidence=confidence,
                reasoning="r",
                query_intent="text_search",
                modality="text",
                complexity="simple",
            )

    def test_serialization_roundtrip(self):
        example = ProfileSelectionExampleSchema(
            query="q",
            available_profiles="a,b",
            selected_profile="a",
            reasoning="r",
            query_intent="video_search",
            modality="video",
            complexity="simple",
        )
        data = example.model_dump()
        assert data["query"] == "q"
        assert data["selected_profile"] == "a"
        rebuilt = ProfileSelectionExampleSchema(**data)
        assert rebuilt == example
        assert "confidence" not in data


class TestRoutingExperienceSchema:
    """Test RoutingExperienceSchema validation and serialization"""

    def test_valid_routing_experience(self):
        """Test creating valid RoutingExperience"""
        experience = RoutingExperienceSchema(
            query="find TensorFlow tutorials",
            entities=[{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            relationships=[],
            enhanced_query="find TensorFlow(TECHNOLOGY) tutorials",
            chosen_agent="search_agent",
            routing_confidence=0.85,
            search_quality=0.78,
            agent_success=True,
        )

        assert experience.query == "find TensorFlow tutorials"
        assert len(experience.entities) == 1
        assert experience.chosen_agent == "search_agent"
        assert experience.routing_confidence == 0.85

    def test_routing_experience_with_satisfaction(self):
        """Test RoutingExperience with user satisfaction"""
        experience = RoutingExperienceSchema(
            query="test",
            entities=[],
            relationships=[],
            enhanced_query="test",
            chosen_agent="agent",
            routing_confidence=0.8,
            search_quality=0.7,
            agent_success=True,
            user_satisfaction=0.9,
        )

        assert experience.user_satisfaction == 0.9

    def test_routing_confidence_bounds(self):
        """Test confidence and quality value bounds"""
        RoutingExperienceSchema(
            query="test",
            entities=[],
            relationships=[],
            enhanced_query="test",
            chosen_agent="agent",
            routing_confidence=0.0,
            search_quality=0.0,
            agent_success=False,
        )

        RoutingExperienceSchema(
            query="test",
            entities=[],
            relationships=[],
            enhanced_query="test",
            chosen_agent="agent",
            routing_confidence=1.0,
            search_quality=1.0,
            agent_success=True,
        )

        with pytest.raises(ValidationError):
            RoutingExperienceSchema(
                query="test",
                entities=[],
                relationships=[],
                enhanced_query="test",
                chosen_agent="agent",
                routing_confidence=1.5,
                search_quality=0.5,
                agent_success=True,
            )

    def test_routing_experience_metadata(self):
        """Test optional metadata field"""
        experience = RoutingExperienceSchema(
            query="test",
            entities=[],
            relationships=[],
            enhanced_query="test",
            chosen_agent="agent",
            routing_confidence=0.8,
            search_quality=0.7,
            agent_success=True,
            metadata={"source": "synthetic", "version": "1.0"},
        )

        assert experience.metadata["source"] == "synthetic"


class TestWorkflowExecutionSchema:
    """Test WorkflowExecutionSchema validation and serialization"""

    def test_valid_workflow_execution(self):
        """Test creating valid WorkflowExecution"""
        workflow = WorkflowExecutionSchema(
            workflow_id="test_001",
            query="summarize video and create report",
            query_type="VIDEO",
            execution_time=3.5,
            success=True,
            agent_sequence=[
                "search_agent",
                "summarizer_agent",
                "detailed_report_agent",
            ],
            task_count=3,
            parallel_efficiency=0.85,
            confidence_score=0.88,
        )

        assert workflow.workflow_id == "test_001"
        assert workflow.agent_sequence == [
            "search_agent",
            "summarizer_agent",
            "detailed_report_agent",
        ]
        assert workflow.task_count == 3

    def test_schema_examples_use_canonical_agent_ids(self):
        routing_example = RoutingExperienceSchema.model_json_schema()["example"]
        workflow_example = WorkflowExecutionSchema.model_json_schema()["example"]

        assert routing_example["chosen_agent"] == "search_agent"
        assert workflow_example["agent_sequence"] == [
            "search_agent",
            "summarizer_agent",
            "detailed_report_agent",
        ]

    def test_workflow_execution_time_validation(self):
        """Test execution time must be non-negative"""
        WorkflowExecutionSchema(
            workflow_id="test",
            query="test",
            query_type="VIDEO",
            execution_time=0.0,
            success=True,
            agent_sequence=["agent"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
        )

        with pytest.raises(ValidationError):
            WorkflowExecutionSchema(
                workflow_id="test",
                query="test",
                query_type="VIDEO",
                execution_time=-1.0,
                success=True,
                agent_sequence=["agent"],
                task_count=1,
                parallel_efficiency=1.0,
                confidence_score=0.9,
            )

    def test_workflow_task_count_validation(self):
        """Test task count must be at least 1"""
        WorkflowExecutionSchema(
            workflow_id="test",
            query="test",
            query_type="VIDEO",
            execution_time=1.0,
            success=True,
            agent_sequence=["agent"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
        )

        with pytest.raises(ValidationError):
            WorkflowExecutionSchema(
                workflow_id="test",
                query="test",
                query_type="VIDEO",
                execution_time=1.0,
                success=True,
                agent_sequence=["agent"],
                task_count=0,
                parallel_efficiency=1.0,
                confidence_score=0.9,
            )

    def test_workflow_with_error(self):
        """Test WorkflowExecution with error details"""
        workflow = WorkflowExecutionSchema(
            workflow_id="test",
            query="test",
            query_type="VIDEO",
            execution_time=1.0,
            success=False,
            agent_sequence=["agent"],
            task_count=1,
            parallel_efficiency=0.0,
            confidence_score=0.5,
            error_details="Agent timeout",
        )

        assert workflow.success is False
        assert workflow.error_details == "Agent timeout"


class TestSyntheticDataRequest:
    """Test SyntheticDataRequest validation"""

    def test_valid_request(self):
        """Test creating valid request"""
        request = SyntheticDataRequest(
            tenant_id="test:unit",
            optimizer="profile",
            count=100,
            vespa_sample_size=200,
            strategy="diverse",
            max_profiles=3,
        )

        assert request.optimizer == "profile"
        assert request.count == 100

    def test_request_count_validation(self):
        """Test count bounds validation"""
        SyntheticDataRequest(tenant_id="test:unit", optimizer="profile", count=1)
        SyntheticDataRequest(tenant_id="test:unit", optimizer="profile", count=10000)

        with pytest.raises(ValidationError):
            SyntheticDataRequest(tenant_id="test:unit", optimizer="profile", count=0)

        with pytest.raises(ValidationError):
            SyntheticDataRequest(
                tenant_id="test:unit", optimizer="profile", count=10001
            )

    def test_request_defaults(self):
        """Test default values for optional fields — tenant_id is required."""
        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=100
        )

        assert request.vespa_sample_size == 200
        assert request.strategy is None
        assert request.max_profiles == 3
        assert request.tenant_id == "test:unit"

    def test_request_rejects_explicit_null_strategy(self):
        with pytest.raises(ValidationError, match="strategy"):
            SyntheticDataRequest.model_validate(
                {
                    "tenant_id": "test:unit",
                    "optimizer": "profile",
                    "count": 1,
                    "strategy": None,
                }
            )

    def test_request_rejects_missing_tenant_id(self):
        """SyntheticDataRequest must raise on missing tenant_id."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SyntheticDataRequest(optimizer="profile", count=1)

    @pytest.mark.parametrize(
        "tenant_id",
        ["", " ", "acme:production:extra", "acme-production"],
    )
    def test_request_rejects_invalid_tenant_id(self, tenant_id):
        with pytest.raises(ValidationError):
            SyntheticDataRequest(
                tenant_id=tenant_id,
                optimizer="profile",
                count=1,
            )

    def test_request_canonicalizes_simple_tenant_id(self):
        request = SyntheticDataRequest(
            tenant_id="acme",
            optimizer="profile",
            count=1,
        )

        assert request.tenant_id == "acme:acme"

    @pytest.mark.parametrize(
        "strategy",
        [
            "",
            "unknown",
        ],
    )
    def test_request_rejects_noncanonical_sampling_strategy(self, strategy):
        with pytest.raises(ValidationError):
            SyntheticDataRequest(
                tenant_id="test:unit",
                optimizer="profile",
                count=1,
                strategy=strategy,
            )

    @pytest.mark.parametrize(
        "strategy",
        [["diverse"], {"name": "diverse"}],
        ids=["list", "object"],
    )
    def test_request_rejects_non_string_sampling_strategy(self, strategy):
        with pytest.raises(ValidationError) as captured:
            SyntheticDataRequest(
                tenant_id="test:unit",
                optimizer="profile",
                count=1,
                strategy=strategy,
            )

        assert captured.value.errors(
            include_url=False,
            include_context=False,
        ) == [
            {
                "type": "value_error",
                "loc": ("strategy",),
                "msg": "Value error, strategy must be a string",
                "input": strategy,
            }
        ]

    def test_request_rejects_obsolete_strategy_list(self):
        with pytest.raises(ValidationError, match="strategies"):
            SyntheticDataRequest.model_validate(
                {
                    "tenant_id": "test:unit",
                    "optimizer": "profile",
                    "count": 1,
                    "strategies": ["diverse", "temporal_recent"],
                }
            )


class TestSyntheticDataSamplingStrategy:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("optimizer", "expected_strategy"),
        [
            ("entity_extraction", "entity_rich"),
            ("query_enhancement", "diverse"),
            ("profile", "diverse"),
            ("routing", "entity_rich"),
            ("workflow", "multi_modal_sequences"),
            ("unified", "multi_modal_sequences"),
            ("cross_modal", "multi_modal_sequences"),
        ],
    )
    async def test_omitted_strategy_uses_optimizer_registry_value(
        self,
        optimizer,
        expected_strategy,
    ):
        recorder = _StrategyRecorder()
        service = _StrategyProbeService(recorder)

        response = await service.generate(
            SyntheticDataRequest(
                tenant_id=f"{optimizer}:{optimizer}",
                optimizer=optimizer,
                count=1,
            )
        )

        assert recorder.calls == [(f"{optimizer}:{optimizer}", expected_strategy)]
        assert response.metadata["backend_query_strategy"] == expected_strategy

    @pytest.mark.asyncio
    async def test_explicit_strategy_overrides_optimizer_registry_value(self):
        recorder = _StrategyRecorder()
        service = _StrategyProbeService(recorder)

        response = await service.generate(
            SyntheticDataRequest(
                tenant_id="routing:routing",
                optimizer="routing",
                count=1,
                strategy="temporal_recent",
            )
        )

        assert recorder.calls == [("routing:routing", "temporal_recent")]
        assert response.metadata["backend_query_strategy"] == "temporal_recent"

    @pytest.mark.asyncio
    async def test_concurrent_requests_keep_resolved_strategies_isolated(self):
        recorder = _StrategyRecorder()
        service = _StrategyProbeService(recorder)
        requests = [
            SyntheticDataRequest(
                tenant_id="routing:routing",
                optimizer="routing",
                count=1,
            ),
            SyntheticDataRequest(
                tenant_id="workflow:workflow",
                optimizer="workflow",
                count=1,
                strategy="temporal_recent",
            ),
            SyntheticDataRequest(
                tenant_id="unified:unified",
                optimizer="unified",
                count=1,
            ),
            SyntheticDataRequest(
                tenant_id="cross:cross",
                optimizer="cross_modal",
                count=1,
                strategy="entity_rich",
            ),
        ]

        responses = await asyncio.gather(
            *(service.generate(request) for request in requests)
        )

        assert [
            response.metadata["backend_query_strategy"] for response in responses
        ] == [
            "entity_rich",
            "temporal_recent",
            "multi_modal_sequences",
            "entity_rich",
        ]
        assert set(recorder.calls) == {
            ("routing:routing", "entity_rich"),
            ("workflow:workflow", "temporal_recent"),
            ("unified:unified", "multi_modal_sequences"),
            ("cross:cross", "entity_rich"),
        }

    @pytest.mark.asyncio
    async def test_backend_failure_preserves_resolved_strategy_context(self):
        recorder = _StrategyRecorder(fail=True)
        service = _StrategyProbeService(recorder)
        request = SyntheticDataRequest(
            tenant_id="routing:routing",
            optimizer="routing",
            count=1,
        )

        with pytest.raises(RuntimeError) as captured:
            await service.generate(request)

        assert str(captured.value) == (
            "Backend sampling failed for tenant 'routing:routing', optimizer "
            "'routing', strategy 'entity_rich': backend unavailable"
        )
        assert type(captured.value.__cause__) is ConnectionError
        assert str(captured.value.__cause__) == "backend unavailable"
        assert recorder.calls == [("routing:routing", "entity_rich")]

    def test_batch_api_forwards_explicit_strategy_to_every_request(self, monkeypatch):
        class CaptureService:
            def __init__(self):
                self.requests = []

            async def generate(self, request):
                self.requests.append(request)
                return SyntheticDataResponse(
                    optimizer=request.optimizer,
                    schema_name="RoutingExperienceSchema",
                    count=request.count,
                    selected_profiles=["source_profile"],
                    profile_selection_reasoning="selected the only deployed profile",
                    data=[
                        {"query": f"unique batch query {index}"}
                        for index in range(request.count)
                    ],
                    metadata={"backend_query_strategy": request.strategy},
                )

        service = CaptureService()
        monkeypatch.setattr(synthetic_api, "_service", service)
        app = FastAPI()
        app.include_router(synthetic_api.router)

        response = TestClient(app).post(
            "/synthetic/batch/generate",
            params={
                "optimizer": "routing",
                "count_per_batch": 1,
                "num_batches": 2,
                "tenant_id": "routing:routing",
                "strategy": "temporal_recent",
            },
        )

        assert response.status_code == 200
        assert [request.strategy for request in service.requests] == ["temporal_recent"]
        assert service.requests[0].count == 2


class TestSyntheticDataResponse:
    """Test SyntheticDataResponse validation"""

    def test_valid_response(self):
        """Test creating valid response"""
        response = SyntheticDataResponse(
            optimizer="profile",
            schema_name="ProfileSelectionExampleSchema",
            count=100,
            selected_profiles=["profile1", "profile2"],
            profile_selection_reasoning="Selected for diversity",
            data=[],
            metadata={"backend_type": "vespa", "generation_time_ms": 1250},
        )

        assert response.optimizer == "profile"
        assert response.schema_name == "ProfileSelectionExampleSchema"
        assert len(response.selected_profiles) == 2
        assert response.metadata["generation_time_ms"] == 1250

    def test_schema_example_names_backend_query_strategy_metadata(self):
        metadata = SyntheticDataResponse.model_json_schema()["example"]["metadata"]

        assert metadata == {
            "backend_type": "vespa",
            "backend_query_strategy": "diverse",
            "generation_time_ms": 1250,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
