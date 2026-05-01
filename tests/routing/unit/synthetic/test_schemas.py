"""
Unit tests for synthetic data schemas
"""

import pytest
from pydantic import ValidationError

from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    RoutingExperienceSchema,
    SyntheticDataRequest,
    SyntheticDataResponse,
    WorkflowExecutionSchema,
)


class TestProfileSelectionExampleSchema:
    """Test ProfileSelectionExampleSchema validation and serialization"""

    def test_valid_profile_selection_example(self):
        example = ProfileSelectionExampleSchema(
            query="find a clip about machine learning",
            available_profiles="video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s",
            selected_profile="video_colqwen_omni_mv_chunk_30s",
            confidence=0.85,
            reasoning="Chunk-based profile fits clip-style queries",
            query_intent="video_search",
            modality="video",
            complexity="medium",
        )

        assert example.query == "find a clip about machine learning"
        assert example.selected_profile == "video_colqwen_omni_mv_chunk_30s"
        assert 0.0 <= example.confidence <= 1.0
        assert example.modality == "video"

    def test_confidence_bounds(self):
        # 0.0 and 1.0 are valid bounds.
        ProfileSelectionExampleSchema(
            query="q",
            available_profiles="a,b",
            selected_profile="a",
            confidence=0.0,
            reasoning="r",
            query_intent="text_search",
            modality="text",
            complexity="simple",
        )
        ProfileSelectionExampleSchema(
            query="q",
            available_profiles="a,b",
            selected_profile="a",
            confidence=1.0,
            reasoning="r",
            query_intent="text_search",
            modality="text",
            complexity="simple",
        )

        with pytest.raises(ValidationError):
            ProfileSelectionExampleSchema(
                query="q",
                available_profiles="a,b",
                selected_profile="a",
                confidence=1.5,
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
            confidence=0.7,
            reasoning="r",
            query_intent="video_search",
            modality="video",
            complexity="simple",
        )
        data = example.model_dump()
        assert data["query"] == "q"
        assert data["selected_profile"] == "a"
        rebuilt = ProfileSelectionExampleSchema(**data)
        assert rebuilt.confidence == example.confidence


class TestRoutingExperienceSchema:
    """Test RoutingExperienceSchema validation and serialization"""

    def test_valid_routing_experience(self):
        """Test creating valid RoutingExperience"""
        experience = RoutingExperienceSchema(
            query="find TensorFlow tutorials",
            entities=[{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            relationships=[],
            enhanced_query="find TensorFlow(TECHNOLOGY) tutorials",
            chosen_agent="video_search_agent",
            routing_confidence=0.85,
            search_quality=0.78,
            agent_success=True,
        )

        assert experience.query == "find TensorFlow tutorials"
        assert len(experience.entities) == 1
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
        # Valid values at boundaries
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

        # Invalid values
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
            agent_sequence=["video_search_agent", "summarizer", "detailed_report"],
            task_count=3,
            parallel_efficiency=0.85,
            confidence_score=0.88,
        )

        assert workflow.workflow_id == "test_001"
        assert len(workflow.agent_sequence) == 3
        assert workflow.task_count == 3

    def test_workflow_execution_time_validation(self):
        """Test execution time must be non-negative"""
        # Valid
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

        # Invalid
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
        # Valid
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

        # Invalid
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
            strategies=["diverse"],
            max_profiles=3,
        )

        assert request.optimizer == "profile"
        assert request.count == 100

    def test_request_count_validation(self):
        """Test count bounds validation"""
        # Valid
        SyntheticDataRequest(tenant_id="test:unit", optimizer="profile", count=1)
        SyntheticDataRequest(tenant_id="test:unit", optimizer="profile", count=10000)

        # Invalid
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
        assert request.strategies == ["diverse"]
        assert request.max_profiles == 3
        assert request.tenant_id == "test:unit"

    def test_request_rejects_missing_tenant_id(self):
        """SyntheticDataRequest must raise on missing tenant_id."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SyntheticDataRequest(optimizer="profile", count=1)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
