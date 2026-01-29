"""
Unit tests for synthetic data schemas
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from cogniverse_synthetic.schemas import (
    FusionHistorySchema,
    ModalityExampleSchema,
    RoutingExperienceSchema,
    SyntheticDataRequest,
    SyntheticDataResponse,
    WorkflowExecutionSchema,
)


class TestModalityExampleSchema:
    """Test ModalityExampleSchema validation and serialization"""

    def test_valid_modality_example(self):
        """Test creating valid ModalityExample"""
        example = ModalityExampleSchema(
            query="show me videos about machine learning",
            modality="VIDEO",
            correct_agent="video_search_agent",
            success=True,
            is_synthetic=True,
            synthetic_source="backend_query",
        )

        assert example.query == "show me videos about machine learning"
        assert example.modality == "VIDEO"
        assert example.correct_agent == "video_search_agent"
        assert example.success is True
        assert example.is_synthetic is True
        assert example.synthetic_source == "backend_query"

    def test_modality_example_with_features(self):
        """Test ModalityExample with modality features"""
        example = ModalityExampleSchema(
            query="test query",
            modality="VIDEO",
            correct_agent="video_search_agent",
            modality_features={"visual_indicators": 0.9, "temporal_context": True},
        )

        assert example.modality_features is not None
        assert example.modality_features["visual_indicators"] == 0.9

    def test_modality_example_defaults(self):
        """Test default values"""
        example = ModalityExampleSchema(
            query="test", modality="VIDEO", correct_agent="agent"
        )

        assert example.success is True
        assert example.is_synthetic is True
        assert example.synthetic_source == "backend_query"
        assert example.modality_features is None

    def test_modality_example_serialization(self):
        """Test JSON serialization"""
        example = ModalityExampleSchema(
            query="test", modality="VIDEO", correct_agent="agent"
        )

        data = example.model_dump()
        assert isinstance(data, dict)
        assert data["query"] == "test"
        assert data["modality"] == "VIDEO"

        # Test round-trip
        example2 = ModalityExampleSchema(**data)
        assert example2.query == example.query


class TestFusionHistorySchema:
    """Test FusionHistorySchema validation and serialization"""

    def test_valid_fusion_history(self):
        """Test creating valid FusionHistory"""
        fusion = FusionHistorySchema(
            primary_modality="VIDEO",
            secondary_modality="DOCUMENT",
            fusion_context={"modality_agreement": 0.75, "query_ambiguity": 0.3},
            success=True,
            improvement=0.25,
        )

        assert fusion.primary_modality == "VIDEO"
        assert fusion.secondary_modality == "DOCUMENT"
        assert fusion.success is True
        assert fusion.improvement == 0.25

    def test_fusion_improvement_validation(self):
        """Test improvement value bounds (0-1)"""
        # Valid values
        FusionHistorySchema(
            primary_modality="VIDEO",
            secondary_modality="DOCUMENT",
            fusion_context={},
            success=True,
            improvement=0.0,
        )

        FusionHistorySchema(
            primary_modality="VIDEO",
            secondary_modality="DOCUMENT",
            fusion_context={},
            success=True,
            improvement=1.0,
        )

        # Invalid values
        with pytest.raises(ValidationError):
            FusionHistorySchema(
                primary_modality="VIDEO",
                secondary_modality="DOCUMENT",
                fusion_context={},
                success=True,
                improvement=-0.1,
            )

        with pytest.raises(ValidationError):
            FusionHistorySchema(
                primary_modality="VIDEO",
                secondary_modality="DOCUMENT",
                fusion_context={},
                success=True,
                improvement=1.1,
            )

    def test_fusion_with_query(self):
        """Test FusionHistory with optional query"""
        fusion = FusionHistorySchema(
            primary_modality="VIDEO",
            secondary_modality="DOCUMENT",
            fusion_context={},
            success=True,
            improvement=0.5,
            query="test query",
        )

        assert fusion.query == "test query"

    def test_fusion_timestamp(self):
        """Test timestamp is automatically set"""
        fusion = FusionHistorySchema(
            primary_modality="VIDEO",
            secondary_modality="DOCUMENT",
            fusion_context={},
            success=True,
            improvement=0.5,
        )

        assert isinstance(fusion.timestamp, datetime)


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
            optimizer="cross_modal",
            count=100,
            vespa_sample_size=200,
            strategies=["diverse"],
            max_profiles=3,
        )

        assert request.optimizer == "cross_modal"
        assert request.count == 100

    def test_request_count_validation(self):
        """Test count bounds validation"""
        # Valid
        SyntheticDataRequest(optimizer="modality", count=1)
        SyntheticDataRequest(optimizer="modality", count=10000)

        # Invalid
        with pytest.raises(ValidationError):
            SyntheticDataRequest(optimizer="modality", count=0)

        with pytest.raises(ValidationError):
            SyntheticDataRequest(optimizer="modality", count=10001)

    def test_request_defaults(self):
        """Test default values"""
        request = SyntheticDataRequest(optimizer="modality", count=100)

        assert request.vespa_sample_size == 200
        assert request.strategies == ["diverse"]
        assert request.max_profiles == 3
        assert request.tenant_id == "default"


class TestSyntheticDataResponse:
    """Test SyntheticDataResponse validation"""

    def test_valid_response(self):
        """Test creating valid response"""
        response = SyntheticDataResponse(
            optimizer="cross_modal",
            schema_name="FusionHistorySchema",
            count=100,
            selected_profiles=["profile1", "profile2"],
            profile_selection_reasoning="Selected for diversity",
            data=[],
            metadata={"backend_type": "vespa", "generation_time_ms": 1250},
        )

        assert response.optimizer == "cross_modal"
        assert response.schema_name == "FusionHistorySchema"
        assert len(response.selected_profiles) == 2
        assert response.metadata["generation_time_ms"] == 1250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
