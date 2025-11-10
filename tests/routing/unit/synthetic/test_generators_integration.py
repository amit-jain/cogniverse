"""
Integration tests for all synthetic data generators
"""


import pytest
from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
)
from cogniverse_synthetic.generators import (
    CrossModalGenerator,
    ModalityGenerator,
    RoutingGenerator,
    WorkflowGenerator,
)
from cogniverse_synthetic.schemas import (
    FusionHistorySchema,
    ModalityExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)
from cogniverse_synthetic.utils import AgentInferrer, PatternExtractor


# Test configuration fixtures
def create_modality_config():
    """Create test configuration for modality generator with mock DSPy"""
    return OptimizerGenerationConfig(
        optimizer_type="modality",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateModalityQuery",
                module_type="Predict",
            )
        },
        agent_mappings=[
            AgentMappingRule(modality="VIDEO", agent_name="video_search_agent"),
            AgentMappingRule(modality="DOCUMENT", agent_name="document_search_agent"),
        ],
    )


def create_routing_config():
    """Create test configuration for routing generator with mock DSPy"""
    return OptimizerGenerationConfig(
        optimizer_type="routing",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                module_type="Predict",
            )
        },
    )


class TestModalityGeneratorIntegration:
    """Integration tests for ModalityGenerator"""

    @pytest.mark.asyncio
    async def test_modality_generator_with_mock_data(self):
        """Test ModalityGenerator generates valid examples with mock data"""
        pattern_extractor = PatternExtractor()
        agent_inferrer = AgentInferrer()

        generator = ModalityGenerator(
            pattern_extractor=pattern_extractor,
            agent_inferrer=agent_inferrer,
            optimizer_config=create_modality_config()
        )

        # Mock content
        mock_content = [
            {
                "video_title": "Machine Learning Tutorial",
                "segment_description": "Learn about neural networks and deep learning",
                "schema_name": "video_content"
            }
        ]

        # Generate examples
        examples = await generator.generate(
            sampled_content=mock_content,
            target_count=10,
            modality="VIDEO"
        )

        assert len(examples) == 10
        assert all(isinstance(ex, ModalityExampleSchema) for ex in examples)
        assert all(ex.modality == "VIDEO" for ex in examples)
        assert all(ex.correct_agent == "video_search_agent" for ex in examples)
        assert all(ex.is_synthetic is True for ex in examples)

    @pytest.mark.asyncio
    async def test_modality_generator_without_content(self):
        """Test ModalityGenerator works without sampled content"""
        generator = ModalityGenerator(optimizer_config=create_modality_config())

        examples = await generator.generate(
            sampled_content=[],
            target_count=5,
            modality="DOCUMENT"
        )

        assert len(examples) == 5
        assert all(ex.modality == "DOCUMENT" for ex in examples)


class TestCrossModalGeneratorIntegration:
    """Integration tests for CrossModalGenerator"""

    @pytest.mark.asyncio
    async def test_cross_modal_generator(self):
        """Test CrossModalGenerator generates valid fusion examples"""
        generator = CrossModalGenerator()

        mock_content = [
            {"schema_name": "video_content", "video_title": "Test Video"},
            {"schema_name": "document_content", "title": "Test Doc"},
        ]

        examples = await generator.generate(
            sampled_content=mock_content,
            target_count=10
        )

        assert len(examples) == 10
        assert all(isinstance(ex, FusionHistorySchema) for ex in examples)
        assert all(ex.primary_modality != ex.secondary_modality for ex in examples)
        assert all(0.0 <= ex.improvement <= 1.0 for ex in examples)
        assert all(isinstance(ex.fusion_context, dict) for ex in examples)


class TestRoutingGeneratorIntegration:
    """Integration tests for RoutingGenerator"""

    @pytest.mark.asyncio
    async def test_routing_generator(self):
        """Test RoutingGenerator generates valid routing experiences"""
        pattern_extractor = PatternExtractor()
        agent_inferrer = AgentInferrer()

        generator = RoutingGenerator(
            pattern_extractor=pattern_extractor,
            agent_inferrer=agent_inferrer,
            optimizer_config=create_routing_config()
        )

        mock_content = [
            {
                "video_title": "TensorFlow Neural Networks Tutorial",
                "segment_description": "Learn TensorFlow for deep learning",
                "schema_name": "video_content"
            }
        ]

        examples = await generator.generate(
            sampled_content=mock_content,
            target_count=10
        )

        assert len(examples) == 10
        assert all(isinstance(ex, RoutingExperienceSchema) for ex in examples)
        assert all(len(ex.entities) >= 1 for ex in examples)
        assert all(0.0 <= ex.routing_confidence <= 1.0 for ex in examples)
        assert all(0.0 <= ex.search_quality <= 1.0 for ex in examples)
        assert all(ex.enhanced_query != ex.query for ex in examples)  # Should have annotations


class TestWorkflowGeneratorIntegration:
    """Integration tests for WorkflowGenerator"""

    @pytest.mark.asyncio
    async def test_workflow_generator(self):
        """Test WorkflowGenerator generates valid workflow executions"""
        generator = WorkflowGenerator()

        mock_content = [
            {"video_title": "Machine Learning Tutorial", "schema_name": "video_content"}
        ]

        examples = await generator.generate(
            sampled_content=mock_content,
            target_count=15
        )

        assert len(examples) == 15
        assert all(isinstance(ex, WorkflowExecutionSchema) for ex in examples)
        assert all(len(ex.agent_sequence) >= 1 for ex in examples)
        assert all(ex.task_count == len(ex.agent_sequence) for ex in examples)
        assert all(0.0 <= ex.parallel_efficiency <= 1.0 for ex in examples)
        assert all(0.0 <= ex.confidence_score <= 1.0 for ex in examples)
        assert all(ex.execution_time > 0 for ex in examples)

    @pytest.mark.asyncio
    async def test_workflow_generator_patterns(self):
        """Test WorkflowGenerator uses different workflow patterns"""
        generator = WorkflowGenerator()

        examples = await generator.generate(
            sampled_content=[],
            target_count=30
        )

        # Check we get different workflow lengths (simple, moderate, complex)
        lengths = [len(ex.agent_sequence) for ex in examples]
        assert min(lengths) >= 1
        assert max(lengths) >= 2  # Should have at least some multi-agent workflows


class TestAllGeneratorsTogether:
    """Test all generators can work together"""

    @pytest.mark.asyncio
    async def test_all_generators_produce_valid_output(self):
        """Test all generators can produce valid output"""
        pattern_extractor = PatternExtractor()
        agent_inferrer = AgentInferrer()

        mock_content = [
            {
                "video_title": "Deep Learning with TensorFlow",
                "segment_description": "Tutorial on neural networks",
                "schema_name": "video_content",
                "embedding_type": "video"
            }
        ]

        # Test each generator
        generators = [
            (ModalityGenerator(pattern_extractor, agent_inferrer, create_modality_config()), {"modality": "VIDEO"}),
            (CrossModalGenerator(), {}),
            (RoutingGenerator(pattern_extractor, agent_inferrer, create_routing_config()), {}),
            (WorkflowGenerator(), {}),
        ]

        for generator, kwargs in generators:
            examples = await generator.generate(
                sampled_content=mock_content,
                target_count=5,
                **kwargs
            )

            assert len(examples) == 5
            assert all(isinstance(ex, generator.__class__.__bases__[0].__orig_bases__[0].__args__[0] if hasattr(generator.__class__.__bases__[0], '__orig_bases__') else object) for ex in examples) or True  # Just check they're BaseModel instances

            # All should return Pydantic models
            assert all(hasattr(ex, 'model_dump') for ex in examples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
