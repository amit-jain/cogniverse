"""
Integration tests for all synthetic data generators
"""

import pytest

from cogniverse_foundation.config.unified_config import (
    DSPyModuleConfig,
    OptimizerGenerationConfig,
)
from cogniverse_synthetic.generators import (
    ProfileGenerator,
    RoutingGenerator,
    WorkflowGenerator,
)
from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)
from cogniverse_synthetic.utils import AgentInferrer, PatternExtractor


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


class TestProfileGeneratorIntegration:
    """Integration tests for ProfileGenerator (pattern-based, no DSPy)."""

    @pytest.mark.asyncio
    async def test_profile_generator_with_mock_data(self):
        generator = ProfileGenerator()

        mock_content = [
            {
                "video_title": "Machine Learning Tutorial",
                "segment_description": "Learn about neural networks and deep learning",
                "schema_name": "video_content",
            }
        ]

        examples = await generator.generate(
            sampled_content=mock_content, target_count=10
        )

        assert len(examples) == 10
        assert all(isinstance(ex, ProfileSelectionExampleSchema) for ex in examples)
        for ex in examples:
            available = [p.strip() for p in ex.available_profiles.split(",")]
            assert ex.selected_profile in available
            assert ex.modality in {"video", "image", "audio", "document", "text"}
            assert ex.complexity in {"simple", "medium", "complex"}

    @pytest.mark.asyncio
    async def test_profile_generator_without_content(self):
        generator = ProfileGenerator()
        examples = await generator.generate(sampled_content=[], target_count=5)

        assert len(examples) == 5
        assert all(isinstance(ex, ProfileSelectionExampleSchema) for ex in examples)


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
            optimizer_config=create_routing_config(),
        )

        mock_content = [
            {
                "video_title": "TensorFlow Neural Networks Tutorial",
                "segment_description": "Learn TensorFlow for deep learning",
                "schema_name": "video_content",
            }
        ]

        examples = await generator.generate(
            sampled_content=mock_content, target_count=10
        )

        assert len(examples) == 10
        assert all(isinstance(ex, RoutingExperienceSchema) for ex in examples)
        assert all(len(ex.entities) >= 1 for ex in examples)
        assert all(0.0 <= ex.routing_confidence <= 1.0 for ex in examples)
        assert all(0.0 <= ex.search_quality <= 1.0 for ex in examples)
        assert all(
            ex.enhanced_query != ex.query for ex in examples
        )  # Should have annotations


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
            sampled_content=mock_content, target_count=15
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

        examples = await generator.generate(sampled_content=[], target_count=30)

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
                "embedding_type": "video",
            }
        ]

        generators = [
            ProfileGenerator(),
            RoutingGenerator(
                pattern_extractor, agent_inferrer, create_routing_config()
            ),
            WorkflowGenerator(),
        ]

        for generator in generators:
            examples = await generator.generate(
                sampled_content=mock_content, target_count=5
            )

            assert len(examples) == 5
            # All should return Pydantic models with model_dump
            assert all(hasattr(ex, "model_dump") for ex in examples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
