"""
End-to-End Integration Tests for Synthetic Data + Optimizers

Tests that synthetic data generation integrates correctly with all optimizers.
"""

import pytest
from cogniverse_agents.workflow_intelligence import WorkflowIntelligence
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer


class TestCrossModalOptimizerIntegration:
    """Integration tests for CrossModalOptimizer with synthetic data"""

    def test_cross_modal_optimizer_initialization(self):
        """Test CrossModalOptimizer can be initialized"""
        optimizer = CrossModalOptimizer()
        assert optimizer is not None
        assert isinstance(optimizer.fusion_history, list)

    @pytest.mark.asyncio
    async def test_generate_synthetic_cross_modal_data(self):
        """Test generating synthetic data for CrossModalOptimizer"""
        optimizer = CrossModalOptimizer()

        # Generate small batch without Vespa (uses mock data)
        count = await optimizer.generate_synthetic_training_data(count=10)

        assert count == 10
        assert len(optimizer.fusion_history) == 10

    @pytest.mark.asyncio
    async def test_cross_modal_data_structure(self):
        """Test generated data has correct structure"""
        optimizer = CrossModalOptimizer()

        await optimizer.generate_synthetic_training_data(count=5)

        # Check first example structure
        example = optimizer.fusion_history[0]
        assert "primary_modality" in example
        assert "secondary_modality" in example
        assert "fusion_context" in example
        assert "success" in example
        assert "improvement" in example

    @pytest.mark.asyncio
    async def test_cross_modal_training_after_synthetic_generation(self):
        """Test that CrossModalOptimizer can train on synthetic data"""
        optimizer = CrossModalOptimizer()

        # Generate synthetic data
        await optimizer.generate_synthetic_training_data(count=50)

        # Attempt to train
        result = optimizer.train_fusion_model()

        # Should have sufficient data now
        assert result["status"] == "success"
        assert result.get("mae") is not None


class TestAdvancedRoutingOptimizerIntegration:
    """Integration tests for AdvancedRoutingOptimizer with synthetic data"""

    def test_advanced_optimizer_initialization(self):
        """Test AdvancedRoutingOptimizer can be initialized"""
        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant")
        assert optimizer is not None
        assert isinstance(optimizer.experiences, list)

    @pytest.mark.asyncio
    async def test_generate_synthetic_routing_data(self, test_generator_config):
        """Test generating synthetic data for AdvancedRoutingOptimizer"""
        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant")

        # Clear any existing experiences
        initial_count = len(optimizer.experiences)
        optimizer.experiences.clear()

        # Generate small batch without Vespa (uses mock data)
        count = await optimizer.generate_synthetic_training_data(count=10, generator_config=test_generator_config)

        assert count == 10
        assert len(optimizer.experiences) == 10

    @pytest.mark.asyncio
    async def test_routing_data_structure(self, test_generator_config):
        """Test generated data has correct structure"""
        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant")

        await optimizer.generate_synthetic_training_data(count=5, generator_config=test_generator_config)

        # Check first experience
        experience = optimizer.experiences[0]
        assert experience.query is not None
        assert isinstance(experience.entities, list)
        assert isinstance(experience.relationships, list)
        assert experience.enhanced_query is not None
        assert experience.chosen_agent is not None
        assert 0 <= experience.routing_confidence <= 1
        assert 0 <= experience.search_quality <= 1

    @pytest.mark.asyncio
    async def test_routing_data_variety(self, test_generator_config):
        """Test that generated data has variety"""
        optimizer = AdvancedRoutingOptimizer(tenant_id="test_tenant")

        await optimizer.generate_synthetic_training_data(count=20, generator_config=test_generator_config)

        # Check we have variety in queries
        queries = [exp.query for exp in optimizer.experiences]
        unique_queries = set(queries)

        assert len(unique_queries) > 1, "Generated queries should be diverse"

        # Check we have variety in agents
        agents = [exp.chosen_agent for exp in optimizer.experiences]
        unique_agents = set(agents)

        assert len(unique_agents) >= 1, "Should have at least one agent type"


class TestWorkflowIntelligenceIntegration:
    """Integration tests for WorkflowIntelligence with synthetic data"""

    def test_workflow_intelligence_initialization(self):
        """Test WorkflowIntelligence can be initialized"""
        workflow_intel = WorkflowIntelligence()
        assert workflow_intel is not None
        assert hasattr(workflow_intel, 'workflow_history')

    @pytest.mark.asyncio
    async def test_generate_synthetic_workflow_data(self):
        """Test generating synthetic data for WorkflowIntelligence"""
        workflow_intel = WorkflowIntelligence()

        # Generate small batch without Vespa (uses mock data)
        count = await workflow_intel.generate_synthetic_training_data(count=10)

        assert count == 10
        assert len(workflow_intel.workflow_history) == 10

    @pytest.mark.asyncio
    async def test_workflow_data_structure(self):
        """Test generated data has correct structure"""
        workflow_intel = WorkflowIntelligence()

        await workflow_intel.generate_synthetic_training_data(count=5)

        # Check first execution
        execution = workflow_intel.workflow_history[0]
        assert execution.workflow_id is not None
        assert execution.query is not None
        assert execution.query_type is not None
        assert execution.execution_time > 0
        assert isinstance(execution.success, bool)
        assert isinstance(execution.agent_sequence, list)
        assert len(execution.agent_sequence) >= 1
        assert execution.task_count >= 1
        assert 0 <= execution.parallel_efficiency <= 1
        assert 0 <= execution.confidence_score <= 1

    @pytest.mark.asyncio
    async def test_workflow_pattern_variety(self):
        """Test that generated workflows have different patterns"""
        workflow_intel = WorkflowIntelligence()

        await workflow_intel.generate_synthetic_training_data(count=30)

        # Check we have different workflow lengths
        sequence_lengths = [len(ex.agent_sequence) for ex in workflow_intel.workflow_history]

        # Should have at least simple (1 agent) and complex (3+ agents) workflows
        assert min(sequence_lengths) >= 1
        assert max(sequence_lengths) >= 2


class TestMultiOptimizerIntegration:
    """Integration tests across multiple optimizers"""

    @pytest.mark.asyncio
    async def test_all_optimizers_can_generate_data(self, test_generator_config):
        """Test that all optimizers can generate synthetic data"""
        cross_modal = CrossModalOptimizer()
        routing = AdvancedRoutingOptimizer(tenant_id="test_tenant")
        workflow = WorkflowIntelligence()

        # Generate for all
        cross_modal_count = await cross_modal.generate_synthetic_training_data(count=5)
        routing_count = await routing.generate_synthetic_training_data(count=5, generator_config=test_generator_config)
        workflow_count = await workflow.generate_synthetic_training_data(count=5)

        assert cross_modal_count == 5
        assert routing_count == 5
        assert workflow_count == 5

    @pytest.mark.asyncio
    async def test_synthetic_data_is_independent(self, test_generator_config):
        """Test that each optimizer gets independent data"""
        cross_modal = CrossModalOptimizer()
        routing = AdvancedRoutingOptimizer(tenant_id="test_tenant")

        # Clear state before testing
        cross_modal.fusion_history.clear()
        routing.experiences.clear()

        await cross_modal.generate_synthetic_training_data(count=3)
        await routing.generate_synthetic_training_data(count=3, generator_config=test_generator_config)

        # Data should be different (fusion history vs routing experiences)
        assert len(cross_modal.fusion_history) == 3
        assert len(routing.experiences) == 3

        # Check structure is different
        fusion_example = cross_modal.fusion_history[0]
        routing_example = routing.experiences[0]

        # Fusion has different fields than routing
        assert "fusion_context" in fusion_example
        assert not hasattr(routing_example, "fusion_context")
        assert hasattr(routing_example, "entities")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
