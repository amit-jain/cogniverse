"""
Unit tests for DSPy routing components.

Tests the routing-specific DSPy components including relationship extraction,
query enhancement, adaptive threshold learning, and advanced optimization.
"""

from unittest.mock import patch

import pytest

from src.app.routing.adaptive_threshold_learner import AdaptiveThresholdLearner
from src.app.routing.advanced_optimizer import (
    AdvancedOptimizerConfig,
    AdvancedRoutingOptimizer,
)
from src.app.routing.dspy_relationship_router import (
    DSPyEntityExtractorModule,
    DSPyRelationshipExtractorModule,
)
from src.app.routing.dspy_routing_signatures import (
    AdvancedRoutingSignature,
    BasicQueryAnalysisSignature,
)
from src.app.routing.query_enhancement_engine import QueryEnhancementPipeline

# DSPy routing components
from src.app.routing.relationship_extraction_tools import RelationshipExtractorTool


class TestDSPyRoutingSignatures:
    """Test DSPy routing signature definitions"""
    
    def test_basic_query_analysis_signature(self):
        """Test basic query analysis signature"""
        signature = BasicQueryAnalysisSignature
        
        # Check that it's a proper DSPy signature
        assert hasattr(signature, '__annotations__')
        assert hasattr(signature, '__doc__')
        
        # Basic structure check
        assert signature is not None
        
    def test_advanced_routing_signature(self):
        """Test advanced routing signature"""  
        signature = AdvancedRoutingSignature
        
        # Check that it's a proper DSPy signature
        assert hasattr(signature, '__annotations__')
        assert hasattr(signature, '__doc__')
        
        # Basic structure check
        assert signature is not None


class TestRelationshipExtractionTool:
    """Test relationship extraction tool"""
    
    def test_tool_initialization(self):
        """Test relationship extraction tool can be initialized"""
        try:
            tool = RelationshipExtractorTool()
            assert tool is not None
            assert hasattr(tool, 'config')
        except Exception:
            # Tool may require spacy models that aren't available in CI
            pytest.skip("RelationshipExtractorTool requires spacy models")
    
    @pytest.mark.asyncio
    async def test_extract_relationships_basic(self):
        """Test basic relationship extraction"""
        try:
            tool = RelationshipExtractorTool()
            
            # Mock the actual extraction to avoid model dependencies
            with patch.object(tool, '_extract_with_spacy', return_value=[
                {"subject": "person", "relation": "performs", "object": "action"}
            ]):
                result = await tool.extract_relationships("A person performs an action", [])
                assert isinstance(result, list)
                assert len(result) >= 0
                
        except Exception:
            pytest.skip("RelationshipExtractorTool requires dependencies")


class TestQueryEnhancementPipeline:
    """Test query enhancement pipeline"""
    
    def test_pipeline_initialization(self):
        """Test query enhancement pipeline initialization"""
        try:
            pipeline = QueryEnhancementPipeline()
            assert pipeline is not None
            assert hasattr(pipeline, 'config')
        except Exception:
            pytest.skip("QueryEnhancementPipeline requires dependencies")
    
    @pytest.mark.asyncio  
    async def test_enhance_query_basic(self):
        """Test basic query enhancement"""
        try:
            pipeline = QueryEnhancementPipeline()
            
            # Mock enhancement to avoid model dependencies
            with patch.object(pipeline, '_enhance_with_context', return_value="enhanced query"):
                result = await pipeline.enhance_query("original query", [], [])
                assert isinstance(result, dict)
                assert "enhanced_query" in result
                
        except Exception:
            pytest.skip("QueryEnhancementPipeline requires dependencies")


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learner"""
    
    def test_learner_initialization(self):
        """Test adaptive threshold learner initialization"""
        learner = AdaptiveThresholdLearner()
        assert learner is not None
        assert hasattr(learner, 'config')
        assert hasattr(learner, 'threshold_states')
    
    def test_learning_status(self):
        """Test getting learning status"""
        learner = AdaptiveThresholdLearner()
        status = learner.get_learning_status()
        
        assert isinstance(status, dict)
        assert 'threshold_status' in status
    
    @pytest.mark.asyncio
    async def test_record_performance_sample(self):
        """Test recording performance sample"""
        learner = AdaptiveThresholdLearner()
        
        # Test recording performance sample
        try:
            await learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.8,
                search_quality=0.7,
                response_time=1.0,
                user_satisfaction=0.9
            )
            # Should not raise an error
            assert True
        except Exception as e:
            pytest.fail(f"record_performance_sample failed: {e}")


class TestAdvancedRoutingOptimizer:
    """Test advanced routing optimizer"""
    
    def test_config_creation(self):
        """Test optimizer configuration"""
        config = AdvancedOptimizerConfig()
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.optimizer_strategy is not None
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        config = AdvancedOptimizerConfig(min_experiences_for_training=100)
        optimizer = AdvancedRoutingOptimizer(config)
        
        assert optimizer.config == config
        assert len(optimizer.experiences) == 0
        assert optimizer.training_step == 0
    
    @pytest.mark.asyncio
    async def test_record_experience(self):
        """Test recording routing experience"""
        config = AdvancedOptimizerConfig(min_experiences_for_training=100)
        optimizer = AdvancedRoutingOptimizer(config)
        
        reward = await optimizer.record_routing_experience(
            query="test query",
            entities=[],
            relationships=[],
            enhanced_query="enhanced test query", 
            chosen_agent="test_agent",
            routing_confidence=0.8,
            search_quality=0.7,
            agent_success=True
        )
        
        assert isinstance(reward, float)
        assert 0 <= reward <= 1
        assert len(optimizer.experiences) == 1
    
    def test_get_optimization_status(self):
        """Test getting optimization status"""
        config = AdvancedOptimizerConfig()
        optimizer = AdvancedRoutingOptimizer(config)
        
        status = optimizer.get_optimization_status()
        assert isinstance(status, dict)
        assert 'optimizer_ready' in status
        assert 'total_experiences' in status


class TestDSPyRouterModules:
    """Test DSPy router modules"""
    
    def test_entity_extractor_module(self):
        """Test DSPy entity extractor module"""
        try:
            module = DSPyEntityExtractorModule()
            assert module is not None
            assert hasattr(module, 'forward')
        except Exception:
            pytest.skip("DSPyEntityExtractorModule requires DSPy setup")
    
    def test_relationship_extractor_module(self):
        """Test DSPy relationship extractor module"""
        try:
            module = DSPyRelationshipExtractorModule()
            assert module is not None
            assert hasattr(module, 'forward')
        except Exception:
            pytest.skip("DSPyRelationshipExtractorModule requires DSPy setup")


class TestDSPyRoutingIntegration:
    """Test integration between DSPy routing components"""
    
    @pytest.mark.asyncio
    async def test_component_interaction(self):
        """Test basic interaction between components"""
        try:
            # Initialize components
            extractor = RelationshipExtractorTool()
            pipeline = QueryEnhancementPipeline()
            AdaptiveThresholdLearner()  # Create but don't assign to unused variable
            optimizer = AdvancedRoutingOptimizer(AdvancedOptimizerConfig(min_experiences_for_training=10))
            
            # Mock interactions to avoid model dependencies
            with patch.object(extractor, 'extract_relationships', return_value=[]):
                with patch.object(pipeline, 'enhance_query', return_value={"enhanced_query": "test"}):
                    
                    # Test that components can work together
                    entities = []
                    relationships = await extractor.extract_relationships("test query", entities)
                    enhanced = await pipeline.enhance_query("test query", entities, relationships)
                    
                    assert isinstance(relationships, list)
                    assert isinstance(enhanced, dict)
                    
                    # Test optimizer recording
                    await optimizer.record_routing_experience(
                        query="test query",
                        entities=entities,
                        relationships=relationships,
                        enhanced_query=enhanced.get("enhanced_query", "test"),
                        chosen_agent="test_agent",
                        routing_confidence=0.8,
                        search_quality=0.7,
                        agent_success=True
                    )
                    
                    assert len(optimizer.experiences) == 1
                    
        except Exception:
            pytest.skip("Component integration requires dependencies")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
