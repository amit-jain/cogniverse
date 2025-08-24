"""
Unit tests for Phase 6 optimization and learning components.

Tests for:
- GRPO routing optimizer
- SIMBA query enhancer  
- Adaptive threshold learner
- MLflow integration
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import numpy as np
from typing import List, Dict, Any, Optional
import tempfile
import os

# Import components to test
from src.app.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer, 
    AdvancedOptimizerConfig,
    RoutingExperience,
    PolicyOptimizationResult
)
from src.app.routing.simba_query_enhancer import (
    SIMBAQueryEnhancer,
    SIMBAConfig, 
    QueryEnhancementPattern,
    EnhancementMemoryMetrics
)
from src.app.routing.adaptive_threshold_learner import (
    AdaptiveThresholdLearner,
    ThresholdConfig,
    PerformanceMetrics,
    AdaptationStrategy,
    ThresholdState
)
from src.app.routing.mlflow_integration import (
    MLflowIntegration,
    MLflowConfig,
    ExperimentConfig,
    PerformanceMetrics
)


class TestAdvancedRoutingOptimizer:
    """Test advanced routing optimization functionality."""
    
    @pytest.fixture
    def optimizer_config(self):
        return AdvancedOptimizerConfig(
            learning_rate=0.001,
            experience_replay_size=100,
            batch_size=16,
            update_frequency=10,
            exploration_epsilon=0.1,
            min_experiences_for_training=20
        )
    
    @pytest.fixture
    def advanced_optimizer(self, optimizer_config):
        with patch('src.app.routing.advanced_optimizer.SentenceTransformer'):
            optimizer = AdvancedRoutingOptimizer(optimizer_config)
            return optimizer
    
    def test_advanced_optimizer_initialization(self, optimizer_config):
        """Test advanced optimizer initializes correctly."""
        with patch('src.app.routing.advanced_optimizer.SentenceTransformer') as mock_st:
            optimizer = AdvancedRoutingOptimizer(optimizer_config)
            
            assert optimizer.config == optimizer_config
            assert len(optimizer.experience_replay) == 0
            assert optimizer.training_step == 0
            mock_st.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_record_routing_experience(self, advanced_optimizer):
        """Test recording routing experience."""
        entities = [{"type": "PERSON", "text": "John", "confidence": 0.9}]
        relationships = [{"subject": "John", "predicate": "works_at", "object": "Company"}]
        
        reward = await advanced_optimizer.record_routing_experience(
            query="Find John's work documents",
            entities=entities,
            relationships=relationships,
            enhanced_query="Find documents related to John who works at Company",
            chosen_agent="video_search_agent",
            routing_confidence=0.85,
            search_quality=0.78,
            agent_success=True
        )
        
        assert isinstance(reward, float)
        assert len(advanced_optimizer.experience_replay) == 1
        
        experience = advanced_optimizer.experience_replay[0]
        assert experience.query == "Find John's work documents"
        assert experience.chosen_agent == "video_search_agent"
        assert experience.routing_confidence == 0.85
        assert experience.search_quality == 0.78
        assert experience.agent_success is True
    
    def test_compute_reward_success_case(self, grpo_optimizer):
        """Test reward computation for successful routing."""
        reward = grpo_optimizer._compute_reward(
            search_quality=0.8,
            agent_success=True,
            routing_confidence=0.9,
            response_time=2.5
        )
        
        # Should be positive for successful routing
        assert reward > 0
        assert isinstance(reward, float)
    
    def test_compute_reward_failure_case(self, grpo_optimizer):
        """Test reward computation for failed routing."""
        reward = grpo_optimizer._compute_reward(
            search_quality=0.3,
            agent_success=False,
            routing_confidence=0.4,
            response_time=8.0
        )
        
        # Should be negative for failed routing
        assert reward < 0
        assert isinstance(reward, float)
    
    @pytest.mark.asyncio
    async def test_optimize_policy_insufficient_data(self, grpo_optimizer):
        """Test policy optimization with insufficient training data."""
        # Add only a few experiences (less than min_experiences_for_training)
        for i in range(5):
            await grpo_optimizer.record_routing_experience(
                query=f"query_{i}",
                entities=[],
                relationships=[],
                enhanced_query=f"enhanced_query_{i}",
                chosen_agent="agent_1",
                routing_confidence=0.5,
                search_quality=0.5,
                agent_success=True
            )
        
        result = await grpo_optimizer.optimize_policy()
        
        assert result.optimization_performed is False
        assert "Insufficient training data" in result.message
    
    @pytest.mark.asyncio
    async def test_get_routing_recommendations(self, grpo_optimizer):
        """Test getting routing recommendations."""
        # Add some training data
        for i in range(25):
            await grpo_optimizer.record_routing_experience(
                query=f"video query {i}",
                entities=[{"type": "CONTENT", "text": "video"}],
                relationships=[],
                enhanced_query=f"enhanced video query {i}",
                chosen_agent="video_search_agent",
                routing_confidence=0.8,
                search_quality=0.75,
                agent_success=True
            )
        
        recommendations = await grpo_optimizer.get_routing_recommendations(
            query="Find video content",
            entities=[{"type": "CONTENT", "text": "video"}],
            relationships=[]
        )
        
        assert isinstance(recommendations, dict)
        assert "recommended_agent" in recommendations
        assert "confidence" in recommendations
        assert "reasoning" in recommendations


class TestSIMBAQueryEnhancer:
    """Test SIMBA query enhancement functionality."""
    
    @pytest.fixture
    def simba_config(self):
        return SIMBAConfig(
            similarity_threshold=0.75,
            max_patterns_stored=500,
            pattern_decay_rate=0.95,
            enhancement_confidence_threshold=0.6,
            max_similar_patterns=5
        )
    
    @pytest.fixture
    def simba_enhancer(self, simba_config):
        with patch('src.app.routing.simba_query_enhancer.SentenceTransformer'):
            enhancer = SIMBAQueryEnhancer(simba_config)
            return enhancer
    
    def test_simba_enhancer_initialization(self, simba_config):
        """Test SIMBA enhancer initializes correctly."""
        with patch('src.app.routing.simba_query_enhancer.SentenceTransformer') as mock_st:
            enhancer = SIMBAQueryEnhancer(simba_config)
            
            assert enhancer.config == simba_config
            assert len(enhancer.patterns) == 0
            mock_st.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_successful_enhancement(self, simba_enhancer):
        """Test storing successful enhancement patterns."""
        entities = [{"type": "PERSON", "text": "Alice"}]
        relationships = [{"subject": "Alice", "predicate": "created", "object": "presentation"}]
        
        await simba_enhancer.store_successful_enhancement(
            original_query="Find Alice's work",
            entities=entities,
            relationships=relationships,
            enhanced_query="Find presentation created by Alice",
            enhancement_quality=0.85
        )
        
        assert len(simba_enhancer.patterns) == 1
        pattern = simba_enhancer.patterns[0]
        assert pattern.original_query == "Find Alice's work"
        assert pattern.enhanced_query == "Find presentation created by Alice"
        assert pattern.enhancement_quality == 0.85
    
    @pytest.mark.asyncio
    async def test_enhance_query_with_patterns_no_matches(self, simba_enhancer):
        """Test query enhancement with no similar patterns."""
        result = await simba_enhancer.enhance_query_with_patterns(
            original_query="Find documents about AI",
            entities=[{"type": "CONCEPT", "text": "AI"}],
            relationships=[]
        )
        
        assert result["enhanced_query"] == "Find documents about AI"
        assert result["confidence"] == 0.0
        assert result["pattern_matches"] == 0
        assert result["enhancement_source"] == "none"
    
    @pytest.mark.asyncio  
    async def test_enhance_query_with_similar_patterns(self, simba_enhancer):
        """Test query enhancement with similar patterns."""
        # Store some patterns first
        await simba_enhancer.store_successful_enhancement(
            original_query="Find AI research papers",
            entities=[{"type": "CONCEPT", "text": "AI"}],
            relationships=[{"subject": "AI", "predicate": "relates_to", "object": "research"}],
            enhanced_query="Find research papers about artificial intelligence and machine learning",
            enhancement_quality=0.9
        )
        
        # Mock similarity calculation to return high similarity
        with patch.object(simba_enhancer, '_calculate_pattern_similarity', return_value=0.85):
            result = await simba_enhancer.enhance_query_with_patterns(
                original_query="Find AI documents",
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[]
            )
        
        assert result["enhanced_query"] != "Find AI documents"
        assert result["confidence"] > 0
        assert result["pattern_matches"] > 0
        assert result["enhancement_source"] == "pattern_matching"
    
    def test_calculate_pattern_similarity(self, simba_enhancer):
        """Test pattern similarity calculation."""
        # Mock embedding calculation
        mock_embedding_1 = np.array([0.1, 0.2, 0.3, 0.4])
        mock_embedding_2 = np.array([0.15, 0.25, 0.35, 0.45])
        
        with patch.object(simba_enhancer.embedding_model, 'encode', side_effect=[mock_embedding_1, mock_embedding_2]):
            similarity = simba_enhancer._calculate_pattern_similarity(
                query1="test query 1",
                entities1=[],
                relationships1=[],
                query2="test query 2", 
                entities2=[],
                relationships2=[]
            )
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    def test_pattern_decay(self, simba_enhancer):
        """Test pattern quality decay over time."""
        pattern = QueryPattern(
            original_query="test",
            entities=[],
            relationships=[],
            enhanced_query="enhanced test",
            enhancement_quality=0.8,
            usage_count=5,
            last_used=datetime.now() - timedelta(days=10)
        )
        
        decayed_quality = simba_enhancer._apply_pattern_decay(pattern)
        assert decayed_quality < pattern.enhancement_quality


class TestAdaptiveThresholdLearner:
    """Test adaptive threshold learning functionality."""
    
    @pytest.fixture
    def threshold_config(self):
        return ThresholdConfig(
            initial_routing_threshold=0.7,
            initial_confidence_threshold=0.6,
            learning_rate=0.01,
            adaptation_window=100,
            min_samples_for_update=20,
            performance_target=0.8,
            optimization_strategies=[
                OptimizationStrategy.GRADIENT_BASED,
                OptimizationStrategy.EVOLUTIONARY
            ]
        )
    
    @pytest.fixture
    def threshold_learner(self, threshold_config):
        learner = AdaptiveThresholdLearner(threshold_config)
        return learner
    
    def test_threshold_learner_initialization(self, threshold_config):
        """Test threshold learner initializes correctly."""
        learner = AdaptiveThresholdLearner(threshold_config)
        
        assert learner.config == threshold_config
        assert learner.current_routing_threshold == 0.7
        assert learner.current_confidence_threshold == 0.6
        assert len(learner.performance_history) == 0
    
    @pytest.mark.asyncio
    async def test_record_performance_sample(self, threshold_learner):
        """Test recording performance samples."""
        await threshold_learner.record_performance_sample(
            routing_success=True,
            routing_confidence=0.85,
            search_quality=0.78,
            response_time=2.3
        )
        
        assert len(threshold_learner.performance_history) == 1
        sample = threshold_learner.performance_history[0]
        assert sample.routing_success is True
        assert sample.routing_confidence == 0.85
        assert sample.search_quality == 0.78
        assert sample.response_time == 2.3
    
    @pytest.mark.asyncio
    async def test_adapt_thresholds_insufficient_data(self, threshold_learner):
        """Test threshold adaptation with insufficient data."""
        # Add only a few samples
        for i in range(5):
            await threshold_learner.record_performance_sample(
                routing_success=True,
                routing_confidence=0.8,
                search_quality=0.75,
                response_time=2.0
            )
        
        result = await threshold_learner.adapt_thresholds()
        
        assert result.thresholds_updated is False
        assert "Insufficient data" in result.message
    
    @pytest.mark.asyncio
    async def test_adapt_thresholds_with_sufficient_data(self, threshold_learner):
        """Test threshold adaptation with sufficient data."""
        # Add sufficient samples with varying performance
        for i in range(25):
            success = i % 3 == 0  # 33% success rate - below target
            await threshold_learner.record_performance_sample(
                routing_success=success,
                routing_confidence=0.6 + (i % 10) * 0.04,  # 0.6 to 1.0
                search_quality=0.5 + (i % 10) * 0.05,      # 0.5 to 0.95  
                response_time=1.0 + (i % 5) * 0.5          # 1.0 to 3.0
            )
        
        result = await threshold_learner.adapt_thresholds()
        
        assert isinstance(result, OptimizationResult)
        # With poor performance, thresholds should be adjusted
        if result.thresholds_updated:
            assert result.new_routing_threshold != threshold_learner.config.initial_routing_threshold
    
    def test_gradient_based_optimization(self, threshold_learner):
        """Test gradient-based threshold optimization."""
        # Create sample data
        samples = [
            PerformanceSample(True, 0.8, 0.75, 2.0, datetime.now()),
            PerformanceSample(False, 0.6, 0.4, 3.0, datetime.now()),
            PerformanceSample(True, 0.9, 0.85, 1.5, datetime.now()),
            PerformanceSample(False, 0.5, 0.3, 4.0, datetime.now()),
            PerformanceSample(True, 0.85, 0.8, 2.2, datetime.now())
        ]
        
        new_routing_threshold, new_confidence_threshold = threshold_learner._gradient_based_optimization(
            samples, 0.7, 0.6
        )
        
        assert isinstance(new_routing_threshold, float)
        assert isinstance(new_confidence_threshold, float)
        assert 0.0 <= new_routing_threshold <= 1.0
        assert 0.0 <= new_confidence_threshold <= 1.0
    
    def test_evolutionary_optimization(self, threshold_learner):
        """Test evolutionary threshold optimization."""
        samples = [
            PerformanceSample(True, 0.8, 0.75, 2.0, datetime.now()),
            PerformanceSample(False, 0.6, 0.4, 3.0, datetime.now()),
            PerformanceSample(True, 0.9, 0.85, 1.5, datetime.now())
        ]
        
        new_routing_threshold, new_confidence_threshold = threshold_learner._evolutionary_optimization(
            samples, 0.7, 0.6
        )
        
        assert isinstance(new_routing_threshold, float)
        assert isinstance(new_confidence_threshold, float)
        assert 0.0 <= new_routing_threshold <= 1.0
        assert 0.0 <= new_confidence_threshold <= 1.0
    
    def test_statistical_analysis(self, threshold_learner):
        """Test statistical analysis of performance data."""
        samples = []
        # Create samples with clear performance difference above/below threshold
        for i in range(30):
            confidence = 0.5 + i * 0.015  # 0.5 to 0.95
            success = confidence > 0.75   # Better performance above 0.75
            samples.append(PerformanceSample(success, confidence, 0.7, 2.0, datetime.now()))
        
        is_significant, effect_size = threshold_learner._statistical_analysis(samples, 0.75)
        
        assert isinstance(is_significant, bool)
        assert isinstance(effect_size, float)


class TestMLflowIntegration:
    """Test MLflow integration functionality."""
    
    @pytest.fixture
    def mlflow_config(self):
        return MLflowConfig(
            tracking_uri="file:///tmp/mlflow_test",
            experiment_name="test_routing_optimization",
            model_registry_name="routing_models",
            enable_autolog=True,
            log_system_metrics=False  # Disable for testing
        )
    
    @pytest.fixture
    def mlflow_integration(self, mlflow_config):
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.create_experiment'):
            integration = MLflowIntegration(mlflow_config)
            return integration
    
    def test_mlflow_integration_initialization(self, mlflow_config):
        """Test MLflow integration initializes correctly."""
        with patch('mlflow.set_tracking_uri') as mock_set_uri, \
             patch('mlflow.set_experiment') as mock_set_exp, \
             patch('mlflow.create_experiment') as mock_create_exp:
            
            integration = MLflowIntegration(mlflow_config)
            
            assert integration.config == mlflow_config
            mock_set_uri.assert_called_once_with(mlflow_config.tracking_uri)
            mock_set_exp.assert_called_once_with(mlflow_config.experiment_name)
    
    @pytest.mark.asyncio
    async def test_log_routing_performance(self, mlflow_integration):
        """Test logging routing performance metrics."""
        with patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.log_params') as mock_log_params:
            
            routing_decision = {
                "chosen_agent": "video_search_agent",
                "confidence": 0.85,
                "reasoning": "High confidence video query"
            }
            
            performance_metrics = {
                "search_quality": 0.78,
                "response_time": 2.3,
                "agent_success": 1.0
            }
            
            await mlflow_integration.log_routing_performance(
                query="Find video about AI",
                routing_decision=routing_decision,
                performance_metrics=performance_metrics
            )
            
            mock_log_metrics.assert_called_once()
            mock_log_params.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_experiment_run(self, mlflow_integration):
        """Test starting experiment run."""
        experiment_config = ExperimentConfig(
            run_name="test_run",
            description="Test experiment run",
            tags={"environment": "test", "version": "1.0"}
        )
        
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.set_tags') as mock_set_tags:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_id"
            mock_start_run.return_value = mock_run
            
            run_id = await mlflow_integration.start_experiment_run(experiment_config)
            
            assert run_id == "test_run_id"
            mock_start_run.assert_called_once()
            mock_set_tags.assert_called_once_with(experiment_config.tags)
    
    @pytest.mark.asyncio
    async def test_register_model(self, mlflow_integration):
        """Test model registration."""
        with patch('mlflow.register_model') as mock_register, \
             patch('mlflow.get_latest_versions') as mock_get_versions:
            mock_model_version = Mock()
            mock_model_version.version = "1"
            mock_register.return_value = mock_model_version
            
            model_version = await mlflow_integration.register_model(
                model_path="models/routing_model",
                model_name="test_routing_model"
            )
            
            assert model_version == "1"
            mock_register.assert_called_once()
    
    @pytest.mark.asyncio  
    async def test_ab_test_setup(self, mlflow_integration):
        """Test A/B test configuration setup."""
        ab_config = {
            "test_name": "routing_strategy_test",
            "variant_a": {"strategy": "confidence_based", "threshold": 0.7},
            "variant_b": {"strategy": "adaptive", "threshold": 0.75},
            "traffic_split": 0.5
        }
        
        with patch('mlflow.log_params') as mock_log_params:
            test_id = await mlflow_integration.setup_ab_test(ab_config)
            
            assert isinstance(test_id, str)
            assert len(test_id) > 0
            mock_log_params.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_performance_dashboard_data(self, mlflow_integration):
        """Test getting performance dashboard data."""
        with patch('mlflow.search_runs') as mock_search:
            mock_df = Mock()
            mock_df.to_dict.return_value = {"metrics": {"accuracy": [0.85, 0.87, 0.89]}}
            mock_search.return_value = mock_df
            
            dashboard_data = await mlflow_integration.get_performance_dashboard_data(
                time_range_hours=24
            )
            
            assert isinstance(dashboard_data, dict)
            assert "metrics" in dashboard_data
            mock_search.assert_called_once()


class TestPhase6Integration:
    """Test integration between Phase 6 components."""
    
    @pytest.fixture
    def integrated_components(self):
        """Create integrated Phase 6 components for testing."""
        with patch('src.app.routing.grpo_optimizer.SentenceTransformer'), \
             patch('src.app.routing.simba_query_enhancer.SentenceTransformer'), \
             patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), \
             patch('mlflow.create_experiment'):
            
            grpo_config = GRPOConfig()
            simba_config = SIMBAConfig()
            threshold_config = ThresholdConfig()
            mlflow_config = MLflowConfig(tracking_uri="file:///tmp/test")
            
            components = {
                'grpo': GRPORoutingOptimizer(grpo_config),
                'simba': SIMBAQueryEnhancer(simba_config),
                'thresholds': AdaptiveThresholdLearner(threshold_config),
                'mlflow': MLflowIntegration(mlflow_config)
            }
            
            return components
    
    @pytest.mark.asyncio
    async def test_optimization_workflow_integration(self, integrated_components):
        """Test integration of optimization workflow across components."""
        grpo = integrated_components['grpo']
        simba = integrated_components['simba']
        thresholds = integrated_components['thresholds']
        mlflow = integrated_components['mlflow']
        
        # Simulate optimization workflow
        query = "Find AI research videos"
        entities = [{"type": "CONCEPT", "text": "AI"}]
        relationships = [{"subject": "AI", "predicate": "relates_to", "object": "research"}]
        
        # 1. Record GRPO experience
        grpo_reward = await grpo.record_routing_experience(
            query=query,
            entities=entities,
            relationships=relationships,
            enhanced_query="Find artificial intelligence and machine learning research videos",
            chosen_agent="video_search_agent",
            routing_confidence=0.85,
            search_quality=0.78,
            agent_success=True
        )
        
        # 2. Store SIMBA pattern
        await simba.store_successful_enhancement(
            original_query=query,
            entities=entities,
            relationships=relationships,
            enhanced_query="Find artificial intelligence and machine learning research videos",
            enhancement_quality=0.82
        )
        
        # 3. Record threshold performance
        await thresholds.record_performance_sample(
            routing_success=True,
            routing_confidence=0.85,
            search_quality=0.78,
            response_time=2.1
        )
        
        # 4. Log to MLflow
        with patch('mlflow.log_metrics'), patch('mlflow.log_params'):
            await mlflow.log_routing_performance(
                query=query,
                routing_decision={
                    "chosen_agent": "video_search_agent",
                    "confidence": 0.85,
                    "reasoning": "High confidence AI query"
                },
                performance_metrics={
                    "search_quality": 0.78,
                    "response_time": 2.1,
                    "agent_success": 1.0,
                    "grpo_reward": grpo_reward
                }
            )
        
        # Verify all components recorded data
        assert len(grpo.experience_replay) == 1
        assert len(simba.patterns) == 1  
        assert len(thresholds.performance_history) == 1
        assert isinstance(grpo_reward, float)
    
    @pytest.mark.asyncio
    async def test_learning_feedback_loop(self, integrated_components):
        """Test learning feedback loop between components."""
        grpo = integrated_components['grpo']
        simba = integrated_components['simba']
        
        # Simulate multiple learning cycles
        queries = [
            "Find machine learning tutorials",
            "Search for deep learning videos", 
            "Locate neural network presentations",
            "Find AI conference talks"
        ]
        
        for i, query in enumerate(queries):
            # SIMBA enhancement
            simba_result = await simba.enhance_query_with_patterns(
                original_query=query,
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[]
            )
            
            # GRPO routing decision  
            grpo_recommendations = await grpo.get_routing_recommendations(
                query=query,
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[]
            )
            
            # Record successful interaction
            await grpo.record_routing_experience(
                query=query,
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[],
                enhanced_query=simba_result["enhanced_query"],
                chosen_agent="video_search_agent",
                routing_confidence=0.8,
                search_quality=0.75,
                agent_success=True
            )
            
            await simba.store_successful_enhancement(
                original_query=query,
                entities=[{"type": "CONCEPT", "text": "AI"}],
                relationships=[],
                enhanced_query=simba_result["enhanced_query"],
                enhancement_quality=0.75
            )
        
        # Verify learning occurred
        assert len(grpo.experience_replay) == len(queries)
        assert len(simba.patterns) == len(queries)
        
        # Test that subsequent queries benefit from learning
        new_query = "Find artificial intelligence research"
        enhanced_result = await simba.enhance_query_with_patterns(
            original_query=new_query,
            entities=[{"type": "CONCEPT", "text": "AI"}],
            relationships=[]
        )
        
        # Should have some enhancement confidence due to stored patterns
        # (exact value depends on similarity calculation which is mocked)
        assert isinstance(enhanced_result["confidence"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])