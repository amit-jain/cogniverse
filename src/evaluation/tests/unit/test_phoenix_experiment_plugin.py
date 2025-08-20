"""
Unit tests for Phoenix experiment plugin.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from src.evaluation.plugins.phoenix_experiment import (
    PhoenixExperimentPlugin,
    register,
    get_phoenix_evaluators
)


class TestPhoenixExperimentPlugin:
    """Test PhoenixExperimentPlugin functionality."""
    
    @pytest.fixture
    def mock_example(self):
        """Mock Phoenix experiment example."""
        example = Mock()
        example.input = {"query": "test query"}
        return example
    
    @pytest.fixture
    def mock_search_service(self):
        """Mock search service."""
        service = Mock()
        result = Mock()
        result.to_dict.return_value = {
            "source_id": "video1",
            "score": 0.95,
            "content": "test content",
            "document_id": "video1_frame_0"
        }
        service.search.return_value = [result]
        return service
    
    @pytest.mark.unit
    def test_wrap_inspect_task_for_phoenix_basic(self, mock_example, mock_search_service):
        """Test basic Phoenix task wrapping."""
        inspect_solver = Mock()
        profiles = ["profile1"]
        strategies = ["strategy1"]
        config = {"top_k": 5}
        
        with patch('src.app.search.service.SearchService', return_value=mock_search_service), \
             patch('src.common.config.get_config', return_value={"vespa_url": "http://localhost"}):
            
            task_func = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
                inspect_solver, profiles, strategies, config
            )
            
            result = task_func(mock_example)
            
            assert result["query"] == "test query"
            assert "results" in result
            assert "profile1_strategy1" in result["results"]
            assert result["results"]["profile1_strategy1"]["success"] == True
            assert len(result["results"]["profile1_strategy1"]["results"]) == 1
            assert result["results"]["profile1_strategy1"]["results"][0]["video_id"] == "video1"
            assert result["results"]["profile1_strategy1"]["results"][0]["score"] == 0.95
            assert "timestamp" in result
    
    @pytest.mark.unit
    def test_wrap_inspect_task_with_dict_example(self, mock_search_service):
        """Test task wrapping with dictionary example (no .input attribute)."""
        example = {"query": "dict query"}
        
        with patch('src.app.search.service.SearchService', return_value=mock_search_service), \
             patch('src.common.config.get_config', return_value={"vespa_url": "http://localhost"}):
            
            task_func = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
                Mock(), ["profile1"], ["strategy1"], {}
            )
            
            result = task_func(example)
            
            assert result["query"] == "dict query"
    
    @pytest.mark.unit
    def test_wrap_inspect_task_multiple_profiles_strategies(self, mock_example, mock_search_service):
        """Test task wrapping with multiple profiles and strategies."""
        profiles = ["profile1", "profile2"]
        strategies = ["strategy1", "strategy2"]
        
        with patch('src.app.search.service.SearchService', return_value=mock_search_service), \
             patch('src.common.config.get_config', return_value={"vespa_url": "http://localhost"}):
            
            task_func = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
                Mock(), profiles, strategies, {}
            )
            
            result = task_func(mock_example)
            
            # Should have 4 result combinations (2 profiles × 2 strategies)
            assert len(result["results"]) == 4
            assert "profile1_strategy1" in result["results"]
            assert "profile1_strategy2" in result["results"]
            assert "profile2_strategy1" in result["results"]
            assert "profile2_strategy2" in result["results"]
    
    @pytest.mark.unit
    def test_wrap_inspect_task_search_failure(self, mock_example):
        """Test task wrapping with search service failure."""
        with patch('src.app.search.service.SearchService', side_effect=Exception("Search failed")), \
             patch('src.common.config.get_config', return_value={"vespa_url": "http://localhost"}):
            
            task_func = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
                Mock(), ["profile1"], ["strategy1"], {}
            )
            
            result = task_func(mock_example)
            
            assert result["results"]["profile1_strategy1"]["success"] == False
            assert result["results"]["profile1_strategy1"]["error"] == "Search failed"
            assert result["results"]["profile1_strategy1"]["results"] == []
    
    @pytest.mark.unit
    def test_wrap_inspect_task_video_id_extraction(self, mock_example):
        """Test video ID extraction from different document ID formats."""
        service = Mock()
        
        # Test different result formats
        results = [
            Mock(**{"to_dict.return_value": {"document_id": "video1_frame_0", "score": 0.9}}),
            Mock(**{"to_dict.return_value": {"source_id": "video2", "score": 0.8}}),
            Mock(**{"to_dict.return_value": {"document_id": "video3", "score": 0.7}})
        ]
        service.search.return_value = results
        
        with patch('src.app.search.service.SearchService', return_value=service), \
             patch('src.common.config.get_config', return_value={"vespa_url": "http://localhost"}):
            
            task_func = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
                Mock(), ["profile1"], ["strategy1"], {}
            )
            
            result = task_func(mock_example)
            
            formatted_results = result["results"]["profile1_strategy1"]["results"]
            assert formatted_results[0]["video_id"] == "video1"  # Extracted from frame ID
            assert formatted_results[1]["video_id"] == "video2"  # From source_id
            assert formatted_results[2]["video_id"] == "video3"  # Direct document_id
    
    @pytest.mark.unit
    def test_wrap_inspect_task_score_handling(self, mock_example):
        """Test score handling and rank assignment."""
        service = Mock()
        
        # Test result without score
        result_without_score = Mock()
        result_without_score.to_dict.return_value = {"source_id": "video1"}
        service.search.return_value = [result_without_score]
        
        with patch('src.app.search.service.SearchService', return_value=service), \
             patch('src.common.config.get_config', return_value={"vespa_url": "http://localhost"}):
            
            task_func = PhoenixExperimentPlugin.wrap_inspect_task_for_phoenix(
                Mock(), ["profile1"], ["strategy1"], {}
            )
            
            result = task_func(mock_example)
            
            formatted_result = result["results"]["profile1_strategy1"]["results"][0]
            assert formatted_result["score"] == 1.0  # Default score (1/rank)
            assert formatted_result["rank"] == 1
    
    @pytest.mark.unit
    def test_run_inspect_with_phoenix_tracking_success(self):
        """Test successful Inspect AI evaluation with Phoenix tracking."""
        mock_client = Mock()
        mock_dataset = Mock()
        mock_client.get_dataset.return_value = mock_dataset
        
        mock_result = {"experiment_id": "test_exp", "results": "success"}
        
        # Create a mock evaluator that will pass Phoenix validation
        mock_evaluator = Mock()
        mock_evaluator.__call__ = Mock(return_value={"score": 0.8})
        
        with patch('phoenix.Client', return_value=mock_client), \
             patch('src.evaluation.plugins.phoenix_experiment.run_experiment', return_value=mock_result) as mock_run_exp, \
             patch('src.evaluation.core.solvers.create_retrieval_solver') as mock_solver:
            
            result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
                dataset_name="test_dataset",
                profiles=["profile1"],
                strategies=["strategy1"],
                evaluators=[],  # Use empty list to avoid validation issues
                config={"top_k": 5}
            )
            
            assert result == mock_result
            mock_client.get_dataset.assert_called_once_with(name="test_dataset")
            mock_solver.assert_called_once_with(["profile1"], ["strategy1"], {"top_k": 5})
            mock_run_exp.assert_called_once()
            
            # Check experiment call arguments
            call_args = mock_run_exp.call_args
            assert call_args[1]["dataset"] == mock_dataset
            assert "task" in call_args[1]
            assert call_args[1]["evaluators"] == []
            assert "inspect_eval_test_dataset_" in call_args[1]["experiment_name"]
            
            metadata = call_args[1]["experiment_metadata"]
            assert metadata["profiles"] == ["profile1"]
            assert metadata["strategies"] == ["strategy1"]
            assert metadata["framework"] == "inspect_ai"
            assert metadata["storage"] == "phoenix"
    
    @pytest.mark.unit
    def test_run_inspect_with_phoenix_tracking_dataset_not_found(self):
        """Test error handling when dataset is not found."""
        mock_client = Mock()
        mock_client.get_dataset.return_value = None
        
        with patch('phoenix.Client', return_value=mock_client):
            with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
                PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
                    dataset_name="nonexistent",
                    profiles=["profile1"],
                    strategies=["strategy1"],
                    evaluators=[]
                )
    
    @pytest.mark.unit
    def test_run_inspect_with_phoenix_tracking_no_config(self):
        """Test running with no config provided."""
        mock_client = Mock()
        # Create a more realistic mock dataset
        mock_dataset = Mock()
        mock_dataset.id = "test_dataset_id"
        mock_dataset.name = "test_dataset"
        mock_dataset.examples = []
        mock_client.get_dataset.return_value = mock_dataset
        mock_result = {"success": True}
        
        with patch('phoenix.Client', return_value=mock_client), \
             patch('src.evaluation.plugins.phoenix_experiment.run_experiment', return_value=mock_result), \
             patch('src.evaluation.core.solvers.create_retrieval_solver'):
            
            result = PhoenixExperimentPlugin.run_inspect_with_phoenix_tracking(
                dataset_name="test_dataset",
                profiles=["profile1"],
                strategies=["strategy1"],
                evaluators=[],
                config={}  # Provide empty dict instead of None
            )
            
            assert result == mock_result


class TestPhoenixExperimentPluginUtilities:
    """Test utility functions in Phoenix experiment plugin."""
    
    @pytest.mark.unit
    def test_register_function(self):
        """Test plugin registration function."""
        result = register()
        assert result == True
    
    @pytest.mark.unit
    def test_get_phoenix_evaluators_no_evaluators(self):
        """Test getting evaluators with no configuration."""
        config = {
            "enable_llm_evaluators": False,
            "enable_quality_evaluators": False
        }
        
        evaluators = get_phoenix_evaluators(config)
        assert evaluators == []
    
    @pytest.mark.unit
    def test_get_phoenix_evaluators_llm_only(self):
        """Test getting LLM evaluators only."""
        config = {
            "enable_llm_evaluators": True,
            "enable_quality_evaluators": False,
            "evaluator_name": "visual_judge",
            "evaluators": {
                "visual_judge": {
                    "provider": "ollama",
                    "model": "test_model",
                    "base_url": "http://localhost",
                    "api_key": "test_key"
                }
            }
        }
        
        with patch('src.evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge') as mock_judge:
            mock_instance = Mock()
            mock_judge.return_value = mock_instance
            
            evaluators = get_phoenix_evaluators(config)
            
            assert len(evaluators) == 1
            assert evaluators[0] == mock_instance
            mock_judge.assert_called_once_with(
                provider="ollama",
                model="test_model",
                base_url="http://localhost",
                api_key="test_key"
            )
    
    @pytest.mark.unit
    def test_get_phoenix_evaluators_quality_only(self):
        """Test getting quality evaluators only."""
        config = {
            "enable_llm_evaluators": False,
            "enable_quality_evaluators": True
        }
        
        mock_quality_evaluators = [Mock(), Mock()]
        
        with patch('src.evaluation.evaluators.sync_reference_free.create_sync_evaluators', 
                   return_value=mock_quality_evaluators):
            
            evaluators = get_phoenix_evaluators(config)
            
            assert len(evaluators) == 2
            assert evaluators == mock_quality_evaluators
    
    @pytest.mark.unit
    def test_get_phoenix_evaluators_both_types(self):
        """Test getting both LLM and quality evaluators."""
        config = {
            "enable_llm_evaluators": True,
            "enable_quality_evaluators": True,
            "evaluator_name": "custom_judge"
        }
        
        mock_visual_judge = Mock()
        mock_quality_evaluators = [Mock(), Mock()]
        
        with patch('src.evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge', 
                   return_value=mock_visual_judge), \
             patch('src.evaluation.evaluators.sync_reference_free.create_sync_evaluators', 
                   return_value=mock_quality_evaluators):
            
            evaluators = get_phoenix_evaluators(config)
            
            assert len(evaluators) == 3
            assert evaluators[0] == mock_visual_judge
            assert evaluators[1:] == mock_quality_evaluators
    
    @pytest.mark.unit
    def test_get_phoenix_evaluators_default_evaluator_config(self):
        """Test getting evaluators with default evaluator configuration."""
        config = {
            "enable_llm_evaluators": True,
            "enable_quality_evaluators": False
            # No evaluator_name or evaluators config
        }
        
        with patch('src.evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge') as mock_judge:
            mock_judge.return_value = Mock()
            
            evaluators = get_phoenix_evaluators(config)
            
            # Should use defaults
            mock_judge.assert_called_once_with(
                provider="ollama",
                model=None,
                base_url=None,
                api_key=None
            )
    
    @pytest.mark.unit  
    def test_get_phoenix_evaluators_custom_evaluator_name(self):
        """Test getting evaluators with custom evaluator name."""
        config = {
            "enable_llm_evaluators": True,
            "evaluator_name": "custom_evaluator",
            "evaluators": {
                "custom_evaluator": {
                    "provider": "openai",
                    "model": "gpt-4"
                }
            }
        }
        
        with patch('src.evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge') as mock_judge:
            mock_judge.return_value = Mock()
            
            evaluators = get_phoenix_evaluators(config)
            
            mock_judge.assert_called_once_with(
                provider="openai",
                model="gpt-4",
                base_url=None,
                api_key=None
            )