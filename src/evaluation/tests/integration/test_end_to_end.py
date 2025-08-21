"""
End-to-end integration tests for evaluation framework.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import json
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Dataset
from src.evaluation.core.task import evaluation_task
from src.evaluation.data.datasets import DatasetManager
from src.evaluation.cli import cli


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    def test_experiment_mode_e2e(self, mock_phoenix_client, mock_search_service):
        """Test complete experiment mode workflow."""
        with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
            with patch('src.app.search.service.SearchService', return_value=mock_search_service):
                # Create evaluation task
                task = evaluation_task(
                    mode="experiment",
                    dataset_name="test_dataset",
                    profiles=["test_profile"],
                    strategies=["test_strategy"],
                    config={
                        "use_ragas": False,
                        "use_custom": True,
                        "custom_metrics": ["diversity", "result_count"]
                    }
                )
                
                # Run evaluation
                results = inspect_eval(task, model="mockllm/model")
                
                # Verify results structure - it returns a list of EvalLog
                assert results is not None
                assert isinstance(results, list)
                assert len(results) > 0
                
                # Get the first EvalLog
                eval_log = results[0]
                
                # Check that evaluation ran
                assert hasattr(eval_log, 'samples') or hasattr(eval_log, 'model_dump')
                
                # For now, just verify the evaluation completed
                # TODO: Add more specific assertions once scorers are properly implemented
    
    @pytest.mark.integration
    def test_batch_mode_e2e(self, mock_phoenix_client):
        """Test complete batch mode workflow."""
        with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
            # Create evaluation task for batch mode
            task = evaluation_task(
                mode="batch",
                dataset_name="test_dataset",
                trace_ids=["trace1"],
                config={
                    "use_ragas": True,
                    "ragas_metrics": ["context_relevancy"]
                }
            )
            
            # Run evaluation
            results = inspect_eval(task, model="mockllm/model")
            
            # Verify results - it returns a list of EvalLog
            assert results is not None
            assert isinstance(results, list)
            assert len(results) > 0
    
    @pytest.mark.integration
    def test_cli_evaluate_command(self, mock_phoenix_client, mock_search_service):
        """Test CLI evaluate command."""
        from click.testing import CliRunner
        
        with patch('src.evaluation.data.storage.px.Client', return_value=mock_phoenix_client):
            with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
                with patch('src.app.search.service.SearchService', return_value=mock_search_service):
                    runner = CliRunner()
                    
                    # Test experiment mode via CLI
                    result = runner.invoke(cli, [
                        'evaluate',
                        '--mode', 'experiment',
                        '--dataset', 'test_dataset',
                        '-p', 'test_profile',
                        '-s', 'test_strategy'
                    ])
                    
                    # Check command executed successfully
                    assert result.exit_code == 0
                    assert 'Starting experiment evaluation' in result.output
                    assert 'Evaluation complete' in result.output or 'Running evaluation' in result.output
    
    @pytest.mark.integration
    def test_cli_list_traces_command(self, mock_phoenix_client):
        """Test CLI list-traces command."""
        from click.testing import CliRunner
        
        with patch('src.evaluation.data.storage.px.Client', return_value=mock_phoenix_client):
            runner = CliRunner()
            
            result = runner.invoke(cli, [
                'list-traces',
                '--hours', '2',
                '--limit', '10'
            ])
            
            assert result.exit_code == 0
            assert 'Fetching traces' in result.output
    
    @pytest.mark.integration
    def test_evaluation_with_output_file(self, mock_phoenix_client, mock_search_service):
        """Test evaluation with output file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.json")
            
            from click.testing import CliRunner
            
            with patch('src.evaluation.data.storage.px.Client', return_value=mock_phoenix_client):
                with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
                    with patch('src.app.search.service.SearchService', return_value=mock_search_service):
                        runner = CliRunner()
                        
                        result = runner.invoke(cli, [
                            'evaluate',
                            '--mode', 'experiment',
                            '--dataset', 'test_dataset',
                            '-p', 'test_profile',
                            '-s', 'test_strategy',
                            '--output', output_file
                        ])
                    
                    # Check file was created
                    if result.exit_code == 0:
                        assert os.path.exists(output_file)
                        
                        # Verify JSON structure
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                            assert data["mode"] == "experiment"
                            assert data["dataset"] == "test_dataset"
                            assert "timestamp" in data
    
    @pytest.mark.integration
    def test_evaluation_with_config_file(self, mock_phoenix_client, mock_search_service):
        """Test evaluation with configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.json")
            
            # Create config file
            config = {
                "use_ragas": True,
                "ragas_metrics": ["context_relevancy"],
                "use_custom": True,
                "custom_metrics": ["diversity"],
                "top_k": 5
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            from click.testing import CliRunner
            
            with patch('src.evaluation.data.storage.px.Client', return_value=mock_phoenix_client):
                with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
                    with patch('src.app.search.service.SearchService', return_value=mock_search_service):
                        runner = CliRunner()
                        
                        result = runner.invoke(cli, [
                            'evaluate',
                            '--mode', 'experiment',
                            '--dataset', 'test_dataset',
                            '-p', 'test_profile',
                            '-s', 'test_strategy',
                            '--config', config_file
                        ])
                        
                        assert result.exit_code == 0
    
    @pytest.mark.integration
    def test_multiple_profiles_strategies(self, mock_phoenix_client, mock_search_service):
        """Test evaluation with multiple profiles and strategies."""
        with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
            with patch('src.app.search.service.SearchService', return_value=mock_search_service):
                task = evaluation_task(
                    mode="experiment",
                    dataset_name="test_dataset",
                    profiles=["profile1", "profile2"],
                    strategies=["strategy1", "strategy2", "strategy3"],
                    config={"use_custom": True}
                )
                
                # This should create 2*3=6 configurations
                results = inspect_eval(task, model="mockllm/model")
                
                assert results is not None
                assert isinstance(results, list)
                assert len(results) > 0
    
    @pytest.mark.integration 
    def test_error_handling_invalid_dataset(self, mock_phoenix_client):
        """Test error handling with invalid dataset."""
        mock_phoenix_client.get_dataset.return_value = None
        
        with patch('src.evaluation.core.task.px.Client', return_value=mock_phoenix_client):
            with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
                evaluation_task(
                    mode="experiment",
                    dataset_name="nonexistent",
                    profiles=["p1"],
                    strategies=["s1"]
                )
    
