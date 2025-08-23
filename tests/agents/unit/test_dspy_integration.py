"""Unit tests for DSPy integration across all agents."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

# DSPy imports
import dspy

# Agent imports
from src.app.agents.routing_agent import RoutingAgent
from src.app.agents.summarizer_agent import SummarizerAgent, SummaryRequest
from src.app.agents.detailed_report_agent import DetailedReportAgent, ReportRequest
from src.app.agents.query_analysis_tool_v3 import QueryAnalysisToolV3
from src.app.agents.dspy_integration_mixin import (
    DSPyIntegrationMixin, DSPyQueryAnalysisMixin, DSPyRoutingMixin,
    DSPySummaryMixin, DSPyDetailedReportMixin
)
from src.app.agents.dspy_agent_optimizer import (
    DSPyAgentPromptOptimizer, DSPyAgentOptimizerPipeline
)


class TestDSPyIntegrationMixin:
    """Test the base DSPy integration mixin."""
    
    def test_mixin_initialization(self):
        """Test mixin initialization without optimized prompts."""
        mixin = DSPyIntegrationMixin()
        
        assert hasattr(mixin, 'dspy_optimized_prompts')
        assert hasattr(mixin, 'dspy_enabled')
        assert hasattr(mixin, 'optimization_cache')
        assert mixin.dspy_optimized_prompts == {}
        assert mixin.dspy_enabled == False
    
    def test_agent_type_detection(self):
        """Test agent type detection from class name."""
        
        class TestRoutingAgent(DSPyIntegrationMixin):
            pass
            
        class TestSummarizerAgent(DSPyIntegrationMixin):
            pass
        
        routing_agent = TestRoutingAgent()
        summarizer_agent = TestSummarizerAgent()
        
        # These should map to appropriate types
        assert routing_agent._get_agent_type() in ['agent_routing', 'query_analysis']
        assert summarizer_agent._get_agent_type() in ['summary_generation', 'query_analysis']
    
    def test_get_optimized_prompt_no_optimization(self):
        """Test prompt retrieval when no optimization is available."""
        mixin = DSPyIntegrationMixin()
        
        default_prompt = "This is a default prompt"
        result = mixin.get_optimized_prompt("test_key", default_prompt)
        
        assert result == default_prompt
    
    def test_get_optimized_prompt_with_optimization(self):
        """Test prompt retrieval with available optimization."""
        mixin = DSPyIntegrationMixin()
        mixin.dspy_enabled = True
        mixin.dspy_optimized_prompts = {
            'compiled_prompts': {
                'test_key': 'This is an optimized prompt'
            }
        }
        
        result = mixin.get_optimized_prompt('test_key', 'default')
        assert result == 'This is an optimized prompt'
        
        # Test fallback to default
        result = mixin.get_optimized_prompt('missing_key', 'default')
        assert result == 'default'
    
    def test_get_dspy_metadata(self):
        """Test DSPy metadata retrieval."""
        mixin = DSPyIntegrationMixin()
        
        # Without optimization
        metadata = mixin.get_dspy_metadata()
        assert metadata['enabled'] == False
        
        # With optimization
        mixin.dspy_enabled = True
        mixin.dspy_optimized_prompts = {
            'metadata': {'test': 'value'},
            'compiled_prompts': {'key1': 'prompt1', 'key2': 'prompt2'}
        }
        
        metadata = mixin.get_dspy_metadata()
        assert metadata['enabled'] == True
        assert metadata['test'] == 'value'
        assert 'agent_type' in metadata
        assert 'prompt_keys' in metadata
        assert len(metadata['prompt_keys']) == 2
    
    def test_apply_dspy_optimization(self):
        """Test DSPy optimization application to prompt templates."""
        mixin = DSPyIntegrationMixin()
        
        template = "Hello {name}, this is a {type} prompt"
        context = {'name': 'Alice', 'type': 'test'}
        
        # Without optimization
        result = mixin.apply_dspy_optimization(template, context)
        assert result == "Hello Alice, this is a test prompt"
        
        # With optimization  
        mixin.dspy_enabled = True
        mixin.dspy_optimized_prompts = {
            'compiled_prompts': {
                'template': 'Optimized hello {name}, {type} prompt'
            }
        }
        
        result = mixin.apply_dspy_optimization(template, context)
        assert result == "Optimized hello Alice, test prompt"
    
    @pytest.mark.asyncio
    async def test_test_dspy_optimization(self):
        """Test the DSPy optimization testing functionality."""
        mixin = DSPyIntegrationMixin()
        
        # Without optimization
        result = await mixin.test_dspy_optimization({'test': 'input'})
        assert 'error' in result
        
        # With optimization
        mixin.dspy_enabled = True
        mixin.dspy_optimized_prompts = {
            'compiled_prompts': {'system': 'optimized system prompt'},
            'metadata': {'test': 'metadata'}
        }
        
        result = await mixin.test_dspy_optimization({'test': 'input'})
        assert result['dspy_enabled'] == True
        assert result['test_completed'] == True
        assert 'prompt_analysis' in result


class TestDSPySpecializedMixins:
    """Test specialized DSPy mixins for different agent types."""
    
    def test_query_analysis_mixin(self):
        """Test DSPy query analysis mixin."""
        mixin = DSPyQueryAnalysisMixin()
        
        prompt = mixin.get_optimized_analysis_prompt(
            "Show me videos of robots", 
            "user context"
        )
        
        assert "Show me videos of robots" in prompt
        assert "user context" in prompt
        assert "intent" in prompt.lower()
        assert "complexity" in prompt.lower()
    
    def test_routing_mixin(self):
        """Test DSPy routing mixin."""
        mixin = DSPyRoutingMixin()
        
        analysis_result = {"intent": "search", "complexity": "simple"}
        available_agents = ["video_search", "text_search"]
        
        prompt = mixin.get_optimized_routing_prompt(
            "Find videos", analysis_result, available_agents
        )
        
        assert "Find videos" in prompt
        assert "search" in prompt
        assert "video_search" in prompt
    
    def test_summary_mixin(self):
        """Test DSPy summary mixin."""
        mixin = DSPySummaryMixin()
        
        prompt = mixin.get_optimized_summary_prompt(
            "Long content to summarize...",
            "brief",
            "executive"
        )
        
        assert "Long content to summarize..." in prompt
        assert "brief" in prompt
        assert "executive" in prompt
    
    def test_detailed_report_mixin(self):
        """Test DSPy detailed report mixin."""
        mixin = DSPyDetailedReportMixin()
        
        search_results = [{"title": "Result 1"}, {"title": "Result 2"}]
        
        prompt = mixin.get_optimized_report_prompt(
            search_results,
            "business analysis",
            "comprehensive"
        )
        
        assert "business analysis" in prompt
        assert "comprehensive" in prompt
        assert "executive summary" in prompt.lower()


class TestDSPyAgentIntegration:
    """Test DSPy integration with actual agent classes."""
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    def test_routing_agent_dspy_integration(self, mock_router):
        """Test DSPy integration in RoutingAgent."""
        # Mock the router to avoid GLiNER loading
        mock_router_instance = Mock()
        mock_router.return_value = mock_router_instance
        
        agent = RoutingAgent()
        
        # Should have DSPy capabilities
        assert hasattr(agent, 'dspy_enabled')
        assert hasattr(agent, 'get_optimized_prompt')
        assert hasattr(agent, 'get_optimized_routing_prompt')
        
        # Test DSPy metadata
        metadata = agent.get_dspy_metadata()
        assert 'enabled' in metadata
        assert metadata['agent_type'] in ['agent_routing', 'query_analysis']
    
    @patch('src.app.agents.summarizer_agent.VLMInterface')
    def test_summarizer_agent_dspy_integration(self, mock_vlm):
        """Test DSPy integration in SummarizerAgent."""
        mock_vlm.return_value = Mock()
        
        agent = SummarizerAgent()
        
        # Should have DSPy capabilities
        assert hasattr(agent, 'dspy_enabled')
        assert hasattr(agent, 'get_optimized_prompt')
        assert hasattr(agent, 'get_optimized_summary_prompt')
        
        # Test DSPy metadata
        metadata = agent.get_dspy_metadata()
        assert 'enabled' in metadata
        assert metadata['agent_type'] in ['summary_generation', 'query_analysis']
    
    @patch('src.app.agents.detailed_report_agent.VLMInterface')
    def test_detailed_report_agent_dspy_integration(self, mock_vlm):
        """Test DSPy integration in DetailedReportAgent."""
        mock_vlm.return_value = Mock()
        
        agent = DetailedReportAgent()
        
        # Should have DSPy capabilities
        assert hasattr(agent, 'dspy_enabled')
        assert hasattr(agent, 'get_optimized_prompt')
        assert hasattr(agent, 'get_optimized_report_prompt')
        
        # Test DSPy metadata
        metadata = agent.get_dspy_metadata()
        assert 'enabled' in metadata
        assert metadata['agent_type'] in ['detailed_report', 'query_analysis']
    
    @patch('src.app.agents.query_analysis_tool_v3.RoutingAgent')
    def test_query_analysis_tool_dspy_integration(self, mock_routing_agent):
        """Test DSPy integration in QueryAnalysisToolV3."""
        mock_routing_agent.return_value = Mock()
        
        tool = QueryAnalysisToolV3(enable_agent_integration=False)
        
        # Should have DSPy capabilities
        assert hasattr(tool, 'dspy_enabled')
        assert hasattr(tool, 'get_optimized_prompt')
        assert hasattr(tool, 'get_optimized_analysis_prompt')
        
        # Test DSPy metadata
        metadata = tool.get_dspy_metadata()
        assert 'enabled' in metadata
        assert metadata['agent_type'] == 'query_analysis'


class TestDSPyAgentOptimizer:
    """Test DSPy agent optimizer and pipeline."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = DSPyAgentPromptOptimizer()
        
        assert optimizer.optimized_prompts == {}
        assert optimizer.lm is None
        assert 'max_bootstrapped_demos' in optimizer.optimization_settings
    
    @patch('dspy.LM')
    def test_language_model_initialization(self, mock_lm_class):
        """Test language model initialization."""
        optimizer = DSPyAgentPromptOptimizer()
        
        mock_lm = Mock()
        mock_lm_class.return_value = mock_lm
        
        result = optimizer.initialize_language_model(
            api_base="http://localhost:11434/v1",
            model="smollm3:8b"
        )
        
        assert result == True
        assert optimizer.lm == mock_lm
        mock_lm_class.assert_called_once()
    
    @patch('dspy.LM')
    def test_language_model_initialization_failure(self, mock_lm_class):
        """Test language model initialization failure."""
        optimizer = DSPyAgentPromptOptimizer()
        
        mock_lm_class.side_effect = Exception("Connection failed")
        
        result = optimizer.initialize_language_model()
        assert result == False
        assert optimizer.lm is None
    
    def test_signature_creation(self):
        """Test DSPy signature creation for different tasks."""
        optimizer = DSPyAgentPromptOptimizer()
        
        # Test query analysis signature
        qa_signature = optimizer.create_query_analysis_signature()
        assert hasattr(qa_signature, '__name__')
        
        # Test agent routing signature
        ar_signature = optimizer.create_agent_routing_signature()
        assert hasattr(ar_signature, '__name__')
        
        # Test summary generation signature
        sg_signature = optimizer.create_summary_generation_signature()
        assert hasattr(sg_signature, '__name__')
        
        # Test detailed report signature
        dr_signature = optimizer.create_detailed_report_signature()
        assert hasattr(dr_signature, '__name__')
    
    def test_pipeline_initialization(self):
        """Test DSPy optimization pipeline initialization."""
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)
        
        assert pipeline.optimizer == optimizer
        assert pipeline.modules == {}
        assert pipeline.compiled_modules == {}
        assert pipeline.training_data == {}
    
    def test_pipeline_module_initialization(self):
        """Test pipeline module initialization."""
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)
        
        pipeline.initialize_modules()
        
        expected_modules = ['query_analysis', 'agent_routing', 'summary_generation', 'detailed_report']
        for module_name in expected_modules:
            assert module_name in pipeline.modules
            assert pipeline.modules[module_name] is not None
    
    def test_training_data_loading(self):
        """Test training data loading."""
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)
        
        training_data = pipeline.load_training_data()
        
        expected_data_types = ['query_analysis', 'agent_routing', 'summary_generation', 'detailed_report']
        for data_type in expected_data_types:
            assert data_type in training_data
            assert len(training_data[data_type]) > 0
            
            # Check first example has required fields
            example = training_data[data_type][0]
            assert hasattr(example, '__dict__') or isinstance(example, dict)
    
    def test_metric_creation(self):
        """Test evaluation metric creation."""
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)
        
        # Test different metric types
        qa_metric = pipeline._create_metric_for_module('query_analysis')
        ar_metric = pipeline._create_metric_for_module('agent_routing')
        sg_metric = pipeline._create_metric_for_module('summary_generation')
        dr_metric = pipeline._create_metric_for_module('detailed_report')
        
        # All should be callable functions
        assert callable(qa_metric)
        assert callable(ar_metric)
        assert callable(sg_metric)
        assert callable(dr_metric)
    
    def test_prompt_extraction(self):
        """Test prompt extraction from DSPy modules."""
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)
        
        # Create a mock module
        mock_module = Mock()
        mock_module.generate_analysis = Mock()
        mock_module.generate_analysis.signature = "Mock signature"
        
        prompts = pipeline._extract_prompts_from_module(mock_module)
        
        assert 'module_type' in prompts
        assert 'compiled_prompts' in prompts
        assert 'metadata' in prompts


class TestDSPyPromptOptimization:
    """Test actual DSPy prompt optimization with mocked components."""
    
    def test_optimized_prompt_loading_from_file(self):
        """Test loading optimized prompts from file."""
        optimized_prompts = {
            'compiled_prompts': {
                'system': 'Optimized system prompt',
                'signature': 'Optimized signature'
            },
            'metadata': {
                'optimization_timestamp': 1234567890,
                'dspy_version': '3.0.2'
            }
        }
        
        # Test with query analysis mixin (should detect as query_analysis type)
        class TestAgent(DSPyQueryAnalysisMixin):
            def _get_agent_type(self):
                return 'test_agent'
        
        # Mock the file system operations
        with patch.object(Path, 'exists') as mock_exists:
            with patch('builtins.open', mock_open(read_data=json.dumps(optimized_prompts))):
                mock_exists.return_value = True
                
                agent = TestAgent()
                
                assert agent.dspy_enabled == True
                assert agent.dspy_optimized_prompts == optimized_prompts
                
                # Test prompt retrieval
                prompt = agent.get_optimized_prompt('system', 'default')
                assert prompt == 'Optimized system prompt'
    
    @pytest.mark.asyncio
    async def test_optimization_integration_test(self):
        """Test integration between optimization and agent usage."""
        # Create mock optimized prompts
        optimized_prompts = {
            'compiled_prompts': {
                'analysis': 'Optimized analysis: {query} with {context}'
            },
            'metadata': {'test': True}
        }
        
        class TestAnalysisAgent(DSPyQueryAnalysisMixin):
            def __init__(self):
                super().__init__()
                self.dspy_enabled = True
                self.dspy_optimized_prompts = optimized_prompts
        
        agent = TestAnalysisAgent()
        
        # Test optimization test
        result = await agent.test_dspy_optimization({'test': 'input'})
        
        assert result['dspy_enabled'] == True
        assert result['test_completed'] == True
        assert 'prompt_analysis' in result
        
        # Test prompt application
        optimized = agent.apply_dspy_optimization(
            "Analysis: {query} with {context}",
            {'query': 'test query', 'context': 'test context'}
        )
        
        # Should use fallback since no 'template' key exists
        assert optimized == "Analysis: test query with test context"


def mock_open_for_path(file_path):
    """Helper to mock open for specific file path."""
    original_open = open
    
    def mock_open(*args, **kwargs):
        if args[0] == file_path or str(args[0]) == file_path:
            with original_open(file_path, 'r') as f:
                content = f.read()
            from io import StringIO
            return StringIO(content)
        return original_open(*args, **kwargs)
    
    return mock_open


@pytest.mark.integration
class TestDSPyEndToEndIntegration:
    """End-to-end integration tests for DSPy optimization."""
    
    @pytest.mark.asyncio
    @patch('dspy.LM')
    @patch('dspy.teleprompt.BootstrapFewShot')
    async def test_full_optimization_pipeline(self, mock_teleprompter, mock_lm_class):
        """Test the complete optimization pipeline."""
        # Mock DSPy components
        mock_lm = Mock()
        mock_lm_class.return_value = mock_lm
        
        mock_compiled_module = Mock()
        mock_teleprompter_instance = Mock()
        mock_teleprompter_instance.compile.return_value = mock_compiled_module
        mock_teleprompter.return_value = mock_teleprompter_instance
        
        # Create optimizer
        optimizer = DSPyAgentPromptOptimizer()
        assert optimizer.initialize_language_model() == True
        
        # Create pipeline
        pipeline = DSPyAgentOptimizerPipeline(optimizer)
        
        # Run optimization (mocked)
        optimized_modules = await pipeline.optimize_all_modules()
        
        # Should have optimized all modules
        expected_modules = ['query_analysis', 'agent_routing', 'summary_generation', 'detailed_report']
        for module_name in expected_modules:
            assert module_name in optimized_modules
    
    @pytest.mark.asyncio
    async def test_agent_integration_with_mocked_optimization(self):
        """Test agent integration with mocked optimization results."""
        
        # Mock optimized prompts for different agent types
        mock_prompts = {
            'query_analysis': {
                'compiled_prompts': {'system': 'Optimized query analysis prompt'},
                'metadata': {'type': 'query_analysis'}
            },
            'agent_routing': {
                'compiled_prompts': {'routing': 'Optimized routing prompt'},
                'metadata': {'type': 'agent_routing'}
            },
            'summary_generation': {
                'compiled_prompts': {'summary': 'Optimized summary prompt'},
                'metadata': {'type': 'summary_generation'}
            },
            'detailed_report': {
                'compiled_prompts': {'report': 'Optimized report prompt'},
                'metadata': {'type': 'detailed_report'}
            }
        }
        
        # Test QueryAnalysisToolV3 with direct prompt setting
        with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent'):
            tool = QueryAnalysisToolV3(enable_agent_integration=False)
            
            # Manually set the optimization data
            tool.dspy_optimized_prompts = mock_prompts['query_analysis']
            tool.dspy_enabled = True
            
            metadata = tool.get_dspy_metadata()
            assert metadata['enabled'] == True
            assert tool.get_optimized_prompt('system', 'default') == 'Optimized query analysis prompt'