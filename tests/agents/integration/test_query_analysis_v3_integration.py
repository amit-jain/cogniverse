"""Integration tests for QueryAnalysisToolV3 (Phase 5) with Ollama models."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.app.agents.query_analysis_tool_v3 import (
    QueryAnalysisToolV3, QueryIntent, QueryComplexity, QueryContext,
    create_enhanced_query_analyzer
)
from src.app.agents.routing_agent import RoutingAgent


@pytest.fixture
def ollama_config():
    """Configuration for Ollama models for query analysis"""
    return {
        "model_provider": "ollama",
        "base_url": "http://localhost:11434",
        "models": {
            "query_analysis": "smollm3:8b",
            "intent_detection": "qwen:7b",
            "reasoning": "qwen:7b"
        },
        "timeout": 30
    }


@pytest.fixture
def mock_routing_agent():
    """Mock routing agent for integration testing"""
    agent = Mock(spec=RoutingAgent)
    agent.analyze_and_route = AsyncMock(return_value={
        "workflow": {
            "type": "detailed_report",
            "steps": [
                {"step": 1, "agent": "video_search", "action": "search"},
                {"step": 2, "agent": "detailed_report", "action": "analyze"}
            ],
            "agents": ["video_search", "detailed_report"]
        }
    })
    return agent


@pytest.fixture 
def sample_conversation_context():
    """Sample conversation context for testing"""
    return QueryContext(
        conversation_history=[
            "Tell me about machine learning",
            "What are neural networks?",
            "Show me recent AI research"
        ],
        user_preferences={
            "preferred_format": "video",
            "analysis_depth": "comprehensive",
            "include_visual": True
        },
        previous_results=[
            {"id": "prev_1", "title": "ML Basics", "score": 0.9},
            {"id": "prev_2", "title": "AI Overview", "score": 0.85}
        ],
        session_metadata={
            "session_id": "test_session",
            "user_id": "test_user",
            "timestamp": "2024-01-20T10:00:00Z"
        }
    )


class MockOllamaQueryClient:
    """Mock Ollama client specifically for query analysis"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def chat(self, model: str, messages: list, **kwargs):
        """Mock chat completion for query analysis"""
        user_message = messages[-1]["content"] if messages else ""
        
        # Simulate different analysis based on query content
        if "analyze" in user_message.lower() or "detailed" in user_message.lower():
            return {
                "message": {
                    "content": "Intent: ANALYZE, Complexity: COMPLEX, Confidence: 0.9, Reasoning: Query requires deep analytical processing with detailed examination of multiple factors."
                }
            }
        elif "summarize" in user_message.lower() or "overview" in user_message.lower():
            return {
                "message": {
                    "content": "Intent: SUMMARIZE, Complexity: MODERATE, Confidence: 0.85, Reasoning: Query seeks summary information with moderate processing requirements."
                }
            }
        elif "compare" in user_message.lower() or "difference" in user_message.lower():
            return {
                "message": {
                    "content": "Intent: COMPARE, Complexity: MODERATE, Confidence: 0.8, Reasoning: Comparative analysis requested requiring evaluation of multiple items."
                }
            }
        else:
            return {
                "message": {
                    "content": "Intent: SEARCH, Complexity: SIMPLE, Confidence: 0.75, Reasoning: Basic search query with straightforward information retrieval needs."
                }
            }


@pytest.fixture
def mock_ollama_query_client():
    """Mock Ollama client fixture for query analysis"""
    return MockOllamaQueryClient("http://localhost:11434")


class TestQueryAnalysisV3OllamaIntegration:
    """Integration tests for QueryAnalysisToolV3 with Ollama models"""
    
    @pytest.mark.asyncio
    async def test_query_analysis_with_smollm3(self, ollama_config, mock_ollama_query_client):
        """Test query analysis with SmolLM3 model"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = {
                **ollama_config,
                "query_analysis_model": "smollm3:8b"
            }
            
            # Mock routing agent initialization to avoid external dependencies
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3(
                    enable_thinking_phase=True,
                    enable_query_expansion=True,
                    enable_agent_integration=False  # Disable for this test
                )
                
                # Test simple search query
                result = await analyzer.analyze("find videos about machine learning")
                
                # Verify basic analysis
                assert result.original_query == "find videos about machine learning"
                assert result.cleaned_query == "find videos about machine learning"
                assert result.primary_intent == QueryIntent.SEARCH
                assert result.complexity_level == QueryComplexity.SIMPLE
                assert result.needs_video_search is True
                assert result.confidence_score > 0.5
                assert len(result.keywords) > 0
                assert "machine" in result.keywords
                assert "learning" in result.keywords
    
    @pytest.mark.asyncio
    async def test_complex_query_analysis_with_qwen(self, ollama_config, mock_ollama_query_client, sample_conversation_context):
        """Test complex query analysis with Qwen model"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = {
                **ollama_config,
                "query_analysis_model": "qwen:7b"
            }
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3(
                    enable_thinking_phase=True,
                    enable_query_expansion=True,
                    enable_agent_integration=False
                )
                
                # Test complex analytical query
                complex_query = "analyze recent developments in artificial intelligence and create a comprehensive detailed report with visual analysis"
                
                result = await analyzer.analyze(complex_query, sample_conversation_context)
                
                # Verify complex analysis
                assert result.primary_intent in [QueryIntent.ANALYZE, QueryIntent.REPORT]
                assert result.complexity_level == QueryComplexity.COMPLEX
                assert QueryIntent.TEMPORAL in result.secondary_intents
                assert result.needs_visual_analysis is True
                assert result.recommended_workflow == "detailed_report"
                assert "detailed_report" in result.required_agents
                assert len(result.thinking_phase.keys()) > 5
                assert result.thinking_phase["has_context"] is True
                assert len(result.thinking_phase["complexity_signals"]) >= 2
    
    @pytest.mark.asyncio
    async def test_thinking_phase_with_ollama_reasoning(self, ollama_config, mock_ollama_query_client):
        """Test thinking phase with Ollama-powered reasoning"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = {
                **ollama_config,
                "reasoning_model": "qwen:7b"
            }
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3(enable_thinking_phase=True)
                
                # Test thinking phase for moderate complexity query
                query = "compare different machine learning frameworks and explain their advantages"
                
                result = await analyzer.analyze(query)
                
                # Verify thinking phase results
                thinking = result.thinking_phase
                assert thinking["query_length"] > 5
                assert len(thinking["complexity_signals"]) > 0
                assert "compare" in thinking["complexity_signals"]
                assert thinking["query_type_indicators"]["summary"] is False  # Not a summary
                assert thinking["reasoning"] is not None
                assert len(thinking["reasoning"]) > 50
                assert "complexity signals" in thinking["reasoning"].lower() or "comparative" in thinking["reasoning"].lower()
    
    @pytest.mark.asyncio
    async def test_query_expansion_with_context(self, ollama_config, sample_conversation_context):
        """Test query expansion using conversation context"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = {
                **ollama_config,
                "expansion_model": "smollm3:8b"
            }
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3(
                    enable_query_expansion=True,
                    max_expanded_queries=5
                )
                
                # Test query that can benefit from context
                query = "latest developments"
                
                result = await analyzer.analyze(query, sample_conversation_context)
                
                # Verify expansion worked
                assert len(result.expanded_queries) > 0
                # Should include context from conversation history
                expanded_text = " ".join(result.expanded_queries)
                assert any(term in expanded_text.lower() for term in ["machine", "neural", "ai"])
    
    @pytest.mark.asyncio
    async def test_routing_agent_integration_with_ollama(self, ollama_config, mock_routing_agent):
        """Test integration with routing agent using Ollama"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = mock_routing_agent
                
                analyzer = QueryAnalysisToolV3(enable_agent_integration=True)
                
                # Test workflow determination with routing agent
                query = "create a detailed analysis report of AI ethics research"
                
                result = await analyzer.analyze(query)
                
                # Verify routing agent integration
                assert result.recommended_workflow == "detailed_report"
                assert "video_search" in result.required_agents
                assert "detailed_report" in result.required_agents
                assert len(result.workflow_steps) == 2
                
                # Verify routing agent was called
                mock_routing_agent.analyze_and_route.assert_called_once()
                call_args = mock_routing_agent.analyze_and_route.call_args
                assert call_args[0][0] == query  # Query was passed
                assert "thinking_phase" in call_args[1]["context"]  # Context included thinking phase
    
    @pytest.mark.asyncio
    async def test_multimodal_query_detection(self, ollama_config):
        """Test detection of multimodal queries"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3()
                
                # Test multimodal query
                multimodal_query = "show me videos and images about neural networks with detailed visual analysis"
                
                result = await analyzer.analyze(multimodal_query)
                
                # Verify multimodal detection
                assert QueryIntent.MULTIMODAL in [result.primary_intent] + result.secondary_intents or \
                       QueryIntent.VISUAL in [result.primary_intent] + result.secondary_intents
                assert result.needs_video_search is True
                assert result.needs_visual_analysis is True
                assert result.thinking_phase["modality_hints"]["video"] is True
    
    @pytest.mark.asyncio
    async def test_temporal_query_analysis(self, ollama_config):
        """Test analysis of temporal queries"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3()
                
                # Test temporal query
                temporal_query = "show me recent AI research from last week"
                
                result = await analyzer.analyze(temporal_query)
                
                # Verify temporal analysis
                assert QueryIntent.TEMPORAL in result.secondary_intents
                assert len(result.temporal_filters) > 0
                assert "start_date" in result.temporal_filters
                assert result.temporal_filters["temporal_term"] in ["recent", "last week"]
                assert result.thinking_phase["temporal_indicators"]
    
    @pytest.mark.asyncio
    async def test_entity_and_keyword_extraction(self, ollama_config):
        """Test entity and keyword extraction capabilities"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3()
                
                # Test query with entities and keywords
                entity_rich_query = 'Find research about "deep learning" and TensorFlow by Google researchers'
                
                result = await analyzer.analyze(entity_rich_query)
                
                # Verify entity extraction
                quoted_entities = [e for e in result.entities if e["type"] == "quoted_phrase"]
                assert len(quoted_entities) >= 1
                assert any(e["text"] == "deep learning" for e in quoted_entities)
                
                proper_noun_entities = [e for e in result.entities if e["type"] == "proper_noun"]
                assert any(e["text"] in ["TensorFlow", "Google"] for e in proper_noun_entities)
                
                # Verify keyword extraction
                assert "research" in result.keywords
                assert "researchers" in result.keywords
                # Stop words should be filtered out
                assert "about" not in result.keywords
                assert "by" not in result.keywords
    
    @pytest.mark.asyncio
    async def test_confidence_scoring_integration(self, ollama_config):
        """Test confidence scoring across different query types"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3()
                
                # Test high-confidence query (clear intent, simple, with entities)
                high_conf_query = 'search for "machine learning" tutorials'
                high_result = await analyzer.analyze(high_conf_query)
                
                # Test low-confidence query (ambiguous, complex, no entities)
                low_conf_query = "find some stuff about things"
                low_result = await analyzer.analyze(low_conf_query)
                
                # Verify confidence differences
                assert high_result.confidence_score > low_result.confidence_score
                assert high_result.confidence_score > 0.7  # Should be high confidence
                assert low_result.confidence_score < 0.6   # Should be lower confidence


class TestQueryAnalysisV3WorkflowIntegration:
    """Integration tests for complete workflow scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_simple_search_workflow(self, ollama_config):
        """Test complete workflow for simple search query"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = create_enhanced_query_analyzer(
                    enable_thinking_phase=True,
                    enable_query_expansion=True,
                    enable_agent_integration=False
                )
                
                # Simulate simple user query
                query = "show me cats"
                
                result = await analyzer.analyze(query)
                
                # Verify complete workflow
                assert result.primary_intent == QueryIntent.SEARCH
                assert result.complexity_level == QueryComplexity.SIMPLE
                assert result.recommended_workflow == "raw_results"
                assert "video_search" in result.required_agents
                assert len(result.workflow_steps) >= 1
                assert result.workflow_steps[0]["agent"] == "video_search"
                assert result.thinking_phase["reasoning"] is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytical_workflow(self, ollama_config, sample_conversation_context):
        """Test complete workflow for analytical query"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = create_enhanced_query_analyzer()
                
                # Simulate complex analytical query
                query = "analyze the evolution of artificial intelligence research over the past year and provide comprehensive insights with visual analysis"
                
                result = await analyzer.analyze(query, sample_conversation_context)
                
                # Verify analytical workflow
                assert result.primary_intent in [QueryIntent.ANALYZE, QueryIntent.REPORT]
                assert result.complexity_level == QueryComplexity.COMPLEX
                assert result.recommended_workflow == "detailed_report"
                assert "detailed_report" in result.required_agents
                assert result.needs_visual_analysis is True
                assert len(result.workflow_steps) >= 2
                assert any(step["agent"] == "detailed_report" for step in result.workflow_steps)
                
                # Verify temporal analysis
                assert len(result.temporal_filters) > 0
                assert QueryIntent.TEMPORAL in result.secondary_intents
    
    @pytest.mark.asyncio
    async def test_statistics_and_monitoring_integration(self, ollama_config):
        """Test statistics collection and monitoring"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = create_enhanced_query_analyzer()
                
                # Perform multiple analyses
                queries = [
                    "simple search",
                    "compare two things", 
                    "analyze complex data and create detailed report"
                ]
                
                results = []
                for query in queries:
                    result = await analyzer.analyze(query)
                    results.append(result)
                
                # Verify statistics
                stats = analyzer.get_statistics()
                assert stats["total_analyses"] == 3
                assert stats["uptime_seconds"] > 0
                assert stats["analyses_per_minute"] >= 0
                assert stats["configuration"]["thinking_phase_enabled"] is True
                assert stats["configuration"]["query_expansion_enabled"] is True
                
                # Verify analysis progression
                assert analyzer.total_analyses == 3
                for i, result in enumerate(results):
                    assert result.analysis_time_ms > 0
                    assert result.routing_method == "enhanced_v3"


class TestQueryAnalysisV3ErrorHandlingIntegration:
    """Integration tests for error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_routing_agent_failure_handling(self, ollama_config):
        """Test handling of routing agent failures"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            # Mock routing agent that fails
            mock_failing_agent = Mock()
            mock_failing_agent.analyze_and_route = AsyncMock(side_effect=Exception("Routing failed"))
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = mock_failing_agent
                
                analyzer = QueryAnalysisToolV3(enable_agent_integration=True)
                
                # Should still complete analysis with fallback workflow
                result = await analyzer.analyze("test query with routing failure")
                
                # Verify fallback behavior
                assert result is not None
                assert result.recommended_workflow in ["raw_results", "summary", "detailed_report"]
                assert len(result.workflow_steps) > 0
    
    @pytest.mark.asyncio
    async def test_thinking_phase_error_handling(self, ollama_config):
        """Test handling of thinking phase errors"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3(enable_thinking_phase=True)
                
                # Mock thinking phase to raise exception
                with patch.object(analyzer, '_thinking_phase', side_effect=Exception("Thinking failed")):
                    # Should still fail gracefully
                    with pytest.raises(Exception) as exc_info:
                        await analyzer.analyze("test thinking error")
                    
                    assert "Thinking failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_empty_and_edge_case_queries(self, ollama_config):
        """Test handling of empty and edge case queries"""
        with patch('src.app.agents.query_analysis_tool_v3.get_config') as mock_config:
            mock_config.return_value = ollama_config
            
            with patch('src.app.agents.query_analysis_tool_v3.RoutingAgent') as mock_routing_class:
                mock_routing_class.return_value = None
                
                analyzer = QueryAnalysisToolV3()
                
                # Test empty query
                empty_result = await analyzer.analyze("")
                assert empty_result.original_query == ""
                assert empty_result.primary_intent == QueryIntent.SEARCH  # Default
                assert empty_result.complexity_level == QueryComplexity.SIMPLE
                
                # Test very long query
                long_query = "analyze " + " ".join(["word"] * 100) + " detailed report"
                long_result = await analyzer.analyze(long_query)
                assert long_result.thinking_phase["query_length"] > 50
                assert "Long query" in long_result.thinking_phase["reasoning"]
                
                # Test special characters
                special_result = await analyzer.analyze("find @#$% videos!!! ???")
                assert special_result.primary_intent == QueryIntent.SEARCH
                assert len(special_result.keywords) >= 1  # Should extract "videos"


# Integration test configuration for Ollama
"""
To run these integration tests with real Ollama:

1. Install and start Ollama:
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve

2. Pull required models:
   ollama pull smollm3:8b
   ollama pull qwen:7b

3. Set environment variables:
   export OLLAMA_BASE_URL=http://localhost:11434
   export OLLAMA_QUERY_MODEL=smollm3:8b
   export OLLAMA_REASONING_MODEL=qwen:7b

4. Run integration tests:
   pytest tests/agents/integration/test_query_analysis_v3_integration.py -v

5. For CI/CD, use conditional skipping:
   @pytest.mark.skipif(not os.getenv('OLLAMA_BASE_URL'), reason="Ollama not configured")
"""