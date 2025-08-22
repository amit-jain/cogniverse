"""
Unit tests for RoutingAgent
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.app.agents.routing_agent import RoutingAgent, RoutingTask
from src.app.routing.base import RoutingDecision, SearchModality, GenerationType
from src.tools.a2a_utils import A2AMessage, DataPart, TextPart


class TestRoutingAgent:
    """Test cases for RoutingAgent class"""
    
    @pytest.fixture
    def mock_system_config(self):
        """Mock system configuration"""
        return {
            "video_agent_url": "http://localhost:8002",
            "text_agent_url": "http://localhost:8003"
        }
    
    @pytest.fixture
    def mock_routing_decision(self):
        """Mock routing decision"""
        return RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.85,
            routing_method="gliner",
            reasoning="Detected video content request"
        )
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    @patch('src.app.agents.routing_agent.get_config')
    def test_routing_agent_initialization(self, mock_get_config, mock_router_class, mock_system_config):
        """Test RoutingAgent initialization"""
        mock_get_config.return_value = mock_system_config
        mock_router_instance = Mock()
        mock_router_class.return_value = mock_router_instance
        
        agent = RoutingAgent()
        
        assert agent.system_config == mock_system_config
        assert agent.agent_registry["video_search"] == "http://localhost:8002"
        assert agent.agent_registry["text_search"] == "http://localhost:8003"
        assert agent.router == mock_router_instance
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_routing_agent_initialization_missing_video_agent(self, mock_get_config):
        """Test RoutingAgent initialization fails when video agent URL missing"""
        mock_get_config.return_value = {"text_agent_url": "http://localhost:8003"}
        
        with pytest.raises(ValueError, match="video_agent_url not configured"):
            RoutingAgent()
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    @patch('src.app.agents.routing_agent.get_config')
    def test_build_routing_config(self, mock_get_config, mock_router_class, mock_system_config):
        """Test routing configuration building"""
        mock_get_config.return_value = mock_system_config
        mock_router_class.return_value = Mock()
        
        agent = RoutingAgent()
        config = agent.routing_config
        
        # Verify configuration structure
        assert hasattr(config, 'routing_mode') or 'routing_mode' in config
        
        # Check if it's a RoutingConfig object or dict
        if hasattr(config, 'routing_mode'):
            assert config.routing_mode == "tiered"
        else:
            assert config["routing_mode"] == "tiered"
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    @patch('src.app.agents.routing_agent.get_config')
    @pytest.mark.asyncio
    async def test_analyze_and_route_video_query(self, mock_get_config, mock_router_class, mock_system_config, mock_routing_decision):
        """Test query analysis and routing for video queries"""
        mock_get_config.return_value = mock_system_config
        mock_router_instance = Mock()
        mock_router_instance.route = AsyncMock(return_value=mock_routing_decision)
        mock_router_class.return_value = mock_router_instance
        
        agent = RoutingAgent()
        
        # Use a simple query that won't be classified as detailed report
        result = await agent.analyze_and_route("find videos")
        
        assert result["query"] == "find videos"
        assert result["workflow_type"] == "raw_results"
        assert "video_search" in result["agents_to_call"]
        assert result["confidence"] == 0.85
        assert result["routing_method"] == "gliner"
        assert len(result["execution_plan"]) > 0
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    @patch('src.app.agents.routing_agent.get_config')
    @pytest.mark.asyncio
    async def test_analyze_and_route_summary_query(self, mock_get_config, mock_router_class, mock_system_config, mock_routing_decision):
        """Test query analysis for summary requests"""
        mock_get_config.return_value = mock_system_config
        mock_router_instance = Mock()
        mock_router_instance.route = AsyncMock(return_value=mock_routing_decision)
        mock_router_class.return_value = mock_router_instance
        
        agent = RoutingAgent()
        
        result = await agent.analyze_and_route("Summarize the latest videos about AI")
        
        assert result["workflow_type"] == "summary"
        assert "video_search" in result["agents_to_call"]
        # Note: summarizer won't be in agents_to_call since it's not available yet
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    @patch('src.app.agents.routing_agent.get_config') 
    @pytest.mark.asyncio
    async def test_analyze_and_route_detailed_report_query(self, mock_get_config, mock_router_class, mock_system_config, mock_routing_decision):
        """Test query analysis for detailed report requests"""
        mock_get_config.return_value = mock_system_config
        mock_router_instance = Mock()
        mock_router_instance.route = AsyncMock(return_value=mock_routing_decision)
        mock_router_class.return_value = mock_router_instance
        
        agent = RoutingAgent()
        
        result = await agent.analyze_and_route("Provide detailed analysis of machine learning trends")
        
        assert result["workflow_type"] == "detailed_report"
        assert "video_search" in result["agents_to_call"]
        # Note: detailed_report agent won't be in agents_to_call since it's not available yet
    
    @patch('src.app.agents.routing_agent.ComprehensiveRouter')
    @patch('src.app.agents.routing_agent.get_config')
    def test_determine_workflow_raw_results(self, mock_get_config, mock_router_class, mock_system_config, mock_routing_decision):
        """Test workflow determination for raw results"""
        mock_get_config.return_value = mock_system_config
        mock_router_class.return_value = Mock()
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("find videos", mock_routing_decision)
        
        assert workflow["type"] == "raw_results"
        assert len(workflow["steps"]) == 1  # Just search step
        assert workflow["steps"][0]["agent"] == "video_search"
        assert workflow["steps"][0]["action"] == "search"
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_determine_workflow_summary(self, mock_get_config, mock_system_config, mock_routing_decision):
        """Test workflow determination for summary requests"""
        mock_get_config.return_value = mock_system_config
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("summarize videos", mock_routing_decision)
        
        assert workflow["type"] == "summary"
        assert len(workflow["steps"]) == 1  # Search step only (summarizer not available)
        assert workflow["steps"][0]["agent"] == "video_search"
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_determine_workflow_detailed_report(self, mock_get_config, mock_system_config, mock_routing_decision):
        """Test workflow determination for detailed report requests"""
        mock_get_config.return_value = mock_system_config
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("detailed analysis of videos", mock_routing_decision)
        
        assert workflow["type"] == "detailed_report"
        assert len(workflow["steps"]) == 1  # Search step only (detailed_report agent not available)
        assert workflow["steps"][0]["agent"] == "video_search"
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_determine_workflow_both_modalities(self, mock_get_config, mock_system_config):
        """Test workflow determination when both video and text search needed"""
        mock_get_config.return_value = mock_system_config
        
        # Create routing decision for both modalities
        both_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="llm"
        )
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("find information", both_decision)
        
        assert workflow["type"] == "raw_results"
        assert len(workflow["steps"]) == 2  # Both video and text search
        assert any(step["agent"] == "video_search" for step in workflow["steps"])
        assert any(step["agent"] == "text_search" for step in workflow["steps"])


class TestRoutingAgentEdgeCases:
    """Test edge cases and error conditions"""
    
    @patch('src.app.agents.routing_agent.get_config')
    @pytest.mark.asyncio
    async def test_router_failure_handling(self, mock_get_config):
        """Test handling when underlying router fails"""
        mock_get_config.return_value = {"video_agent_url": "http://localhost:8002"}
        
        agent = RoutingAgent()
        agent.router.route = AsyncMock(side_effect=Exception("Router failed"))
        
        with pytest.raises(Exception, match="Router failed"):
            await agent.analyze_and_route("test query")
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_workflow_with_missing_text_agent(self, mock_get_config):
        """Test workflow determination when text agent is not available"""
        # Don't include text_agent_url in config
        mock_get_config.return_value = {"video_agent_url": "http://localhost:8002"}
        
        both_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="llm"
        )
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("find information", both_decision)
        
        # Should only have video search step, not text search
        assert len(workflow["steps"]) == 1
        assert workflow["steps"][0]["agent"] == "video_search"
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_validate_agent_registry_success(self, mock_get_config):
        """Test successful agent registry validation"""
        mock_get_config.return_value = {
            "video_agent_url": "http://localhost:8002",
            "text_agent_url": "http://localhost:8003"
        }
        
        # Should not raise any exception
        agent = RoutingAgent()
        assert agent.agent_registry["video_search"] == "http://localhost:8002"
    
    @patch('src.app.agents.routing_agent.get_config')
    @pytest.mark.asyncio
    async def test_analyze_and_route_with_context(self, mock_get_config):
        """Test query analysis with additional context"""
        mock_get_config.return_value = {"video_agent_url": "http://localhost:8002"}
        
        mock_decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.9,
            routing_method="contextual"
        )
        
        agent = RoutingAgent()
        agent.router.route = AsyncMock(return_value=mock_decision)
        
        context = {"user_id": "test_user", "session_id": "session_123"}
        result = await agent.analyze_and_route("test query", context)
        
        # Verify context was passed to router
        agent.router.route.assert_called_once_with("test query", context)
        assert result["confidence"] == 0.9
        assert result["routing_method"] == "contextual"


class TestWorkflowStepGeneration:
    """Test workflow step generation logic"""
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_workflow_step_parameters(self, mock_get_config):
        """Test that workflow steps contain correct parameters"""
        mock_get_config.return_value = {"video_agent_url": "http://localhost:8002"}
        
        decision = RoutingDecision(
            search_modality=SearchModality.VIDEO,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="test"
        )
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("find videos about cats", decision)
        
        search_step = workflow["steps"][0]
        assert search_step["step"] == 1
        assert search_step["agent"] == "video_search"
        assert search_step["action"] == "search"
        assert search_step["parameters"]["query"] == "find videos about cats"
        assert search_step["parameters"]["top_k"] == 10
    
    @patch('src.app.agents.routing_agent.get_config')
    def test_workflow_step_numbering(self, mock_get_config):
        """Test that workflow steps are numbered correctly"""
        mock_get_config.return_value = {
            "video_agent_url": "http://localhost:8002",
            "text_agent_url": "http://localhost:8003"
        }
        
        decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.8,
            routing_method="test"
        )
        
        agent = RoutingAgent()
        workflow = agent._determine_workflow("find information", decision)
        
        # Should have 2 steps numbered 1 and 2
        step_numbers = [step["step"] for step in workflow["steps"]]
        assert step_numbers == [1, 2]