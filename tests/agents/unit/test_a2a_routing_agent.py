"""
Unit tests for A2ARoutingAgent
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cogniverse_agents.a2a_routing_agent import (
    A2ARoutingAgent,
    AgentEndpoint,
)
from cogniverse_core.common.a2a_utils import A2AMessage, DataPart, Task, TextPart


@pytest.mark.unit
class TestA2ARoutingAgent:
    """Test cases for A2ARoutingAgent class"""

    @pytest.fixture
    def mock_routing_agent(self):
        """Mock routing agent"""
        mock_agent = Mock()
        mock_agent.route_query = AsyncMock(
            return_value={
                "query": "test query",
                "routing_decision": {"search_modality": "video"},
                "workflow_type": "raw_results",
                "agents_to_call": ["video_search"],
                "execution_plan": [
                    {
                        "step": 1,
                        "agent": "video_search",
                        "action": "search",
                        "parameters": {"query": "test query", "top_k": 10},
                    }
                ],
            }
        )
        return mock_agent

    @pytest.fixture
    def mock_agent_registry(self):
        """Mock agent registry"""
        mock_registry = Mock()
        mock_registry.get_agent.return_value = AgentEndpoint(
            name="video_search",
            url="http://localhost:8002",
            capabilities=["video_search"],
        )
        return mock_registry

    @pytest.fixture
    def sample_task(self):
        """Sample A2A task"""
        message = A2AMessage(
            role="user",
            parts=[DataPart(data={"query": "find videos about AI", "top_k": 5})],
        )
        return Task(id="test_task_123", messages=[message])

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_a2a_routing_agent_initialization(
        self, mock_get_config, mock_registry_class, mock_routing_agent
    ):
        """Test A2ARoutingAgent initialization"""
        mock_get_config.return_value = {}
        mock_registry_instance = Mock()
        mock_registry_class.return_value = mock_registry_instance

        agent = A2ARoutingAgent(mock_routing_agent)

        assert agent.routing_agent == mock_routing_agent
        assert agent.agent_registry == mock_registry_instance
        assert agent.http_client is not None

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_extract_query_and_context_with_data_part(self, mock_get_config, mock_registry_class):
        """Test query and context extraction from DataPart"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()
        mock_routing_agent = Mock()

        agent = A2ARoutingAgent(mock_routing_agent)

        message = A2AMessage(
            role="user",
            parts=[
                DataPart(
                    data={"query": "test query", "top_k": 10, "user_id": "test_user"}
                )
            ],
        )
        task = Task(id="test_task", messages=[message])

        query, context = agent._extract_query_and_context(task)

        assert query == "test query"
        assert context["task_id"] == "test_task"
        assert context["top_k"] == 10
        assert context["user_id"] == "test_user"
        assert "query" not in context

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_extract_query_and_context_with_text_part(self, mock_get_config, mock_registry_class):
        """Test query extraction from TextPart"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        message = A2AMessage(
            role="user", parts=[TextPart(text="find videos about machine learning")]
        )
        task = Task(id="test_task", messages=[message])

        query, context = agent._extract_query_and_context(task)

        assert query == "find videos about machine learning"
        assert context["task_id"] == "test_task"

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_extract_query_and_context_no_messages(self, mock_get_config, mock_registry_class):
        """Test extraction with no messages"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        task = Task(id="test_task", messages=[])

        with pytest.raises(ValueError, match="Task has no messages"):
            agent._extract_query_and_context(task)

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_extract_query_and_context_no_query(self, mock_get_config, mock_registry_class):
        """Test extraction with no query found"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        message = A2AMessage(
            role="user", parts=[DataPart(data={"other_field": "value"})]
        )
        task = Task(id="test_task", messages=[message])

        with pytest.raises(ValueError, match="No query found in task messages"):
            agent._extract_query_and_context(task)

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_create_agent_task(self, mock_get_config, mock_registry_class):
        """Test agent task creation"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        parameters = {"query": "test query", "top_k": 10}
        task_id = "main_task"
        step = 1

        agent_task = agent._create_agent_task(parameters, task_id, step)

        assert agent_task.id == "main_task_step_1"
        assert len(agent_task.messages) == 1

        message = agent_task.messages[0]
        assert message.role == "user"
        assert len(message.parts) == 1

        data_part = message.parts[0]
        assert isinstance(data_part, DataPart)
        assert data_part.data == parameters

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    @pytest.mark.asyncio
    async def test_send_to_agent_success(self, mock_get_config, mock_registry_class):
        """Test successful agent communication"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        # Mock HTTP client
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "results": []}
        mock_response.raise_for_status = Mock()

        agent.http_client.post = AsyncMock(return_value=mock_response)

        agent_endpoint = AgentEndpoint(
            name="test_agent", url="http://localhost:8002", capabilities=["test"]
        )

        message = A2AMessage(role="user", parts=[DataPart(data={"query": "test"})])
        task = Task(id="test_task", messages=[message])

        result = await agent._send_to_agent(agent_endpoint, task)

        assert result == {"status": "success", "results": []}
        agent.http_client.post.assert_called_once()

    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @pytest.mark.asyncio
    async def test_send_to_agent_http_error(self, mock_registry_class, mock_get_config):
        """Test agent communication with HTTP error"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        # Mock HTTP error
        from httpx import HTTPStatusError

        mock_response = Mock()
        mock_response.status_code = 500

        agent.http_client.post = AsyncMock(
            side_effect=HTTPStatusError(
                "Server error", request=Mock(), response=mock_response
            )
        )

        agent_endpoint = AgentEndpoint(
            name="test_agent", url="http://localhost:8002", capabilities=["test"]
        )

        message = A2AMessage(role="user", parts=[DataPart(data={"query": "test"})])
        task = Task(id="test_task", messages=[message])

        with pytest.raises(Exception, match="Agent test_agent returned 500"):
            await agent._send_to_agent(agent_endpoint, task)

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_aggregate_results_raw_results(self, mock_get_config, mock_registry_class):
        """Test result aggregation for raw results workflow"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        routing_analysis = {
            "query": "test query",
            "workflow_type": "raw_results",
            "routing_decision": {"search_modality": "video"},
        }

        agent_responses = {
            "video_search": {
                "status": "success",
                "results": [
                    {"id": "1", "title": "Result 1", "score": 0.9},
                    {"id": "2", "title": "Result 2", "score": 0.8},
                ],
            }
        }

        final_result = agent._aggregate_results(routing_analysis, agent_responses)

        assert final_result["workflow_type"] == "raw_results"
        assert final_result["query"] == "test query"
        assert len(final_result["search_results"]) == 2
        assert final_result["search_results"][0]["title"] == "Result 1"

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_aggregate_results_summary_with_summarizer(self, mock_get_config, mock_registry_class):
        """Test result aggregation for summary workflow with summarizer agent"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        routing_analysis = {
            "query": "summarize videos",
            "workflow_type": "summary",
            "routing_decision": {"search_modality": "video"},
        }

        agent_responses = {
            "video_search": {"results": [{"id": "1", "title": "Video 1"}]},
            "summarizer": {"summary": "This is a summary of the video content."},
        }

        final_result = agent._aggregate_results(routing_analysis, agent_responses)

        assert final_result["workflow_type"] == "summary"
        assert final_result["summary"] == "This is a summary of the video content."

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_aggregate_results_summary_fallback(self, mock_get_config, mock_registry_class):
        """Test result aggregation for summary workflow without summarizer agent"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        routing_analysis = {
            "query": "summarize videos",
            "workflow_type": "summary",
            "routing_decision": {"search_modality": "video"},
        }

        agent_responses = {
            "video_search": {
                "results": [
                    {"id": "1", "title": "Video 1", "source_id": "source1"},
                    {"id": "2", "title": "Video 2", "source_id": "source2"},
                ]
            }
        }

        final_result = agent._aggregate_results(routing_analysis, agent_responses)

        assert final_result["workflow_type"] == "summary"
        assert "summary" in final_result
        assert "Found 2 results" in final_result["summary"]

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_create_fallback_summary(self, mock_get_config, mock_registry_class):
        """Test fallback summary creation"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        search_results = [
            {"source_id": "video1", "title": "AI Tutorial"},
            {"source_id": "video2", "title": "ML Basics"},
            {"source_id": "video1", "title": "Advanced AI"},  # Duplicate source
        ]

        summary = agent._create_fallback_summary(search_results)

        assert "Found 3 results" in summary
        assert "2 sources" in summary
        assert "video1" in summary

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_create_fallback_summary_empty(self, mock_get_config, mock_registry_class):
        """Test fallback summary with empty results"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        summary = agent._create_fallback_summary([])

        assert summary == "No results found."

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_create_fallback_report(self, mock_get_config, mock_registry_class):
        """Test fallback detailed report creation"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        search_results = [
            {
                "source_id": "video1",
                "content_type": "video",
                "score": 0.95,
                "metadata": {"start_time": "00:00", "end_time": "05:30"},
            },
            {"source_id": "document1", "content_type": "document", "score": 0.87},
        ]

        report = agent._create_fallback_report(search_results)

        assert "Detailed Analysis of 2 Results" in report
        assert "video1 (video)" in report
        assert "Relevance Score: 0.950" in report
        assert "Time Range: 00:00 - 05:30" in report
        assert "document1 (document)" in report

    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    def test_create_fallback_report_empty(self, mock_get_config, mock_registry_class):
        """Test fallback report with empty results"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        mock_routing_agent = Mock()
        agent = A2ARoutingAgent(mock_routing_agent)

        report = agent._create_fallback_report([])

        assert report == "No results available for detailed analysis."


@pytest.mark.unit
class TestA2ARoutingAgentIntegration:
    """Integration tests for A2ARoutingAgent workflow execution"""

    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @pytest.mark.asyncio
    async def test_process_task_success(self, mock_registry_class, mock_get_config):
        """Test successful task processing end-to-end"""
        mock_get_config.return_value = {}

        # Mock agent registry
        mock_registry = Mock()
        mock_agent_endpoint = AgentEndpoint(
            name="video_search",
            url="http://localhost:8002",
            capabilities=["video_search"],
        )
        mock_registry.get_agent.return_value = mock_agent_endpoint
        mock_registry_class.return_value = mock_registry

        # Mock routing agent
        from cogniverse_agents.routing_agent import RoutingDecision
        mock_routing_agent = Mock()
        mock_routing_decision = RoutingDecision(
            query="find AI videos",
            recommended_agent="video_search_agent",
            confidence=0.9,
            reasoning="Test routing",
            fallback_agents=[],
            enhanced_query="find AI videos",
            entities=[],
            relationships=[],
            metadata={
                "workflow_type": "raw_results",
                "agents_to_call": ["video_search"],
                "execution_plan": [
                    {
                        "step": 1,
                        "agent": "video_search",
                        "action": "search",
                        "parameters": {"query": "find AI videos", "top_k": 10},
                    }
                ],
            }
        )
        mock_routing_agent.analyze_and_route = AsyncMock(return_value=mock_routing_decision)

        # Create A2A routing agent
        a2a_agent = A2ARoutingAgent(mock_routing_agent)

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "results": [{"id": "1", "title": "AI Video"}],
        }
        mock_response.raise_for_status = Mock()
        a2a_agent.http_client.post = AsyncMock(return_value=mock_response)

        # Create task
        message = A2AMessage(
            role="user", parts=[DataPart(data={"query": "find AI videos", "top_k": 5})]
        )
        task = Task(id="test_task", messages=[message])

        # Process task
        result = await a2a_agent.process_task(task)

        # Verify result
        assert result.success is True
        assert result.task_id == "test_task"
        assert result.routing_decision.metadata.get("workflow_type") == "raw_results"
        assert "video_search" in result.agent_responses
        assert len(result.final_result["search_results"]) == 1
        assert result.execution_time > 0

    @patch("cogniverse_agents.a2a_routing_agent.get_config")
    @patch("cogniverse_agents.agent_registry.AgentRegistry")
    @pytest.mark.asyncio
    async def test_process_task_routing_failure(
        self, mock_registry_class, mock_get_config
    ):
        """Test task processing when routing fails"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        # Mock routing agent to fail
        mock_routing_agent = Mock()
        mock_routing_agent.analyze_and_route = AsyncMock(
            side_effect=Exception("Routing failed")
        )

        a2a_agent = A2ARoutingAgent(mock_routing_agent)

        # Create task
        message = A2AMessage(
            role="user", parts=[DataPart(data={"query": "test query"})]
        )
        task = Task(id="test_task", messages=[message])

        # Process task
        result = await a2a_agent.process_task(task)

        # Verify failure handling
        assert result.success is False
        assert result.error == "Routing failed"
        assert result.task_id == "test_task"
