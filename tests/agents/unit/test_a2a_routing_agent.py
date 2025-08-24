"""
Unit tests for A2ARoutingAgent
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.app.agents.a2a_routing_agent import (
    A2ARoutingAgent,
    AgentEndpoint,
)
from src.tools.a2a_utils import A2AMessage, DataPart, Task, TextPart


@pytest.mark.unit
class TestA2ARoutingAgent:
    """Test cases for A2ARoutingAgent class"""

    @pytest.fixture
    def mock_routing_agent(self):
        """Mock routing agent"""
        mock_agent = Mock()
        mock_agent.analyze_and_route = AsyncMock(
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

    @patch("src.app.agents.agent_registry.AgentRegistry")
    @patch("src.app.agents.a2a_routing_agent.get_config")
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

    def test_extract_query_and_context_with_data_part(self):
        """Test query and context extraction from DataPart"""
        agent = A2ARoutingAgent()

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

    def test_extract_query_and_context_with_text_part(self):
        """Test query extraction from TextPart"""
        agent = A2ARoutingAgent()

        message = A2AMessage(
            role="user", parts=[TextPart(text="find videos about machine learning")]
        )
        task = Task(id="test_task", messages=[message])

        query, context = agent._extract_query_and_context(task)

        assert query == "find videos about machine learning"
        assert context["task_id"] == "test_task"

    def test_extract_query_and_context_no_messages(self):
        """Test extraction with no messages"""
        agent = A2ARoutingAgent()

        task = Task(id="test_task", messages=[])

        with pytest.raises(ValueError, match="Task has no messages"):
            agent._extract_query_and_context(task)

    def test_extract_query_and_context_no_query(self):
        """Test extraction with no query found"""
        agent = A2ARoutingAgent()

        message = A2AMessage(
            role="user", parts=[DataPart(data={"other_field": "value"})]
        )
        task = Task(id="test_task", messages=[message])

        with pytest.raises(ValueError, match="No query found in task messages"):
            agent._extract_query_and_context(task)

    def test_create_agent_task(self):
        """Test agent task creation"""
        agent = A2ARoutingAgent()

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

    @patch("src.app.agents.agent_registry.AgentRegistry")
    @patch("src.app.agents.a2a_routing_agent.get_config")
    @pytest.mark.asyncio
    async def test_send_to_agent_success(self, mock_get_config, mock_registry_class):
        """Test successful agent communication"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        agent = A2ARoutingAgent()

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

    @patch("src.app.agents.a2a_routing_agent.get_config")
    @patch("src.app.agents.agent_registry.AgentRegistry")
    @pytest.mark.asyncio
    async def test_send_to_agent_http_error(self, mock_registry_class, mock_get_config):
        """Test agent communication with HTTP error"""
        mock_get_config.return_value = {}
        mock_registry_class.return_value = Mock()

        agent = A2ARoutingAgent()

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

    def test_aggregate_results_raw_results(self):
        """Test result aggregation for raw results workflow"""
        agent = A2ARoutingAgent()

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

    def test_aggregate_results_summary_with_summarizer(self):
        """Test result aggregation for summary workflow with summarizer agent"""
        agent = A2ARoutingAgent()

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

    def test_aggregate_results_summary_fallback(self):
        """Test result aggregation for summary workflow without summarizer agent"""
        agent = A2ARoutingAgent()

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

    def test_create_fallback_summary(self):
        """Test fallback summary creation"""
        agent = A2ARoutingAgent()

        search_results = [
            {"source_id": "video1", "title": "AI Tutorial"},
            {"source_id": "video2", "title": "ML Basics"},
            {"source_id": "video1", "title": "Advanced AI"},  # Duplicate source
        ]

        summary = agent._create_fallback_summary(search_results)

        assert "Found 3 results" in summary
        assert "2 sources" in summary
        assert "video1" in summary

    def test_create_fallback_summary_empty(self):
        """Test fallback summary with empty results"""
        agent = A2ARoutingAgent()

        summary = agent._create_fallback_summary([])

        assert summary == "No results found."

    def test_create_fallback_report(self):
        """Test fallback detailed report creation"""
        agent = A2ARoutingAgent()

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

    def test_create_fallback_report_empty(self):
        """Test fallback report with empty results"""
        agent = A2ARoutingAgent()

        report = agent._create_fallback_report([])

        assert report == "No results available for detailed analysis."


@pytest.mark.unit
class TestA2ARoutingAgentIntegration:
    """Integration tests for A2ARoutingAgent workflow execution"""

    @patch("src.app.agents.a2a_routing_agent.get_config")
    @patch("src.app.agents.agent_registry.AgentRegistry")
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
        mock_routing_agent = Mock()
        mock_routing_agent.analyze_and_route = AsyncMock(
            return_value={
                "query": "find AI videos",
                "routing_decision": {"search_modality": "video"},
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
        assert result.routing_decision["workflow_type"] == "raw_results"
        assert "video_search" in result.agent_responses
        assert len(result.final_result["search_results"]) == 1
        assert result.execution_time > 0

    @patch("src.app.agents.a2a_routing_agent.get_config")
    @patch("src.app.agents.agent_registry.AgentRegistry")
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
