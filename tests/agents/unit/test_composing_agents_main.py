"""
Comprehensive unit tests for composing_agents_main.py

Tests the central orchestrator component that coordinates multiple specialized agents
for video content analysis and search.

Tests cover:
- EnhancedA2AClientTool: A2A communication with specialized agents
- QueryAnalysisTool: Query analysis and routing logic
- Direct routing and execution functions
- Web interface startup and configuration validation
"""

import asyncio
import datetime
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock external modules that aren't available in test environment
mock_modules = {
    "google": MagicMock(),
    "google.adk": MagicMock(),
    "google.adk.agents": MagicMock(),
    "google.adk.runners": MagicMock(),
    "google.adk.sessions": MagicMock(),
    "google.adk.tools": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "gliner": MagicMock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module


# Set up base classes with proper mock behavior
class MockBaseTool:
    def __init__(self, name=None, description=None):
        self.name = name
        self.description = description


sys.modules["google.adk.tools"].BaseTool = MockBaseTool
sys.modules["google.adk.agents"].LlmAgent = MagicMock
sys.modules["google.adk.runners"].Runner = MagicMock
sys.modules["google.adk.sessions"].InMemorySessionService = MagicMock
sys.modules["gliner"].GLiNER = MagicMock
sys.modules["google.genai.types"].Part = MagicMock


# Mock config before any imports to handle module-level initialization
class MockConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def validate_required_config(self):
        # Return empty dict to indicate no missing config
        return {}


mock_config = MockConfig(
    {
        "query_inference_engine": {
            "mode": "keyword",
            "current_gliner_model": "urchade/gliner_large-v2.1",
        },
        "timeout": 60.0,
        "video_agent_url": "http://localhost:8001",
        "text_agent_url": "http://localhost:8002",
        "composing_agent_port": 8000,
        "search_backend": "vespa",
    }
)

# Apply multiple config mocks to handle different access patterns
config_patcher = patch("cogniverse_core.config.utils.get_config", return_value=mock_config)
config_patcher.start()

# Also mock the config import in the composing_agents_main module specifically
composing_config_patcher = patch(
    "cogniverse_agents.composing_agents_main.get_config", return_value=mock_config
)
composing_config_patcher.start()


class TestEnhancedA2AClientTool:
    """Test the enhanced A2A client tool for agent communication"""

    @patch("cogniverse_agents.tools.a2a_utils.A2AClient")
    @patch("cogniverse_core.config.utils.get_config")
    def test_tool_initialization(self, mock_get_config, mock_a2a_client):
        """Test basic tool initialization"""
        # Mock config
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        # Mock A2A client
        mock_client_instance = MagicMock()
        mock_a2a_client.return_value = mock_client_instance

        # Now import and test
        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool(
            name="TestAgent",
            description="Test agent description",
            agent_url="http://localhost:8001",
            result_type="video",
        )

        assert tool.name == "TestAgent"
        assert tool.description == "Test agent description"
        assert tool.agent_url == "http://localhost:8001"
        assert tool.result_type == "video"
        assert hasattr(tool, "client")

    @patch("cogniverse_agents.tools.a2a_utils.A2AClient")
    @patch("cogniverse_core.config.utils.get_config")
    def test_tool_initialization_default_result_type(
        self, mock_get_config, mock_a2a_client
    ):
        """Test tool initialization with default result type"""
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool(
            name="TestAgent",
            description="Test description",
            agent_url="http://localhost:8001",
        )

        assert tool.result_type == "generic"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    @patch("cogniverse_agents.composing_agents_main.format_search_results")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_execute_success(self, mock_get_config, mock_format):
        """Test successful execution with proper result formatting"""
        # Mock config
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        mock_results = [
            {"video_id": "test1", "relevance": 0.9, "frame_id": 1},
            {"video_id": "test2", "relevance": 0.8, "frame_id": 2},
        ]

        mock_format.return_value = "Formatted video results"

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool(
            name="TestAgent",
            description="Test agent",
            agent_url="http://localhost:8001",
            result_type="video",
        )

        # Mock the client that was created in the tool's __init__
        mock_client_instance = AsyncMock()
        mock_client_instance.send_task.return_value = {"results": mock_results}
        tool.client = mock_client_instance

        result = await tool.execute(query="test query", top_k=5)

        assert result["success"] is True
        assert result["results"] == mock_results
        assert result["formatted_results"] == "Formatted video results"
        assert result["result_count"] == 2
        assert result["agent_used"] == "TestAgent"

        # Verify A2A client was called with correct parameters
        mock_client_instance.send_task.assert_called_once_with(
            "http://localhost:8001", query="test query", top_k=5
        )

    @pytest.mark.asyncio
    @patch("cogniverse_agents.composing_agents_main.format_search_results")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_execute_with_temporal_parameters(self, mock_get_config, mock_format):
        """Test execution with start_date and end_date parameters"""
        # Mock config
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        mock_format.return_value = "Formatted results"

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool(
            name="TestAgent",
            description="Test agent",
            agent_url="http://localhost:8001",
        )

        # Mock the client that was created in the tool's __init__
        mock_client_instance = AsyncMock()
        mock_client_instance.send_task.return_value = {"results": []}
        tool.client = mock_client_instance

        await tool.execute(
            query="test query",
            top_k=10,
            start_date="2024-01-01",
            end_date="2024-01-31",
            preferred_agent="http://custom:8002",
        )

        # Should use preferred_agent instead of default agent_url
        mock_client_instance.send_task.assert_called_once_with(
            "http://custom:8002",
            query="test query",
            top_k=10,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

    @pytest.mark.asyncio
    @patch("cogniverse_core.config.utils.get_config")
    async def test_execute_agent_error_response(self, mock_get_config):
        """Test handling of error response from agent"""
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool(
            name="TestAgent",
            description="Test agent",
            agent_url="http://localhost:8001",
        )

        # Mock the client that was created in the tool's __init__
        mock_client_instance = AsyncMock()
        mock_client_instance.send_task.return_value = {"error": "Connection failed"}
        tool.client = mock_client_instance

        result = await tool.execute(query="test query")

        assert result["success"] is False
        assert result["error"] == "Connection failed"
        assert "Error from TestAgent: Connection failed" in result["formatted_results"]

    @pytest.mark.asyncio
    @patch("cogniverse_core.config.utils.get_config")
    async def test_execute_exception_handling(self, mock_get_config):
        """Test exception handling during execution"""
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool(
            name="TestAgent",
            description="Test agent",
            agent_url="http://localhost:8001",
        )

        # Mock the client that was created in the tool's __init__
        mock_client_instance = AsyncMock()
        mock_client_instance.send_task.side_effect = Exception("Network timeout")
        tool.client = mock_client_instance

        result = await tool.execute(query="test query")

        assert result["success"] is False
        assert result["error"] == "Network timeout"
        assert "Error from TestAgent: Network timeout" in result["formatted_results"]


class TestQueryAnalysisTool:
    """Test the query analysis tool for intent detection and routing"""

    @patch("cogniverse_core.config.utils.get_config")
    def test_tool_initialization_keyword_mode(self, mock_get_config):
        """Test initialization in keyword mode"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()

        assert tool.name == "QueryAnalyzer"
        assert tool.inference_mode == "keyword"

    @patch("gliner.GLiNER")
    def test_tool_initialization_gliner_mode(self, mock_gliner):
        """Test initialization in GLiNER mode"""
        mock_model = MagicMock()
        mock_gliner.from_pretrained.return_value = mock_model

        # Create a new tool instance with GLiNER config
        with patch("cogniverse_agents.composing_agents_main.config") as mock_config:
            mock_config.get.return_value = {
                "mode": "gliner_only",
                "current_gliner_model": "gliner-test-model",
            }

            from cogniverse_agents.composing_agents_main import QueryAnalysisTool

            tool = QueryAnalysisTool()

        assert tool.inference_mode == "gliner_only"
        assert hasattr(tool, "gliner_model")
        assert tool.gliner_model == mock_model

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    @patch("cogniverse_core.config.utils.get_config")
    async def test_keyword_based_analysis_video_intent(self, mock_get_config):
        """Test keyword-based analysis detecting video search intent"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()
        result = await tool.execute("show me video clips of meetings")

        assert result["original_query"] == "show me video clips of meetings"
        assert result["needs_video_search"] is True
        assert result["needs_text_search"] is False
        assert result["routing_method"] == "keyword"
        assert result["cleaned_query"] == "show me video clips of meetings"

    @pytest.mark.asyncio
    @patch("cogniverse_core.config.utils.get_config")
    async def test_keyword_based_analysis_text_intent(self, mock_get_config):
        """Test keyword-based analysis detecting text search intent"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()
        result = await tool.execute("find document reports about analysis")

        assert result["needs_video_search"] is False
        assert result["needs_text_search"] is True
        assert result["routing_method"] == "keyword"

    @pytest.mark.asyncio
    async def test_keyword_based_analysis_default_both(self):
        """Test keyword-based analysis defaulting to both search types when no keywords match"""
        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        # Use a query with no specific keywords to trigger the fallback to both
        tool = QueryAnalysisTool()
        result = await tool.execute("find stuff about artificial intelligence")

        # No specific keywords should trigger fallback to both search types
        assert result["needs_video_search"] is True
        assert result["needs_text_search"] is True
        assert result["routing_method"] == "keyword"

    @patch("cogniverse_core.config.utils.get_config")
    def test_extract_temporal_info_yesterday(self, mock_get_config):
        """Test temporal extraction for 'yesterday' pattern"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        temporal_info = tool._extract_temporal_info("find videos from yesterday")

        assert temporal_info["start_date"] == yesterday.strftime("%Y-%m-%d")
        assert temporal_info["end_date"] == today.strftime("%Y-%m-%d")
        assert temporal_info["detected_pattern"] == r"yesterday"

    @patch("cogniverse_core.config.utils.get_config")
    def test_extract_temporal_info_specific_dates(self, mock_get_config):
        """Test temporal extraction for specific date patterns"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()

        # Single date
        temporal_info = tool._extract_temporal_info("videos from 2024-01-15")
        assert temporal_info["start_date"] == "2024-01-15"
        assert "end_date" not in temporal_info

        # Date range
        temporal_info = tool._extract_temporal_info(
            "videos from 2024-01-15 to 2024-01-20"
        )
        assert temporal_info["start_date"] == "2024-01-15"
        assert temporal_info["end_date"] == "2024-01-20"

    @patch("cogniverse_core.config.utils.get_config")
    def test_extract_temporal_info_no_pattern(self, mock_get_config):
        """Test temporal extraction when no temporal patterns are found"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()
        temporal_info = tool._extract_temporal_info("find videos about cats")

        assert temporal_info == {}


class TestRouteAndExecuteQuery:
    """Test the main query routing and execution function"""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    @patch("cogniverse_agents.tools.a2a_utils.A2AClient")
    @patch("cogniverse_agents.tools.a2a_utils.format_search_results")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_manual_routing(self, mock_get_config, mock_format, mock_a2a_client):
        """Test manual routing to a specific agent"""
        preferred_agent = "http://localhost:9001"

        # Mock config
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        # Mock A2A client for manual agent
        mock_client_instance = AsyncMock()
        mock_a2a_client.return_value = mock_client_instance
        mock_client_instance.send_task.return_value = {
            "results": [{"video_id": "test", "relevance": 0.9}]
        }
        mock_format.return_value = "Formatted results"

        from cogniverse_agents.composing_agents_main import route_and_execute_query

        result = await route_and_execute_query(
            query="test query", top_k=5, preferred_agent=preferred_agent
        )

        assert result["execution_type"] == "manual_routed"
        assert result["query_analysis"]["manual_routing"] is True
        assert result["query_analysis"]["preferred_agent"] == preferred_agent
        assert result["success"] is True
        assert len(result["agents_called"]) == 1
        assert f"Manual: {preferred_agent}" in result["agents_called"]

    @pytest.mark.asyncio
    @patch("cogniverse_agents.composing_agents_main.query_analyzer")
    @patch("cogniverse_agents.composing_agents_main.video_search_tool")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_automatic_routing_video_search(
        self, mock_get_config, mock_video_tool, mock_analyzer
    ):
        """Test automatic routing to video search agent"""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_analysis = {
            "needs_video_search": True,
            "needs_text_search": False,
            "temporal_info": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        }

        mock_analyzer.execute = AsyncMock(return_value=mock_analysis)
        mock_video_tool.execute = AsyncMock(
            return_value={
                "success": True,
                "results": [{"video_id": "test"}],
                "result_count": 1,
            }
        )

        from cogniverse_agents.composing_agents_main import route_and_execute_query

        result = await route_and_execute_query("test video query", top_k=10)

        assert result["execution_type"] == "routed"
        assert result["query_analysis"] == mock_analysis
        assert result["success"] is True
        assert "VideoSearchAgent" in result["agents_called"]
        assert "video_search_results" in result

        # Verify video tool was called with temporal parameters
        mock_video_tool.execute.assert_called_once_with(
            query="test video query",
            top_k=10,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

    @pytest.mark.asyncio
    @patch("cogniverse_agents.composing_agents_main.query_analyzer")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_automatic_routing_text_search_unavailable(
        self, mock_get_config, mock_analyzer
    ):
        """Test automatic routing when text search is needed but unavailable"""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_analysis = {
            "needs_video_search": False,
            "needs_text_search": True,
            "temporal_info": {},
        }

        mock_analyzer.execute = AsyncMock(return_value=mock_analysis)

        from cogniverse_agents.composing_agents_main import route_and_execute_query

        result = await route_and_execute_query("find text documents")

        assert result["execution_type"] == "routed"
        assert result["success"] is True
        assert "TextSearchAgent (unavailable)" in result["agents_called"]
        assert "text_search_results" in result
        assert "not available" in result["text_search_results"]["error"]

    @pytest.mark.asyncio
    @patch("cogniverse_agents.composing_agents_main.query_analyzer")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_routing_exception_handling(self, mock_get_config, mock_analyzer):
        """Test exception handling in routing"""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_analyzer.execute = AsyncMock(side_effect=Exception("Analysis failed"))

        from cogniverse_agents.composing_agents_main import route_and_execute_query

        result = await route_and_execute_query("test query")

        assert result["execution_type"] == "routed"
        assert result["success"] is False
        assert "error" in result
        assert "Analysis failed" in result["error"]


class TestWebInterfaceAndConfiguration:
    """Test web interface startup and configuration validation"""

    @patch("cogniverse_agents.composing_agents_main.os.system")
    @patch("cogniverse_agents.composing_agents_main.config")
    def test_start_web_interface_success(self, mock_config, mock_os_system):
        """Test successful web interface startup"""
        # Mock configuration validation
        mock_config.validate_required_config.return_value = {}  # No missing config
        mock_config.get.side_effect = lambda key, default=None: {
            "search_backend": "vespa",
            "text_agent_url": "http://localhost:8002",
            "video_agent_url": "http://localhost:8001",
            "composing_agent_port": 8000,
        }.get(key, default)

        from cogniverse_agents.composing_agents_main import start_web_interface

        start_web_interface()

        # Verify configuration was validated
        mock_config.validate_required_config.assert_called_once()

        # Verify ADK web command was executed
        mock_os_system.assert_called_once_with("adk web")

    @patch("cogniverse_agents.composing_agents_main.config")
    def test_start_web_interface_config_errors(self, mock_config):
        """Test web interface startup with configuration errors"""
        # Mock configuration validation with missing config
        mock_config.validate_required_config.return_value = {
            "video_agent_url": "Video agent URL is required",
            "search_backend": "Search backend must be configured",
        }

        from cogniverse_agents.composing_agents_main import start_web_interface

        # Should return early without starting ADK
        with patch("cogniverse_agents.composing_agents_main.os.system") as mock_os_system:
            start_web_interface()
            mock_os_system.assert_not_called()

        # Verify configuration was validated
        mock_config.validate_required_config.assert_called_once()


class TestADKIntegration:
    """Test ADK framework integration"""

    @pytest.mark.asyncio
    @patch("cogniverse_agents.composing_agents_main.InMemorySessionService")
    @patch("cogniverse_agents.composing_agents_main.Runner")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_run_query_programmatically(
        self, mock_get_config, mock_runner_class, mock_session_service
    ):
        """Test programmatic query execution"""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        # Mock the ADK session service
        mock_session_instance = AsyncMock()
        mock_session_instance.create_session.return_value = "test_session_id"
        mock_session_service.return_value = mock_session_instance

        # Mock the ADK runner
        mock_runner = MagicMock()
        mock_events = [
            {"type": "start", "message": "Starting"},
            {"type": "result", "message": "Query completed successfully"},
        ]

        # Create an async generator function
        async def mock_event_generator(*args, **kwargs):
            for event in mock_events:
                yield event

        # Set the run method directly to the async generator function
        mock_runner.run = mock_event_generator
        mock_runner_class.return_value = mock_runner

        from cogniverse_agents.composing_agents_main import run_query_programmatically

        result = await run_query_programmatically("test query")

        assert result["query"] == "test query"
        assert result["session_id"] == "test_session_id"
        assert len(result["events"]) == 2
        assert result["final_response"]["message"] == "Query completed successfully"

        # Verify session creation
        mock_session_instance.create_session.assert_called_once_with(
            user_id="programmatic_user", app_name="multi_agent_rag"
        )


class TestPerformanceAndEdgeCases:
    """Test performance considerations and edge cases"""

    @pytest.mark.asyncio
    @patch("cogniverse_agents.composing_agents_main.format_search_results")
    @patch("cogniverse_core.config.utils.get_config")
    async def test_concurrent_tool_execution(self, mock_get_config, mock_format):
        """Test handling of concurrent tool executions"""
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        mock_format.return_value = "Formatted results"

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool1 = EnhancedA2AClientTool("Agent1", "Test 1", "http://localhost:8001")
        tool2 = EnhancedA2AClientTool("Agent2", "Test 2", "http://localhost:8002")

        # Create separate mock instances for each tool and inject them
        mock_client_1 = AsyncMock()
        mock_client_2 = AsyncMock()

        mock_client_1.send_task.return_value = {"results": [{"id": 1}]}
        mock_client_2.send_task.return_value = {"results": [{"id": 2}]}

        tool1.client = mock_client_1
        tool2.client = mock_client_2

        # Execute both tools concurrently
        results = await asyncio.gather(
            tool1.execute(query="test1"), tool2.execute(query="test2")
        )

        assert len(results) == 2
        assert all(result["success"] for result in results)
        assert results[0]["agent_used"] == "Agent1"
        assert results[1]["agent_used"] == "Agent2"

    @patch("cogniverse_agents.tools.a2a_utils.A2AClient")
    @patch("cogniverse_core.config.utils.get_config")
    def test_tool_memory_efficiency(self, mock_get_config, mock_a2a_client):
        """Test that tools don't retain unnecessary state"""
        mock_config = MagicMock()
        mock_config.get.return_value = 60.0
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import EnhancedA2AClientTool

        tool = EnhancedA2AClientTool("TestAgent", "Test", "http://localhost:8001")

        # Tool should only have essential attributes
        expected_attrs = {"name", "description", "agent_url", "result_type", "client"}
        actual_attrs = {attr for attr in dir(tool) if not attr.startswith("_")}

        # Should not have excessive attributes (allow for some inherited ones)
        assert len(actual_attrs) < 20, f"Tool has too many attributes: {actual_attrs}"
        assert expected_attrs.issubset(actual_attrs)

    @patch("cogniverse_core.config.utils.get_config")
    def test_temporal_info_edge_cases(self, mock_get_config):
        """Test temporal information extraction edge cases"""
        mock_config = MagicMock()
        mock_config.get.return_value = {"mode": "keyword"}
        mock_get_config.return_value = mock_config

        from cogniverse_agents.composing_agents_main import QueryAnalysisTool

        tool = QueryAnalysisTool()

        # Test multiple temporal patterns (should take first match)
        temporal_info = tool._extract_temporal_info("yesterday last week videos")
        assert temporal_info["detected_pattern"] == r"yesterday"

        # Test malformed dates (should still extract)
        temporal_info = tool._extract_temporal_info("videos from 2024-13-45")
        assert temporal_info["start_date"] == "2024-13-45"

        # Test this week pattern
        temporal_info = tool._extract_temporal_info("show me videos from this week")
        assert "start_date" in temporal_info
        assert "end_date" in temporal_info
        assert temporal_info["detected_pattern"] == r"this week"
