"""Tests for DetailedReportAgent with DSPy integration."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
    ReportRequest,
    ReportResult,
    ThinkingPhase,
    VLMInterface,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager

# Endpoint the agent routes per request when _initialize_vlm_client is mocked.
_GATEWAY_TEST_ENDPOINT = LLMEndpointConfig(
    model="openai/test-model", api_base="http://localhost:11434/v1"
)


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        {
            "id": "1",
            "title": "Sample Video 1",
            "description": "Description 1",
            "score": 0.9,
            "duration": 120,
            "thumbnail": "/path/to/thumb1.jpg",
            "format": "mp4",
            "resolution": "1080p",
            "content_type": "video",
        },
        {
            "id": "2",
            "title": "Sample Video 2",
            "description": "Description 2",
            "score": 0.7,
            "duration": 300,
            "thumbnail": "/path/to/thumb2.jpg",
            "format": "mp4",
            "resolution": "720p",
            "content_type": "video",
        },
        {
            "id": "3",
            "title": "Sample Image 1",
            "description": "Description 3",
            "score": 0.6,
            "image_path": "/path/to/image1.jpg",
            "content_type": "image",
        },
    ]


@pytest.fixture
def sample_report_request(sample_search_results):
    """Sample report request for testing"""
    return ReportRequest(
        query="test query",
        search_results=sample_search_results,
        report_type="comprehensive",
        include_visual_analysis=True,
        include_technical_details=True,
        include_recommendations=True,
        max_results_to_analyze=10,
    )


@pytest.fixture
def mock_config():
    """Mock configuration for DSPy"""
    return {
        "llm": {
            "model_name": "test-model",
            "base_url": "http://localhost:11434",
            "api_key": None,
        }
    }


@pytest.mark.unit
class TestVLMInterface:
    """Test VLM interface with DSPy integration"""

    @patch("cogniverse_foundation.config.llm_factory.create_dspy_lm")
    @patch("cogniverse_core.common.vlm_interface.get_config")
    @pytest.mark.ci_fast
    def test_vlm_interface_initialization_success(
        self, mock_get_config, mock_create_dspy_lm
    ):
        """Test VLM interface initialization with proper config"""
        mock_config = Mock()
        mock_llm_config = Mock()
        mock_endpoint = Mock()
        mock_endpoint.model = "test-model"
        mock_endpoint.api_base = "http://localhost:11434"
        mock_llm_config.resolve.return_value = mock_endpoint
        mock_config.get_llm_config.return_value = mock_llm_config
        mock_get_config.return_value = mock_config
        mock_create_dspy_lm.return_value = Mock()

        vlm = VLMInterface(
            config_manager=create_default_config_manager(), tenant_id="test:unit"
        )

        assert vlm.config is not None
        mock_create_dspy_lm.assert_called_once_with(mock_endpoint)

    @patch("cogniverse_core.common.vlm_interface.get_config")
    def test_vlm_interface_initialization_missing_config(self, mock_get_config):
        """Test VLM interface initialization fails with missing config"""
        mock_config = Mock()
        mock_config.get_llm_config.side_effect = ValueError("LLM configuration missing")
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="LLM configuration missing"):
            VLMInterface(
                config_manager=create_default_config_manager(), tenant_id="test:unit"
            )


@pytest.mark.unit
class TestDetailedReportAgent:
    """Test cases for DetailedReportAgent class"""

    @patch("cogniverse_foundation.config.llm_factory.create_dspy_lm")
    @patch("cogniverse_foundation.config.utils.get_config")
    @patch("cogniverse_agents.detailed_report_agent.VLMInterface")
    @pytest.mark.ci_fast
    def test_detailed_report_agent_initialization(
        self, mock_vlm_class, mock_get_config, mock_create_dspy_lm
    ):
        """Test DetailedReportAgent initialization"""
        mock_sys_config = Mock()
        mock_llm_config = Mock()
        mock_endpoint = Mock()
        mock_endpoint.model = "test-model"
        mock_endpoint.api_base = "http://localhost:11434"
        mock_llm_config.resolve.return_value = mock_endpoint
        mock_sys_config.get_llm_config.return_value = mock_llm_config
        mock_get_config.return_value = mock_sys_config
        mock_create_dspy_lm.return_value = Mock()
        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        agent = DetailedReportAgent(deps=DetailedReportDeps(), config_manager=Mock())

        assert agent.config is not None
        assert agent.vlm == mock_vlm_instance
        assert agent.max_report_length == 2000
        assert agent.thinking_enabled is True
        assert agent.visual_analysis_enabled is True
        assert agent.technical_analysis_enabled is True

    @patch("cogniverse_foundation.config.llm_factory.create_dspy_lm")
    @patch("cogniverse_foundation.config.utils.get_config")
    @patch("cogniverse_agents.detailed_report_agent.VLMInterface")
    def test_detailed_report_agent_custom_config(
        self, mock_vlm_class, mock_get_config, mock_create_dspy_lm
    ):
        """Test DetailedReportAgent with custom configuration"""
        mock_sys_config = Mock()
        mock_llm_config = Mock()
        mock_endpoint = Mock()
        mock_endpoint.model = "test-model"
        mock_endpoint.api_base = "http://localhost:11434"
        mock_llm_config.resolve.return_value = mock_endpoint
        mock_sys_config.get_llm_config.return_value = mock_llm_config
        mock_get_config.return_value = mock_sys_config
        mock_create_dspy_lm.return_value = Mock()
        mock_vlm_class.return_value = Mock()

        agent = DetailedReportAgent(
            deps=DetailedReportDeps(
                max_report_length=1500,
                thinking_enabled=False,
                visual_analysis_enabled=False,
                technical_analysis_enabled=False,
            ),
            config_manager=Mock(),
        )

        assert agent.deps.max_report_length == 1500
        assert agent.deps.thinking_enabled is False
        assert agent.deps.visual_analysis_enabled is False
        assert agent.deps.technical_analysis_enabled is False

    @pytest.mark.ci_fast
    def test_thinking_phase_creation(self):
        """Test ThinkingPhase data structure"""
        thinking = ThinkingPhase(
            content_analysis={"type": "video", "count": 2},
            visual_assessment={"quality": "high", "clarity": "good"},
            technical_findings=["finding1", "finding2"],
            patterns_identified=["pattern1", "pattern2"],
            gaps_and_limitations=["gap1", "gap2"],
            reasoning="Test reasoning",
        )

        assert thinking.content_analysis["type"] == "video"
        assert len(thinking.technical_findings) == 2
        assert thinking.reasoning == "Test reasoning"

    def test_report_request_validation(self, sample_search_results):
        """Test ReportRequest validation"""
        request = ReportRequest(
            query="test query",
            search_results=sample_search_results,
            report_type="comprehensive",
            include_visual_analysis=True,
        )

        assert request.query == "test query"
        assert len(request.search_results) == 3
        assert request.report_type == "comprehensive"
        assert request.include_visual_analysis is True

    def test_report_result_structure(self):
        """Test ReportResult data structure"""
        thinking = ThinkingPhase(
            content_analysis={},
            visual_assessment={},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning="",
        )

        result = ReportResult(
            executive_summary="Summary",
            detailed_findings=[{"finding": "test"}],
            visual_analysis=[{"analysis": "visual"}],
            technical_details=[{"detail": "tech"}],
            recommendations=["rec1", "rec2"],
            confidence_assessment={"overall": 0.8},
            thinking_phase=thinking,
            metadata={"processed": True},
        )

        assert result.executive_summary == "Summary"
        assert len(result.detailed_findings) == 1
        assert len(result.recommendations) == 2
        assert result.confidence_assessment["overall"] == 0.8

    @patch("cogniverse_agents.detailed_report_agent.VLMInterface")
    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_process_a2a_task_success(self, mock_vlm_class):
        """Test processing report generation request successfully"""
        mock_vlm_class.return_value = Mock()

        # Mock the _initialize_vlm_client to avoid DSPy config issues
        with patch.object(DetailedReportAgent, "_initialize_vlm_client"):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=Mock()
            )
            agent._llm_config = _GATEWAY_TEST_ENDPOINT

        # Pin the only LLM-dependent step so the assertion proves the DSPy
        # output is carried verbatim into the result (was unstubbed, so the
        # old test silently exercised the fallback path).
        agent.call_dspy = AsyncMock(
            return_value=Mock(
                executive_summary="report for test query",
                recommendations="deepen coverage, add benchmarks",
            )
        )

        # Create report request
        request = ReportRequest(
            query="test query",
            search_results=[{"id": "1", "title": "Test", "score": 0.8}],
            report_type="comprehensive",
            include_visual_analysis=False,  # Skip visual analysis for speed
        )

        result = await agent._generate_report(request)

        assert result.executive_summary == "report for test query"
        # Thinking phase is derived deterministically from the single result.
        ca = result.thinking_phase.content_analysis
        assert ca["total_results"] == 1
        assert ca["avg_relevance"] == pytest.approx(0.8)
        assert isinstance(result.detailed_findings, list) and result.detailed_findings
        # The LM's recommendations reach the report — not the canned templates.
        assert result.recommendations == ["deepen coverage", "add benchmarks"]
        assert isinstance(result.confidence_assessment, dict)

    @patch("cogniverse_agents.detailed_report_agent.VLMInterface")
    @pytest.mark.asyncio
    async def test_process_a2a_task_no_messages(self, mock_vlm_class):
        """Test report generation with empty search results"""
        mock_vlm_class.return_value = Mock()

        # Mock the _initialize_vlm_client to avoid DSPy config issues
        with patch.object(DetailedReportAgent, "_initialize_vlm_client"):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=Mock()
            )
            agent._llm_config = _GATEWAY_TEST_ENDPOINT

        # _generate_executive_summary calls call_dspy even for empty results, so
        # pin it — else the assertion passed identically whether the DSPy path
        # ran or connection-errored into the fabricated fallback string.
        agent.call_dspy = AsyncMock(
            return_value=Mock(
                executive_summary="no results found for test query",
                recommendations="broaden the query",
            )
        )

        # Empty search results should still generate a report
        request = ReportRequest(
            query="test query",
            search_results=[],
            report_type="comprehensive",
            include_visual_analysis=False,
        )

        result = await agent._generate_report(request)

        # The mocked DSPy summary/recommendations are carried verbatim.
        assert result.executive_summary == "no results found for test query"
        assert result.recommendations == ["broaden the query"]
        assert result.thinking_phase.content_analysis["total_results"] == 0


@pytest.mark.unit
class TestDetailedReportAgentEdgeCases:
    """Test edge cases and error conditions"""

    @patch("cogniverse_agents.detailed_report_agent.VLMInterface")
    @pytest.mark.asyncio
    async def test_generate_report_empty_results(self, mock_vlm_class):
        """_generate_report must return a well-formed ReportResult for empty
        search results. The old test only inspected the ReportRequest it built
        and never called the agent at all."""
        mock_vlm_class.return_value = Mock()
        with patch.object(DetailedReportAgent, "_initialize_vlm_client"):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=Mock()
            )
            agent._llm_config = _GATEWAY_TEST_ENDPOINT

        # Pin the exec-summary DSPy step so the assertion proves the DSPy output
        # is carried through, not the fabricated LM-down fallback string.
        agent.call_dspy = AsyncMock(
            return_value=Mock(
                executive_summary="empty report for test query",
                recommendations="add more sources",
            )
        )

        request = ReportRequest(
            query="test query",
            search_results=[],  # Empty results
            report_type="comprehensive",
            include_visual_analysis=False,
        )

        result = await agent._generate_report(request)

        # Empty input still produces a structured report with the DSPy summary.
        assert result.thinking_phase.content_analysis["total_results"] == 0
        assert result.executive_summary == "empty report for test query"
        assert result.recommendations == ["add more sources"]
        assert isinstance(result.detailed_findings, list)
        assert isinstance(result.confidence_assessment, dict)


@pytest.mark.unit
class TestDetailedReportAgentCoreFunctionality:
    """Test core report generation functionality that was missing coverage"""

    @pytest.fixture
    def agent_with_mocks(self):
        """Create agent with properly mocked dependencies"""
        with (
            patch(
                "cogniverse_agents.detailed_report_agent.VLMInterface"
            ) as mock_vlm_class,
            patch.object(DetailedReportAgent, "_initialize_vlm_client"),
        ):
            mock_vlm = Mock()
            mock_vlm.analyze_visual_content = AsyncMock(
                return_value={
                    "visual_elements": ["person", "object"],
                    "scene_description": "Test scene",
                    "technical_quality": "high",
                    "confidence_score": 0.9,
                }
            )
            mock_vlm_class.return_value = mock_vlm

            agent = DetailedReportAgent(
                deps=DetailedReportDeps(), config_manager=Mock()
            )
            agent._llm_config = _GATEWAY_TEST_ENDPOINT
            agent.vlm = mock_vlm
            return agent

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_thinking_phase_functionality(
        self, agent_with_mocks, sample_report_request
    ):
        """Test the thinking phase logic that drives report generation"""
        agent = agent_with_mocks

        # Test the private _thinking_phase method
        thinking_phase = await agent._thinking_phase(sample_report_request)

        assert isinstance(thinking_phase, ThinkingPhase)
        assert thinking_phase.reasoning is not None
        assert len(thinking_phase.reasoning) > 0
        assert isinstance(thinking_phase.content_analysis, dict)
        assert "total_results" in thinking_phase.content_analysis

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_visual_analysis_functionality(
        self, agent_with_mocks, sample_report_request
    ):
        """Test visual analysis performs actual VLM calls"""
        agent = agent_with_mocks

        # Create proper thinking_phase with visual_assessment
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "content_types": {"video": 2, "image": 1},
            },
            visual_assessment={
                "has_visual_content": True,
                "visual_elements": {"images": 1},
            },
            technical_findings=["HD quality"],
            patterns_identified=["educational"],
            gaps_and_limitations=[],
            reasoning="test",
        )

        # Mock the visual analysis to avoid method signature issues
        image_results = [
            r
            for r in sample_report_request.search_results
            if r.get("content_type") == "image"
        ]

        if image_results:
            # Create a mock request for visual analysis
            visual_request = ReportRequest(
                query="test query",
                search_results=image_results,
                report_type="comprehensive",
                include_visual_analysis=True,
            )
            visual_analysis = await agent._perform_visual_analysis(
                visual_request, thinking_phase
            )

            assert isinstance(visual_analysis, list)
            # Should call the VLM client
            # Visual analysis is called via vlm.analyze_visual_content
            # Just verify that visual analysis happened (check results have insights)
            assert (
                len(visual_analysis) > 0 or not visual_request.include_visual_analysis
            )

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_visual_analysis_reads_relevance_score_as_confidence(self):
        """The VLM emits relevance_score, not confidence — the report used to read
        the absent 'confidence' key and reported 0.0 for every visual result."""
        from types import SimpleNamespace

        from cogniverse_agents.detailed_report_agent import (
            DetailedReportAgent,
            ReportRequest,
        )

        agent = object.__new__(DetailedReportAgent)
        agent.visual_analysis_enabled = True
        agent.vlm = Mock()
        agent.vlm.analyze_visual_content = AsyncMock(
            return_value={"insights": ["a chart"], "relevance_score": 0.85}
        )

        request = ReportRequest(
            query="q",
            search_results=[{"id": "r1", "image_path": "/tmp/x.jpg"}],
            report_type="comprehensive",
            include_visual_analysis=True,
        )
        thinking = SimpleNamespace(visual_assessment={"has_visual_content": True})
        out = await agent._perform_visual_analysis(request, thinking)

        assert len(out) == 1
        assert out[0]["confidence"] == 0.85
        assert out[0]["insights"] == ["a chart"]

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_generate_report_full_workflow(
        self, agent_with_mocks, sample_report_request
    ):
        """The full report pipeline derives the thinking phase deterministically
        from the 3 fixture results (2 video + 1 image) and surfaces the DSPy
        executive summary verbatim. Assert exact derived values, not just types.
        """
        agent = agent_with_mocks
        # The only LLM-dependent step is the executive summary (via call_dspy);
        # pin it so the assertion is deterministic and proves the wiring carries
        # the DSPy output through to the result.
        agent.call_dspy = AsyncMock(
            return_value=Mock(executive_summary="3 results analyzed for test query")
        )

        result = await agent._generate_report(sample_report_request)

        assert isinstance(result, ReportResult)
        assert result.executive_summary == "3 results analyzed for test query"

        # Thinking phase is computed from the search results, NOT from any LLM:
        # 2 videos + 1 image, durations 120/300/none, scores 0.9/0.7/0.6.
        ca = result.thinking_phase.content_analysis
        assert ca["total_results"] == 3
        assert ca["content_types"] == {"video": 2, "image": 1}
        assert ca["duration_distribution"] == {"short": 1, "medium": 1, "long": 1}
        assert ca["quality_metrics"] == {"high": 1, "medium": 2, "low": 0}
        assert ca["avg_relevance"] == pytest.approx((0.9 + 0.7 + 0.6) / 3)

        # First detailed finding is always the content-analysis summary; its
        # significance is "high" because avg_relevance (0.733) > 0.7.
        first = result.detailed_findings[0]
        assert first["category"] == "Content Analysis"
        assert first["finding"] == "Analyzed 3 results"
        assert first["significance"] == "high"
        assert first["details"] == ca

        # avg_relevance 0.733 sits between the 0.5 and 0.8 thresholds, so NEITHER
        # relevance-based recommendation is emitted.
        assert (
            "Consider refining the search query to improve result relevance"
            not in result.recommendations
        )
        assert (
            "High relevance scores indicate good query formulation"
            not in result.recommendations
        )

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_generate_executive_summary_logic(
        self, agent_with_mocks, sample_report_request
    ):
        """Test executive summary generation with actual content"""
        agent = agent_with_mocks

        # Create a mock thinking phase with all required fields
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "avg_relevance": 0.8,
                "content_types": {"video": 2, "image": 1},
                "duration_distribution": {"short": 1, "medium": 2, "long": 0},
            },
            visual_assessment={"has_visual_content": True, "overall_quality": "high"},
            technical_findings=["HD quality", "good audio"],
            patterns_identified=["educational content"],
            gaps_and_limitations=["limited time range"],
            reasoning="Comprehensive analysis completed",
        )

        # Mock DSPy module to return proper result instead of using fallback
        mock_dspy_result = Mock()
        mock_dspy_result.executive_summary = "Comprehensive analysis of 3 results for test query, covering key topics in HD quality with educational content"
        mock_dspy_result.recommendations = "expand coverage, add benchmarks"

        with patch.object(
            agent.report_module, "forward", return_value=mock_dspy_result
        ):
            summary, recommendations = await agent._generate_executive_summary(
                sample_report_request, thinking_phase
            )

        assert isinstance(summary, str)
        assert len(summary) > 30  # Should be substantive
        assert (
            "3" in summary or "three" in summary.lower()
        )  # Should mention result count
        # The LM's recommendations come back parsed alongside the summary.
        assert recommendations == ["expand coverage", "add benchmarks"]

    @pytest.mark.ci_fast
    def test_generate_detailed_findings_logic(
        self, agent_with_mocks, sample_report_request
    ):
        """Test detailed findings generation"""
        agent = agent_with_mocks

        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "content_types": {"video": 2, "image": 1},
                "quality_metrics": {"high": 2, "medium": 1, "low": 0},
                "avg_relevance": 0.8,
            },
            visual_assessment={
                "has_visual_content": True,
                "quality_score": 0.8,
                "visual_coverage": 7,
                "visual_elements": {"images": 2, "videos": 1},
            },
            technical_findings=["HD quality"],
            patterns_identified=["educational"],
            gaps_and_limitations=[],
            reasoning="test",
        )

        findings = agent._generate_detailed_findings(
            sample_report_request, thinking_phase
        )

        assert isinstance(findings, list)
        assert len(findings) > 0
        # Should create findings based on search results
        for finding in findings:
            assert "category" in finding
            assert "finding" in finding
            assert "details" in finding

    @pytest.mark.ci_fast
    def test_generate_recommendations_logic(
        self, agent_with_mocks, sample_report_request
    ):
        """Test recommendations generation"""
        agent = agent_with_mocks

        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "content_types": {"video": 2, "image": 1},
                "quality_metrics": {"high": 1, "medium": 2, "low": 0},
                "avg_relevance": 0.7,
            },
            visual_assessment={
                "has_visual_content": True,
                "has_thumbnails": True,
                "visual_coverage": 8,
            },
            technical_findings=["mixed quality"],
            patterns_identified=["educational content"],
            gaps_and_limitations=["could use more recent content"],
            reasoning="analysis complete",
        )

        recommendations = agent._generate_recommendations(
            sample_report_request, thinking_phase
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should provide actionable recommendations
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 10

    @pytest.mark.ci_fast
    def test_analyze_content_structure_functionality(
        self, agent_with_mocks, sample_search_results
    ):
        """Test content structure analysis method"""
        agent = agent_with_mocks

        content_analysis = agent._analyze_content_structure(sample_search_results)

        assert isinstance(content_analysis, dict)
        assert "total_results" in content_analysis
        assert content_analysis["total_results"] == 3
        assert "content_types" in content_analysis
        assert "content_types" in content_analysis
        assert "video" in content_analysis["content_types"]
        assert "image" in content_analysis["content_types"]

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_process_impl_uses_enrichment_from_input(self, agent_with_mocks):
        """_process_impl consumes enrichment (entities, relationships,
        enhanced_query) directly from DetailedReportInput — the orchestrator
        threads preprocessing outputs onto the input."""
        agent = agent_with_mocks

        from cogniverse_agents.detailed_report_agent import DetailedReportInput

        # Capture the ReportRequest the agent assembles so we can prove the
        # enrichment fields were actually folded in (not merely accepted).
        captured = {}
        real_generate_report = agent._generate_report

        async def _spy_generate_report(request):
            captured["request"] = request
            return await real_generate_report(request)

        agent._generate_report = _spy_generate_report
        agent.call_dspy = AsyncMock(
            return_value=Mock(executive_summary="enriched report for AI")
        )

        input_data = DetailedReportInput(
            tenant_id="test_tenant",
            query="test query",
            search_results=[{"title": "AI video", "content_type": "video"}],
            enhanced_query="test query enhanced",
            entities=[{"text": "AI", "type": "topic"}],
            relationships=[{"type": "semantic", "entities": ["AI", "technology"]}],
            include_visual_analysis=False,
            report_type="comprehensive",
        )

        result = await agent._process_impl(input_data)

        # enhanced_query overrides the raw query on the downstream request.
        req = captured["request"]
        assert req.query == "test query enhanced"
        # entities + relationships are merged into the request context verbatim.
        assert req.context["entities"] == [{"text": "AI", "type": "topic"}]
        assert req.context["relationships"] == [
            {"type": "semantic", "entities": ["AI", "technology"]}
        ]
        # The DSPy executive summary flows through to the output.
        assert result.executive_summary == "enriched report for AI"


@pytest.mark.unit
class TestDetailedReportDepsConfiguration:
    """Deps-level knobs (thinking_enabled, max_report_length,
    technical_analysis_enabled) change behavior."""

    _NEUTRAL_CONTENT_ANALYSIS = {
        "total_results": 1,
        "content_types": {},
        "duration_distribution": {"short": 0, "medium": 0, "long": 0},
        "quality_metrics": {"high": 0, "medium": 0, "low": 0},
        "avg_relevance": 0.0,
    }

    def _make_agent(self, deps: DetailedReportDeps) -> DetailedReportAgent:
        with (
            patch("cogniverse_agents.detailed_report_agent.VLMInterface"),
            patch.object(DetailedReportAgent, "_initialize_vlm_client"),
        ):
            agent = DetailedReportAgent(deps=deps, config_manager=Mock())
            agent._llm_config = _GATEWAY_TEST_ENDPOINT
            return agent

    @staticmethod
    def _request(include_technical_details: bool = True) -> ReportRequest:
        return ReportRequest(
            query="AI overview",
            search_results=[
                {
                    "id": "1",
                    "title": "AI Demo",
                    "content_type": "video",
                    "score": 0.8,
                    "duration": 120,
                }
            ],
            report_type="comprehensive",
            include_visual_analysis=False,
            include_technical_details=include_technical_details,
        )

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_thinking_disabled_skips_thinking_phase(self):
        """thinking_enabled=False bypasses _thinking_phase entirely and carries
        a neutral ThinkingPhase (empty analyses, zeroed metrics) to the result."""
        agent = self._make_agent(DetailedReportDeps(thinking_enabled=False))
        agent.call_dspy = AsyncMock(
            return_value=Mock(executive_summary="Report without thinking.")
        )

        with patch.object(
            DetailedReportAgent, "_thinking_phase", autospec=True
        ) as mock_think:
            result = await agent._generate_report(self._request())

        mock_think.assert_not_called()
        assert result.executive_summary == "Report without thinking."
        assert result.thinking_phase.content_analysis == self._NEUTRAL_CONTENT_ANALYSIS
        assert result.thinking_phase.visual_assessment == {
            "has_visual_content": False,
            "visual_elements": {"thumbnails": 0, "keyframes": 0, "images": 0},
            "visual_coverage": 0,
            "visual_analysis_feasible": False,
        }
        assert result.thinking_phase.technical_findings == []
        assert result.thinking_phase.patterns_identified == []
        assert result.thinking_phase.gaps_and_limitations == []
        assert result.thinking_phase.reasoning == ""
        # Downstream sections still assemble from the neutral phase.
        assert result.recommendations == [
            "Consider refining the search query to improve result relevance",
            "Expand result set to identify more meaningful patterns",
        ]
        assert [d["category"] for d in result.technical_details] == [
            "Content Distribution",
            "Quality Metrics",
        ]

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_thinking_enabled_invokes_thinking_phase(self):
        """thinking_enabled=True (default) awaits _thinking_phase exactly once
        and carries its result through verbatim."""
        agent = self._make_agent(DetailedReportDeps(thinking_enabled=True))
        agent.call_dspy = AsyncMock(
            return_value=Mock(executive_summary="Report with thinking.")
        )

        thinking = ThinkingPhase(
            content_analysis={
                "total_results": 1,
                "content_types": {"video": 1},
                "duration_distribution": {"short": 0, "medium": 1, "long": 0},
                "quality_metrics": {"high": 0, "medium": 1, "low": 0},
                "avg_relevance": 0.8,
            },
            visual_assessment={"has_visual_content": False},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning="one video result",
        )
        with patch.object(
            DetailedReportAgent,
            "_thinking_phase",
            autospec=True,
            return_value=thinking,
        ) as mock_think:
            result = await agent._generate_report(self._request())

        mock_think.assert_awaited_once()
        assert result.executive_summary == "Report with thinking."
        assert result.thinking_phase is thinking

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_max_report_length_truncates_at_word_boundary(self):
        """An over-long executive summary is truncated at the last word
        boundary within max_report_length, with an ellipsis appended."""
        agent = self._make_agent(DetailedReportDeps(max_report_length=50))
        long_summary = ("word " * 40).strip()  # 199 chars
        agent.call_dspy = AsyncMock(return_value=Mock(executive_summary=long_summary))

        result = await agent._generate_report(self._request())

        # text[:50] ends mid-boundary after the 10th "word"; the cut lands on
        # the last space, keeping 10 whole words plus the ellipsis.
        assert result.executive_summary == "word " * 9 + "word…"
        assert len(result.executive_summary) == 50

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_max_report_length_passes_short_summary_verbatim(self):
        """An executive summary within max_report_length is returned untouched."""
        agent = self._make_agent(DetailedReportDeps(max_report_length=50))
        agent.call_dspy = AsyncMock(
            return_value=Mock(executive_summary="Fits in the limit.")
        )

        result = await agent._generate_report(self._request())

        assert result.executive_summary == "Fits in the limit."

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_technical_analysis_disabled_overrides_request_flag(self):
        """technical_analysis_enabled=False suppresses the technical section
        even when the request asks for it, and metadata reports it disabled."""
        agent = self._make_agent(DetailedReportDeps(technical_analysis_enabled=False))
        agent.call_dspy = AsyncMock(return_value=Mock(executive_summary="Report."))

        result = await agent._generate_report(
            self._request(include_technical_details=True)
        )

        assert result.technical_details == []
        assert result.metadata["technical_analysis_enabled"] is False

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_technical_analysis_enabled_with_request_flag_produces_section(self):
        """technical_analysis_enabled=True + include_technical_details=True
        produces the two technical subsections."""
        agent = self._make_agent(DetailedReportDeps(technical_analysis_enabled=True))
        agent.call_dspy = AsyncMock(return_value=Mock(executive_summary="Report."))

        result = await agent._generate_report(
            self._request(include_technical_details=True)
        )

        assert [d["category"] for d in result.technical_details] == [
            "Content Distribution",
            "Quality Metrics",
        ]
        assert result.technical_details[0]["metrics"] == {"video": 1}
        assert result.technical_details[1]["metrics"] == {
            "high": 0,
            "medium": 1,
            "low": 0,
        }
        assert result.metadata["technical_analysis_enabled"] is True


class TestEnforceMaxLength:
    """_enforce_max_length must reject a non-positive bound like its summarizer
    sibling, instead of returning a one-char "…" for max_length=0."""

    def test_enforce_max_length_zero_raises(self):
        with pytest.raises(ValueError, match=r"max_length must be positive, got 0"):
            DetailedReportAgent._enforce_max_length("some text", 0)

    def test_enforce_max_length_negative_raises(self):
        with pytest.raises(ValueError, match=r"max_length must be positive, got -5"):
            DetailedReportAgent._enforce_max_length("some text", -5)

    def test_enforce_max_length_truncates_at_word_boundary(self):
        out = DetailedReportAgent._enforce_max_length("alpha beta gamma", 9)
        assert out == "alpha…"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
