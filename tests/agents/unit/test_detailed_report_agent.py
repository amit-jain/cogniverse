"""Tests for DetailedReportAgent with VLM integration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.app.agents.detailed_report_agent import (
    DetailedReportAgent, VLMInterface, ReportRequest, 
    ThinkingPhase, ReportResult
)
from src.tools.a2a_utils import A2AMessage, DataPart, Task


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
            "content_type": "video"
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
            "content_type": "video"
        },
        {
            "id": "3",
            "title": "Sample Image 1",
            "description": "Description 3",
            "score": 0.6,
            "image_path": "/path/to/image1.jpg",
            "content_type": "image"
        }
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
        max_results_to_analyze=10
    )


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "openai_api_key": "test_key",
        "anthropic_api_key": None
    }


class TestVLMInterface:
    """Test VLM interface functionality"""
    
    @patch('builtins.__import__')
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_vlm_interface_initialization_mock(self, mock_get_config, mock_import):
        """Test VLM interface initialization with mock client"""
        mock_get_config.return_value = {}
        mock_import.side_effect = ImportError("No API clients available")
        
        vlm = VLMInterface()
        
        assert vlm.client_type == "mock"
        assert vlm.client is None
    
    @patch('builtins.__import__')
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_vlm_interface_initialization_openai(self, mock_get_config, mock_import):
        """Test VLM interface initialization with OpenAI"""
        mock_get_config.return_value = {"openai_api_key": "test_key"}
        
        # Mock OpenAI module and client
        mock_openai_module = Mock()
        mock_client = Mock()
        mock_openai_module.OpenAI.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == "openai":
                return mock_openai_module
            return Mock()
        
        mock_import.side_effect = import_side_effect
        
        vlm = VLMInterface()
        
        assert vlm.client_type == "openai"
        assert vlm.client == mock_client
    
    @patch('builtins.__import__')
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_vlm_interface_initialization_anthropic(self, mock_get_config, mock_import):
        """Test VLM interface initialization with Anthropic"""
        mock_get_config.return_value = {"anthropic_api_key": "test_key"}
        
        # Mock Anthropic module and client
        mock_anthropic_module = Mock()
        mock_client = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        def import_side_effect(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("OpenAI not available")
            elif name == "anthropic":
                return mock_anthropic_module
            return Mock()
        
        mock_import.side_effect = import_side_effect
        
        vlm = VLMInterface()
        
        assert vlm.client_type == "anthropic"
        assert vlm.client == mock_client
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_analyze_visual_content_detailed_mock(self, mock_get_config):
        """Test detailed visual content analysis with mock client"""
        mock_get_config.return_value = {}
        vlm = VLMInterface()
        
        result = await vlm.analyze_visual_content_detailed(["/path/to/image.jpg"], "test query")
        
        assert "detailed_descriptions" in result
        assert "technical_analysis" in result
        assert "visual_patterns" in result
        assert "quality_assessment" in result
        assert "annotations" in result
        assert len(result["detailed_descriptions"]) == 1


class TestDetailedReportAgent:
    """Test DetailedReportAgent functionality"""
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_detailed_report_agent_initialization(self, mock_get_config, mock_config):
        """Test agent initialization"""
        mock_get_config.return_value = mock_config
        
        agent = DetailedReportAgent()
        
        assert agent.max_report_length == 2000
        assert agent.thinking_enabled is True
        assert agent.visual_analysis_enabled is True
        assert agent.technical_analysis_enabled is True
        assert agent.vlm is not None
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_detailed_report_agent_custom_config(self, mock_get_config, mock_config):
        """Test agent initialization with custom configuration"""
        mock_get_config.return_value = mock_config
        
        agent = DetailedReportAgent(
            max_report_length=1000,
            thinking_enabled=False,
            visual_analysis_enabled=False,
            technical_analysis_enabled=False
        )
        
        assert agent.max_report_length == 1000
        assert agent.thinking_enabled is False
        assert agent.visual_analysis_enabled is False
        assert agent.technical_analysis_enabled is False
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_analyze_content_structure(self, mock_get_config, mock_config, sample_search_results):
        """Test content structure analysis"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        analysis = agent._analyze_content_structure(sample_search_results)
        
        assert analysis["total_results"] == 3
        assert "content_types" in analysis
        assert "duration_distribution" in analysis
        assert "quality_metrics" in analysis
        assert "avg_relevance" in analysis
        assert analysis["content_types"]["video"] == 2
        assert analysis["content_types"]["image"] == 1
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_assess_visual_content(self, mock_get_config, mock_config, sample_search_results):
        """Test visual content assessment"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        assessment = agent._assess_visual_content(sample_search_results)
        
        assert assessment["has_visual_content"] is True
        assert assessment["visual_elements"]["thumbnails"] == 2
        assert assessment["visual_elements"]["images"] == 1
        assert assessment["visual_coverage"] == 3
        assert assessment["visual_analysis_feasible"] is True
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_identify_technical_findings(self, mock_get_config, mock_config, sample_search_results):
        """Test technical findings identification"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        findings = agent._identify_technical_findings(sample_search_results)
        
        assert len(findings) >= 2  # Should find format and resolution info
        assert any("mp4" in finding for finding in findings)
        assert any("1080p" in finding or "720p" in finding for finding in findings)
    
    @patch('src.app.agents.detailed_report_agent.get_config') 
    def test_identify_patterns(self, mock_get_config, mock_config, sample_search_results):
        """Test pattern identification"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        patterns = agent._identify_patterns(sample_search_results)
        
        assert isinstance(patterns, list)
        # Should identify some patterns based on the sample data
        assert len(patterns) >= 1
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_identify_gaps_and_limitations(self, mock_get_config, mock_config, sample_search_results):
        """Test gaps and limitations identification"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        gaps = agent._identify_gaps_and_limitations(sample_search_results)
        
        assert isinstance(gaps, list)
        # Should identify missing timestamp information
        assert any("temporal" in gap.lower() for gap in gaps)
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_extract_common_terms(self, mock_get_config, mock_config):
        """Test common terms extraction"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        texts = ["Sample Video Content", "Video Sample Data", "Content Analysis Video"]
        terms = agent._extract_common_terms(texts)
        
        assert "video" in terms
        assert "sample" in terms
        assert "content" in terms
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_thinking_phase(self, mock_get_config, mock_config, sample_report_request):
        """Test thinking phase generation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = await agent._thinking_phase(sample_report_request)
        
        assert isinstance(thinking_phase, ThinkingPhase)
        assert thinking_phase.content_analysis["total_results"] == 3
        assert thinking_phase.visual_assessment["has_visual_content"] is True
        assert isinstance(thinking_phase.technical_findings, list)
        assert isinstance(thinking_phase.patterns_identified, list)
        assert isinstance(thinking_phase.gaps_and_limitations, list)
        assert thinking_phase.reasoning
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_perform_visual_analysis_disabled(self, mock_get_config, mock_config, sample_report_request):
        """Test visual analysis when disabled"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent(visual_analysis_enabled=False)
        
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={"has_visual_content": True},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        sample_report_request.include_visual_analysis = False
        result = await agent._perform_visual_analysis(sample_report_request, thinking_phase)
        
        assert result == []
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_perform_visual_analysis_no_content(self, mock_get_config, mock_config, sample_report_request):
        """Test visual analysis with no visual content"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={"has_visual_content": False},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        result = await agent._perform_visual_analysis(sample_report_request, thinking_phase)
        
        assert result == []
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_perform_visual_analysis_with_content(self, mock_get_config, mock_config, sample_report_request):
        """Test visual analysis with visual content"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={"has_visual_content": True},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning="test reasoning"
        )
        
        with patch.object(agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "detailed_descriptions": ["Description 1", "Description 2"],
                "technical_analysis": ["Tech 1", "Tech 2"], 
                "quality_assessment": {"overall": 0.8}
            }
            
            result = await agent._perform_visual_analysis(sample_report_request, thinking_phase)
            
            assert len(result) == 2
            assert result[0]["detailed_description"] == "Description 1"
            assert result[0]["technical_assessment"] == "Tech 1"
            assert result[0]["quality_score"] == 0.8
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, mock_get_config, mock_config, sample_report_request):
        """Test executive summary generation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "avg_relevance": 0.75,
                "content_types": {"video": 2, "image": 1},
                "duration_distribution": {"short": 1, "medium": 2, "long": 0}
            },
            visual_assessment={"has_visual_content": True},
            technical_findings=["Finding 1", "Finding 2"],
            patterns_identified=["Pattern 1", "Pattern 2"],
            gaps_and_limitations=["Gap 1"],
            reasoning="test reasoning"
        )
        
        summary = await agent._generate_executive_summary(sample_report_request, thinking_phase)
        
        assert "test query" in summary
        assert "3" in summary  # total results
        assert "0.75" in summary or "0.8" in summary  # avg relevance
        assert len(summary) > 100  # Should be substantial
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_generate_executive_summary_no_results(self, mock_get_config, mock_config, sample_report_request):
        """Test executive summary with no results"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={"total_results": 0, "avg_relevance": 0, "content_types": {}, "duration_distribution": {}},
            visual_assessment={"has_visual_content": False},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        summary = await agent._generate_executive_summary(sample_report_request, thinking_phase)
        
        assert "No relevant results found" in summary
        assert "test query" in summary
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_generate_detailed_findings(self, mock_get_config, mock_config, sample_report_request):
        """Test detailed findings generation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "content_types": {"video": 2, "image": 1},
                "quality_metrics": {"high": 2, "medium": 1, "low": 0}
            },
            visual_assessment={
                "has_visual_content": True,
                "visual_coverage": 15,
                "visual_elements": {"thumbnails": 2, "images": 1}
            },
            technical_findings=["Finding 1"],
            patterns_identified=["Pattern 1", "Pattern 2"],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        findings = agent._generate_detailed_findings(sample_report_request, thinking_phase)
        
        assert len(findings) >= 3  # Should have content, quality, visual findings
        assert any(f["category"] == "Content Composition" for f in findings)
        assert any(f["category"] == "Quality Assessment" for f in findings)
        assert any(f["category"] == "Visual Content" for f in findings)
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_generate_technical_details(self, mock_get_config, mock_config, sample_report_request):
        """Test technical details generation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "quality_metrics": {"high": 2, "medium": 1, "low": 0},
                "avg_relevance": 0.75
            },
            visual_assessment={
                "has_visual_content": True,
                "visual_coverage": 3,
                "visual_elements": {"thumbnails": 2, "images": 1}
            },
            technical_findings=["Finding 1", "Finding 2"],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        details = agent._generate_technical_details(sample_report_request, thinking_phase)
        
        assert len(details) >= 2  # Should have system and content metrics
        assert any(d["section"] == "System Analysis" for d in details)
        assert any(d["section"] == "Content Metrics" for d in details)
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_generate_technical_details_disabled(self, mock_get_config, mock_config, sample_report_request):
        """Test technical details when disabled"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent(technical_analysis_enabled=False)
        
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        sample_report_request.include_technical_details = False
        details = agent._generate_technical_details(sample_report_request, thinking_phase)
        
        assert details == []
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_generate_recommendations(self, mock_get_config, mock_config, sample_report_request):
        """Test recommendations generation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 3,
                "quality_metrics": {"high": 0, "medium": 1, "low": 2},  # Low quality ratio
                "content_types": {"video": 3}  # Single content type
            },
            visual_assessment={"has_visual_content": False},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=["Incomplete metadata", "Limited source diversity"],
            reasoning=""
        )
        
        recommendations = agent._generate_recommendations(sample_report_request, thinking_phase)
        
        assert len(recommendations) >= 3  # Should generate multiple recommendations
        assert any("search criteria" in rec.lower() for rec in recommendations)
        assert any("visual content" in rec.lower() for rec in recommendations)
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_generate_recommendations_disabled(self, mock_get_config, mock_config, sample_report_request):
        """Test recommendations when disabled"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        sample_report_request.include_recommendations = False
        recommendations = agent._generate_recommendations(sample_report_request, thinking_phase)
        
        assert recommendations == []
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    def test_calculate_confidence_assessment(self, mock_get_config, mock_config, sample_report_request):
        """Test confidence assessment calculation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={
                "total_results": 10,
                "avg_relevance": 0.8
            },
            visual_assessment={
                "has_visual_content": True,
                "visual_coverage": 5
            },
            technical_findings=["Finding 1", "Finding 2", "Finding 3"],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        confidence = agent._calculate_confidence_assessment(sample_report_request, thinking_phase)
        
        assert "overall" in confidence
        assert "content_analysis" in confidence
        assert "visual_analysis" in confidence
        assert "technical_analysis" in confidence
        assert 0 <= confidence["overall"] <= 1
        assert 0 <= confidence["content_analysis"] <= 1
        assert 0 <= confidence["visual_analysis"] <= 1
        assert 0 <= confidence["technical_analysis"] <= 1
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_generate_report_comprehensive(self, mock_get_config, mock_config, sample_report_request):
        """Test comprehensive report generation"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        with patch.object(agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "detailed_descriptions": ["Description 1"],
                "technical_analysis": ["Tech 1"],
                "quality_assessment": {"overall": 0.8}
            }
            
            result = await agent.generate_report(sample_report_request)
            
            assert isinstance(result, ReportResult)
            assert result.executive_summary
            assert isinstance(result.detailed_findings, list)
            assert isinstance(result.visual_analysis, list)
            assert isinstance(result.technical_details, list)
            assert isinstance(result.recommendations, list)
            assert isinstance(result.confidence_assessment, dict)
            assert result.metadata["results_analyzed"] == 3
            assert result.metadata["report_type"] == "comprehensive"
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_process_a2a_task(self, mock_get_config, mock_config, sample_search_results):
        """Test A2A task processing"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        # Create A2A task
        data_part = DataPart(data={
            "query": "test query",
            "search_results": sample_search_results,
            "report_type": "comprehensive",
            "include_visual_analysis": True
        })
        message = A2AMessage(role="user", parts=[data_part])
        task = Task(id="test_task", messages=[message])
        
        with patch.object(agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "detailed_descriptions": ["Description 1"],
                "technical_analysis": ["Tech 1"],
                "quality_assessment": {"overall": 0.8}
            }
            
            result = await agent.process_a2a_task(task)
            
            assert result["task_id"] == "test_task"
            assert result["status"] == "completed"
            assert "result" in result
            assert "executive_summary" in result["result"]
            assert "detailed_findings" in result["result"]
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_process_a2a_task_no_messages(self, mock_get_config, mock_config):
        """Test A2A task with no messages"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        task = Task(id="test_task", messages=[])
        
        with pytest.raises(Exception) as exc_info:
            await agent.process_a2a_task(task)
        
        assert "No messages in task" in str(exc_info.value)
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_process_a2a_task_no_data_part(self, mock_get_config, mock_config):
        """Test A2A task with no data part"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        message = A2AMessage(role="user", parts=[])
        task = Task(id="test_task", messages=[message])
        
        with pytest.raises(Exception) as exc_info:
            await agent.process_a2a_task(task)
        
        assert "No data in message" in str(exc_info.value)


class TestDetailedReportAgentEdgeCases:
    """Test edge cases for DetailedReportAgent"""
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_generate_report_empty_results(self, mock_get_config, mock_config):
        """Test report generation with empty results"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        request = ReportRequest(
            query="test query",
            search_results=[],
            report_type="comprehensive"
        )
        
        result = await agent.generate_report(request)
        
        assert isinstance(result, ReportResult)
        assert "No relevant results found" in result.executive_summary
        assert result.confidence_assessment["overall"] == 0.0
    
    @patch('src.app.agents.detailed_report_agent.get_config')
    @pytest.mark.asyncio
    async def test_visual_analysis_failure(self, mock_get_config, mock_config, sample_report_request):
        """Test visual analysis failure handling"""
        mock_get_config.return_value = mock_config
        agent = DetailedReportAgent()
        
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={"has_visual_content": True},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        with patch.object(agent.vlm, 'analyze_visual_content_detailed', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = Exception("VLM failure")
            
            result = await agent._perform_visual_analysis(sample_report_request, thinking_phase)
            
            assert len(result) == 1
            assert "Visual analysis unavailable" in result[0]["analysis"]


class TestDataClasses:
    """Test data classes and models"""
    
    def test_report_request_defaults(self):
        """Test ReportRequest with default values"""
        request = ReportRequest(
            query="test",
            search_results=[]
        )
        
        assert request.report_type == "comprehensive"
        assert request.include_visual_analysis is True
        assert request.include_technical_details is True
        assert request.include_recommendations is True
        assert request.max_results_to_analyze == 20
        assert request.context is None
    
    def test_thinking_phase_creation(self):
        """Test ThinkingPhase creation"""
        thinking_phase = ThinkingPhase(
            content_analysis={"test": "value"},
            visual_assessment={"has_visual": True},
            technical_findings=["finding1"],
            patterns_identified=["pattern1"],
            gaps_and_limitations=["gap1"],
            reasoning="test reasoning"
        )
        
        assert thinking_phase.content_analysis["test"] == "value"
        assert thinking_phase.visual_assessment["has_visual"] is True
        assert thinking_phase.technical_findings == ["finding1"]
        assert thinking_phase.reasoning == "test reasoning"
    
    def test_report_result_creation(self):
        """Test ReportResult creation"""
        thinking_phase = ThinkingPhase(
            content_analysis={},
            visual_assessment={},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning=""
        )
        
        result = ReportResult(
            executive_summary="test summary",
            detailed_findings=[{"finding": "test"}],
            visual_analysis=[{"visual": "test"}],
            technical_details=[{"tech": "test"}],
            recommendations=["rec1"],
            confidence_assessment={"overall": 0.8},
            thinking_phase=thinking_phase,
            metadata={"test": "value"}
        )
        
        assert result.executive_summary == "test summary"
        assert len(result.detailed_findings) == 1
        assert len(result.visual_analysis) == 1
        assert len(result.technical_details) == 1
        assert len(result.recommendations) == 1
        assert result.confidence_assessment["overall"] == 0.8
        assert result.metadata["test"] == "value"