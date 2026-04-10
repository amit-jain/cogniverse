"""Unit tests for GatewayAgent."""

from unittest.mock import MagicMock, Mock

import pytest

from cogniverse_agents.gateway_agent import (
    ALL_LABELS,
    GENERATION_LABELS,
    MODALITY_LABELS,
    SIMPLE_ROUTE_MAP,
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
    GatewayOutput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entity(text: str, label: str, score: float) -> dict:
    """Helper to build a GLiNER-style entity dict."""
    return {"text": text, "label": label, "score": score}


@pytest.fixture
def mock_gliner_model():
    """A mock GLiNER model whose predict_entities can be configured per-test."""
    model = MagicMock()
    model.predict_entities.return_value = []
    return model


def _make_gateway(**kwargs) -> GatewayAgent:
    deps = GatewayDeps(**kwargs)
    return GatewayAgent(deps=deps)


@pytest.fixture
def gateway_agent(mock_gliner_model):
    """GatewayAgent with a pre-injected mock GLiNER model (no real download)."""
    agent = _make_gateway()
    agent._gliner_model = mock_gliner_model
    return agent


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestGatewayAgentInit:
    def test_initialization(self):
        """Agent initializes with default deps."""
        agent = _make_gateway()
        assert agent.agent_name == "gateway_agent"
        assert "gateway" in agent.capabilities
        assert "classification" in agent.capabilities
        assert agent._gliner_model is None  # lazy-loaded

    def test_custom_deps(self):
        deps = GatewayDeps(
            gliner_model_name="custom/model",
            gliner_threshold=0.5,
            fast_path_confidence_threshold=0.9,
        )
        agent = GatewayAgent(deps=deps)
        assert agent.deps.gliner_model_name == "custom/model"
        assert agent.deps.gliner_threshold == 0.5
        assert agent.deps.fast_path_confidence_threshold == 0.9


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------


class TestGatewayModels:
    def test_input_requires_query(self):
        with pytest.raises(Exception):
            GatewayInput()  # type: ignore[call-arg]

    def test_input_with_query(self):
        inp = GatewayInput(query="find me a video")
        assert inp.query == "find me a video"
        assert inp.tenant_id is None

    def test_input_with_tenant(self):
        inp = GatewayInput(query="hello", tenant_id="acme")
        assert inp.tenant_id == "acme"

    def test_output_fields(self):
        out = GatewayOutput(
            query="test",
            complexity="simple",
            modality="video",
            generation_type="raw_results",
            routed_to="search_agent",
            confidence=0.95,
            reasoning="test reasoning",
        )
        assert out.complexity == "simple"
        assert out.modality == "video"
        assert out.generation_type == "raw_results"
        assert out.routed_to == "search_agent"
        assert out.confidence == 0.95

    def test_output_rejects_invalid_complexity(self):
        with pytest.raises(Exception):
            GatewayOutput(
                query="q",
                complexity="medium",  # type: ignore[arg-type]
                modality="video",
                generation_type="raw_results",
                routed_to="x",
                confidence=0.5,
                reasoning="r",
            )


# ---------------------------------------------------------------------------
# Label / route map coverage
# ---------------------------------------------------------------------------


class TestLabelMappings:
    def test_all_labels_flattened(self):
        """ALL_LABELS contains every label from modality + generation dicts."""
        expected = set()
        for labels in MODALITY_LABELS.values():
            expected.update(labels)
        for labels in GENERATION_LABELS.values():
            expected.update(labels)
        assert set(ALL_LABELS) == expected

    def test_route_map_covers_all_modality_gen_combos(self):
        """Every (modality, gen_type) pair has a route."""
        modalities = list(MODALITY_LABELS.keys())
        gen_types = ["raw_results", "summary", "detailed_report"]
        for mod in modalities:
            for gt in gen_types:
                assert (mod, gt) in SIMPLE_ROUTE_MAP, (
                    f"Missing route for ({mod}, {gt})"
                )


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


class TestClassification:
    def test_classify_modality_video(self, gateway_agent):
        entities = [_make_entity("video", "video_content", 0.9)]
        modality, conf = gateway_agent._classify_modality(entities)
        assert modality == "video"
        assert conf == 0.9

    def test_classify_modality_no_entities(self, gateway_agent):
        modality, conf = gateway_agent._classify_modality([])
        assert modality == "video"  # default
        assert conf == 0.0

    def test_classify_modality_multiple_returns_both(self, gateway_agent):
        entities = [
            _make_entity("video", "video_content", 0.8),
            _make_entity("document", "document_content", 0.7),
        ]
        modality, conf = gateway_agent._classify_modality(entities)
        assert modality == "both"
        assert conf == 0.8

    def test_classify_generation_type_default(self, gateway_agent):
        gen_type, conf = gateway_agent._classify_generation_type([])
        assert gen_type == "raw_results"
        assert conf == 1.0

    def test_classify_generation_type_summary(self, gateway_agent):
        entities = [_make_entity("summarize", "summary_request", 0.85)]
        gen_type, conf = gateway_agent._classify_generation_type(entities)
        assert gen_type == "summary"
        assert conf == 0.85

    def test_classify_generation_type_detailed_report(self, gateway_agent):
        entities = [_make_entity("report", "detailed_report_request", 0.75)]
        gen_type, conf = gateway_agent._classify_generation_type(entities)
        assert gen_type == "detailed_report"
        assert conf == 0.75


class TestComplexityDecision:
    def test_no_entities_is_always_complex(self, gateway_agent):
        """No entities → always complex, regardless of query content."""
        assert gateway_agent._is_complex("analyze the data", "video", "raw_results", [], 0.0) is True
        assert gateway_agent._is_complex("find videos", "video", "raw_results", [], 0.0) is True
        assert gateway_agent._is_complex("hello", "video", "raw_results", [], 0.0) is True

    def test_low_confidence_is_complex(self, gateway_agent):
        entities = [_make_entity("vid", "video_content", 0.35)]
        assert gateway_agent._is_complex("find videos", "video", "raw_results", entities, 0.35) is True

    def test_both_modality_is_complex(self, gateway_agent):
        entities = [_make_entity("vid", "video_content", 0.9)]
        assert gateway_agent._is_complex("find stuff", "both", "raw_results", entities, 0.9) is True

    def test_high_confidence_single_modality_is_simple(self, gateway_agent):
        entities = [_make_entity("vid", "video_content", 0.9)]
        assert gateway_agent._is_complex("find videos", "video", "raw_results", entities, 0.9) is False

    def test_detailed_report_is_always_complex(self, gateway_agent):
        """detailed_report generation type always requires orchestration."""
        entities = [_make_entity("doc", "document_content", 0.9)]
        assert gateway_agent._is_complex("find docs", "document", "detailed_report", entities, 0.9) is True

    def test_analysis_keyword_is_complex(self, gateway_agent):
        """Queries with analysis/synthesis verbs need orchestration."""
        entities = [_make_entity("vid", "video_content", 0.9)]
        assert gateway_agent._is_complex("analyze the video trends", "video", "raw_results", entities, 0.9) is True

    def test_multi_step_marker_is_complex(self, gateway_agent):
        """Multi-step queries need orchestration."""
        entities = [_make_entity("vid", "video_content", 0.9)]
        assert gateway_agent._is_complex("find videos then summarize them", "video", "raw_results", entities, 0.9) is True

    def test_compound_query_is_complex(self, gateway_agent):
        """Queries with many clauses need orchestration."""
        entities = [_make_entity("vid", "video_content", 0.9)]
        query = "find videos about cats and dogs and birds and fish"
        assert gateway_agent._is_complex(query, "video", "raw_results", entities, 0.9) is True

    def test_simple_search_is_not_complex(self, gateway_agent):
        """Plain search query with good confidence is simple."""
        entities = [_make_entity("vid", "video_content", 0.9)]
        assert gateway_agent._is_complex("find cat videos", "video", "raw_results", entities, 0.9) is False


# ---------------------------------------------------------------------------
# End-to-end _process_impl
# ---------------------------------------------------------------------------


class TestProcessImpl:
    @pytest.mark.asyncio
    async def test_simple_video_query(self, gateway_agent, mock_gliner_model):
        """Video query with high-confidence entity -> search_agent."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "cooking videos", "label": "video_content", "score": 0.92},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="Show me cooking videos")
        )
        assert result.complexity == "simple"
        assert result.modality == "video"
        assert result.generation_type == "raw_results"
        assert result.routed_to == "search_agent"
        assert result.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_complex_multimodal_query(self, gateway_agent, mock_gliner_model):
        """Multi-modal entities -> orchestrator_agent."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content", "score": 0.85},
            {"text": "documents", "label": "document_content", "score": 0.80},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="Compare the video with the PDF document")
        )
        assert result.complexity == "complex"
        assert result.modality == "both"
        assert result.routed_to == "orchestrator_agent"

    @pytest.mark.asyncio
    async def test_no_entities_is_always_complex(self, gateway_agent, mock_gliner_model):
        """No entities → always complex, regardless of query."""
        mock_gliner_model.predict_entities.return_value = []
        result = await gateway_agent._process_impl(
            GatewayInput(query="hello world")
        )
        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"

    @pytest.mark.asyncio
    async def test_low_confidence_complex(self, gateway_agent, mock_gliner_model):
        """Low-confidence entity -> complex."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "something", "label": "video_content", "score": 0.35},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="something vague")
        )
        assert result.complexity == "complex"
        assert result.routed_to == "orchestrator_agent"

    @pytest.mark.asyncio
    async def test_summary_request(self, gateway_agent, mock_gliner_model):
        """Summary generation type with simple query -> summarizer_agent."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.90},
            {"text": "summary", "label": "summary_request", "score": 0.88},
        ]
        # "brief of the cooking video" — no analysis keywords, single modality
        result = await gateway_agent._process_impl(
            GatewayInput(query="brief of the cooking video")
        )
        assert result.complexity == "simple"
        assert result.generation_type == "summary"
        assert result.routed_to == "summarizer_agent"

    @pytest.mark.asyncio
    async def test_image_search(self, gateway_agent, mock_gliner_model):
        """Image modality raw_results -> image_search_agent."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "photo", "label": "image_content", "score": 0.91},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="Find photos of cats")
        )
        assert result.complexity == "simple"
        assert result.modality == "image"
        assert result.routed_to == "image_search_agent"

    @pytest.mark.asyncio
    async def test_audio_analysis(self, gateway_agent, mock_gliner_model):
        """Audio modality -> audio_analysis_agent."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "podcast", "label": "audio_content", "score": 0.87},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="Find podcast episodes about AI")
        )
        assert result.complexity == "simple"
        assert result.modality == "audio"
        assert result.routed_to == "audio_analysis_agent"

    @pytest.mark.asyncio
    async def test_document_detailed_report_is_complex(self, gateway_agent, mock_gliner_model):
        """Document + detailed_report -> always complex (needs search→analyze→report)."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "spreadsheet", "label": "document_content", "score": 0.82},
            {"text": "report", "label": "detailed_report_request", "score": 0.78},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="full writeup of the spreadsheet data")
        )
        assert result.complexity == "complex"
        assert result.generation_type == "detailed_report"
        assert result.routed_to == "orchestrator_agent"
        assert "detailed report" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_tenant_id_passthrough(self, gateway_agent, mock_gliner_model):
        """tenant_id from input is used in span emission."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.92},
        ]
        result = await gateway_agent._process_impl(
            GatewayInput(query="find videos", tenant_id="acme")
        )
        assert result.query == "find videos"


# ---------------------------------------------------------------------------
# Telemetry span emission
# ---------------------------------------------------------------------------


class TestTelemetrySpan:
    @pytest.mark.asyncio
    async def test_span_emitted(self, gateway_agent, mock_gliner_model):
        """When telemetry_manager is set, _emit_gateway_span calls span()."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.92},
        ]
        mock_tm = MagicMock()
        mock_span = MagicMock()
        mock_tm.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tm.span.return_value.__exit__ = Mock(return_value=False)
        gateway_agent.telemetry_manager = mock_tm

        await gateway_agent._process_impl(
            GatewayInput(query="cooking videos", tenant_id="acme")
        )

        mock_tm.span.assert_called_once()
        call_kwargs = mock_tm.span.call_args
        assert call_kwargs[0][0] == "cogniverse.gateway"
        assert call_kwargs[1]["tenant_id"] == "acme"
        attrs = call_kwargs[1]["attributes"]
        assert attrs["gateway.query"] == "cooking videos"
        assert attrs["gateway.complexity"] == "simple"
        assert attrs["gateway.modality"] == "video"
        assert attrs["gateway.routed_to"] == "search_agent"

    @pytest.mark.asyncio
    async def test_no_telemetry_manager(self, gateway_agent, mock_gliner_model):
        """No telemetry_manager -> no error, no span."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.92},
        ]
        gateway_agent.telemetry_manager = None
        result = await gateway_agent._process_impl(
            GatewayInput(query="cooking videos")
        )
        assert result.routed_to == "search_agent"

    @pytest.mark.asyncio
    async def test_default_tenant_id_in_span(self, gateway_agent, mock_gliner_model):
        """When tenant_id is None, span uses 'default'."""
        mock_gliner_model.predict_entities.return_value = [
            {"text": "video", "label": "video_content", "score": 0.92},
        ]
        mock_tm = MagicMock()
        mock_tm.span.return_value.__enter__ = Mock(return_value=MagicMock())
        mock_tm.span.return_value.__exit__ = Mock(return_value=False)
        gateway_agent.telemetry_manager = mock_tm

        await gateway_agent._process_impl(
            GatewayInput(query="videos", tenant_id=None)
        )

        call_kwargs = mock_tm.span.call_args
        assert call_kwargs[1]["tenant_id"] == "default"
