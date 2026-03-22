"""
Unit tests for DeepResearchAgent.

Tests:
1. Task decomposition produces sub-questions
2. Parallel search dispatches for each sub-question
3. Evidence evaluation determines sufficiency
4. Synthesis produces structured report
5. Iteration loop respects max_iterations
6. Citations extracted from evidence
7. Full process() pipeline with mock DSPy
"""

from unittest.mock import MagicMock

import pytest

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchDeps,
    DeepResearchInput,
    DeepResearchOutput,
)


@pytest.fixture
def mock_search_fn():
    async def search(query: str, tenant_id: str):
        return [
            {"document_id": f"doc-{hash(query) % 100}", "description": f"Result for: {query}"}
        ]
    return search


@pytest.fixture
def mock_dspy_modules():
    """Mock all DSPy modules to avoid needing a real LM."""
    decompose_result = MagicMock()
    decompose_result.sub_questions = [
        "What are the key visual features?",
        "What audio content is present?",
        "What temporal patterns exist?",
    ]

    eval_sufficient = MagicMock()
    eval_sufficient.has_sufficient_evidence = True
    eval_sufficient.gaps = []
    eval_sufficient.confidence = 0.85

    eval_insufficient = MagicMock()
    eval_insufficient.has_sufficient_evidence = False
    eval_insufficient.gaps = ["What temporal patterns exist?"]
    eval_insufficient.confidence = 0.4

    synth_result = MagicMock()
    synth_result.summary = "Research synthesis: visual features include X, audio content is Y."

    return {
        "decompose": decompose_result,
        "eval_sufficient": eval_sufficient,
        "eval_insufficient": eval_insufficient,
        "synthesis": synth_result,
    }


class TestDeepResearchAgentDecomposition:
    @pytest.mark.asyncio
    async def test_decompose_produces_sub_questions(self, mock_search_fn, mock_dspy_modules):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=mock_search_fn)

        agent._decomposer = MagicMock()
        agent._decomposer.forward = MagicMock(return_value=mock_dspy_modules["decompose"])

        result = await agent._decompose("Analyze video content patterns")
        assert len(result) == 3
        assert "visual features" in result[0].lower()

    @pytest.mark.asyncio
    async def test_decompose_handles_string_response(self, mock_search_fn):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=mock_search_fn)

        string_result = MagicMock()
        string_result.sub_questions = "Question 1\nQuestion 2\nQuestion 3"
        agent._decomposer = MagicMock()
        agent._decomposer.forward = MagicMock(return_value=string_result)

        result = await agent._decompose("test query")
        assert len(result) == 3


class TestDeepResearchAgentSearch:
    @pytest.mark.asyncio
    async def test_parallel_search_dispatches(self, mock_search_fn):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=mock_search_fn)

        evidence = await agent._search_parallel(
            ["q1", "q2", "q3"], tenant_id="test"
        )
        assert len(evidence) == 3
        assert all(e["source"] == "search" for e in evidence)
        assert all(len(e["results"]) > 0 for e in evidence)

    @pytest.mark.asyncio
    async def test_search_without_fn_raises(self):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=None)

        with pytest.raises(ValueError, match="no search_fn provided"):
            await agent._search_parallel(["q1"], tenant_id="test")


class TestDeepResearchAgentCitations:
    def test_extract_citations_from_results(self):
        evidence = [
            {
                "question": "What visual features?",
                "results": [
                    {"document_id": "doc-1", "description": "Visual analysis"},
                    {"video_id": "vid-2", "transcript": "Audio content"},
                ],
            }
        ]
        citations = DeepResearchAgent._extract_citations(evidence)
        assert len(citations) == 2
        assert citations[0]["source"] == "doc-1"
        assert citations[1]["source"] == "vid-2"

    def test_extract_citations_empty_results(self):
        evidence = [{"question": "q", "results": []}]
        citations = DeepResearchAgent._extract_citations(evidence)
        assert citations == []


class TestDeepResearchAgentPipeline:
    @pytest.mark.asyncio
    async def test_full_process_with_mocked_dspy(
        self, mock_search_fn, mock_dspy_modules
    ):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=mock_search_fn)

        agent._decomposer = MagicMock()
        agent._decomposer.forward = MagicMock(return_value=mock_dspy_modules["decompose"])

        agent._evaluator = MagicMock()
        agent._evaluator.forward = MagicMock(return_value=mock_dspy_modules["eval_sufficient"])

        agent._synthesizer = MagicMock()
        agent._synthesizer.forward = MagicMock(return_value=mock_dspy_modules["synthesis"])

        result = await agent.process(
            DeepResearchInput(query="Analyze video content", tenant_id="test")
        )

        assert isinstance(result, DeepResearchOutput)
        assert len(result.sub_questions) == 3
        assert result.iterations_used >= 1
        assert len(result.evidence) > 0
        assert "synthesis" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_iterates_on_insufficient_evidence(
        self, mock_search_fn, mock_dspy_modules
    ):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=mock_search_fn)

        agent._decomposer = MagicMock()
        agent._decomposer.forward = MagicMock(return_value=mock_dspy_modules["decompose"])

        call_count = 0

        def eval_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return mock_dspy_modules["eval_insufficient"]
            return mock_dspy_modules["eval_sufficient"]

        agent._evaluator = MagicMock()
        agent._evaluator.forward = MagicMock(side_effect=eval_side_effect)

        agent._synthesizer = MagicMock()
        agent._synthesizer.forward = MagicMock(return_value=mock_dspy_modules["synthesis"])

        result = await agent.process(
            DeepResearchInput(
                query="Complex analysis", tenant_id="test", max_iterations=3
            )
        )

        assert result.iterations_used == 2

    @pytest.mark.asyncio
    async def test_respects_max_iterations(
        self, mock_search_fn, mock_dspy_modules
    ):
        deps = DeepResearchDeps(tenant_id="test")
        agent = DeepResearchAgent(deps=deps, search_fn=mock_search_fn)

        agent._decomposer = MagicMock()
        agent._decomposer.forward = MagicMock(return_value=mock_dspy_modules["decompose"])

        agent._evaluator = MagicMock()
        agent._evaluator.forward = MagicMock(
            return_value=mock_dspy_modules["eval_insufficient"]
        )

        agent._synthesizer = MagicMock()
        agent._synthesizer.forward = MagicMock(return_value=mock_dspy_modules["synthesis"])

        result = await agent.process(
            DeepResearchInput(
                query="Hard question", tenant_id="test", max_iterations=2
            )
        )

        assert result.iterations_used == 2
        assert len(result.gaps_remaining) > 0


class TestDeepResearchOutputModel:
    def test_output_serializes(self):
        output = DeepResearchOutput(
            summary="Test summary",
            sub_questions=["q1", "q2"],
            evidence=[{"question": "q1", "results": []}],
            citations=[{"source": "doc-1", "text": "evidence"}],
            iterations_used=1,
            confidence=0.8,
        )
        d = output.model_dump()
        assert d["summary"] == "Test summary"
        assert d["iterations_used"] == 1
        assert d["confidence"] == 0.8
