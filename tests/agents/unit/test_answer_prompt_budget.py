"""Grounding hits reach an answer LM only as far as its context window allows.

The summarizer, the detailed report and the deep research synthesis each put
every grounding hit into one prompt field. The field keeps the leading hits
that fit the window of the endpoint the request is sent to, most relevant
first, and cuts the next one to fit. Requests are observed on the wire: a
student endpoint publishes its window and counts on ``/tokenize`` (four
tokens per message plus one per word), and a router endpoint receives the
completion request.
"""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import Mock, patch

import dspy
import pytest

from cogniverse_agents.deep_research_agent import DeepResearchAgent, DeepResearchDeps
from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
    ReportRequest,
)
from cogniverse_agents.detailed_report_agent import (
    ThinkingPhase as ReportThinking,
)
from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummarizerDeps,
    SummaryRequest,
    ThinkingPhase,
)
from cogniverse_core.agents.base import PromptLines
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.token_budget import (
    ContextWindowUnavailableError,
    TokenCountUnavailableError,
    clear_context_window_memo,
    line_cut_notice,
)
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.dspy import LenientJSONAdapter
from tests.utils.llm_endpoint_stub import serve_llm_endpoint

pytestmark = [pytest.mark.unit]

STUDENT = "openai/google/gemma-4-e4b-it"
TENANT = "acme:prod"
# The window and completion reservation of the student the summarizer
# overflowed in production: 8192 - 1000 leaves 7192 prompt tokens, less the
# routed margin of 32.
WINDOW = 8192
RESERVED = 1000
ALLOWANCE = WINDOW - RESERVED - 32

SUMMARY_REPLY = json.dumps(
    {
        "reasoning": "r",
        "summary": "The harbour clips agree.",
        "key_points": "tugboats",
        "confidence_score": "0.9",
    }
)
REPORT_REPLY = json.dumps(
    {
        "reasoning": "r",
        "executive_summary": "Report.",
        "key_findings": "f",
        "recommendations": "r",
        "confidence_score": "0.9",
    }
)
SYNTHESIS_REPLY = json.dumps({"reasoning": "r", "summary": "Synthesis."})

_FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ## |\Z)", re.DOTALL)


def _word_count(messages: list[dict]) -> int:
    return sum(4 + len(str(message["content"]).split()) for message in messages)


def _field(request: dict, name: str) -> str:
    user = request["messages"][-1]["content"]
    return dict(_FIELD.findall(user))[name]


def _hit(index: int, words: int) -> dict:
    """A flattened grounding hit whose description is ``words`` words."""
    return {
        "id": f"doc{index}",
        "title": f"wiki page {index}",
        "content_type": "wiki",
        "score": 0.9 - index / 100,
        "description": " ".join(f"p{index:02d}w{word:03d}" for word in range(words)),
    }


@pytest.fixture(autouse=True)
def _fresh_windows():
    clear_context_window_memo()
    yield
    clear_context_window_memo()


@pytest.fixture
def endpoints(request):
    reply = getattr(request, "param", SUMMARY_REPLY)
    student = serve_llm_endpoint(context_window=WINDOW)
    router = serve_llm_endpoint(reply=reply)
    yield student, router
    student.close()
    router.close()


def _routed(student_api_base: str, router_api_base: str, call_site: str, tier="free"):
    return create_routed_lm(
        endpoint=LLMEndpointConfig(
            model=STUDENT,
            api_base=student_api_base,
            max_tokens=RESERVED,
            num_retries=0,
        ),
        config=SemanticRouterConfig(enabled=True, semantic_router_url=router_api_base),
        tenant_id=TENANT,
        tier=tier,
        call_site=call_site,
    )


def _summarizer() -> SummarizerAgent:
    with (
        patch("cogniverse_agents.summarizer_agent.VLMInterface"),
        patch.object(SummarizerAgent, "_initialize_vlm_client"),
    ):
        agent = SummarizerAgent(
            deps=SummarizerDeps(multimodal_generation_enabled=False),
            config_manager=Mock(),
        )
    return agent


def _thinking(themes=("wiki_content",)) -> ThinkingPhase:
    return ThinkingPhase(
        key_themes=list(themes),
        content_categories=["text"],
        relevance_scores={},
        visual_elements=[],
        reasoning="",
    )


async def _summarize(agent, hits, lm):
    request = SummaryRequest(
        query="summarize what machine learning is", search_results=hits
    )
    with dspy.context(lm=lm, adapter=LenientJSONAdapter()):
        return await agent._generate_comprehensive_summary(
            request, hits, _thinking(), [], []
        )


def _summary_line(hit: dict) -> str:
    return (
        f"- {hit['title']} (wiki, relevance {hit['score']:.2f}): {hit['description']}"
    )


# Nine hits of 680 words and one of 787: the whole prompt the summarizer
# builds from them is 7193 tokens, one past the 7192 an 8192 window leaves
# after 1000 reserved.
OVERFLOWING_HITS = [_hit(index, 680) for index in range(9)] + [_hit(9, 787)]


class TestTheSummaryFitsTheWindowItIsSentTo:
    async def test_the_overflowing_prompt_is_7193_tokens_when_every_hit_fits(self):
        student = serve_llm_endpoint(context_window=2 * WINDOW)
        router = serve_llm_endpoint(reply=SUMMARY_REPLY)
        try:
            await _summarize(
                _summarizer(),
                OVERFLOWING_HITS,
                _routed(student.api_base, router.api_base, "summarizer_agent"),
            )
        finally:
            student.close()
            router.close()

        [sent] = router.posted("/v1/chat/completions")
        assert _field(sent, "content") == "\n".join(
            [*map(_summary_line, OVERFLOWING_HITS), "Key themes: wiki_content"]
        )
        assert _word_count(sent["messages"]) == 7193

    async def test_the_7193_token_prompt_now_fits_the_8192_token_window(
        self, endpoints
    ):
        student, router = endpoints
        summary, _ = await _summarize(
            _summarizer(),
            OVERFLOWING_HITS,
            _routed(student.api_base, router.api_base, "summarizer_agent"),
        )

        [sent] = router.posted("/v1/chat/completions")
        # Nine hits whole, then 5950 of the tenth's 6333 characters with the
        # fifteen-word notice: exactly the 7160 tokens the window allows.
        tenth = _summary_line(OVERFLOWING_HITS[9])
        assert len(tenth) == 6333
        assert _field(sent, "content") == "\n".join(
            [
                *map(_summary_line, OVERFLOWING_HITS[:9]),
                tenth[:5950] + line_cut_notice(5950, 6333),
                "Key themes: wiki_content",
            ]
        )
        assert _word_count(sent["messages"]) == ALLOWANCE
        assert sent["max_tokens"] == RESERVED
        assert summary == "The harbour clips agree."

    async def test_hits_within_the_window_reach_the_lm_whole(self, endpoints):
        student, router = endpoints
        hits = [_hit(index, 40) for index in range(3)]
        await _summarize(
            _summarizer(),
            hits,
            _routed(student.api_base, router.api_base, "summarizer_agent"),
        )

        [sent] = router.posted("/v1/chat/completions")
        assert _field(sent, "content") == "\n".join(
            [*map(_summary_line, hits), "Key themes: wiki_content"]
        )
        # One count, with the whole prompt.
        assert len(student.posted("/tokenize")) == 1

    async def test_concurrent_summaries_each_send_their_own_fitted_hits(
        self, endpoints
    ):
        student, router = endpoints
        agent = _summarizer()
        lm = _routed(student.api_base, router.api_base, "summarizer_agent")
        corpora = {
            run: [
                {**_hit(index, 700), "title": f"run {run} page {index}"}
                for index in range(10)
            ]
            for run in range(6)
        }

        await asyncio.gather(
            *(_summarize(agent, hits, lm) for hits in corpora.values())
        )

        sent = {
            _field(request, "content").split(" page ", 1)[0]: request
            for request in router.posted("/v1/chat/completions")
        }
        assert sorted(sent) == [f"- run {run}" for run in range(6)]
        for run, hits in corpora.items():
            request = sent[f"- run {run}"]
            content = _field(request, "content")
            assert content.startswith("\n".join(_summary_line(hit) for hit in hits[:8]))
            assert all(
                f"run {other} " not in content for other in corpora if other != run
            )
            assert _word_count(request["messages"]) <= ALLOWANCE

    async def test_a_dispatched_prompt_overlay_is_counted_in_the_fit(self, endpoints):
        student, router = endpoints
        agent = _summarizer()
        instructions = "Answer only from the results. " * 200
        agent.get_dispatched_prompts = lambda: {"summarizer": instructions}
        await _summarize(
            agent,
            OVERFLOWING_HITS,
            _routed(student.api_base, router.api_base, "summarizer_agent"),
        )

        [sent] = router.posted("/v1/chat/completions")
        # A thousand words of overlay instructions leave room for one hit fewer.
        ninth = _summary_line(OVERFLOWING_HITS[8])
        assert len(ninth) == 5477
        assert instructions.strip() in sent["messages"][0]["content"]
        assert _field(sent, "content") == "\n".join(
            [
                *map(_summary_line, OVERFLOWING_HITS[:8]),
                ninth[:3494] + line_cut_notice(3494, 5477),
                "Key themes: wiki_content",
            ]
        )
        assert _word_count(sent["messages"]) == ALLOWANCE


class TestTheSummaryBudgetFaultContract:
    async def test_a_student_that_cannot_count_sends_nothing(self):
        student = serve_llm_endpoint(context_window=WINDOW)
        router = serve_llm_endpoint(reply=SUMMARY_REPLY)
        dead = student.api_base
        student.close()
        lm = create_routed_lm(
            endpoint=LLMEndpointConfig(
                model=STUDENT,
                api_base=dead,
                max_tokens=RESERVED,
                num_retries=0,
                context_window=WINDOW,
            ),
            config=SemanticRouterConfig(
                enabled=True, semantic_router_url=router.api_base
            ),
            tenant_id=TENANT,
            tier="free",
            call_site="summarizer_agent",
        )
        try:
            with pytest.raises(TokenCountUnavailableError) as raised:
                await _summarize(_summarizer(), OVERFLOWING_HITS, lm)
        finally:
            router.close()

        assert str(raised.value) == (
            f"{dead.removesuffix('/v1')}/tokenize is unreachable, so the "
            f"request cannot be counted with the tokenizer {STUDENT} is served with"
        )
        assert router.received == []

    async def test_an_lm_whose_serving_endpoints_are_unknown_sends_nothing(
        self, endpoints
    ):
        student, router = endpoints
        # A pro call on the free-form entry with no pro endpoint known.
        lm = _routed(student.api_base, router.api_base, "summarizer_agent", "pro")

        with pytest.raises(ContextWindowUnavailableError) as raised:
            await _summarize(_summarizer(), OVERFLOWING_HITS, lm)

        assert str(raised.value) == (
            "'openai/auto' does not know the context window of the endpoints "
            "serving it, so the SummarizerAgent content input cannot be fitted"
        )
        assert router.received == []

    async def test_two_fitted_inputs_in_one_call_are_refused(self, endpoints):
        student, router = endpoints
        agent = _summarizer()
        lm = _routed(student.api_base, router.api_base, "summarizer_agent")

        with dspy.context(lm=lm, adapter=LenientJSONAdapter()):
            with pytest.raises(ValueError) as raised:
                await agent.call_dspy(
                    agent.summarization_module,
                    output_field="summary",
                    content=PromptLines(["a"]),
                    query=PromptLines(["b"]),
                    summary_type="brief",
                    keyframes=[],
                )

        assert str(raised.value) == (
            "only one input is fitted per call, got ['content', 'query']"
        )
        assert (student.received, router.received) == ([], [])


def _report_line(hit: dict) -> str:
    return f"- {hit['title']} (wiki, score: {hit['score']:.2f}): {hit['description']}"


class TestTheReportFitsTheWindowItIsSentTo:
    @pytest.mark.parametrize("endpoints", [REPORT_REPLY], indirect=True)
    async def test_an_overflowing_report_keeps_its_leading_results(self, endpoints):
        student, router = endpoints
        with (
            patch("cogniverse_agents.detailed_report_agent.VLMInterface"),
            patch.object(DetailedReportAgent, "_initialize_vlm_client"),
        ):
            agent = DetailedReportAgent(
                deps=DetailedReportDeps(multimodal_generation_enabled=False),
                config_manager=Mock(),
            )
        hits = OVERFLOWING_HITS
        request = ReportRequest(
            query="summarize what machine learning is", search_results=hits
        )
        thinking = ReportThinking(
            content_analysis={"total_results": len(hits)},
            visual_assessment={},
            technical_findings=[],
            patterns_identified=[],
            gaps_and_limitations=[],
            reasoning="",
        )
        with dspy.context(
            lm=_routed(student.api_base, router.api_base, "detailed_report_agent"),
            adapter=LenientJSONAdapter(),
        ):
            summary, _ = await agent._generate_executive_summary(request, thinking)

        [sent] = router.posted("/v1/chat/completions")
        tenth = _report_line(hits[9])
        assert len(tenth) == 6330
        assert _field(sent, "content") == "\n".join(
            [
                "Total Results: 10",
                *map(_report_line, hits[:9]),
                tenth[:5843] + line_cut_notice(5843, 6330),
            ]
        )
        assert _word_count(sent["messages"]) == ALLOWANCE
        assert summary == "Report."


class TestTheSynthesisCoversEverySubQuestionFirst:
    @pytest.mark.parametrize("endpoints", [SYNTHESIS_REPLY], indirect=True)
    async def test_each_question_s_first_hit_comes_before_any_second(self, endpoints):
        student, router = endpoints
        agent = DeepResearchAgent(
            deps=DeepResearchDeps(
                tenant_id=TENANT, multimodal_generation_enabled=False
            ),
            config_manager=Mock(),
        )
        evidence = [
            {
                "question": "what is a tugboat",
                "results": [_hit(0, 1800), _hit(1, 1800)],
            },
            {"question": "what is a barge", "results": [_hit(2, 1800), _hit(3, 1800)]},
            {"question": "what is a pier", "results": []},
        ]
        with dspy.context(
            lm=_routed(student.api_base, router.api_base, "deep_research_agent"),
            adapter=LenientJSONAdapter(),
        ):
            summary = await agent._synthesize("ships", evidence)

        def line(question: str, hit: dict) -> str:
            return (
                f"[{question}] {hit['title']} (wiki, relevance {hit['score']:.2f}): "
                f"{hit['description']}"
            )

        [sent] = router.posted("/v1/chat/completions")
        lines = [
            line("what is a tugboat", _hit(0, 1800)),
            line("what is a barge", _hit(2, 1800)),
            "[what is a pier] No results",
            line("what is a tugboat", _hit(1, 1800)),
            line("what is a barge", _hit(3, 1800)),
        ]
        assert agent._evidence_lines(evidence) == lines
        # Both questions' first hits, the empty question and the tugboat's
        # second hit whole, then 12842 of the barge's second hit.
        assert len(lines[4]) == 15253
        assert _field(sent, "evidence") == "\n".join(
            [*lines[:4], lines[4][:12842] + line_cut_notice(12842, 15253)]
        )
        assert _word_count(sent["messages"]) == ALLOWANCE
        assert summary == "Synthesis."
