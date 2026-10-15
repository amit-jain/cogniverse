"""A summary of real grounding hits fits the window of the real model it is
sent to.

The corpus is ten hits the e2e cluster's tenant answered "summarize what
machine learning is" with, flattened by the dispatcher (wiki pages carry their
page content), in rank order across the profiles searched. Sent whole, the
summary prompt overflows both chat models on Modal: the student's 8192-token
window and the teacher's 4096. The summarizer is bound to each endpoint
directly; the endpoint's own usage report says what reached it.

Re-record the corpus from a runtime with
``python -m tests.agents.integration.test_answer_prompt_budget_real <runtime>``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import dspy
import pytest

from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummarizerDeps,
    SummaryRequest,
    ThinkingPhase,
)
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.token_budget import (
    clear_context_window_memo,
    line_cut_notice,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.dspy import LenientJSONAdapter
from cogniverse_runtime.agent_dispatcher import _HIT_CONTENT_TYPES

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).parent / "fixtures" / "answer_grounding_hits.json"
QUERY = "summarize what machine learning is"
TENANT = "flywheel_org:production"
PROFILES = (
    "wiki_semantic",
    "document_text_semantic",
    "video_colpali_smol500_mv_frame",
    "audio_clap_semantic",
)
STRATEGIES = {"wiki_semantic": "hybrid"}
RECORD_KEYS = {"profile", "id", "title", "content_type", "score", "description"}
RESERVED = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)["llm_config"]["primary"]["max_tokens"]


def _corpus() -> list[dict]:
    return json.loads(CORPUS.read_text())


def _line(hit: dict) -> str:
    return (
        f"- {hit['title']} ({hit['content_type']}, relevance {hit['score']:.2f}): "
        f"{hit['description']}"
    )


@pytest.fixture(autouse=True)
def _fresh_windows():
    clear_context_window_memo()
    yield
    clear_context_window_memo()


def _endpoint(role: str) -> LLMEndpointConfig:
    configured = json.loads(Path(os.environ["COGNIVERSE_CONFIG"]).read_text())[
        "llm_config"
    ][role]
    return LLMEndpointConfig(
        model=configured["model"],
        api_base=configured["api_base"],
        temperature=0.0,
        max_tokens=RESERVED,
        request_timeout=300.0,
        num_retries=1,
    )


async def _summarize(endpoint: LLMEndpointConfig):
    with (
        patch("cogniverse_agents.summarizer_agent.VLMInterface"),
        patch.object(SummarizerAgent, "_initialize_vlm_client"),
    ):
        agent = SummarizerAgent(
            deps=SummarizerDeps(multimodal_generation_enabled=False),
            config_manager=Mock(),
        )
    # The order the summarizer reads them in: three videos, then the wiki
    # index page.
    hits = sorted(_corpus(), key=lambda hit: hit["score"], reverse=True)
    lm = create_dspy_lm(endpoint)
    thinking = ThinkingPhase(
        key_themes=["wiki_content", "document_content", "video_content"],
        content_categories=["text", "video"],
        relevance_scores={},
        visual_elements=[],
        reasoning="",
    )
    with dspy.context(lm=lm, adapter=LenientJSONAdapter()):
        await agent._generate_comprehensive_summary(
            SummaryRequest(query=QUERY, search_results=hits), hits, thinking, [], []
        )
    [call] = lm.history
    return hits, lm, call


def _content(call) -> str:
    user = call["messages"][-1]["content"]
    return user.split("[[ ## content ## ]]\n", 1)[1].split("\n\n[[ ## ", 1)[0]


def _whole(call, hits) -> list[dict]:
    """The request ``call`` would have been with every hit in it."""
    whole = "\n".join(
        [*map(_line, hits), "Key themes: wiki_content, document_content, video_content"]
    )
    messages = [dict(message) for message in call["messages"]]
    messages[-1]["content"] = messages[-1]["content"].replace(_content(call), whole)
    return messages


class TestTheRecordedCorpus:
    def test_the_corpus_is_ten_flattened_hits_of_the_searched_profiles(self):
        corpus = _corpus()
        assert len(corpus) == 10
        assert [set(hit) for hit in corpus] == [RECORD_KEYS] * 10
        assert {hit["profile"] for hit in corpus} == set(PROFILES)
        produced = {kind for _, kind in _HIT_CONTENT_TYPES}
        assert {hit["content_type"] for hit in corpus} == {
            "wiki",
            "document",
            "video",
            "audio",
        }
        assert {hit["content_type"] for hit in corpus} <= produced
        assert [hit["profile"] for hit in corpus[:4]] == [
            "wiki_semantic",
            "document_text_semantic",
            "video_colpali_smol500_mv_frame",
            "audio_clap_semantic",
        ]
        # The wiki index page alone is longer than either window.
        assert max(len(hit["description"]) for hit in corpus) == 88477


@pytest.mark.requires_lm
class TestTheSummaryFitsTheStudent:
    async def test_the_student_answers_a_summary_cut_to_its_window(self):
        endpoint = _endpoint("primary")
        hits, lm, call = await _summarize(endpoint)

        allowance = lm.prompt_allowance()
        assert (allowance.budget.context_window, allowance.tokens) == (
            8192,
            8192 - RESERVED,
        )
        # Whole, the prompt overflows the student.
        assert allowance.count_tokens(_whole(call, hits)) > allowance.tokens
        lines = list(map(_line, hits))
        assert len(lines[3]) == 88540
        assert _content(call) == "\n".join(
            [
                *lines[:3],
                lines[3][:24059] + line_cut_notice(24059, 88540),
                "Key themes: wiki_content, document_content, video_content",
            ]
        )
        # The student counted what it received as within its window.
        assert call["usage"]["prompt_tokens"] <= allowance.tokens
        assert call["usage"]["prompt_tokens"] == allowance.count_tokens(
            call["messages"]
        )


@pytest.mark.requires_teacher_model
class TestTheSummaryFitsTheTeacher:
    async def test_the_teacher_answers_a_summary_cut_to_its_window(self):
        endpoint = _endpoint("teacher")
        hits, lm, call = await _summarize(endpoint)

        allowance = lm.prompt_allowance()
        assert (allowance.budget.context_window, allowance.tokens) == (
            4096,
            4096 - RESERVED,
        )
        assert allowance.count_tokens(_whole(call, hits)) > allowance.tokens
        lines = list(map(_line, hits))
        assert _content(call) == "\n".join(
            [
                *lines[:3],
                lines[3][:8086] + line_cut_notice(8086, 88540),
                "Key themes: wiki_content, document_content, video_content",
            ]
        )
        assert call["usage"]["prompt_tokens"] <= allowance.tokens
        assert call["usage"]["prompt_tokens"] == allowance.count_tokens(
            call["messages"]
        )


def _record(runtime: str) -> None:
    import httpx

    from cogniverse_runtime.agent_dispatcher import _flatten_search_hit

    ranked = {}
    for profile in PROFILES:
        response = httpx.post(
            f"{runtime}/search/",
            json={
                "query": QUERY,
                "profile": profile,
                "strategy": STRATEGIES.get(profile, "default"),
                "top_k": 5,
                "tenant_id": TENANT,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        ranked[profile] = [
            _flatten_search_hit(hit) for hit in response.json()["results"]
        ]
    corpus = [
        {"profile": profile, **{key: hit.get(key) for key in RECORD_KEYS - {"profile"}}}
        for rank in range(5)
        for profile in PROFILES
        if rank < len(ranked[profile])
        for hit in [ranked[profile][rank]]
    ][:10]
    CORPUS.write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _record(sys.argv[1])
