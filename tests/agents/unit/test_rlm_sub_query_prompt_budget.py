"""An RLM sub-query prompt is bounded to the window of the endpoint answering it.

The prompt ``llm_query`` receives is built by the generated program and can
carry the whole context variable. Requests are observed on the wire: the
serving endpoint counts, the router receives the completion request.
"""

from __future__ import annotations

import threading

import dspy
import pytest

from cogniverse_agents.inference.tolerant_interpreter import TolerantRLM
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.token_budget import (
    ContextWindowEndpointUnreachableError,
    TokenCountUnavailableError,
    clear_context_window_memo,
    cut_notice,
)
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from tests.utils.llm_endpoint_stub import serve_llm_endpoint

pytestmark = [pytest.mark.unit]

STUDENT = "openai/google/gemma-4-e4b-it"
# 232 served - 100 reserved - 32 routed margin = 100 prompt tokens. The stub
# counts 4 template tokens plus one per word, and the cut notice is 15 words,
# so 81 five-character words (405 characters) are the most that fit.
WINDOW = 232
RESERVED = 100
KEPT = 405


def _text(tag: str, words: int = 400) -> str:
    """``words`` five-character words unique to ``tag``."""
    return f"{tag[:4]:_<4} " * words


@pytest.fixture(autouse=True)
def _fresh_windows():
    clear_context_window_memo()
    yield
    clear_context_window_memo()


@pytest.fixture
def endpoints():
    serving = serve_llm_endpoint(context_window=WINDOW)
    router = serve_llm_endpoint()
    yield serving, router
    serving.close()
    router.close()


def _lm(serving_api_base: str, router_api_base: str, call_site: str, **endpoint):
    return create_routed_lm(
        endpoint=LLMEndpointConfig(
            model=STUDENT,
            api_base=serving_api_base,
            max_tokens=RESERVED,
            num_retries=0,
            **endpoint,
        ),
        config=SemanticRouterConfig(enabled=True, semantic_router_url=router_api_base),
        tenant_id="acme:prod",
        tier="pro",
        call_site=call_site,
    )


def _tools():
    return TolerantRLM("context, query -> answer", max_iterations=1)._make_llm_tools()


def _sent(router) -> list[str]:
    return [
        body["messages"][0]["content"] for body in router.posted("/v1/chat/completions")
    ]


class TestSubQueryPromptIsBounded:
    def test_an_overflowing_prompt_is_cut_before_it_is_sent(self, endpoints):
        serving, router = endpoints
        text = _text("over")
        with dspy.context(
            lm=_lm(serving.api_base, router.api_base, "orchestrator_agent")
        ):
            answer = _tools()["llm_query"](text)

        assert answer == "ok"
        assert len(text) == 2000
        assert _sent(router) == [text[:KEPT] + cut_notice(KEPT, 2000)]
        request = router.posted("/v1/chat/completions")[0]
        assert request["max_tokens"] == RESERVED
        assert request["model"] == "cogniverse-classification"
        # The router is never asked to count.
        assert router.posted("/tokenize") == []
        assert len(serving.posted("/tokenize")) == 11

    def test_a_prompt_inside_the_allowance_is_sent_byte_identical(self, endpoints):
        serving, router = endpoints
        text = "Summarize: the robot arm lifts a red block.\n\tKeep 'quotes'."
        with dspy.context(
            lm=_lm(serving.api_base, router.api_base, "orchestrator_agent")
        ):
            _tools()["llm_query"](text)

        assert _sent(router) == [text]
        assert serving.posted("/tokenize") == []

    def test_each_prompt_of_a_batch_is_bounded_on_its_own(self, endpoints):
        serving, router = endpoints
        small = "batch: name the colour of the block."
        big = _text("many")
        with dspy.context(
            lm=_lm(serving.api_base, router.api_base, "orchestrator_agent")
        ):
            answers = _tools()["llm_query_batched"]([small, big])

        assert answers == ["ok", "ok"]
        assert sorted(_sent(router)) == sorted(
            [small, big[:KEPT] + cut_notice(KEPT, 2000)]
        )

    def test_an_lm_whose_upstream_the_router_picks_sends_the_prompt_as_built(
        self, endpoints
    ):
        serving, router = endpoints
        text = _text("auto")
        with dspy.context(
            lm=_lm(serving.api_base, router.api_base, "summarizer_agent")
        ):
            _tools()["llm_query"](text)

        assert _sent(router) == [text]
        assert serving.received == []

    def test_concurrent_sub_queries_each_send_their_own_cut_prompt(self, endpoints):
        serving, router = endpoints
        lm = _lm(serving.api_base, router.api_base, "orchestrator_agent")
        threads = 8
        barrier = threading.Barrier(threads)
        texts = {index: _text(f"c{index}") for index in range(threads)}
        answers: dict[int, str] = {}

        def worker(index: int) -> None:
            with dspy.context(lm=lm):
                tools = _tools()
                barrier.wait()
                answers[index] = tools["llm_query"](texts[index])

        pool = [
            threading.Thread(target=worker, args=(index,)) for index in range(threads)
        ]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join()

        assert answers == {index: "ok" for index in range(threads)}
        assert sorted(_sent(router)) == sorted(
            text[:KEPT] + cut_notice(KEPT, 2000) for text in texts.values()
        )
        # Eight first touches read the served window once.
        assert serving.received.count(("/v1/models", None)) == 1


class TestSubQueryBudgetFaultContract:
    def test_a_tokenizer_that_cannot_count_stops_the_sub_query(self):
        serving = serve_llm_endpoint(context_window=WINDOW)
        router = serve_llm_endpoint()
        dead = serving.api_base
        serving.close()
        lm = _lm(dead, router.api_base, "orchestrator_agent", context_window=WINDOW)
        try:
            with dspy.context(lm=lm):
                with pytest.raises(TokenCountUnavailableError) as raised:
                    _tools()["llm_query"](_text("down"))
        finally:
            router.close()

        assert str(raised.value) == (
            f"{dead.removesuffix('/v1')}/tokenize is unreachable, so the "
            f"request cannot be counted with the tokenizer {STUDENT} is served with"
        )
        assert router.received == []

    def test_an_unknown_window_stops_the_sub_query(self):
        serving = serve_llm_endpoint(context_window=WINDOW)
        router = serve_llm_endpoint()
        dead = serving.api_base
        serving.close()
        lm = _lm(dead, router.api_base, "orchestrator_agent")
        try:
            with dspy.context(lm=lm):
                with pytest.raises(ContextWindowEndpointUnreachableError) as raised:
                    _tools()["llm_query"]("short prompt")
        finally:
            router.close()

        assert str(raised.value) == (
            f"{STUDENT} at {dead} is unreachable and declares no context "
            "window, so no request budget can be derived"
        )
        assert router.received == []
