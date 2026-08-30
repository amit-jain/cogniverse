"""``LearnedReranker`` hands litellm the real inference bearer for ``api_base``.

An OpenAI-compatible reranker behind ``reranking.api_base`` was called with
``api_base`` alone, so a Modal-hosted server answered 401 and every rerank
raised.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from litellm.types.rerank import RerankResponse

from cogniverse_agents.search.learned_reranker import LearnedReranker
from cogniverse_agents.search.types import RerankerSearchResult

pytestmark = pytest.mark.unit

MODAL = "https://amit-jain--cogniverse-reranker-inference.modal.run/v1"
MODEL = "openai/bge-reranker-v2-m3"


def _reranker(api_base: str | None) -> LearnedReranker:
    with patch(
        "cogniverse_agents.search.learned_reranker.get_config_value",
        return_value={"api_base": api_base, "top_n": 2, "max_results_to_rerank": 100},
    ):
        return LearnedReranker(
            model=MODEL, tenant_id="test:unit", config_manager=Mock()
        )


def _results() -> list[RerankerSearchResult]:
    return [
        RerankerSearchResult(
            id="doc-1",
            title="Machine Learning",
            content="Introduction to ML",
            modality="text",
            score=0.8,
            metadata={},
        ),
        RerankerSearchResult(
            id="doc-2",
            title="Deep Learning",
            content="Neural networks guide",
            modality="text",
            score=0.7,
            metadata={},
        ),
    ]


@pytest.mark.asyncio
async def test_modal_api_base_hands_litellm_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    reranker = _reranker(MODAL)

    with patch(
        "cogniverse_agents.search.learned_reranker.arerank",
        return_value=RerankResponse(
            id="rr", results=[{"index": 1, "relevance_score": 0.95}]
        ),
    ) as arerank:
        reranked = await reranker.rerank("neural networks", _results())

    assert [r.id for r in reranked] == ["doc-2"]
    assert arerank.call_args.kwargs == {
        "model": MODEL,
        "query": "neural networks",
        "documents": [
            "Machine Learning Introduction to ML",
            "Deep Learning Neural networks guide",
        ],
        "top_n": 2,
        "api_base": MODAL,
        "api_key": "real-bearer",
    }


def test_without_an_api_base_litellm_gets_no_endpoint_kwargs(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    reranker = _reranker(None)

    with patch(
        "cogniverse_agents.search.learned_reranker.rerank",
        return_value=RerankResponse(
            id="rr", results=[{"index": 0, "relevance_score": 0.9}]
        ),
    ) as rerank:
        reranker.rerank_sync("intro", _results())

    assert rerank.call_args.kwargs == {
        "model": MODEL,
        "query": "intro",
        "documents": [
            "Machine Learning Introduction to ML",
            "Deep Learning Neural networks guide",
        ],
        "top_n": 2,
    }


def test_modal_api_base_without_a_bearer_fails_before_the_call(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    reranker = _reranker(MODAL)

    with (
        patch("cogniverse_agents.search.learned_reranker.rerank") as rerank,
        pytest.raises(
            RuntimeError,
            match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
        ),
    ):
        reranker.rerank_sync("intro", _results())

    assert rerank.call_count == 0
