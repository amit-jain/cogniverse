"""
Real integration tests for QueryEnhancementModule with real LLM inference.

Tests verify the DSPy-powered query enhancement actually expands and
enriches queries — not that the class initializes without error.
"""

import logging

import dspy
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration]


# Runtime gate via the requires_lm marker — see
# tests/agents/integration/conftest.py (an import-time skipif latches the
# pre-session-fixture endpoint state).
from tests.agents.integration.conftest import skip_if_no_lm  # noqa: F401


@pytest.fixture(scope="module")
def dspy_lm(ensure_host_ollama):
    """Module-scoped DSPy LM on the session's provisioned primary endpoint."""
    import json
    import os
    from pathlib import Path

    config_path = Path(os.environ["COGNIVERSE_CONFIG"])
    with open(config_path) as f:
        config = json.load(f)
    primary = config.get("llm_config", {}).get("primary", {})
    model = primary.get("model")
    api_base = primary.get("api_base")

    extra_body = None
    if model and ("qwen3" in model or "qwen-3" in model):
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=api_base,
        temperature=0.0,
        max_tokens=1000,
        extra_body=extra_body,
    )
    return create_dspy_lm(endpoint)


@pytest.fixture
def enhancement_module(dspy_lm, caplog):
    """QueryEnhancementModule (DSPy module, not the full A2A agent) on the
    session LM with the runtime's adapter; every test reaches the LM, so a
    fallback for a failed call fails the test."""
    from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule
    from cogniverse_foundation.dspy import LenientJSONAdapter

    with (
        caplog.at_level(
            logging.WARNING, logger="cogniverse_agents.query_enhancement_agent"
        ),
        dspy.context(lm=dspy_lm, adapter=LenientJSONAdapter()),
    ):
        yield QueryEnhancementModule()
    assert [
        record.getMessage()
        for record in caplog.records
        if "reason=DSPy failure" in record.getMessage()
    ] == []


@skip_if_no_lm
def test_enhances_short_query(enhancement_module):
    """A single-word query is either rewritten by the LM or, when the LM
    echoes it, searched exactly as asked: never padded with words that change
    it."""
    result = enhancement_module.forward(query="cats")

    enhanced = result.enhanced_query
    assert (result.path_used, enhanced == "cats") in {
        ("lm", False),
        ("heuristic_fallback", True),
    }, (result.path_used, enhanced)
    assert "cat" in enhanced.lower(), enhanced


@skip_if_no_lm
def test_preserves_intent(enhancement_module):
    """Enhancement of 'machine learning tutorials' must keep ML semantics."""
    result = enhancement_module.forward(query="machine learning tutorials")

    enhanced = result.enhanced_query.lower()
    expansion = result.expansion_terms.lower()
    synonyms = result.synonyms.lower()

    all_output = f"{enhanced} {expansion} {synonyms}"

    ml_terms = [
        "machine learning",
        "ml",
        "deep learning",
        "neural",
        "algorithm",
        "model",
        "training",
        "learning",
    ]
    matched = [t for t in ml_terms if t in all_output]
    assert matched, (
        f"Enhanced output does not preserve ML intent. "
        f"enhanced_query={result.enhanced_query!r}, "
        f"expansion_terms={result.expansion_terms!r}, "
        f"synonyms={result.synonyms!r}"
    )


@skip_if_no_lm
def test_holdout_scoring_sends_one_request_per_distinct_input(
    enhancement_module, dspy_lm, monkeypatch
):
    """A served holdout repeats the same call; scoring it reaches the LM once
    per distinct input, as many times as scoring the distinct inputs alone."""
    from cogniverse_runtime.optimization_cli import (
        _query_enhancement_example,
        _query_enhancement_scores,
    )

    def _row(query: str, source_text: str) -> dict:
        return {
            "query": query,
            "source_text": source_text,
            "grounding_context": "",
            "enhanced_query": "",
            "expansion_terms": [],
            "synonyms": [],
            "context": [],
            "confidence": 0.0,
        }

    robots = _query_enhancement_example(
        _row(
            "find robots",
            "A humanoid robot assembles car parts on a factory floor while "
            "an engineer calibrates its servo motors.",
        )
    )
    reefs = _query_enhancement_example(
        _row(
            "coral reef footage",
            "Divers film bleached coral reefs and schools of parrotfish near "
            "the Great Barrier Reef.",
        )
    )
    requests = []
    send = dspy_lm.forward

    def _counted(*args, **kwargs):
        live_turn = kwargs["messages"][-1]["content"]
        requests.append(live_turn.split("\n")[1])
        return send(*args, **kwargs)

    monkeypatch.setattr(dspy_lm, "forward", _counted)

    _, distinct_rows = _query_enhancement_scores(enhancement_module, [robots, reefs])
    distinct_requests = list(requests)
    requests.clear()
    _, repeated_rows = _query_enhancement_scores(
        enhancement_module, [robots] * 4 + [reefs] * 2
    )

    assert (distinct_rows, repeated_rows) == (2, 6)
    assert requests == distinct_requests == ["find robots", "coral reef footage"]
