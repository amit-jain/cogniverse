"""The search query rewrite is a bounded, single-field transformation.

Every grounded search runs this rewrite before retrieval, so its cost is paid
on the live path once per search. These pins hold the served module to the
shape and the latency that make that affordable: one output field, an output
ceiling the served prompt cannot exceed, and a p90 measured against the real
serving endpoint.
"""

from __future__ import annotations

import statistics
import time

import dspy
import pytest

from cogniverse_agents.search_agent import (
    QUERY_REWRITE_MAX_OUTPUT_TOKENS,
    QUERY_REWRITE_P90_BUDGET_S,
    SEARCH_STAGE_TIMING_KEYS,
    SearchOptimizationModule,
    SearchOptimizationSignature,
)
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

pytestmark = [pytest.mark.integration, pytest.mark.requires_lm, pytest.mark.benchmark]

# Runtime gate via the requires_lm marker - see
# tests/agents/integration/conftest.py (an import-time skipif latches the
# pre-session-fixture endpoint state).
from tests.agents.integration.conftest import skip_if_no_lm  # noqa: E402,F401

# Fixed, hardest-known inputs drawn from the shipped evaluation corpus:
# short keyword, long natural-language question, cross-modal lateral probe.
FIXED_QUERIES = (
    "lifting",
    "man wearing",
    "What is the man doing in the video and what is he wearing?",
    "What happens after the biker rides towards the middle of the dirt field?",
    "Can you describe the video in detail?",
    "video where the activity demonstrated requires the same equipment as snowboarding",
)

SAMPLES_PER_QUERY = 3


def _percentile(values, p):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@pytest.fixture(scope="module")
def serving_lm():
    """The exact model this deployment serves, bound with the cache off.

    Resolved through ``ensure_llm`` - the same path every other real-LM
    integration test in this directory takes - so the budget below is pinned
    against the endpoint that answers a live search. A cached rewrite would
    measure the cache, not the rewrite.
    """
    from tests.utils.hermetic_llm import MODEL, ensure_llm

    base = ensure_llm()
    if base is None:
        pytest.fail(
            "the serving LM did not resolve; the rewrite latency pins measure "
            "the real endpoint and cannot run without it"
        )

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    lm = create_dspy_lm(
        LLMEndpointConfig(
            model=f"openai/{MODEL}",
            api_base=base,
            temperature=0.1,
            max_tokens=QUERY_REWRITE_MAX_OUTPUT_TOKENS,
        )
    )
    lm.cache = False
    return lm


class TestRewriteShape:
    def test_signature_emits_only_the_rewritten_query(self):
        """One output field. Every extra field is generated tokens on the
        live path that nothing downstream reads."""
        assert set(SearchOptimizationSignature.output_fields) == {"enhanced_query"}

    def test_served_predictor_generates_no_reasoning(self):
        """A ChainOfThought emits a reasoning field ahead of the answer; the
        rewrite is a transformation and must not pay for one."""
        module = SearchOptimizationModule()
        predictor = module.search_optimizer
        assert type(predictor) is dspy.Predict
        assert set(predictor.signature.output_fields) == {"enhanced_query"}

    def test_stage_timing_keys_are_exactly_these(self):
        assert SEARCH_STAGE_TIMING_KEYS == (
            "search.stage.context_injection_ms",
            "search.stage.query_rewrite_lm_ms",
            "search.stage.query_rewrite_ms",
            "search.stage.retrieval_ms",
        )


class TestRewriteLatency:
    """Measured against the real serving endpoint, cache off."""

    @pytest.fixture(scope="class")
    def samples(self, serving_lm):
        rows = []
        with dspy.context(lm=serving_lm):
            module = SearchOptimizationModule()
            for query in FIXED_QUERIES:
                for _ in range(SAMPLES_PER_QUERY):
                    before = len(serving_lm.history)
                    start = time.perf_counter()
                    prediction = module(query=query, modality="video", top_k=10)
                    elapsed = time.perf_counter() - start
                    usage = {}
                    if len(serving_lm.history) > before:
                        usage = serving_lm.history[-1].get("usage") or {}
                    rows.append(
                        {
                            "query": query,
                            "seconds": elapsed,
                            "completion_tokens": usage.get("completion_tokens"),
                            "enhanced_query": str(prediction.enhanced_query or ""),
                        }
                    )
        return rows

    def test_p90_within_measured_budget(self, samples):
        seconds = [row["seconds"] for row in samples]
        p90 = _percentile(seconds, 0.90)
        assert p90 <= QUERY_REWRITE_P90_BUDGET_S, (
            f"rewrite p90 {p90:.3f}s exceeds the {QUERY_REWRITE_P90_BUDGET_S}s "
            f"budget; samples={[round(s, 3) for s in sorted(seconds)]}"
        )

    def test_every_sample_stays_under_the_output_ceiling(self, samples):
        over = [
            (row["query"], row["completion_tokens"])
            for row in samples
            if (row["completion_tokens"] or 0) > QUERY_REWRITE_MAX_OUTPUT_TOKENS
        ]
        assert over == []

    def test_every_rewrite_is_a_non_empty_single_line_query(self, samples):
        """The rewrite feeds a retrieval encoder: a blank or multi-field blob
        searches the wrong thing."""
        bad = [
            (row["query"], row["enhanced_query"])
            for row in samples
            if not row["enhanced_query"].strip()
            or "\n\n" in row["enhanced_query"].strip()
        ]
        assert bad == []

    def test_median_output_is_smaller_than_the_ceiling(self, samples):
        """The ceiling is a bound, not the operating point: a median that sits
        at the ceiling means the model is being truncated, not bounded."""
        tokens = [
            row["completion_tokens"] for row in samples if row["completion_tokens"]
        ]
        assert statistics.median(tokens) < QUERY_REWRITE_MAX_OUTPUT_TOKENS
