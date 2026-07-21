"""ClaimExtractor must bind the per-tenant LM, not the ambient one."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy

from cogniverse_agents.graph.claim_extractor import ClaimExtractor
from cogniverse_agents.graph.graph_schema import Mention


class _CapturingModule:
    """Records the active ``dspy.settings.lm`` at invocation time."""

    def __init__(self) -> None:
        self.captured_lm: object = None
        self.call_count: int = 0

    def __call__(self, **_):
        self.captured_lm = dspy.settings.lm
        self.call_count += 1
        return MagicMock(claims=[])


def test_per_tenant_lm_wraps_module_call() -> None:
    sentinel = MagicMock(name="per_tenant_lm")
    ambient = MagicMock(name="ambient_global_lm")
    extractor = ClaimExtractor(llm_config=MagicMock())
    extractor._cot_module = _CapturingModule()

    with dspy.context(lm=ambient):
        # Patch where the direct path binds the factory (semantic_router
        # imports the name at module level) — patching llm_factory only
        # works if this test happens to trigger semantic_router's first
        # import, which made the result depend on test order.
        with patch(
            "cogniverse_foundation.config.semantic_router.create_dspy_lm",
            return_value=sentinel,
        ):
            extractor._invoke(
                text="some short text",
                entity_hints=["Alice"],
                modality_hint="text",
                tenant_id="acme",
            )

    assert extractor._cot_module.call_count == 1
    assert extractor._cot_module.captured_lm is sentinel, (
        f"Expected per-tenant LM; got {extractor._cot_module.captured_lm!r}"
    )


def test_no_llm_config_falls_through_to_ambient() -> None:
    ambient = MagicMock(name="ambient_global_lm")
    extractor = ClaimExtractor(llm_config=None)
    extractor._cot_module = _CapturingModule()

    with dspy.context(lm=ambient):
        extractor._invoke(
            text="hi",
            entity_hints=[],
            modality_hint="text",
            tenant_id="acme",
        )

    assert extractor._cot_module.captured_lm is ambient


class _ClaimsModule:
    """Returns a fixed claims list, mimicking a real LM's loose output."""

    def __init__(self, claims: list[dict]) -> None:
        self._claims = claims

    def __call__(self, **_):
        return dspy.Prediction(claims=self._claims)


def _anchor() -> Mention:
    return Mention(
        source_doc_id="doc1",
        segment_id="seg1",
        ts_start=0.0,
        ts_end=1.0,
        modality="text",
        evidence_span="Marie Curie was born in Warsaw",
    )


def test_non_numeric_confidence_maps_to_band_instead_of_crashing() -> None:
    """Non-numeric LM confidence maps to a band: "high"->0.9, "85%"->0.85."""
    text = "Marie Curie was born in Warsaw, Poland."
    extractor = ClaimExtractor(llm_config=None)
    extractor._cot_module = _ClaimsModule(
        [
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": "Warsaw",
                "confidence": "high",
                "evidence_span": "Marie Curie was born in Warsaw",
            },
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": "Poland",
                "confidence": "85%",
                "evidence_span": "born in Warsaw, Poland",
            },
        ]
    )

    edges = extractor.extract(
        text=text,
        entity_hints=["Marie Curie"],
        modality_hint="text",
        segment_anchor=_anchor(),
        tenant_id="acme:acme",
        source_doc_id="doc1",
    )

    assert [e.confidence for e in edges] == [0.9, 0.85]
    assert [e.target for e in edges] == ["Warsaw", "Poland"]


def test_out_of_range_and_missing_confidence_are_clamped() -> None:
    """Numeric > 1 saturates at 1.0; a missing field falls back to 1.0."""
    text = "Marie Curie discovered radium."
    extractor = ClaimExtractor(llm_config=None)
    extractor._cot_module = _ClaimsModule(
        [
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": "Warsaw",
                "confidence": 1.5,
            },
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": "Paris",
            },
        ]
    )

    edges = extractor.extract(
        text=text,
        entity_hints=["Marie Curie"],
        modality_hint="text",
        segment_anchor=_anchor(),
        tenant_id="acme:acme",
        source_doc_id="doc1",
    )

    assert [e.confidence for e in edges] == [1.0, 1.0]


def test_output_budget_raised_above_endpoint_default() -> None:
    """The signature's three output fields overflow the endpoint-default
    1000-token cap: the LM truncates before ``rationale``, the parse fails,
    and every segment silently yields zero claims (an empty KG that reads as
    success). The extractor must guarantee itself an adequate budget."""
    from cogniverse_agents.graph.claim_extractor import (
        CLAIM_EXTRACTION_MIN_OUTPUT_TOKENS,
        ClaimExtractor,
    )
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    capped = LLMEndpointConfig(model="openai/auto", max_tokens=1000)
    extractor = ClaimExtractor(llm_config=capped)
    assert extractor._llm_config.max_tokens == CLAIM_EXTRACTION_MIN_OUTPUT_TOKENS
    # Everything else carries over unchanged.
    assert extractor._llm_config.model == "openai/auto"

    roomy = LLMEndpointConfig(model="openai/auto", max_tokens=8000)
    assert ClaimExtractor(llm_config=roomy)._llm_config.max_tokens == 8000

    assert ClaimExtractor(llm_config=None)._llm_config is None
