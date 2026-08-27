"""ClaimExtractor must bind the per-tenant LM, not the ambient one."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dspy
import pytest

from cogniverse_agents.graph.claim_extractor import ClaimExtractor
from cogniverse_agents.graph.graph_schema import Mention

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


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
                source_doc_id="doc1",
                segment_anchor=_anchor(),
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
            source_doc_id="doc1",
            segment_anchor=_anchor(),
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
    """Non-numeric LM confidence maps to a band: "0.9"->0.9, "85%"->0.85."""
    text = "Marie Curie was born in Warsaw, Poland."
    extractor = ClaimExtractor(llm_config=None)
    extractor._cot_module = _ClaimsModule(
        [
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": "Warsaw",
                "confidence": "0.9",
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


def test_non_string_claim_fields_coerce_or_drop_instead_of_crashing() -> None:
    """LMs emit JSON scalars where the signature asks for strings ("object":
    1867 for born_in). A numeric subject/object must keep its text form and
    the claim must survive; list/dict/bool/None fields drop only that claim.
    Any non-string used to crash the whole segment's edge build with
    AttributeError, and ingestion's KG stage swallowed it — the document
    lost its entire graph while the ingest reported success."""
    text = "Marie Curie was born in 1867. Radium glows."
    extractor = ClaimExtractor(llm_config=None)
    extractor._cot_module = _ClaimsModule(
        [
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": 1867,
                "evidence_span": "Marie Curie was born in 1867",
            },
            {
                "subject": "Marie Curie",
                "predicate": "born_in",
                "object": "Warsaw",
                "evidence_span": 99999,
            },
            {"subject": ["Radium"], "predicate": "born_in", "object": "light"},
            {"subject": "Radium", "predicate": {"r": "born_in"}, "object": "light"},
            {"subject": "Radium", "predicate": "born_in", "object": True},
            {"subject": None, "predicate": "born_in", "object": "light"},
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

    assert [(e.source, e.relation, e.target) for e in edges] == [
        ("Marie Curie", "born_in", "1867"),
        ("Marie Curie", "born_in", "Warsaw"),
    ]
    # "99999" appears nowhere in the text, so the evidence falls back to
    # the leading span of the segment text.
    assert edges[1].evidence_span == text


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


class _StaticResponse:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.choices = [
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                    provider_specific_fields={},
                ),
            )
        ]
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.model = "dummy/static"
        self._hidden_params = {}

    def __getitem__(self, key: str):
        return getattr(self, key)


class _StaticCompletionLM(dspy.LM):
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        super().__init__(
            "openai/static",
            temperature=0.0,
            max_tokens=1000,
            cache=False,
            num_retries=0,
        )
        self._response = _StaticResponse(content, finish_reason)

    def forward(self, prompt=None, messages=None, **kwargs):
        del prompt, messages, kwargs
        return self._response


def test_signature_contract_is_reasoning_plus_claims_only() -> None:
    from cogniverse_agents.graph.claim_extractor import (
        CLAIM_EXTRACTION_MAX_CLAIMS,
        CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS,
        CLAIM_EXTRACTION_REASONING_TOKENS,
        CLAIM_EXTRACTION_TOKENS_PER_CLAIM,
    )
    from cogniverse_agents.graph.dspy_signatures import ClaimExtractionSignature

    assert list(ClaimExtractionSignature.output_fields.keys()) == ["claims"]
    assert list(
        dspy.ChainOfThought(ClaimExtractionSignature).predict.signature.output_fields
    ) == [
        "reasoning",
        "claims",
    ]
    assert CLAIM_EXTRACTION_MAX_CLAIMS == 4
    assert CLAIM_EXTRACTION_TOKENS_PER_CLAIM == 80
    assert CLAIM_EXTRACTION_REASONING_TOKENS == 192
    assert CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS == 512
    assert (
        "Return at most four claims."
        in ClaimExtractionSignature.output_fields["claims"].json_schema_extra["desc"]
    )
    assert (
        "claims is the final output field."
        in ClaimExtractionSignature.output_fields["claims"].json_schema_extra["desc"]
    )
    assert (
        "evidence_span must be a verbatim substring"
        in ClaimExtractionSignature.output_fields["claims"].json_schema_extra["desc"]
    )


def test_output_budget_is_clamped_to_contract_cap() -> None:
    """The LM budget is derived from the four-claim contract."""
    from cogniverse_agents.graph.claim_extractor import (
        CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS,
        ClaimExtractor,
    )
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    capped = LLMEndpointConfig(model="openai/auto", max_tokens=1000)
    extractor = ClaimExtractor(llm_config=capped)
    assert extractor._llm_config.max_tokens == CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS
    assert extractor._llm_config.model == "openai/auto"

    roomy = LLMEndpointConfig(model="openai/auto", max_tokens=8000)
    assert ClaimExtractor(llm_config=roomy)._llm_config.max_tokens == (
        CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS
    )
    assert ClaimExtractor(llm_config=None)._llm_config is None


def test_extraction_decodes_greedily_regardless_of_tenant_temperature() -> None:
    """Sampling temperatures let the model mis-attribute claim subjects (a 4B
    model at 0.1 swaps the SPO subject on real transcript segments), so the
    extractor pins its decoding temperature to 0.0 while carrying every other
    endpoint field through unchanged."""
    from cogniverse_agents.graph.claim_extractor import ClaimExtractor
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    sampled = LLMEndpointConfig(
        model="openai/auto",
        api_base="http://llm.test:8000/v1",
        temperature=0.1,
        max_tokens=8000,
    )
    extractor = ClaimExtractor(llm_config=sampled)
    assert extractor._llm_config.temperature == 0.0
    assert extractor._llm_config.model == "openai/auto"
    assert extractor._llm_config.api_base == "http://llm.test:8000/v1"
    assert extractor._llm_config.max_tokens == 512

    greedy = LLMEndpointConfig(model="openai/auto", temperature=0.0, max_tokens=8000)
    greedy_extractor = ClaimExtractor(llm_config=greedy)
    assert greedy_extractor._llm_config.temperature == 0.0
    assert greedy_extractor._llm_config.max_tokens == 512


def test_length_capped_completion_raises_with_source_and_segment() -> None:
    from cogniverse_agents.graph.claim_extractor import ClaimExtractor
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    content = (
        "[[ ## reasoning ## ]]\n"
        "The segment has one claim.\n\n"
        "[[ ## claims ## ]]\n"
        '[{"subject":"Marie Curie","predicate":"discovered",'
        '"object":"radium","evidence_span":"Marie Curie discovered radium",'
        '"confidence":0.97}]\n\n'
        "[[ ## completed ## ]]\n"
    )
    lm = _StaticCompletionLM(content, finish_reason="length")
    extractor = ClaimExtractor(
        llm_config=LLMEndpointConfig(model="openai/auto", max_tokens=512)
    )

    with patch(
        "cogniverse_foundation.config.semantic_router.create_dspy_lm",
        return_value=lm,
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                r"^Claim extraction failed for source 'doc1' segment 'seg1': "
                r"LM response hit max_tokens for source 'doc1' segment 'seg1'$"
            ),
        ):
            extractor.extract(
                text="Marie Curie discovered radium.",
                entity_hints=["Marie Curie", "radium"],
                modality_hint="text",
                segment_anchor=_anchor(),
                tenant_id="acme:acme",
                source_doc_id="doc1",
            )


def test_well_formed_completion_parses_to_exact_edges() -> None:
    from cogniverse_agents.graph.claim_extractor import ClaimExtractor

    content = (
        "[[ ## reasoning ## ]]\n"
        "The text contains two claims.\n\n"
        "[[ ## claims ## ]]\n"
        "["
        '{"subject":"Marie Curie","predicate":"discovered","object":"radium",'
        '"evidence_span":"Marie Curie discovered radium","confidence":0.97},'
        '{"subject":"Marie Curie","predicate":"won","object":"Nobel Prize",'
        '"evidence_span":"won the Nobel Prize.","confidence":0.93}'
        "]\n\n"
        "[[ ## completed ## ]]\n"
    )
    lm = _StaticCompletionLM(content, finish_reason="stop")
    extractor = ClaimExtractor(llm_config=None)
    text = "Marie Curie discovered radium and won the Nobel Prize."
    anchor = Mention(
        source_doc_id="doc1",
        segment_id="seg1",
        ts_start=12.0,
        ts_end=18.5,
        modality="text",
        evidence_span=text,
    )

    with dspy.context(lm=lm):
        edges = extractor.extract(
            text=text,
            entity_hints=["Marie Curie", "radium", "Nobel Prize"],
            modality_hint="text",
            segment_anchor=anchor,
            tenant_id="acme:acme",
            source_doc_id="doc1",
        )

    actual = [asdict(edge) for edge in edges]
    for edge in actual:
        edge.pop("created_at", None)
    assert actual == [
        {
            "tenant_id": "acme:acme",
            "source": "Marie Curie",
            "target": "radium",
            "relation": "discovered",
            "evidence_span": "Marie Curie discovered radium",
            "segment_id": "seg1",
            "ts_start": 12.0,
            "ts_end": 18.5,
            "modality": "text",
            "provenance": "EXTRACTED",
            "source_doc_id": "doc1",
            "confidence": 0.97,
        },
        {
            "tenant_id": "acme:acme",
            "source": "Marie Curie",
            "target": "Nobel Prize",
            "relation": "won",
            "evidence_span": "won the Nobel Prize.",
            "segment_id": "seg1",
            "ts_start": 12.0,
            "ts_end": 18.5,
            "modality": "text",
            "provenance": "EXTRACTED",
            "source_doc_id": "doc1",
            "confidence": 0.93,
        },
    ]
