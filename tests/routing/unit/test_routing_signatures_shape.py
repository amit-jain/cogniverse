"""Shape coverage for the DSPy routing signatures + structured-output
data models. Construct each model with valid kwargs, assert the
declared fields round-trip, and confirm validation rejects missing
required fields.

The module is otherwise uncovered in routing-tests CI; the constructions
here lift coverage out of the 0% band so the routing job's 15%
threshold holds without depending on cross-package unit tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cogniverse_agents.routing.dspy_routing_signatures import (
    AdaptiveThresholdSignature,
    AdvancedRoutingSignature,
    BasicQueryAnalysisSignature,
    EntityInfo,
    MetaRoutingSignature,
    MultiAgentOrchestrationSignature,
    QueryReformulationSignature,
    RelationshipRoutingDecision,
    RelationshipTuple,
    TemporalInfo,
    UnifiedExtractionReformulationSignature,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pydantic data models — exact field round-trip.
# ---------------------------------------------------------------------------


def test_entity_info_round_trip() -> None:
    e = EntityInfo(text="Marie Curie", label="PERSON", confidence=0.9)
    assert e.text == "Marie Curie"
    assert e.label == "PERSON"
    assert e.confidence == 0.9
    assert e.start_pos is None
    assert e.end_pos is None


def test_entity_info_accepts_positions() -> None:
    e = EntityInfo(text="Paris", label="LOC", confidence=0.8, start_pos=10, end_pos=15)
    assert e.start_pos == 10
    assert e.end_pos == 15


def test_entity_info_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        EntityInfo()  # type: ignore[call-arg]


def test_relationship_tuple_round_trip() -> None:
    r = RelationshipTuple(
        subject="Marie Curie",
        relation="discovered",
        object="Radium",
        confidence=0.95,
    )
    assert r.subject == "Marie Curie"
    assert r.relation == "discovered"
    assert r.object == "Radium"
    assert r.confidence == 0.95
    assert r.subject_type is None
    assert r.object_type is None


def test_relationship_tuple_typed_subject_and_object() -> None:
    r = RelationshipTuple(
        subject="Marie Curie",
        relation="discovered",
        object="Radium",
        confidence=0.95,
        subject_type="PERSON",
        object_type="CHEMICAL",
    )
    assert r.subject_type == "PERSON"
    assert r.object_type == "CHEMICAL"


def test_temporal_info_round_trip() -> None:
    t = TemporalInfo(
        time_references=["yesterday", "last week"],
        date_patterns=["2026-01-01"],
        temporal_context="recent",
        has_temporal_constraints=True,
    )
    assert t.time_references == ["yesterday", "last week"]
    assert t.date_patterns == ["2026-01-01"]
    assert t.temporal_context == "recent"
    assert t.has_temporal_constraints is True


def test_relationship_routing_decision_round_trip() -> None:
    d = RelationshipRoutingDecision(
        search_modality="video_only",
        generation_type="summary",
        primary_agent="search_agent",
        secondary_agents=["summarizer_agent"],
        execution_mode="sequential",
        confidence=0.7,
        reasoning="single-modality query",
    )
    assert d.search_modality == "video_only"
    assert d.generation_type == "summary"
    assert d.primary_agent == "search_agent"
    assert d.secondary_agents == ["summarizer_agent"]
    assert d.execution_mode == "sequential"
    assert d.confidence == 0.7
    assert "single-modality" in d.reasoning


# ---------------------------------------------------------------------------
# DSPy signature classes — verify the declared input/output fields by
# inspecting ``model_fields`` (DSPy stores them on the underlying
# pydantic model).
# ---------------------------------------------------------------------------


def _field_names(sig_cls) -> set[str]:
    """Return the set of declared fields on a DSPy signature class."""
    fields = getattr(sig_cls, "model_fields", None)
    if fields is None:
        # Fallback for older DSPy releases that put them under `__fields__`.
        fields = getattr(sig_cls, "__fields__", {})
    return set(fields.keys())


def test_basic_query_analysis_signature_fields() -> None:
    names = _field_names(BasicQueryAnalysisSignature)
    assert "query" in names
    assert "primary_intent" in names
    assert "complexity_level" in names


def test_query_reformulation_signature_fields() -> None:
    names = _field_names(QueryReformulationSignature)
    assert "original_query" in names
    assert "enhanced_query" in names
    assert "query_variants" in names


def test_unified_extraction_reformulation_signature_fields() -> None:
    names = _field_names(UnifiedExtractionReformulationSignature)
    assert "original_query" in names
    assert "enhanced_query" in names
    assert "entities" in names
    assert "relationships" in names


def test_multi_agent_orchestration_signature_fields() -> None:
    names = _field_names(MultiAgentOrchestrationSignature)
    assert "query" in names


def test_advanced_routing_signature_fields() -> None:
    names = _field_names(AdvancedRoutingSignature)
    assert "query" in names


def test_meta_routing_signature_fields() -> None:
    names = _field_names(MetaRoutingSignature)
    assert "query" in names


def test_adaptive_threshold_signature_fields() -> None:
    names = _field_names(AdaptiveThresholdSignature)
    assert names  # at least one declared field


def test_module_demo_reports_declared_fields_for_every_signature() -> None:
    """The module ``__main__`` demo (now ``_demo()``) reports the declared
    input/output fields of all seven routing signatures. Drive it directly so
    the demo body is covered surface, not a script nothing imports; assert the
    exact reported shape."""
    from cogniverse_agents.routing.dspy_routing_signatures import _demo

    fields_by_signature = _demo()

    assert set(fields_by_signature) == {
        "BasicQueryAnalysis",
        "QueryReformulation",
        "UnifiedExtractionReformulation",
        "MultiAgentOrchestration",
        "AdvancedRouting",
        "MetaRouting",
        "AdaptiveThreshold",
    }
    # Every signature reports a non-empty, sorted field list.
    for name, names in fields_by_signature.items():
        assert names, f"{name} reported no fields"
        assert names == sorted(names)
    # One full pin plus the required AdaptiveThreshold outputs that make a bare
    # SignatureClass() raise (why the demo inspects model_fields, not instances).
    assert fields_by_signature["BasicQueryAnalysis"] == [
        "complexity_level",
        "confidence_score",
        "context",
        "needs_multimodal",
        "needs_text_search",
        "needs_video_search",
        "primary_intent",
        "query",
        "reasoning",
        "recommended_agent",
    ]
    assert {
        "fast_path_threshold",
        "slow_path_threshold",
        "escalation_threshold",
    } <= set(fields_by_signature["AdaptiveThreshold"])
