"""Native confidence extraction for canonical synthetic item schemas."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_core.approval.interfaces import ApprovalStatus
from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)
from cogniverse_synthetic.generators.profile import ProfileGenerator
from cogniverse_synthetic.generators.query_enhancement import (
    QueryEnhancementGenerator,
)
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    QueryEnhancementExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

pytestmark = pytest.mark.unit

ROUTING_SEMANTICS = {
    "routing_confidence": "observed_gateway_confidence",
    "search_quality": "unobserved_zero_sentinel",
    "agent_success": "unobserved_false_sentinel",
    "processing_time": "unobserved_zero_sentinel",
}
WORKFLOW_UNOBSERVED_SEMANTICS = {
    "execution_time": "unobserved_zero_sentinel",
    "success": "unobserved_false_sentinel",
    "parallel_efficiency": "unobserved_zero_sentinel",
    "confidence_score": "unobserved_zero_sentinel",
}
WORKFLOW_OBSERVED_SEMANTICS = {
    "execution_time": "observed_duration_seconds",
    "success": "observed_execution_outcome",
    "parallel_efficiency": "observed_parallel_efficiency",
    "confidence_score": "observed_confidence_score",
}


def _profile() -> dict:
    return ProfileSelectionExampleSchema(
        query="find a TensorFlow guide",
        available_profiles="document_text",
        selected_profile="document_text",
        reasoning="The document profile matches the requested guide.",
        query_intent="document_search",
        modality="document",
        complexity="simple",
    ).model_dump()


def _query_enhancement() -> dict:
    return QueryEnhancementExampleSchema(
        query="TensorFlow guide",
        enhanced_query="TensorFlow deployment guide",
        expansion_terms=["deployment"],
        synonyms=["manual"],
        context="machine learning",
        reasoning="Deployment narrows the requested TensorFlow material.",
    ).model_dump()


def _routing(confidence: float = 0.84) -> dict:
    return RoutingExperienceSchema(
        query="find TensorFlow documentation",
        entities=[{"text": "TensorFlow", "type": "TECHNOLOGY"}],
        relationships=[],
        enhanced_query="find TensorFlow(TECHNOLOGY) documentation",
        chosen_agent="document_agent",
        routing_confidence=confidence,
        search_quality=0.0,
        agent_success=False,
        user_satisfaction=None,
        processing_time=0.0,
        reward=None,
        metadata={
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": ROUTING_SEMANTICS,
            }
        },
    ).model_dump()


def _workflow(*, observed: bool, confidence: float) -> dict:
    return WorkflowExecutionSchema(
        workflow_id="workflow-1",
        query="summarize TensorFlow documentation",
        query_type="DOCUMENT",
        execution_time=1.25 if observed else 0.0,
        success=observed,
        agent_sequence=["document_agent", "summarizer_agent"],
        task_count=2,
        parallel_efficiency=0.75 if observed else 0.0,
        confidence_score=confidence,
        user_satisfaction=0.9 if observed else None,
        error_details=None,
        metadata={
            "_outcome_metadata": {
                "observed": observed,
                "required_field_semantics": (
                    WORKFLOW_OBSERVED_SEMANTICS
                    if observed
                    else WORKFLOW_UNOBSERVED_SEMANTICS
                ),
            }
        },
    ).model_dump()


def test_each_canonical_schema_uses_only_observed_native_confidence() -> None:
    extractor = SyntheticDataConfidenceExtractor()

    assert [
        extractor.extract(item)
        for item in (
            _profile(),
            _query_enhancement(),
            _routing(),
            _workflow(observed=True, confidence=0.91),
        )
    ] == [0.0, 0.0, 0.84, 0.91]


@pytest.mark.asyncio
async def test_pattern_generators_emit_unobserved_targets_requiring_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_fabricated_confidence(*_args, **_kwargs) -> float:
        raise AssertionError("synthetic generators must not fabricate confidence")

    monkeypatch.setattr("random.uniform", reject_fabricated_confidence)

    async def label_profile(query: str, profiles: list[str], tenant_id: str):
        assert tenant_id == "acme:synthetic"
        return {
            "query": query,
            "selected_profile": "document_semantic",
            "reasoning": "Production selector chose document retrieval.",
            "query_intent": "research_lookup",
            "modality": "document",
            "complexity": "medium",
        }

    profile_examples = await ProfileGenerator(profile_labeler=label_profile).generate(
        sampled_content=[{"title": "quantum computing applications"}],
        target_count=1,
        profile_configs={
            "document_semantic": {
                "type": "document",
                "schema_name": "document_pages",
                "embedding_type": "single_vector",
                "pipeline_config": {},
            }
        },
        tenant_id="acme:synthetic",
    )

    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert source_text == "quantum computing applications\ndeployment guide"
        return {
            "original_query": query,
            "enhanced_query": f"{query} deployment guide",
            "expansion_terms": ["deployment", "guide"],
            "synonyms": [],
            "reasoning": "Production enhancement used exact source terms.",
        }

    query_examples = await QueryEnhancementGenerator(
        query_enhancer=enhance_query
    ).generate(
        sampled_content=[
            {
                "title": "quantum computing applications",
                "description": "deployment guide",
                "content_type": "document",
            }
        ],
        target_count=1,
        tenant_id="acme:synthetic",
    )

    profile = profile_examples[0].model_dump()
    query = query_examples[0].model_dump()
    assert set(profile) == {
        "available_profiles",
        "complexity",
        "modality",
        "query",
        "query_intent",
        "reasoning",
        "selected_profile",
    }
    assert set(query) == {
        "context",
        "enhanced_query",
        "expansion_terms",
        "query",
        "reasoning",
        "synonyms",
    }
    extractor = SyntheticDataConfidenceExtractor()
    assert [extractor.get_confidence_breakdown(item) for item in (profile, query)] == [
        {
            "schema": "ProfileSelectionExampleSchema",
            "confidence_field": None,
            "final_confidence": 0.0,
            "outcome_observed": None,
            "requires_human_review": True,
        },
        {
            "schema": "QueryEnhancementExampleSchema",
            "confidence_field": None,
            "final_confidence": 0.0,
            "outcome_observed": None,
            "requires_human_review": True,
        },
    ]


@pytest.mark.asyncio
async def test_unobserved_workflow_requires_human_review_at_zero_confidence() -> None:
    extractor = SyntheticDataConfidenceExtractor()
    item = _workflow(observed=False, confidence=0.0)

    assert extractor.extract(item) == 0.0

    batch = await HumanApprovalAgent(
        confidence_extractor=extractor,
        confidence_threshold=0.85,
    ).process_batch(
        [item],
        "workflow-batch",
        {"optimizer": "workflow", "agent_type": "workflow"},
    )

    assert len(batch.items) == 1
    assert batch.items[0].confidence == 0.0
    assert batch.items[0].status is ApprovalStatus.PENDING_REVIEW
    assert batch.auto_approved == []
    assert batch.pending_review == [batch.items[0]]


@pytest.mark.asyncio
async def test_entity_extraction_without_native_confidence_requires_review() -> None:
    item = EntityExtractionExampleSchema(
        query="TensorFlow was developed by Google Brain",
        entities=[{"text": "TensorFlow", "type": "TECHNOLOGY"}],
        entity_types="TECHNOLOGY",
        relationships=[],
    ).model_dump()
    extractor = SyntheticDataConfidenceExtractor()

    assert extractor.extract(item) == 0.0
    assert extractor.get_confidence_breakdown(item) == {
        "schema": "EntityExtractionExampleSchema",
        "confidence_field": None,
        "final_confidence": 0.0,
        "outcome_observed": None,
        "requires_human_review": True,
    }

    batch = await HumanApprovalAgent(
        confidence_extractor=extractor,
        confidence_threshold=0.85,
    ).process_batch(
        [item],
        "entity-batch",
        {"optimizer": "entity_extraction", "agent_type": "entity_extraction"},
    )

    assert batch.items[0].confidence == 0.0
    assert batch.items[0].status is ApprovalStatus.PENDING_REVIEW
    assert batch.auto_approved == []
    assert batch.pending_review == [batch.items[0]]


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(
            {"query": "find TensorFlow", "confidence": 0.99},
            id="partial-shape",
        ),
        pytest.param(
            {**_routing(), "confidence": 0.99},
            id="mixed-confidence-fields",
        ),
        pytest.param([], id="non-dict"),
    ],
)
def test_noncanonical_or_mixed_shapes_are_rejected(item) -> None:
    with pytest.raises(
        ValueError,
        match="^confidence item must match exactly one canonical synthetic schema",
    ):
        SyntheticDataConfidenceExtractor().extract(item)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda item: item.update(metadata={}),
            "RoutingExperienceSchema.metadata must contain _outcome_metadata",
        ),
        (
            lambda item: item.update(metadata=[]),
            "RoutingExperienceSchema.metadata must be a dict",
        ),
        (
            lambda item: item["metadata"].update(_outcome_metadata=[]),
            "RoutingExperienceSchema.metadata._outcome_metadata must be a dict",
        ),
        (
            lambda item: item["metadata"]["_outcome_metadata"].update(observed="false"),
            (
                "RoutingExperienceSchema.metadata._outcome_metadata.observed "
                "must be a bool"
            ),
        ),
        (
            lambda item: item["metadata"]["_outcome_metadata"].update(
                required_field_semantics=[]
            ),
            (
                "RoutingExperienceSchema.metadata._outcome_metadata."
                "required_field_semantics must be a dict"
            ),
        ),
        (
            lambda item: item["metadata"]["_outcome_metadata"].update(extra=True),
            (
                "RoutingExperienceSchema.metadata._outcome_metadata must contain "
                "exactly: observed, required_field_semantics"
            ),
        ),
        (
            lambda item: item["metadata"]["_outcome_metadata"].update(
                required_field_semantics={
                    **ROUTING_SEMANTICS,
                    "routing_confidence": "unobserved_zero_sentinel",
                }
            ),
            (
                "RoutingExperienceSchema.metadata._outcome_metadata."
                "required_field_semantics must exactly match the routing contract"
            ),
        ),
    ],
)
def test_routing_outcome_metadata_faults_raise_exact_context(mutate, message) -> None:
    item = _routing()
    mutate(item)

    with pytest.raises(ValueError) as error:
        SyntheticDataConfidenceExtractor().extract(item)

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_time", 0.1),
        ("success", True),
        ("parallel_efficiency", 0.1),
        ("confidence_score", 0.1),
    ],
)
def test_unobserved_workflow_rejects_non_sentinel_outcomes(field, value) -> None:
    item = _workflow(observed=False, confidence=0.0)
    item[field] = value

    with pytest.raises(ValueError) as error:
        SyntheticDataConfidenceExtractor().extract(item)

    assert str(error.value) == (
        f"WorkflowExecutionSchema.{field} must match its unobserved sentinel"
    )


@pytest.mark.parametrize("item", [_profile(), _query_enhancement()])
def test_unobserved_target_rejects_injected_confidence(item) -> None:
    item["confidence"] = 0.99

    with pytest.raises(ValueError) as error:
        SyntheticDataConfidenceExtractor().extract(item)

    assert str(error.value).startswith(
        "confidence item must match exactly one canonical synthetic schema; keys: "
    )


def test_concurrent_mixed_items_keep_schema_dispatch_isolated() -> None:
    worker_count = 16
    start = threading.Barrier(worker_count)
    extractor = SyntheticDataConfidenceExtractor()
    items = (
        (_profile(), 0.0),
        (_query_enhancement(), 0.0),
        (_routing(), 0.84),
        (_workflow(observed=True, confidence=0.59), 0.59),
    )
    work = [items[index % len(items)] for index in range(worker_count)]

    def extract(entry: tuple[dict, float]) -> float:
        start.wait()
        return extractor.extract(entry[0])

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        actual = list(pool.map(extract, work))

    assert actual == [expected for _, expected in work]


def test_breakdown_names_exact_schema_field_and_review_state() -> None:
    extractor = SyntheticDataConfidenceExtractor()

    assert extractor.get_confidence_breakdown(
        _workflow(observed=False, confidence=0.0)
    ) == {
        "schema": "WorkflowExecutionSchema",
        "confidence_field": "confidence_score",
        "final_confidence": 0.0,
        "outcome_observed": False,
        "requires_human_review": True,
    }
