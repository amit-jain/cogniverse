"""
Tests for Human-in-the-Loop Approval System

Tests approval interfaces, agents, confidence extraction, and feedback handling.
"""

import asyncio
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cogniverse_agents.approval import (
    ApprovalBatch,
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
)
from cogniverse_synthetic.approval import (
    SyntheticDataConfidenceExtractor,
    SyntheticDataFeedbackHandler,
)
from cogniverse_synthetic.dspy_modules import (
    ValidatedEntityQueryGenerator,
    ValidatedSyntheticExampleRegenerator,
)
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    QueryEnhancementExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

pytestmark = [pytest.mark.unit]


class _BoundTestQueryGenerator(ValidatedEntityQueryGenerator):
    def __init__(self, forward):
        super().__init__(max_retries=3)
        self.lm = SimpleNamespace(model="test-lm")
        self._test_forward = forward

    def forward(self, **kwargs):
        return self._test_forward(**kwargs)


class _BoundTestRegenerator(ValidatedSyntheticExampleRegenerator):
    def __init__(self, forward):
        super().__init__(max_retries=3)
        self.lm = SimpleNamespace(model="test-lm")
        self._test_forward = forward

    def forward(self, **kwargs):
        return self._test_forward(**kwargs)


class TestApprovalInterfaces:
    """Test approval data structures"""

    def test_review_item_creation(self):
        """Test creating ReviewItem"""
        item = ReviewItem(
            item_id="test_001",
            data={"query": "find TensorFlow tutorial", "entities": ["TensorFlow"]},
            confidence=0.9,
        )

        assert item.item_id == "test_001"
        assert item.confidence == 0.9
        assert item.status == ApprovalStatus.PENDING_REVIEW
        assert item.created_at is not None

    def test_review_decision_creation(self):
        """Test creating ReviewDecision"""
        decision = ReviewDecision(
            item_id="test_001",
            approved=True,
            feedback="Looks good",
            corrections={},
            reviewer="test_user",
        )

        assert decision.item_id == "test_001"
        assert decision.approved is True
        assert decision.feedback == "Looks good"
        assert decision.timestamp is not None

    def test_approval_batch_properties(self):
        """Test ApprovalBatch property methods"""
        items = [
            ReviewItem(
                item_id="test_001",
                data={"query": "query1"},
                confidence=0.95,
                status=ApprovalStatus.AUTO_APPROVED,
            ),
            ReviewItem(
                item_id="test_002",
                data={"query": "query2"},
                confidence=0.7,
                status=ApprovalStatus.PENDING_REVIEW,
            ),
            ReviewItem(
                item_id="test_003",
                data={"query": "query3"},
                confidence=0.6,
                status=ApprovalStatus.PENDING_REVIEW,
            ),
        ]

        batch = ApprovalBatch(batch_id="batch_001", items=items, context={})

        assert len(batch.auto_approved) == 1
        assert len(batch.pending_review) == 2
        assert len(batch.approved) == 0
        assert len(batch.rejected) == 0
        assert batch.approval_rate == pytest.approx(1 / 3)


class TestConfidenceExtractor:
    @staticmethod
    def _routing_record():
        return {
            "query": "find TensorFlow tutorials",
            "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "relationships": [],
            "enhanced_query": "find TensorFlow(TECHNOLOGY) tutorials",
            "chosen_agent": "video_search_agent",
            "routing_confidence": 0.84,
            "search_quality": 0.0,
            "agent_success": False,
            "user_satisfaction": None,
            "processing_time": 0.0,
            "reward": None,
            "timestamp": datetime(2026, 8, 5, 1, 2, 3, tzinfo=timezone.utc),
            "metadata": {
                "_outcome_metadata": {
                    "observed": True,
                    "required_field_semantics": {
                        "routing_confidence": "observed_gateway_confidence",
                        "search_quality": "unobserved_zero_sentinel",
                        "agent_success": "unobserved_false_sentinel",
                        "processing_time": "unobserved_zero_sentinel",
                    },
                }
            },
        }

    @staticmethod
    def _workflow_record(*, observed):
        if observed:
            outcome_values = {
                "execution_time": 6.25,
                "success": True,
                "parallel_efficiency": 0.8,
                "confidence_score": 0.91,
            }
            semantics = {
                "execution_time": "observed_duration_seconds",
                "success": "observed_execution_outcome",
                "parallel_efficiency": "observed_parallel_efficiency",
                "confidence_score": "observed_confidence_score",
            }
        else:
            outcome_values = {
                "execution_time": 0.0,
                "success": False,
                "parallel_efficiency": 0.0,
                "confidence_score": 0.0,
            }
            semantics = {
                "execution_time": "unobserved_zero_sentinel",
                "success": "unobserved_false_sentinel",
                "parallel_efficiency": "unobserved_zero_sentinel",
                "confidence_score": "unobserved_zero_sentinel",
            }
        return {
            "workflow_id": "workflow-17",
            "query": "summarize the radium video",
            "query_type": "VIDEO",
            **outcome_values,
            "agent_sequence": ["video_search_agent", "summarizer_agent"],
            "task_count": 2,
            "user_satisfaction": None,
            "error_details": None,
            "timestamp": datetime(2026, 8, 5, 2, 3, 4, tzinfo=timezone.utc),
            "metadata": {
                "_outcome_metadata": {
                    "observed": observed,
                    "required_field_semantics": semantics,
                }
            },
        }

    @pytest.mark.parametrize(
        ("record", "confidence", "breakdown"),
        [
            pytest.param(
                {
                    "query": "find radium footage",
                    "available_profiles": "video_colpali,video_colqwen",
                    "selected_profile": "video_colqwen",
                    "reasoning": "The query requires temporal video context.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "medium",
                },
                0.0,
                {
                    "schema": "ProfileSelectionExampleSchema",
                    "confidence_field": None,
                    "final_confidence": 0.0,
                    "outcome_observed": None,
                    "requires_human_review": True,
                },
                id="profile-unobserved-confidence",
            ),
            pytest.param(
                {
                    "query": "radium discovery",
                    "enhanced_query": "radium discovery Marie Curie",
                    "expansion_terms": ["Marie Curie"],
                    "synonyms": ["radium isolation"],
                    "context": "science history",
                    "reasoning": "The person name disambiguates the discovery.",
                },
                0.0,
                {
                    "schema": "QueryEnhancementExampleSchema",
                    "confidence_field": None,
                    "final_confidence": 0.0,
                    "outcome_observed": None,
                    "requires_human_review": True,
                },
                id="enhancement-unobserved-confidence",
            ),
        ],
    )
    def test_schema_confidence_is_returned_exactly(self, record, confidence, breakdown):
        extractor = SyntheticDataConfidenceExtractor()

        assert extractor.extract(record) == confidence
        assert extractor.get_confidence_breakdown(record) == breakdown

    def test_routing_preserves_observed_confidence_with_unobserved_outcomes(self):
        extractor = SyntheticDataConfidenceExtractor()
        record = self._routing_record()

        assert RoutingExperienceSchema.model_validate(record)
        assert extractor.extract(record) == 0.84
        assert extractor.get_confidence_breakdown(record) == {
            "schema": "RoutingExperienceSchema",
            "confidence_field": "routing_confidence",
            "final_confidence": 0.84,
            "outcome_observed": True,
            "requires_human_review": False,
        }

    def test_routing_unobserved_confidence_requires_human_review(self):
        extractor = SyntheticDataConfidenceExtractor()
        record = self._routing_record()
        record["routing_confidence"] = 0.0
        record["metadata"]["_outcome_metadata"] = {
            "observed": False,
            "required_field_semantics": {
                "routing_confidence": "unobserved_zero_sentinel",
                "search_quality": "unobserved_zero_sentinel",
                "agent_success": "unobserved_false_sentinel",
                "processing_time": "unobserved_zero_sentinel",
            },
        }

        assert extractor.extract(record) == 0.0
        assert extractor.get_confidence_breakdown(record) == {
            "schema": "RoutingExperienceSchema",
            "confidence_field": "routing_confidence",
            "final_confidence": 0.0,
            "outcome_observed": False,
            "requires_human_review": True,
        }

    @pytest.mark.parametrize("observed", [True, False])
    def test_workflow_observation_state_has_exact_review_outcome(self, observed):
        extractor = SyntheticDataConfidenceExtractor()
        record = self._workflow_record(observed=observed)
        expected_confidence = 0.91 if observed else 0.0

        assert WorkflowExecutionSchema.model_validate(record)
        assert extractor.extract(record) == expected_confidence
        assert extractor.get_confidence_breakdown(record) == {
            "schema": "WorkflowExecutionSchema",
            "confidence_field": "confidence_score",
            "final_confidence": expected_confidence,
            "outcome_observed": observed,
            "requires_human_review": not observed,
        }

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            pytest.param(
                lambda record: record["metadata"].pop("_outcome_metadata"),
                "RoutingExperienceSchema.metadata must contain _outcome_metadata",
                id="missing-outcome",
            ),
            pytest.param(
                lambda record: record["metadata"]["_outcome_metadata"].update(
                    {"observed": False}
                ),
                (
                    "RoutingExperienceSchema.metadata._outcome_metadata."
                    "required_field_semantics must exactly match the routing contract"
                ),
                id="routing-observation-semantics-mismatch",
            ),
            pytest.param(
                lambda record: record.update({"search_quality": 0.5}),
                "RoutingExperienceSchema.search_quality must match its unobserved sentinel",
                id="non-sentinel-outcome",
            ),
            pytest.param(
                lambda record: record.update({"routing_confidence": 1}),
                "RoutingExperienceSchema.routing_confidence must be a finite float between 0 and 1",
                id="coercible-integer-confidence",
            ),
            pytest.param(
                lambda record: record.update({"obsolete_score": 0.99}),
                (
                    "confidence item must match exactly one canonical synthetic "
                    "schema; keys: agent_success, chosen_agent, enhanced_query, "
                    "entities, metadata, obsolete_score, processing_time, query, "
                    "relationships, reward, routing_confidence, search_quality, "
                    "timestamp, user_satisfaction"
                ),
                id="extra-field",
            ),
        ],
    )
    def test_malformed_canonical_record_is_rejected_exactly(self, mutate, message):
        extractor = SyntheticDataConfidenceExtractor()
        record = self._routing_record()
        mutate(record)

        for consumer in (extractor.extract, extractor.get_confidence_breakdown):
            with pytest.raises(ValueError) as error:
                consumer(record)
            assert str(error.value) == message

    def test_entity_extraction_schema_uses_explicit_review_confidence(self):
        extractor = SyntheticDataConfidenceExtractor()
        record = {
            "query": "Marie Curie isolated radium",
            "entities": [
                {"text": "Marie Curie", "type": "PERSON"},
                {"text": "radium", "type": "CONCEPT"},
            ],
            "entity_types": "PERSON,CONCEPT",
            "relationships": [],
        }

        assert EntityExtractionExampleSchema.model_validate(record)
        assert extractor.extract(record) == 0.0
        assert extractor.get_confidence_breakdown(record) == {
            "schema": "EntityExtractionExampleSchema",
            "confidence_field": None,
            "final_confidence": 0.0,
            "outcome_observed": None,
            "requires_human_review": True,
        }


class TestFeedbackHandler:
    """Regeneration uses one schema-aware generator with strict boundaries."""

    @staticmethod
    def _handler(forward, *, attempts=2, timeout=0.5):
        return SyntheticDataFeedbackHandler(
            generator=_BoundTestRegenerator(forward),
            max_regeneration_attempts=attempts,
            generation_timeout_seconds=timeout,
        )

    def test_feedback_handler_requires_bound_generator_and_finite_deadline(self):
        generator = _BoundTestRegenerator(
            lambda **kwargs: pytest.fail(f"unexpected generation call: {kwargs}")
        )
        handler = SyntheticDataFeedbackHandler(
            generator=generator,
            max_regeneration_attempts=2,
            generation_timeout_seconds=7.5,
        )

        assert handler.generator is generator
        assert handler.max_attempts == 2
        assert handler.generation_timeout_seconds == 7.5

        for invalid in (True, False, 0, -1, float("inf"), "7.5", None):
            with pytest.raises(
                ValueError,
                match="^generation_timeout_seconds must be finite and positive$",
            ):
                SyntheticDataFeedbackHandler(
                    generator=generator,
                    generation_timeout_seconds=invalid,
                )

    def test_regenerator_serializes_exact_context_for_dspy(self):
        calls = []

        def regenerate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                updates_json=(
                    '{"enhanced_query":"exact PyTorch deployment tutorial",'
                    '"reasoning":"The rewrite names deployment."}'
                ),
                reasoning="Applied the exact reviewer instruction.",
            )

        generator = ValidatedSyntheticExampleRegenerator(max_retries=1)
        generator.lm = SimpleNamespace(model="test-lm")
        generator.regenerate = regenerate
        result = generator.forward(
            schema_name="QueryEnhancementExampleSchema",
            source_context={"query": "PyTorch tutorial"},
            reviewer_instruction="Name deployment explicitly.",
            corrections={"enhanced_query": "exact PyTorch deployment tutorial"},
            schema_contract={"required": ["query", "enhanced_query"]},
        )

        assert calls == [
            {
                "schema_name": "QueryEnhancementExampleSchema",
                "source_context_json": '{"query":"PyTorch tutorial"}',
                "reviewer_instruction": "Name deployment explicitly.",
                "corrections_json": (
                    '{"enhanced_query":"exact PyTorch deployment tutorial"}'
                ),
                "schema_contract_json": ('{"required":["query","enhanced_query"]}'),
            }
        ]
        assert result.updates == {
            "enhanced_query": "exact PyTorch deployment tutorial",
            "reasoning": "The rewrite names deployment.",
        }
        assert result.reasoning == "Applied the exact reviewer instruction."
        assert result._retry_count == 0
        assert result._max_retries == 1

    @pytest.mark.parametrize(
        ("schema", "original", "updates", "corrections", "expected"),
        [
            pytest.param(
                EntityExtractionExampleSchema,
                {
                    "query": "TensorFlow was created by Google Brain",
                    "entities": [
                        {"text": "TensorFlow", "type": "TECHNOLOGY"},
                        {"text": "Google Brain", "type": "ORG"},
                    ],
                    "entity_types": "TECHNOLOGY,ORG",
                    "relationships": [],
                },
                {
                    "query": "PyTorch was created by Meta AI",
                    "entities": [
                        {"text": "PyTorch", "type": "TECHNOLOGY"},
                        {"text": "Meta AI", "type": "ORG"},
                    ],
                },
                {
                    "entities": [
                        {"text": "PyTorch", "type": "TECHNOLOGY"},
                        {"text": "Meta AI", "type": "ORG"},
                    ]
                },
                {
                    "query": "PyTorch was created by Meta AI",
                    "entities": [
                        {"text": "PyTorch", "type": "TECHNOLOGY"},
                        {"text": "Meta AI", "type": "ORG"},
                    ],
                    "entity_types": "TECHNOLOGY,ORG",
                    "relationships": [],
                },
                id="entity",
            ),
            pytest.param(
                RoutingExperienceSchema,
                {
                    "query": "find TensorFlow tutorial",
                    "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
                    "relationships": [],
                    "enhanced_query": "find TensorFlow(TECHNOLOGY) tutorial",
                    "chosen_agent": "video_search_agent",
                    "routing_confidence": 0.84,
                    "search_quality": 0.7,
                    "agent_success": True,
                    "user_satisfaction": 0.9,
                    "processing_time": 1.25,
                    "reward": 0.8,
                    "metadata": {"source": "reviewed trace"},
                },
                {
                    "query": "find exact PyTorch tutorial",
                    "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
                    "chosen_agent": "document_agent",
                },
                {
                    "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
                    "chosen_agent": "document_agent",
                },
                {
                    "query": "find exact PyTorch tutorial",
                    "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
                    "relationships": [],
                    "enhanced_query": "find exact PyTorch(TECHNOLOGY) tutorial",
                    "chosen_agent": "document_agent",
                    "routing_confidence": 0.0,
                    "search_quality": 0.0,
                    "agent_success": False,
                    "user_satisfaction": None,
                    "processing_time": 0.0,
                    "reward": None,
                    "metadata": {
                        "source": "reviewed trace",
                        "_outcome_metadata": {
                            "observed": False,
                            "required_field_semantics": {
                                "routing_confidence": "unobserved_zero_sentinel",
                                "search_quality": "unobserved_zero_sentinel",
                                "agent_success": "unobserved_false_sentinel",
                                "processing_time": "unobserved_zero_sentinel",
                            },
                        },
                        "_generation_metadata": {
                            "retry_count": 0,
                            "max_retries": 3,
                            "regeneration_attempt": 1,
                            "max_regeneration_attempts": 2,
                            "regeneration": True,
                            "original_query": "find TensorFlow tutorial",
                            "human_feedback": "Apply the reviewed correction exactly.",
                            "corrections_applied": {
                                "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
                                "chosen_agent": "document_agent",
                            },
                            "reasoning": "Applied the reviewer instruction.",
                        },
                    },
                },
                id="routing",
            ),
            pytest.param(
                ProfileSelectionExampleSchema,
                {
                    "query": "find transformer clips",
                    "available_profiles": "video_colpali,video_colqwen",
                    "selected_profile": "video_colpali",
                    "reasoning": "Still frames were preferred.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "medium",
                },
                {
                    "selected_profile": "video_colqwen",
                    "reasoning": "Temporal chunks answer the clip request.",
                },
                {"selected_profile": "video_colqwen"},
                {
                    "query": "find transformer clips",
                    "available_profiles": "video_colpali,video_colqwen",
                    "selected_profile": "video_colqwen",
                    "reasoning": "Temporal chunks answer the clip request.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "medium",
                },
                id="profile",
            ),
            pytest.param(
                QueryEnhancementExampleSchema,
                {
                    "query": "PyTorch tutorial",
                    "enhanced_query": "PyTorch beginner tutorial",
                    "expansion_terms": ["beginner"],
                    "synonyms": ["guide"],
                    "context": "machine learning",
                    "reasoning": "Beginner narrows the request.",
                },
                {
                    "enhanced_query": "PyTorch deployment tutorial",
                    "expansion_terms": ["deployment"],
                    "reasoning": "Deployment is the requested focus.",
                },
                {"enhanced_query": "PyTorch deployment tutorial"},
                {
                    "query": "PyTorch tutorial",
                    "enhanced_query": "PyTorch deployment tutorial",
                    "expansion_terms": ["deployment"],
                    "synonyms": ["guide"],
                    "context": "machine learning",
                    "reasoning": "Deployment is the requested focus.",
                },
                id="query-enhancement",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_regeneration_passes_instruction_and_source_to_actual_generator(
        self, schema, original, updates, corrections, expected
    ):
        calls = []

        def regenerate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                updates=updates,
                reasoning="Applied the reviewer instruction.",
                _retry_count=0,
                _max_retries=3,
            )

        item = ReviewItem(item_id="reviewed_item", data=original, confidence=0.4)
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=False,
            feedback="Apply the reviewed correction exactly.",
            corrections=corrections,
        )
        regenerated = await self._handler(regenerate).process_rejection(item, decision)

        assert len(calls) == 1
        assert calls[0]["schema_name"] == schema.__name__
        assert calls[0]["source_context"] == original
        assert calls[0]["reviewer_instruction"] == decision.feedback
        assert calls[0]["corrections"] == corrections
        assert calls[0]["schema_contract"] == schema.model_json_schema()
        assert regenerated.data == expected
        assert regenerated.item_id == "reviewed_item_regen_0"
        assert regenerated.confidence == 0.0
        assert regenerated.status is ApprovalStatus.REGENERATED
        assert regenerated.metadata == {
            "original_item_id": "reviewed_item",
            "regeneration_attempt": 1,
            "feedback": decision.feedback,
            "generation": {
                "retry_count": 0,
                "max_retries": 3,
                "reasoning": "Applied the reviewer instruction.",
            },
        }

    @pytest.mark.parametrize(
        ("data", "schema_name"),
        [
            pytest.param(
                {
                    "query": "TensorFlow tutorial",
                    "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
                    "entity_types": "TECHNOLOGY",
                    "relationships": [],
                },
                "EntityExtractionExampleSchema",
                id="entity",
            ),
            pytest.param(
                {
                    "query": "TensorFlow tutorial",
                    "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
                    "relationships": [],
                    "enhanced_query": "TensorFlow(TECHNOLOGY) tutorial",
                    "chosen_agent": "search_agent",
                    "routing_confidence": 0.8,
                    "search_quality": 0.0,
                    "agent_success": False,
                    "processing_time": 0.0,
                },
                "RoutingExperienceSchema",
                id="routing",
            ),
            pytest.param(
                {
                    "query": "transformer clips",
                    "available_profiles": "video_colpali,video_colqwen",
                    "selected_profile": "video_colpali",
                    "reasoning": "Frames match.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "simple",
                },
                "ProfileSelectionExampleSchema",
                id="profile",
            ),
            pytest.param(
                {
                    "query": "PyTorch tutorial",
                    "enhanced_query": "PyTorch beginner tutorial",
                    "expansion_terms": ["beginner"],
                    "synonyms": ["guide"],
                    "context": "machine learning",
                    "reasoning": "Beginner narrows the query.",
                },
                "QueryEnhancementExampleSchema",
                id="query-enhancement",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_regeneration_rejects_unchanged_outputs_for_every_live_schema(
        self, data, schema_name
    ):
        query = data["query"]

        def generator(**_):
            return SimpleNamespace(
                updates={"query": query},
                reasoning="Returned the source unchanged.",
                _retry_count=0,
                _max_retries=3,
            )

        item = ReviewItem(item_id="unchanged", data=data, confidence=0.4)
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=False,
            feedback="Make a material correction.",
        )

        with pytest.raises(RuntimeError) as error:
            await self._handler(generator).process_rejection(item, decision)

        assert str(error.value) == (
            "Failed to regenerate unchanged after 2 regeneration attempts"
        )
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            f"item=unchanged schema={schema_name} "
            "regeneration did not change any training value"
        )

    @pytest.mark.asyncio
    async def test_regeneration_rejects_generator_that_ignores_exact_correction(self):
        original = {
            "query": "transformer clips",
            "available_profiles": "video_colpali,video_colqwen",
            "selected_profile": "video_colpali",
            "reasoning": "Frames match.",
            "query_intent": "video_search",
            "modality": "video",
            "complexity": "simple",
        }

        def generator(**_):
            return SimpleNamespace(
                updates={"reasoning": "Changed only the explanation."},
                reasoning="Ignored the selected-profile correction.",
                _retry_count=0,
                _max_retries=3,
            )

        decision = ReviewDecision(
            item_id="ignored",
            approved=False,
            feedback="Use the temporal profile.",
            corrections={"selected_profile": "video_colqwen"},
        )

        with pytest.raises(RuntimeError) as error:
            await self._handler(generator, attempts=1).process_rejection(
                ReviewItem(item_id="ignored", data=original, confidence=0.4),
                decision,
            )

        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == (
            "item=ignored schema=ProfileSelectionExampleSchema regenerated data "
            "does not apply correction selected_profile='video_colqwen'"
        )

    @pytest.mark.asyncio
    async def test_prompt_only_topics_guide_routing_without_becoming_schema_data(self):
        original = {
            "query": "find Curie footage",
            "entities": [{"text": "Curie", "type": "PERSON"}],
            "relationships": [],
            "enhanced_query": "find Curie(PERSON) footage",
            "chosen_agent": "search_agent",
            "routing_confidence": 0.8,
            "search_quality": 0.0,
            "agent_success": False,
            "processing_time": 0.0,
            "metadata": {
                "_outcome_metadata": {
                    "observed": True,
                    "required_field_semantics": {
                        "routing_confidence": "observed_gateway_confidence",
                        "search_quality": "unobserved_zero_sentinel",
                        "agent_success": "unobserved_false_sentinel",
                        "processing_time": "unobserved_zero_sentinel",
                    },
                }
            },
        }
        topics = ["radioactivity research"]

        def regenerate(**_):
            return SimpleNamespace(
                updates={
                    "query": "find Marie Curie radioactivity research",
                    "entities": [{"text": "Marie Curie", "type": "PERSON"}],
                    "topics": topics,
                },
                reasoning="Used the prompt-only topic to produce a schema query.",
                _retry_count=0,
                _max_retries=3,
            )

        regenerated = await self._handler(regenerate).process_rejection(
            ReviewItem(item_id="topic_guidance", data=original, confidence=0.4),
            ReviewDecision(
                item_id="topic_guidance",
                approved=False,
                feedback="Use the exact scientist and research topic.",
                corrections={
                    "entities": [{"text": "Marie Curie", "type": "PERSON"}],
                    "topics": topics,
                },
            ),
        )

        assert regenerated.data["query"] == ("find Marie Curie radioactivity research")
        assert regenerated.data["enhanced_query"] == (
            "find Marie Curie(PERSON) radioactivity research"
        )
        assert "topics" not in regenerated.data

    @pytest.mark.asyncio
    async def test_hung_regenerator_is_bounded_by_configured_deadline(self):
        release = threading.Event()

        def hang(**_):
            release.wait(timeout=2)
            raise AssertionError("hung generator continued past its deadline")

        item = ReviewItem(
            item_id="hung_profile",
            data={
                "query": "find transformer clips",
                "available_profiles": "video_colpali,video_colqwen",
                "selected_profile": "video_colpali",
                "reasoning": "Frames match.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "simple",
            },
            confidence=0.4,
        )
        started = time.monotonic()
        try:
            with pytest.raises(RuntimeError) as error:
                await self._handler(hang, attempts=1, timeout=0.05).process_rejection(
                    item,
                    ReviewDecision(
                        item_id=item.item_id,
                        approved=False,
                        feedback="Use temporal chunks.",
                    ),
                )
        finally:
            release.set()

        assert time.monotonic() - started < 0.5
        assert isinstance(error.value.__cause__, TimeoutError)
        assert str(error.value.__cause__) == (
            "synthetic feedback regeneration timed out after 0.05 seconds for "
            "item=hung_profile schema=ProfileSelectionExampleSchema attempt=1/1"
        )
        assert item.status is ApprovalStatus.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_concurrent_regenerations_keep_source_and_feedback_isolated(self):
        barrier = threading.Barrier(2)
        calls = []

        def regenerate(**kwargs):
            calls.append(kwargs)
            barrier.wait(timeout=1)
            source_query = kwargs["source_context"]["query"]
            framework = source_query.split()[0]
            return SimpleNamespace(
                updates={"query": f"{framework} exact tutorial"},
                reasoning=f"Applied {framework} instruction.",
                _retry_count=0,
                _max_retries=3,
            )

        handler = self._handler(regenerate)

        def item(framework):
            return ReviewItem(
                item_id=framework.lower(),
                data={
                    "query": f"{framework} tutorial",
                    "entities": [{"text": framework, "type": "TECHNOLOGY"}],
                    "entity_types": "TECHNOLOGY",
                    "relationships": [],
                },
                confidence=0.4,
            )

        def decision(framework):
            return ReviewDecision(
                item_id=framework.lower(),
                approved=False,
                feedback=f"Make only {framework} exact.",
            )

        pytorch, tensorflow = await asyncio.gather(
            handler.process_rejection(item("PyTorch"), decision("PyTorch")),
            handler.process_rejection(item("TensorFlow"), decision("TensorFlow")),
        )

        assert pytorch.data["query"] == "PyTorch exact tutorial"
        assert tensorflow.data["query"] == "TensorFlow exact tutorial"
        assert {
            (call["source_context"]["query"], call["reviewer_instruction"])
            for call in calls
        } == {
            ("PyTorch tutorial", "Make only PyTorch exact."),
            ("TensorFlow tutorial", "Make only TensorFlow exact."),
        }

    @pytest.mark.asyncio
    async def test_workflow_corrections_remain_explicit_observed_values(self):
        calls = []
        original = {
            "workflow_id": "workflow-17",
            "query": "summarize a video and write a report",
            "query_type": "VIDEO",
            "execution_time": 8.5,
            "success": False,
            "agent_sequence": ["video_search_agent", "summarizer"],
            "task_count": 2,
            "parallel_efficiency": 0.25,
            "confidence_score": 0.45,
            "error_details": "report generation was omitted",
            "metadata": {"run_id": "run-17"},
        }
        corrections = {
            "execution_time": 5.25,
            "success": True,
            "task_count": 3,
            "confidence_score": 0.91,
            "error_details": None,
        }
        result = await self._handler(
            lambda **kwargs: calls.append(kwargs)
        ).process_rejection(
            ReviewItem(item_id="workflow", data=original, confidence=0.4),
            ReviewDecision(
                item_id="workflow",
                approved=False,
                feedback="Apply the measured run.",
                corrections=corrections,
            ),
        )

        assert result.data == original | corrections
        assert calls == []


class TestApprovalConfig:
    """Test ApprovalConfig"""

    def test_approval_config_creation(self):
        """Test creating ApprovalConfig"""
        from cogniverse_foundation.config.unified_config import ApprovalConfig

        config = ApprovalConfig(
            enabled=True, confidence_threshold=0.9, storage_backend="phoenix"
        )

        assert config.enabled is True
        assert config.confidence_threshold == 0.9
        assert config.storage_backend == "phoenix"

    def test_approval_config_from_dict(self):
        """Test creating ApprovalConfig from dict"""
        from cogniverse_foundation.config.unified_config import ApprovalConfig

        data = {
            "enabled": True,
            "confidence_threshold": 0.88,
            "storage_backend": "database",
            "phoenix_project_name": "my_project",
        }

        config = ApprovalConfig.from_dict(data)

        assert config.enabled is True
        assert config.confidence_threshold == 0.88
        assert config.storage_backend == "database"
        assert config.phoenix_project_name == "my_project"

    def test_approval_config_to_dict(self):
        """Test converting ApprovalConfig to dict"""
        from cogniverse_foundation.config.unified_config import ApprovalConfig

        config = ApprovalConfig(
            enabled=True, confidence_threshold=0.92, reviewer_email="test@example.com"
        )

        config_dict = config.to_dict()

        assert config_dict["enabled"] is True
        assert config_dict["confidence_threshold"] == 0.92
        assert config_dict["reviewer_email"] == "test@example.com"

    @pytest.mark.parametrize(
        "value",
        [True, False, "0.8", float("nan"), float("inf"), -0.01, 1.01],
    )
    def test_approval_config_rejects_invalid_confidence_threshold(self, value):
        from cogniverse_foundation.config.unified_config import ApprovalConfig

        with pytest.raises(
            ValueError,
            match=(
                r"approval confidence_threshold must be a finite number in \[0, 1\]"
            ),
        ):
            ApprovalConfig(confidence_threshold=value)

        with pytest.raises(
            ValueError,
            match=(
                r"approval confidence_threshold must be a finite number in \[0, 1\]"
            ),
        ):
            ApprovalConfig.from_dict({"confidence_threshold": value})
