"""
Tests for Human-in-the-Loop Approval System

Tests approval interfaces, agents, confidence extraction, and feedback handling.
"""

import asyncio
import copy
import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cogniverse_agents.approval import (
    ApprovalBatch,
    ApprovalStatus,
    HumanApprovalAgent,
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

    def test_regenerated_item_remains_pending_for_human_review(self):
        original = ReviewItem(
            item_id="original",
            data={"query": "Find a framework tutorial"},
            confidence=0.4,
            status=ApprovalStatus.REJECTED,
        )
        replacement = ReviewItem(
            item_id="original_regen_0",
            data={"query": "Find exact PyTorch tutorials"},
            confidence=0.91,
            status=ApprovalStatus.REGENERATED,
        )
        approved = ReviewItem(
            item_id="approved",
            data={"query": "Find exact JAX tutorials"},
            confidence=0.88,
            status=ApprovalStatus.APPROVED,
        )

        batch = ApprovalBatch(
            batch_id="regenerated-review",
            items=[original, replacement, approved],
        )

        assert [(item.item_id, item.status) for item in batch.pending_review] == [
            (replacement.item_id, ApprovalStatus.REGENERATED)
        ]


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


class TestHumanApprovalAgent:
    """Test HumanApprovalAgent"""

    def test_agent_initialization(self):
        """Test initializing approval agent"""
        extractor = SyntheticDataConfidenceExtractor()
        agent = HumanApprovalAgent(
            confidence_extractor=extractor, confidence_threshold=0.85, storage=None
        )

        assert agent.confidence_extractor is not None
        assert agent.threshold == 0.85

    def test_approval_stats(self):
        """Test get_approval_stats"""
        extractor = SyntheticDataConfidenceExtractor()
        agent = HumanApprovalAgent(
            confidence_extractor=extractor, confidence_threshold=0.85, storage=None
        )

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
        ]

        batch = ApprovalBatch(batch_id="batch_001", items=items, context={})

        stats = agent.get_approval_stats(batch)

        assert stats["total_items"] == 2
        assert stats["auto_approved"] == 1
        assert stats["pending_review"] == 1
        assert stats["overall_approval_rate"] == 0.5
        assert "avg_confidence" in stats

    @pytest.mark.asyncio
    async def test_from_approval_config_threshold_drives_auto_approval(self):
        """The auto-approval threshold comes from ApprovalConfig and actually
        gates auto-approval: an item at confidence 0.75 auto-approves under a
        0.70 threshold but needs review under 0.80."""
        from cogniverse_foundation.config.unified_config import ApprovalConfig

        class _FixedConfidence:
            def extract(self, data):
                return 0.75

        items = [{"query": "q"}]

        agent_low = HumanApprovalAgent.from_approval_config(
            ApprovalConfig(confidence_threshold=0.70),
            confidence_extractor=_FixedConfidence(),
        )
        assert agent_low.threshold == 0.70
        batch_low = await agent_low.process_batch(
            items,
            "b_low",
            {"agent_type": "routing"},
        )
        assert len(batch_low.auto_approved) == 1
        assert len(batch_low.pending_review) == 0

        agent_high = HumanApprovalAgent.from_approval_config(
            ApprovalConfig(confidence_threshold=0.80),
            confidence_extractor=_FixedConfidence(),
        )
        assert agent_high.threshold == 0.80
        batch_high = await agent_high.process_batch(
            items,
            "b_high",
            {"agent_type": "routing"},
        )
        assert len(batch_high.auto_approved) == 0
        assert len(batch_high.pending_review) == 1
        assert [item.metadata for item in batch_high.items] == [
            {"agent_type": "routing", "batch_id": "b_high", "index": 0}
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("agent_type", [None, "", "  ", 17])
    async def test_process_batch_requires_canonical_agent_type(self, agent_type):
        class _RejectExtraction:
            def extract(self, data):
                raise AssertionError(f"confidence extraction ran for {data}")

        agent = HumanApprovalAgent(confidence_extractor=_RejectExtraction())

        with pytest.raises(
            ValueError,
            match="^context.agent_type must be a non-empty string$",
        ):
            await agent.process_batch(
                [{"query": "q"}],
                "invalid-agent-type",
                {"agent_type": agent_type},
            )

    @pytest.mark.asyncio
    async def test_submit_for_review_classifies_and_persists_prebuilt_batch(self):
        """submit_for_review re-classifies a caller-built batch against the
        threshold (>= auto-approve, else pending) using each item's own
        confidence, and persists it so the dashboard surfaces it. This is the
        path the finetuning synthetic-data flow uses."""

        class _Extractor:
            def extract(self, data):
                return 0.0

        class _FakeStorage:
            def __init__(self):
                self.tenant_id = "acme:prod"
                self.saved = []
                self.persisted = []

            async def save_batch(self, batch):
                self.saved.append(batch.batch_id)
                return batch.batch_id

            async def persist_approved_item(self, **kwargs):
                persisted = copy.deepcopy(kwargs["item"])
                assert persisted.status is ApprovalStatus.AUTO_APPROVED
                persisted.reviewed_at = kwargs["decision"].timestamp
                self.persisted.append(kwargs)
                return persisted

        storage = _FakeStorage()
        agent = HumanApprovalAgent(
            confidence_extractor=_Extractor(),
            confidence_threshold=0.85,
            storage=storage,
        )
        batch = ApprovalBatch(
            batch_id="synthetic_b1",
            items=[
                ReviewItem(
                    item_id="i_hi",
                    data={"query": "find alpha", "chosen_agent": "search"},
                    confidence=0.9,
                    metadata={"agent_type": "routing"},
                ),
                ReviewItem(
                    item_id="i_lo",
                    data={"query": "find beta", "chosen_agent": "search"},
                    confidence=0.8,
                    metadata={"agent_type": "routing"},
                ),
            ],
            context={"tenant_id": "acme:prod", "agent_type": "routing"},
        )

        result = await agent.submit_for_review(batch)

        assert result is batch
        assert [i.item_id for i in batch.auto_approved] == ["i_hi"]
        assert batch.approved == []
        assert [i.item_id for i in batch.pending_review] == ["i_lo"]
        assert batch.approved_count == 1
        assert storage.saved == ["synthetic_b1"]
        assert len(storage.persisted) == 1
        persisted = storage.persisted[0]
        assert persisted["batch_id"] == "synthetic_b1"
        assert persisted["dataset_name"] == "approved_synthetic_data-acme:prod"
        assert persisted["item"].item_id == "i_hi"
        assert persisted["decision"].reviewer == "cogniverse:auto-approval"
        assert persisted["decision"].timestamp == batch.items[0].created_at

    @pytest.mark.asyncio
    async def test_pending_items_include_their_owning_batch_id(self):
        first = ReviewItem(
            item_id="first",
            data={"query": "first"},
            confidence=0.4,
            metadata={"source": "phoenix"},
        )
        second = ReviewItem(
            item_id="second",
            data={"query": "second"},
            confidence=0.5,
        )

        class _Storage:
            async def get_pending_batches(self, context_filter):
                assert context_filter == {"tenant_id": "acme"}
                return [
                    ApprovalBatch(batch_id="batch-a", items=[first]),
                    ApprovalBatch(batch_id="batch-b", items=[second]),
                ]

        agent = HumanApprovalAgent(
            confidence_extractor=SyntheticDataConfidenceExtractor(),
            storage=_Storage(),
        )

        pending = await agent.get_pending_items({"tenant_id": "acme"})

        assert [item.item_id for item in pending] == ["first", "second"]
        assert [item.metadata for item in pending] == [
            {"source": "phoenix", "approval_batch_id": "batch-a"},
            {"approval_batch_id": "batch-b"},
        ]

    @pytest.mark.asyncio
    async def test_approved_decision_uses_failure_safe_storage_transaction(self):
        import copy

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        reviewed_at = datetime(2026, 8, 5, 4, 5, 6, tzinfo=timezone.utc)
        pending = ReviewItem(
            item_id="approval-item",
            data={"query": "find exact radium footage"},
            confidence=0.875,
        )
        batch = ApprovalBatch(
            batch_id="approval-batch",
            items=[pending],
            context={"tenant_id": "acme:approval", "optimizer": "routing"},
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        calls = []
        selections = []

        async def get_batch(batch_id):
            assert batch_id == batch.batch_id
            return batch

        async def persist_approved_item(**kwargs):
            calls.append(kwargs)
            approved = copy.deepcopy(kwargs["item"])
            approved.status = ApprovalStatus.APPROVED
            approved.reviewed_at = kwargs["decision"].timestamp
            return approved

        async def select_review_decision(**kwargs):
            selections.append(kwargs)

        storage.get_batch = get_batch
        storage.select_review_decision = select_review_decision
        storage.persist_approved_item = persist_approved_item
        agent = HumanApprovalAgent(
            confidence_extractor=SyntheticDataConfidenceExtractor(),
            storage=storage,
        )
        decision = ReviewDecision(
            item_id=pending.item_id,
            approved=True,
            reviewer="reviewer@example.com",
            timestamp=reviewed_at,
        )

        approved = await agent.apply_decision(batch.batch_id, decision)

        assert pending.status is ApprovalStatus.PENDING_REVIEW
        assert pending.reviewed_at is None
        assert approved is not pending
        assert approved.status is ApprovalStatus.APPROVED
        assert approved.reviewed_at == reviewed_at
        assert selections == [
            {
                "batch_id": batch.batch_id,
                "original_item_id": pending.item_id,
                "decision": decision,
            }
        ]
        assert calls == [
            {
                "batch_id": batch.batch_id,
                "dataset_name": "approved_synthetic_data-acme:approval",
                "item": pending,
                "decision": decision,
                "project_context": batch.context,
            }
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("approved", [True, False])
    async def test_decision_rejects_batch_tenant_mismatch_before_any_write(
        self, approved
    ):
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        pending = ReviewItem(
            item_id="mismatched-item",
            data={"query": "find exact radium footage"},
            confidence=0.5,
        )
        batch = ApprovalBatch(
            batch_id="mismatched-batch",
            items=[pending],
            context={"tenant_id": "other:tenant", "optimizer": "routing"},
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        writes = []

        async def get_batch(batch_id):
            assert batch_id == batch.batch_id
            return batch

        async def record_write(*args, **kwargs):
            writes.append((args, kwargs))

        storage.get_batch = get_batch
        storage.select_review_decision = record_write
        storage.persist_approved_item = record_write
        storage.replace_item = record_write
        storage.update_item = record_write
        storage.get_item_span_id = record_write
        storage.log_approval_decision = record_write
        agent = HumanApprovalAgent(
            confidence_extractor=SyntheticDataConfidenceExtractor(),
            storage=storage,
        )
        decision = ReviewDecision(
            item_id=pending.item_id,
            approved=approved,
            reviewer="reviewer@example.com",
            timestamp=datetime(2026, 8, 5, 4, 5, 6, tzinfo=timezone.utc),
        )

        with pytest.raises(
            ValueError,
            match=(
                "Approval batch tenant does not match its storage: "
                "batch=mismatched-batch context_tenant=other:tenant "
                "storage_tenant=acme:approval"
            ),
        ):
            await agent.apply_decision(batch.batch_id, decision)

        assert writes == []
        assert pending.status is ApprovalStatus.PENDING_REVIEW
        assert pending.reviewed_at is None


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


class TestApprovalStorageContract:
    """The ApprovalStorage ABC must declare the contract its callers use.

    human_approval_agent calls update_item(item, batch_id=...); the ABC
    previously declared update_item(item) only, so a faithful subclass would
    break those call sites.
    """

    def test_update_item_abc_declares_batch_id(self):
        import inspect

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
        from cogniverse_core.approval.interfaces import ApprovalStorage

        abc_params = inspect.signature(ApprovalStorage.update_item).parameters
        impl_params = inspect.signature(ApprovalStorageImpl.update_item).parameters

        assert "batch_id" in abc_params
        assert "batch_id" in impl_params

    @pytest.mark.asyncio
    async def test_integrity_store_preserves_replace_dataset_contract(self):
        import pandas as pd

        from cogniverse_agents.approval.approval_storage import (
            _ApprovedDatasetIntegrityStore,
        )
        from cogniverse_foundation.telemetry.providers.base import DatasetStore

        class _Delegate:
            def __init__(self):
                self.calls = []

            async def get_dataset(self, name):
                self.calls.append(("get", name))
                return pd.DataFrame([{"value": "old"}])

            async def delete_dataset(self, name):
                self.calls.append(("delete", name))
                return True

            async def create_dataset(self, name, data, metadata=None):
                self.calls.append(("create", name, data.to_dict("records"), metadata))
                return "replacement-id"

            async def append_to_dataset(self, name, data, metadata=None):
                raise AssertionError("replace must not append")

        delegate = _Delegate()
        store = _ApprovedDatasetIntegrityStore(delegate)
        replacement = pd.DataFrame([{"value": "new"}])

        dataset_id = await store.replace_dataset(
            "optimizer-artifact",
            replacement,
            {"purpose": "active"},
        )

        assert isinstance(store, DatasetStore)
        assert dataset_id == "replacement-id"
        assert delegate.calls == [
            ("get", "optimizer-artifact"),
            ("delete", "optimizer-artifact"),
            (
                "create",
                "optimizer-artifact",
                [{"value": "new"}],
                {"purpose": "active"},
            ),
        ]

    def test_integrity_validation_accepts_exact_numeric_boundary_strings(self):
        from cogniverse_agents.approval.approval_storage import (
            _provider_value_matches_canonical,
        )

        assert _provider_value_matches_canonical("0.25", 0.25) is True
        assert _provider_value_matches_canonical("0", 0) is True
        assert _provider_value_matches_canonical("1", 1) is True
        assert (
            _provider_value_matches_canonical(
                "{'reviewer': 'reviewer@example.com', 'corrections': {}}",
                {"reviewer": "reviewer@example.com", "corrections": {}},
            )
            is True
        )

    @pytest.mark.parametrize(
        ("observed", "canonical"),
        [
            ("0.26", 0.25),
            (" 0.25", 0.25),
            ("NaN", 0.25),
            ("Infinity", 0.25),
            ("00", 0),
            ("1.0", 1),
            (True, 1),
            ("true", True),
            (
                "{'reviewer': 'mallory@example.com', 'corrections': {}}",
                {"reviewer": "reviewer@example.com", "corrections": {}},
            ),
            ("__import__('os').system('false')", {"corrections": {}}),
        ],
    )
    def test_integrity_validation_rejects_changed_numeric_boundary_values(
        self, observed, canonical
    ):
        from cogniverse_agents.approval.approval_storage import (
            _provider_value_matches_canonical,
        )

        assert _provider_value_matches_canonical(observed, canonical) is False


class TestApprovedItemPersistence:
    @staticmethod
    def _storage(monkeypatch, events, dataset_error=None):
        import pandas as pd

        import cogniverse_agents.approval.approval_storage as storage_module
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
        from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

        class _Redis:
            async def set(self, key, token, *, nx, px):
                assert key == (
                    "cogniverse:approval:dataset-lock:"
                    "acme:approval:approved_synthetic_data-acme:approval"
                )
                assert isinstance(token, str) and len(token) == 32
                assert nx is True
                assert px == 120_000
                events.append("lock")
                return True

            async def eval(self, script, key_count, key, token):
                assert "redis.call('del'" in script
                assert key_count == 1
                assert key.endswith("approved_synthetic_data-acme:approval")
                assert isinstance(token, str) and len(token) == 32
                events.append("unlock")
                return 1

            async def aclose(self):
                return None

        class _Datasets:
            def __init__(self):
                self.frame = None
                self.write_count = 0

            async def get_dataset(self, name):
                assert name == "approved_synthetic_data-acme:approval"
                if self.frame is None:
                    raise DatasetNotFoundError(name)
                return self.frame.copy(deep=True)

            async def create_dataset(self, name, data):
                events.append("dataset")
                if dataset_error is not None:
                    raise dataset_error
                self.write_count += 1
                self.frame = pd.DataFrame([{"input": data.iloc[0].to_dict()}])
                return "dataset-id"

            async def append_to_dataset(self, name, data):
                events.append("dataset")
                if dataset_error is not None:
                    raise dataset_error
                self.write_count += 1
                appended = pd.DataFrame([{"input": data.iloc[0].to_dict()}])
                self.frame = pd.concat([self.frame, appended], ignore_index=True)

        redis = _Redis()

        def from_url(url, **options):
            assert url == "redis://approval:6379/0"
            assert options == {
                "decode_responses": True,
                "socket_connect_timeout": 2.0,
                "socket_timeout": 2.0,
                "retry_on_timeout": False,
            }
            return redis

        monkeypatch.setattr(
            storage_module,
            "aioredis",
            SimpleNamespace(from_url=from_url),
            raising=False,
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.redis_url = "redis://approval:6379/0"
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.provider = SimpleNamespace(datasets=_Datasets())

        class _ReplacementRecords:
            def __init__(self):
                self.selected_decision = None

            async def select_review_decision(self, **kwargs):
                from cogniverse_agents.approval.replacement_store import (
                    CanonicalReplacementRecord,
                )

                if self.selected_decision is None:
                    self.selected_decision = dict(kwargs["candidate"])
                payload = dict(self.selected_decision)
                record_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return CanonicalReplacementRecord(
                    payload=payload,
                    json=record_json,
                    sha256=hashlib.sha256(record_json.encode()).hexdigest(),
                )

        storage._replacement_records = _ReplacementRecords()

        async def get_item_span_id(item_id, batch_id=None):
            assert (item_id, batch_id) == ("approval-item", "approval-batch")
            return "approval-span"

        async def log_approval_decision(**kwargs):
            events.append("decision")
            assert kwargs == {
                "span_id": "approval-span",
                "item_id": "approval-item",
                "approved": True,
                "feedback": "The result is exact.",
                "reviewer": "reviewer@example.com",
                "decision_timestamp": datetime(
                    2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc
                ),
            }
            return True

        async def update_item(item, batch_id=None):
            events.append("status")
            assert item.status is ApprovalStatus.APPROVED
            assert item.reviewed_at == datetime(
                2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc
            )
            assert batch_id == "approval-batch"

        storage.get_item_span_id = get_item_span_id
        storage.log_approval_decision = log_approval_decision
        storage.update_item = update_item
        return storage

    @pytest.mark.asyncio
    async def test_dataset_lock_uses_bounded_redis_socket_timeouts(self, monkeypatch):
        import cogniverse_agents.approval.approval_storage as storage_module
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        connection_options = {}

        class Redis:
            async def set(self, _key, _token, *, nx, px):
                assert nx is True
                assert px == 120_000
                return True

            async def eval(self, script, _key_count, _key, _token):
                assert "redis.call('del'" in script
                return 1

            async def aclose(self):
                return None

        def from_url(url, **options):
            connection_options["url"] = url
            connection_options.update(options)
            return Redis()

        monkeypatch.setattr(
            storage_module,
            "aioredis",
            SimpleNamespace(from_url=from_url),
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.redis_url = "redis://approval:6379/0"

        async with storage._approval_dataset_lock(
            "approved_synthetic_data-acme:approval"
        ):
            pass

        assert connection_options == {
            "url": "redis://approval:6379/0",
            "decode_responses": True,
            "socket_connect_timeout": 2.0,
            "socket_timeout": 2.0,
            "retry_on_timeout": False,
        }

    @pytest.mark.asyncio
    async def test_dataset_lock_renewal_error_aborts_protected_work(self, monkeypatch):
        from redis.exceptions import ConnectionError as RedisConnectionError

        import cogniverse_agents.approval.approval_storage as storage_module
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        events = []
        redis_error = RedisConnectionError("connection closed during renewal")

        class Redis:
            async def set(self, _key, _token, *, nx, px):
                assert nx is True
                assert px == 60
                events.append("acquired")
                return True

            async def eval(self, script, _key_count, _key, _token, *args):
                if "pexpire" in script:
                    assert args == (60,)
                    events.append("renewal_failed")
                    raise redis_error
                assert "redis.call('del'" in script
                events.append("released")
                return 1

            async def aclose(self):
                events.append("closed")

        monkeypatch.setattr(
            storage_module,
            "aioredis",
            SimpleNamespace(from_url=lambda _url, **_options: Redis()),
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.redis_url = "redis://approval:6379/0"
        storage._DATASET_LOCK_LEASE_MS = 60

        with pytest.raises(RuntimeError) as error:
            async with storage._approval_dataset_lock(
                "approved_synthetic_data-acme:approval"
            ):
                events.append("protected_started")
                await asyncio.sleep(0.25)
                events.append("protected_continued")

        assert str(error.value) == (
            "Failed to renew approved dataset lock: "
            "tenant=acme:approval dataset=approved_synthetic_data-acme:approval"
        )
        assert error.value.__cause__ is redis_error
        assert events == [
            "acquired",
            "protected_started",
            "renewal_failed",
            "released",
            "closed",
        ]

    @pytest.mark.asyncio
    async def test_dataset_lock_ownership_loss_aborts_protected_work(self, monkeypatch):
        import cogniverse_agents.approval.approval_storage as storage_module
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        events = []

        class Redis:
            async def set(self, _key, _token, *, nx, px):
                assert (nx, px) == (True, 60)
                events.append("acquired")
                return True

            async def eval(self, script, _key_count, _key, _token, *args):
                if "pexpire" in script:
                    assert args == (60,)
                    events.append("renewal_rejected")
                    return 0
                assert "redis.call('del'" in script
                events.append("release_rejected")
                return 0

            async def aclose(self):
                events.append("closed")

        monkeypatch.setattr(
            storage_module,
            "aioredis",
            SimpleNamespace(from_url=lambda _url, **_options: Redis()),
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.redis_url = "redis://approval:6379/0"
        storage._DATASET_LOCK_LEASE_MS = 60

        with pytest.raises(RuntimeError) as error:
            async with storage._approval_dataset_lock(
                "approved_synthetic_data-acme:approval"
            ):
                events.append("protected_started")
                await asyncio.sleep(0.25)
                events.append("protected_continued")

        assert str(error.value) == (
            "Approved dataset lock ownership was lost during renewal: "
            "tenant=acme:approval dataset=approved_synthetic_data-acme:approval"
        )
        assert events == [
            "acquired",
            "protected_started",
            "renewal_rejected",
            "release_rejected",
            "closed",
        ]

    @pytest.mark.asyncio
    async def test_dataset_commit_precedes_decision_and_status(self, monkeypatch):
        events = []
        storage = self._storage(monkeypatch, events)
        reviewed_at = datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc)
        item = ReviewItem(
            item_id="approval-item",
            data={
                "query": "find Marie Curie laboratory footage",
                "chosen_agent": "video_search_agent",
            },
            confidence=0.875,
            metadata={"agent_type": "routing", "source": "vespa"},
            created_at=datetime(2026, 8, 5, 1, 2, 3, tzinfo=timezone.utc),
        )
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            feedback="The result is exact.",
            corrections={"query": "find Marie Curie laboratory footage"},
            reviewer="reviewer@example.com",
            timestamp=reviewed_at,
        )

        approved = await storage.persist_approved_item(
            batch_id="approval-batch",
            dataset_name="approved_synthetic_data-acme:approval",
            item=item,
            decision=decision,
            project_context={
                "tenant_id": "acme:approval",
                "optimizer": "routing",
            },
        )

        assert events == ["lock", "dataset", "decision", "status", "unlock"]
        assert item.status is ApprovalStatus.PENDING_REVIEW
        assert item.reviewed_at is None
        assert approved is not item
        assert approved.status is ApprovalStatus.APPROVED
        assert approved.reviewed_at == reviewed_at
        assert approved.metadata == {
            "agent_type": "routing",
            "source": "vespa",
            "decision": {
                "reviewer": "reviewer@example.com",
                "feedback": "The result is exact.",
                "corrections": {"query": "find Marie Curie laboratory footage"},
                "timestamp": "2026-08-05T03:04:05+00:00",
            },
        }
        record = storage.provider.datasets.frame.iloc[0]["input"]
        assert record["item_id"] == item.item_id
        assert record["status"] == ApprovalStatus.APPROVED.value
        assert record["reviewed_at"] == "2026-08-05T03:04:05+00:00"
        assert record["metadata.decision"] == approved.metadata["decision"]
        assert record["metadata.approval_decision_sha256"] == (
            "de98a4111a1477b7740d03a9782e11348ddb32f62e56dfb6076b7005f6672b9b"
        )
        assert record["metadata.approval_record_json"] == (
            '{"chosen_agent":"video_search_agent","confidence":0.875,'
            '"context.optimizer":"routing","context.tenant_id":"acme:approval",'
            '"created_at":"2026-08-05T01:02:03+00:00","item_id":"approval-item",'
            '"metadata.agent_type":"routing",'
            '"metadata.approval_decision_sha256":'
            '"de98a4111a1477b7740d03a9782e11348ddb32f62e56dfb6076b7005f6672b9b",'
            '"metadata.approval_decision_timestamp":"2026-08-05T03:04:05+00:00",'
            '"metadata.decision":{"corrections":'
            '{"query":"find Marie Curie laboratory footage"},'
            '"feedback":"The result is exact.",'
            '"reviewer":"reviewer@example.com",'
            '"timestamp":"2026-08-05T03:04:05+00:00"},'
            '"metadata.source":"vespa",'
            '"query":"find Marie Curie laboratory footage",'
            '"reviewed_at":"2026-08-05T03:04:05+00:00","status":"approved"}'
        )
        assert record["metadata.approval_record_sha256"] == (
            "ce551c817f4194fe48716e636520ea69dbe3027bba3f56d47db2263019a57adf"
        )

    @pytest.mark.parametrize(
        ("payload_factory", "cause_message"),
        [
            (
                lambda _pd: None,
                "Approved dataset payload is not a pandas DataFrame: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval got=NoneType",
            ),
            (
                lambda _pd: [{"input": {}}],
                "Approved dataset payload is not a pandas DataFrame: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval got=list",
            ),
            (
                lambda pd: pd.DataFrame([{"output": {}}]),
                "Approved dataset payload has no input column: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval",
            ),
            (
                lambda pd: pd.DataFrame([{"input": None}]),
                "Approved dataset row has no input record: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_existing_dataset_rejects_invalid_snapshot_before_write(
        self,
        monkeypatch,
        payload_factory,
        cause_message,
    ):
        import pandas as pd

        events = []
        storage = self._storage(monkeypatch, events)

        async def get_dataset(name):
            assert name == "approved_synthetic_data-acme:approval"
            return payload_factory(pd)

        storage.provider.datasets.get_dataset = get_dataset
        item = ReviewItem(
            item_id="approval-item",
            data={"query": "find Marie Curie", "chosen_agent": "video_search_agent"},
            confidence=0.875,
            metadata={"agent_type": "routing"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc),
        )

        with pytest.raises(RuntimeError) as error:
            await storage.append_to_training_dataset(
                dataset_name="approved_synthetic_data-acme:approval",
                items=[item],
                project_context={"tenant_id": "acme:approval"},
            )

        assert str(error.value) == (
            "Failed to append items to training dataset: "
            "tenant=acme:approval dataset=approved_synthetic_data-acme:approval"
        )
        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == cause_message
        assert storage.provider.datasets.write_count == 0
        assert events == ["lock", "unlock"]

    @pytest.mark.asyncio
    async def test_existing_dataset_rejects_unrelated_duplicate_ids(self, monkeypatch):
        import pandas as pd

        events = []
        storage = self._storage(monkeypatch, events)
        reviewed_at = datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc)
        first = ReviewItem(
            item_id="approval-item",
            data={"query": "find Marie Curie", "chosen_agent": "video_search_agent"},
            confidence=0.875,
            metadata={"agent_type": "routing"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=reviewed_at,
        )
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data-acme:approval",
            items=[first],
            project_context={"tenant_id": "acme:approval"},
        )
        storage.provider.datasets.frame = pd.concat(
            [storage.provider.datasets.frame, storage.provider.datasets.frame],
            ignore_index=True,
        )
        second = ReviewItem(
            item_id="approval-item-2",
            data={"query": "find JAX", "chosen_agent": "video_search_agent"},
            confidence=0.9,
            metadata={"agent_type": "routing"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=reviewed_at,
        )

        with pytest.raises(RuntimeError) as error:
            await storage.append_to_training_dataset(
                dataset_name="approved_synthetic_data-acme:approval",
                items=[second],
                project_context={"tenant_id": "acme:approval"},
            )

        assert str(error.value.__cause__) == (
            "Approved dataset contains duplicate item records: "
            "tenant=acme:approval dataset=approved_synthetic_data-acme:approval "
            "item=approval-item count=2"
        )
        assert storage.provider.datasets.write_count == 1
        assert events == ["lock", "dataset", "unlock", "lock", "unlock"]

    @pytest.mark.parametrize(
        ("corruption", "cause_message"),
        [
            (
                "missing-record-json",
                "Approved dataset item has invalid "
                "metadata.approval_record_json: tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0 "
                "item=approval-item",
            ),
            (
                "invalid-record-hash",
                "Approved dataset item has invalid "
                "metadata.approval_record_sha256: tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0 "
                "item=approval-item",
            ),
            (
                "content-hash-mismatch",
                "Approved dataset item canonical content hash does not match: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0 "
                "item=approval-item",
            ),
            (
                "noncanonical-content",
                "Approved dataset item content is not canonical JSON: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0 "
                "item=approval-item",
            ),
            (
                "decision-content-mismatch",
                "Approved dataset item decision content hash does not match: "
                "tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0 "
                "item=approval-item",
            ),
            (
                "naive-decision-timestamp",
                "Approved dataset item has naive "
                "metadata.approval_decision_timestamp: tenant=acme:approval "
                "dataset=approved_synthetic_data-acme:approval row=0 "
                "item=approval-item",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_existing_dataset_rejects_corrupt_integrity_before_write(
        self,
        monkeypatch,
        corruption,
        cause_message,
    ):
        import hashlib
        import json

        events = []
        storage = self._storage(monkeypatch, events)
        reviewed_at = datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc)
        first = ReviewItem(
            item_id="approval-item",
            data={"query": "find Marie Curie", "chosen_agent": "video_search_agent"},
            confidence=0.875,
            metadata={"agent_type": "routing"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=reviewed_at,
        )
        await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data-acme:approval",
            items=[first],
            project_context={"tenant_id": "acme:approval"},
        )
        record = storage.provider.datasets.frame.iloc[0]["input"]
        if corruption == "missing-record-json":
            record.pop("metadata.approval_record_json")
        elif corruption == "invalid-record-hash":
            record["metadata.approval_record_sha256"] = "not-a-sha256"
        elif corruption == "content-hash-mismatch":
            record["metadata.approval_record_json"] = record[
                "metadata.approval_record_json"
            ].replace("Marie Curie", "Pierre Curie", 1)
        else:
            canonical = json.loads(record["metadata.approval_record_json"])
            if corruption == "noncanonical-content":
                canonical_json = json.dumps(canonical, sort_keys=True)
            elif corruption == "decision-content-mismatch":
                canonical["metadata.decision"] = {"reviewer": "mallory@example.com"}
                canonical_json = json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            elif corruption == "naive-decision-timestamp":
                naive_timestamp = "2026-08-05T03:04:05"
                canonical["reviewed_at"] = naive_timestamp
                canonical["metadata.approval_decision_timestamp"] = naive_timestamp
                record["metadata.approval_decision_timestamp"] = naive_timestamp
                canonical_json = json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            else:
                raise AssertionError(f"unsupported corruption {corruption}")
            record["metadata.approval_record_json"] = canonical_json
            record["metadata.approval_record_sha256"] = hashlib.sha256(
                canonical_json.encode("utf-8")
            ).hexdigest()

        snapshot_before_attempt = storage.provider.datasets.frame.to_dict("records")
        second = ReviewItem(
            item_id="approval-item-2",
            data={"query": "find JAX", "chosen_agent": "video_search_agent"},
            confidence=0.9,
            metadata={"agent_type": "routing"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=reviewed_at,
        )

        with pytest.raises(RuntimeError) as error:
            await storage.append_to_training_dataset(
                dataset_name="approved_synthetic_data-acme:approval",
                items=[second],
                project_context={"tenant_id": "acme:approval"},
            )

        assert str(error.value.__cause__) == cause_message
        assert storage.provider.datasets.frame.to_dict("records") == (
            snapshot_before_attempt
        )
        assert storage.provider.datasets.write_count == 1
        assert events == ["lock", "dataset", "unlock", "lock", "unlock"]

    @pytest.mark.asyncio
    async def test_native_data_confidence_mismatch_is_rejected_before_write(
        self, monkeypatch
    ):
        events = []
        storage = self._storage(monkeypatch, events)
        item = ReviewItem(
            item_id="profile-item",
            data={
                "query": "find exact slide text",
                "available_profiles": "video_colpali,video_colqwen",
                "selected_profile": "video_colpali",
                "reasoning": "The frame profile preserves exact slide text.",
                "query_intent": "text_search",
                "modality": "video",
                "complexity": "simple",
                "confidence": 0.6,
            },
            confidence=0.7,
            metadata={"agent_type": "profile_selection"},
            status=ApprovalStatus.APPROVED,
            reviewed_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        )

        with pytest.raises(RuntimeError) as error:
            await storage.append_to_training_dataset(
                dataset_name="approved_synthetic_data-acme:approval",
                items=[item],
            )

        assert str(error.value.__cause__) == (
            "Training dataset item 'profile-item' data confidence must exactly "
            "match ReviewItem.confidence: data=0.6 item=0.7"
        )
        assert storage.provider.datasets.write_count == 0
        assert events == ["lock", "unlock"]

    @pytest.mark.asyncio
    async def test_dataset_failure_prevents_decision_and_status(self, monkeypatch):
        events = []
        dataset_error = ConnectionError("Phoenix dataset write failed")
        storage = self._storage(monkeypatch, events, dataset_error=dataset_error)
        item = ReviewItem(
            item_id="approval-item",
            data={
                "query": "find Marie Curie laboratory footage",
                "chosen_agent": "video_search_agent",
            },
            confidence=0.875,
            metadata={"agent_type": "routing"},
        )
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            feedback="The result is exact.",
            reviewer="reviewer@example.com",
            timestamp=datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc),
        )

        with pytest.raises(RuntimeError) as error:
            await storage.persist_approved_item(
                batch_id="approval-batch",
                dataset_name="approved_synthetic_data-acme:approval",
                item=item,
                decision=decision,
                project_context={"tenant_id": "acme:approval"},
            )

        assert str(error.value) == (
            "Failed to persist approved item: tenant=acme:approval "
            "dataset=approved_synthetic_data-acme:approval "
            "batch=approval-batch item=approval-item"
        )
        assert error.value.__cause__ is dataset_error
        assert events == ["lock", "dataset", "unlock"]
        assert item.status is ApprovalStatus.PENDING_REVIEW
        assert item.reviewed_at is None

    @pytest.mark.asyncio
    async def test_fresh_retry_reuses_first_persisted_decision_timestamp(
        self, monkeypatch
    ):
        events = []
        storage = self._storage(monkeypatch, events)
        item = ReviewItem(
            item_id="approval-item",
            data={
                "query": "find Marie Curie laboratory footage",
                "chosen_agent": "video_search_agent",
            },
            confidence=0.875,
            metadata={"agent_type": "routing"},
        )
        first_timestamp = datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc)
        retry_timestamp = datetime(2026, 8, 5, 3, 9, 10, tzinfo=timezone.utc)

        async def fail_after_dataset_commit(**_kwargs):
            events.append("decision-failure")
            raise ConnectionError("approval annotation unavailable")

        successful_log = storage.log_approval_decision
        storage.log_approval_decision = fail_after_dataset_commit
        with pytest.raises(RuntimeError) as first_error:
            await storage.persist_approved_item(
                batch_id="approval-batch",
                dataset_name="approved_synthetic_data-acme:approval",
                item=item,
                decision=ReviewDecision(
                    item_id=item.item_id,
                    approved=True,
                    feedback="The result is exact.",
                    corrections={"query": "find Marie Curie laboratory footage"},
                    reviewer="reviewer@example.com",
                    timestamp=first_timestamp,
                ),
                project_context={
                    "tenant_id": "acme:approval",
                    "optimizer": "routing",
                },
            )

        assert isinstance(first_error.value.__cause__, ConnectionError)
        assert item.status is ApprovalStatus.PENDING_REVIEW
        assert item.reviewed_at is None

        storage.log_approval_decision = successful_log
        approved = await storage.persist_approved_item(
            batch_id="approval-batch",
            dataset_name="approved_synthetic_data-acme:approval",
            item=item,
            decision=ReviewDecision(
                item_id=item.item_id,
                approved=True,
                feedback="The result is exact.",
                corrections={"query": "find Marie Curie laboratory footage"},
                reviewer="reviewer@example.com",
                timestamp=retry_timestamp,
            ),
            project_context={
                "tenant_id": "acme:approval",
                "optimizer": "routing",
            },
        )

        assert storage.provider.datasets.write_count == 1
        record = storage.provider.datasets.frame.iloc[0]["input"]
        assert record["reviewed_at"] == first_timestamp.isoformat()
        assert record["metadata.decision"]["timestamp"] == first_timestamp.isoformat()
        assert approved.reviewed_at == first_timestamp
        assert approved.metadata["decision"]["timestamp"] == first_timestamp.isoformat()
        assert events == [
            "lock",
            "dataset",
            "decision-failure",
            "unlock",
            "lock",
            "decision",
            "status",
            "unlock",
        ]

    @pytest.mark.asyncio
    async def test_dataset_write_is_idempotent_and_rejects_conflicting_record(
        self, monkeypatch
    ):
        events = []
        storage = self._storage(monkeypatch, events)
        item = ReviewItem(
            item_id="approval-item",
            data={
                "query": "find Marie Curie laboratory footage",
                "chosen_agent": "video_search_agent",
            },
            confidence=0.875,
            metadata={"agent_type": "routing"},
        )
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            feedback="The result is exact.",
            reviewer="reviewer@example.com",
            timestamp=datetime(2026, 8, 5, 3, 4, 5, tzinfo=timezone.utc),
        )
        approved = storage._approved_item_copy(item, decision)

        first = await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data-acme:approval",
            items=[approved],
            project_context={"tenant_id": "acme:approval"},
        )
        second = await storage.append_to_training_dataset(
            dataset_name="approved_synthetic_data-acme:approval",
            items=[approved],
            project_context={"tenant_id": "acme:approval"},
        )

        assert first is True
        assert second is True
        assert storage.provider.datasets.write_count == 1
        assert storage.provider.datasets.frame["input"].map(
            lambda record: record["item_id"]
        ).tolist() == ["approval-item"]
        assert events == ["lock", "dataset", "unlock", "lock", "unlock"]

        conflicting = storage._approved_item_copy(item, decision)
        conflicting.data["query"] = "find an unrelated video"
        with pytest.raises(RuntimeError) as error:
            await storage.append_to_training_dataset(
                dataset_name="approved_synthetic_data-acme:approval",
                items=[conflicting],
                project_context={"tenant_id": "acme:approval"},
            )

        assert str(error.value) == (
            "Failed to append items to training dataset: "
            "tenant=acme:approval dataset=approved_synthetic_data-acme:approval"
        )
        assert str(error.value.__cause__) == (
            "Approved dataset item conflicts with immutable record: "
            "tenant=acme:approval dataset=approved_synthetic_data-acme:approval "
            "item=approval-item"
        )
        assert storage.provider.datasets.write_count == 1
        assert events == [
            "lock",
            "dataset",
            "unlock",
            "lock",
            "unlock",
            "lock",
            "unlock",
        ]


class TestApprovalStorageEventLoop:
    """Telemetry-indexing delays must not block the event loop.

    get_batch / get_pending_batches / get_item_span_id paused with a blocking
    time.sleep inside async methods, freezing every other coroutine on the
    worker for the full indexing-lag window. They must await asyncio.sleep.
    """

    @pytest.mark.asyncio
    async def test_save_batch_rejects_tenant_mismatch_before_span_export(self):
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        span_calls = []
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.project_name = "synthetic_data"
        storage.telemetry_manager = SimpleNamespace(
            span=lambda **kwargs: span_calls.append(kwargs)
        )
        batch = ApprovalBatch(
            batch_id="wrong-tenant-batch",
            items=[
                ReviewItem(
                    item_id="routing-item",
                    data={"query": "find the launch"},
                    confidence=0.25,
                )
            ],
            context={"tenant_id": "other:tenant", "agent_type": "routing"},
        )

        with pytest.raises(
            ValueError,
            match=(
                "Approval batch tenant does not match its storage: "
                "batch=wrong-tenant-batch context_tenant=other:tenant "
                "storage_tenant=acme:approval"
            ),
        ):
            await storage.save_batch(batch)

        assert span_calls == []

    @pytest.mark.asyncio
    async def test_get_batch_rejects_span_response_without_batch_id(self):
        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"

        with pytest.raises(
            RuntimeError,
            match="Approval span response is missing attributes.batch_id",
        ):
            await storage.get_batch(
                "batch-a",
                spans_df=pd.DataFrame(
                    [{"name": "approval_batch", "context.span_id": "batch-span-a"}]
                ),
            )

    @pytest.mark.asyncio
    async def test_get_batch_collapses_identical_retry_roots_and_items(self):
        import json
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.provider = SimpleNamespace(
            annotations=SimpleNamespace(
                get_annotations=AsyncMock(return_value=pd.DataFrame())
            )
        )
        created_at = datetime(2026, 8, 5, tzinfo=timezone.utc).isoformat()
        root_attributes = {
            "attributes.batch_id": "duplicate-batch",
            "attributes.total_items": 1,
            "attributes.auto_approved": 0,
            "attributes.pending_review": 1,
            "attributes.context": '{"tenant_id":"acme:approval"}',
            "attributes.created_at": created_at,
        }
        item_attributes = {
            "attributes.item_id": "retry-item",
            "attributes.status": "pending_review",
            "attributes.created_at": created_at,
            "attributes.reviewed_at": None,
            "attributes.data": json.dumps({"query": "find the incident"}),
            "attributes.metadata": json.dumps({"approval_batch_id": "duplicate-batch"}),
            "attributes.confidence": 0.4,
        }
        spans = pd.DataFrame(
            [
                {
                    "name": "approval_batch",
                    **root_attributes,
                    "context.span_id": "root-one",
                    "parent_id": None,
                },
                {
                    "name": "approval_batch",
                    **root_attributes,
                    "context.span_id": "root-two",
                    "parent_id": None,
                },
                {
                    "name": "approval_item",
                    **item_attributes,
                    "context.span_id": "item-one",
                    "parent_id": "root-one",
                },
                {
                    "name": "approval_item",
                    **item_attributes,
                    "context.span_id": "item-two",
                    "parent_id": "root-two",
                },
            ]
        )

        batch = await storage.get_batch("duplicate-batch", spans_df=spans)

        assert batch.batch_id == "duplicate-batch"
        assert batch.context == {"tenant_id": "acme:approval"}
        assert [(item.item_id, item.data, item.confidence) for item in batch.items] == [
            ("retry-item", {"query": "find the incident"}, 0.4)
        ]

    @pytest.mark.asyncio
    async def test_get_batch_rejects_conflicting_retry_roots(self):
        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        spans = pd.DataFrame(
            [
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "conflicting-batch",
                    "attributes.context": '{"source":"first"}',
                    "context.span_id": "root-one",
                },
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "conflicting-batch",
                    "attributes.context": '{"source":"second"}',
                    "context.span_id": "root-two",
                },
            ]
        )

        with pytest.raises(RuntimeError, match="conflicting root spans"):
            await storage.get_batch("conflicting-batch", spans_df=spans)

    @pytest.mark.asyncio
    async def test_get_batch_rejects_malformed_item_without_returning_partial_view(
        self,
    ):
        import json
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.provider = SimpleNamespace(
            annotations=SimpleNamespace(
                get_annotations=AsyncMock(return_value=pd.DataFrame())
            )
        )
        created_at = datetime(2026, 8, 5, tzinfo=timezone.utc).isoformat()
        spans = pd.DataFrame(
            [
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "corrupt-batch",
                    "attributes.context": '{"tenant_id":"acme:approval"}',
                    "context.span_id": "batch-root",
                    "parent_id": None,
                },
                {
                    "name": "approval_item",
                    "attributes.batch_id": None,
                    "attributes.item_id": "valid-item",
                    "attributes.status": "pending_review",
                    "attributes.created_at": created_at,
                    "attributes.data": '{"query":"valid"}',
                    "attributes.metadata": "{}",
                    "attributes.confidence": 0.75,
                    "context.span_id": "valid-span",
                    "parent_id": "batch-root",
                },
                {
                    "name": "approval_item",
                    "attributes.batch_id": None,
                    "attributes.item_id": "broken-item",
                    "attributes.status": "pending_review",
                    "attributes.created_at": created_at,
                    "attributes.data": "{broken-json",
                    "attributes.metadata": "{}",
                    "attributes.confidence": 0.5,
                    "context.span_id": "broken-span",
                    "parent_id": "batch-root",
                },
            ]
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Approval batch 'corrupt-batch' contains malformed item 'broken-item'"
            ),
        ) as exc_info:
            await storage.get_batch("corrupt-batch", spans_df=spans)

        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)

    def test_reconstruct_item_rejects_missing_required_confidence(self):
        from datetime import datetime, timezone

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        row = pd.Series(
            {
                "attributes.item_id": "missing-confidence",
                "attributes.status": "pending_review",
                "attributes.created_at": datetime(
                    2026, 8, 5, tzinfo=timezone.utc
                ).isoformat(),
                "attributes.data": '{"query":"exact"}',
                "attributes.metadata": "{}",
            }
        )

        with pytest.raises(
            ValueError,
            match="approval item missing-confidence is missing attributes.confidence",
        ):
            ApprovalStorageImpl._reconstruct_item(
                object.__new__(ApprovalStorageImpl), row, pd.DataFrame()
            )

    def test_human_approval_history_does_not_change_pending_status(self):
        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        row = pd.Series(
            {
                "attributes.item_id": "partially-approved-item",
                "attributes.status": "pending_review",
                "attributes.created_at": "2026-08-05T03:04:00+00:00",
                "attributes.data": '{"query":"find Curie"}',
                "attributes.metadata": "{}",
                "attributes.confidence": 0.8,
            }
        )
        annotations = pd.DataFrame(
            [
                {
                    "annotation_name": "human_approval",
                    "metadata": {
                        "item_id": "partially-approved-item",
                        "reviewed_at": "2026-08-05T03:04:05+00:00",
                        "reviewer": "reviewer@example.com",
                    },
                    "result.label": "approved",
                    "created_at": "2026-08-05T03:04:06+00:00",
                }
            ]
        )

        item = ApprovalStorageImpl._reconstruct_item(
            object.__new__(ApprovalStorageImpl), row, annotations
        )

        assert (item.status, item.reviewed_at) == (
            ApprovalStatus.PENDING_REVIEW,
            None,
        )

    def test_reconstruct_item_rejects_invalid_annotation_status(self):
        from datetime import datetime, timezone

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        row = pd.Series(
            {
                "attributes.item_id": "annotated-item",
                "attributes.status": "pending_review",
                "attributes.created_at": datetime(
                    2026, 8, 5, tzinfo=timezone.utc
                ).isoformat(),
                "attributes.data": '{"query":"exact"}',
                "attributes.metadata": "{}",
                "attributes.confidence": 0.8,
            }
        )
        annotations = pd.DataFrame(
            [
                {
                    "annotation_name": "item_status_update",
                    "metadata": {"item_id": "annotated-item"},
                    "result.label": "obsolete_status",
                    "created_at": datetime(2026, 8, 5, tzinfo=timezone.utc),
                }
            ]
        )

        with pytest.raises(
            ValueError,
            match="'obsolete_status' is not a valid ApprovalStatus",
        ):
            ApprovalStorageImpl._reconstruct_item(
                object.__new__(ApprovalStorageImpl), row, annotations
            )

    @pytest.mark.asyncio
    async def test_get_pending_batches_yields_during_indexing_delay(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:acme-synthetic_data"
        storage.tenant_id = "acme:acme"
        storage.project_name = "synthetic_data"
        storage.provider = SimpleNamespace(
            traces=SimpleNamespace(get_all_spans=AsyncMock(return_value=pd.DataFrame()))
        )

        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(100):
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        result = await storage.get_pending_batches()
        task.cancel()

        assert result == []
        # A blocking time.sleep(0.5) would freeze the loop so the ticker could
        # not advance; awaiting asyncio.sleep lets it tick many times.
        assert ticks >= 5

    @pytest.mark.asyncio
    async def test_get_pending_batches_reuses_spans_single_fetch(self):
        """get_pending_batches must reconstruct every batch from one project
        span fetch, not re-query the whole project per batch (N+1)."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        spans = pd.DataFrame(
            [
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "b1",
                    "attributes.total_items": 1,
                    "attributes.auto_approved": 0,
                    "attributes.pending_review": 1,
                    "attributes.context": "{}",
                    "attributes.created_at": "2026-08-05T00:00:00+00:00",
                    "context.span_id": "s1",
                    "parent_id": None,
                },
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "b2",
                    "attributes.total_items": 1,
                    "attributes.auto_approved": 0,
                    "attributes.pending_review": 1,
                    "attributes.context": "{}",
                    "attributes.created_at": "2026-08-05T00:00:01+00:00",
                    "context.span_id": "s2",
                    "parent_id": None,
                },
                {
                    "name": "approval_item",
                    "attributes.batch_id": None,
                    "attributes.item_id": "b1-item",
                    "attributes.status": "pending_review",
                    "attributes.created_at": "2026-08-05T00:00:02+00:00",
                    "attributes.reviewed_at": None,
                    "attributes.data": '{"query":"find b1"}',
                    "attributes.metadata": "{}",
                    "attributes.confidence": 0.4,
                    "context.span_id": "b1-item-span",
                    "parent_id": "s1",
                },
                {
                    "name": "approval_item",
                    "attributes.batch_id": None,
                    "attributes.item_id": "b2-item",
                    "attributes.status": "pending_review",
                    "attributes.created_at": "2026-08-05T00:00:03+00:00",
                    "attributes.reviewed_at": None,
                    "attributes.data": '{"query":"find b2"}',
                    "attributes.metadata": "{}",
                    "attributes.confidence": 0.45,
                    "context.span_id": "b2-item-span",
                    "parent_id": "s2",
                },
            ]
        )
        get_all_spans = AsyncMock(return_value=spans)
        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:acme-synthetic_data"
        storage.tenant_id = "acme:acme"
        storage.project_name = "synthetic_data"
        storage.provider = SimpleNamespace(
            traces=SimpleNamespace(get_all_spans=get_all_spans),
            annotations=SimpleNamespace(
                get_annotations=AsyncMock(return_value=pd.DataFrame())
            ),
        )

        batches = await storage.get_pending_batches()

        assert {b.batch_id for b in batches} == {"b1", "b2"}
        get_all_spans.assert_awaited_once_with(
            project=storage.full_project_name,
            filters={
                "name": [
                    "approval_batch",
                    "approval_item",
                    "approval_item_replacement",
                ]
            },
        )

    @pytest.mark.asyncio
    async def test_item_span_lookup_scopes_duplicate_item_id_to_batch(self):
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        spans = pd.DataFrame(
            [
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "batch-a",
                    "attributes.item_id": None,
                    "context.span_id": "batch-span-a",
                    "parent_id": None,
                    "start_time": "2026-08-05T00:00:00Z",
                },
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "batch-b",
                    "attributes.item_id": None,
                    "context.span_id": "batch-span-b",
                    "parent_id": None,
                    "start_time": "2026-08-05T00:00:01Z",
                },
                {
                    "name": "approval_item",
                    "attributes.batch_id": None,
                    "attributes.item_id": "shared-item",
                    "context.span_id": "item-span-a",
                    "parent_id": "batch-span-a",
                    "start_time": "2026-08-05T00:00:02Z",
                },
                {
                    "name": "approval_item",
                    "attributes.batch_id": None,
                    "attributes.item_id": "shared-item",
                    "context.span_id": "item-span-b",
                    "parent_id": "batch-span-b",
                    "start_time": "2026-08-05T00:00:03Z",
                },
            ]
        )
        get_all_spans = AsyncMock(return_value=spans)
        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.provider = SimpleNamespace(
            traces=SimpleNamespace(get_all_spans=get_all_spans)
        )

        span_id = await storage.get_item_span_id(
            "shared-item",
            batch_id="batch-a",
        )

        assert span_id == "item-span-a"
        get_all_spans.assert_awaited_once_with(
            project=storage.full_project_name,
            filters={"name": ["approval_batch", "approval_item"]},
        )

    @pytest.mark.asyncio
    async def test_save_batch_yields_while_checked_export_is_blocked(self):
        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        export_entered = threading.Event()
        loop_advanced = threading.Event()

        class _Span:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                export_entered.set()
                assert loop_advanced.wait(timeout=1)
                return False

            def set_status(self, status):
                self.status = status

        class _TelemetryManager:
            def __init__(self):
                self.export_thread = None

            def span(self, **kwargs):
                assert kwargs["require_export"] is True
                self.export_thread = threading.get_ident()
                return _Span()

        manager = _TelemetryManager()
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.project_name = "synthetic_data"
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.telemetry_manager = manager
        storage._replacement_records = None
        batch = ApprovalBatch(
            batch_id="nonblocking_save",
            items=[
                ReviewItem(
                    item_id="nonblocking_item",
                    data={"query": "find PyTorch"},
                    confidence=0.4,
                )
            ],
            context={"tenant_id": "acme:approval"},
        )

        async def ticker():
            while not export_entered.is_set():
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            loop_advanced.set()

        saved_batch_id, _ = await asyncio.gather(storage.save_batch(batch), ticker())

        assert saved_batch_id == batch.batch_id
        assert loop_advanced.is_set()
        assert manager.export_thread != threading.get_ident()

    @pytest.mark.asyncio
    async def test_replace_item_yields_while_checked_export_is_blocked(self):
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        export_entered = threading.Event()
        loop_advanced = threading.Event()

        class _Span:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                export_entered.set()
                assert loop_advanced.wait(timeout=1)
                return False

            def set_status(self, status):
                self.status = status

        class _TelemetryManager:
            def __init__(self):
                self.export_thread = None

            def span(self, **kwargs):
                assert kwargs["require_export"] is True
                self.export_thread = threading.get_ident()
                return _Span()

        class _ReplacementRecords:
            async def select_review_decision(self, **kwargs):
                from cogniverse_agents.approval.replacement_store import (
                    CanonicalReplacementRecord,
                )

                payload = dict(kwargs["candidate"])
                record_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return CanonicalReplacementRecord(
                    payload=payload,
                    json=record_json,
                    sha256=hashlib.sha256(record_json.encode()).hexdigest(),
                )

            async def select_canonical(self, **kwargs):
                from cogniverse_agents.approval.replacement_store import (
                    CanonicalReplacementRecord,
                )

                payload = dict(kwargs["candidate"])
                record_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return CanonicalReplacementRecord(
                    payload=payload,
                    json=record_json,
                    sha256=hashlib.sha256(record_json.encode()).hexdigest(),
                )

            @asynccontextmanager
            async def replacement_event_lock(self, **kwargs):
                yield

        original = ReviewItem(
            item_id="flush_original",
            data={"query": "original"},
            confidence=0.4,
        )
        replacement = ReviewItem(
            item_id="flush_replacement",
            data={"query": "replacement"},
            confidence=0.8,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "reviewer",
                    "feedback": "replace it",
                    "corrections": {"query": "replacement"},
                    "timestamp": "2026-08-05T00:00:00+00:00",
                },
            },
        )
        spans = pd.DataFrame(
            [
                {
                    "attributes.batch_id": "flush_batch",
                    "attributes.original_item_id": original.item_id,
                    "attributes.replacement_item_id": replacement.item_id,
                    "attributes.replacement_record_json": json.dumps(
                        {
                            "item_id": replacement.item_id,
                            "data": replacement.data,
                            "confidence": replacement.confidence,
                            "status": replacement.status.value,
                            "metadata": replacement.metadata,
                            "created_at": replacement.created_at.isoformat(),
                            "reviewed_at": None,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            ]
        )
        spans["attributes.replacement_record_sha256"] = spans[
            "attributes.replacement_record_json"
        ].apply(lambda value: hashlib.sha256(value.encode()).hexdigest())
        manager = _TelemetryManager()
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.project_name = "synthetic_data"
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.telemetry_manager = manager
        storage.provider = SimpleNamespace(
            traces=SimpleNamespace(
                get_all_spans=AsyncMock(side_effect=[pd.DataFrame(), spans])
            )
        )
        storage._replacement_records = _ReplacementRecords()

        async def ticker():
            while not export_entered.is_set():
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            loop_advanced.set()

        await asyncio.gather(
            storage.replace_item("flush_batch", original, replacement),
            ticker(),
        )

        assert loop_advanced.is_set()
        assert manager.export_thread != threading.get_ident()

    @pytest.mark.asyncio
    async def test_replace_item_preserves_checked_export_failure_as_cause(self):
        from unittest.mock import AsyncMock

        import pandas as pd

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        export_error = TimeoutError("telemetry export timed out")

        class _Span:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                raise export_error

            def set_status(self, status):
                self.status = status

        class _TelemetryManager:
            def span(self, **kwargs):
                assert kwargs["require_export"] is True
                return _Span()

        class _ReplacementRecords:
            async def select_review_decision(self, **kwargs):
                from cogniverse_agents.approval.replacement_store import (
                    CanonicalReplacementRecord,
                )

                payload = dict(kwargs["candidate"])
                record_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return CanonicalReplacementRecord(
                    payload=payload,
                    json=record_json,
                    sha256=hashlib.sha256(record_json.encode()).hexdigest(),
                )

            async def select_canonical(self, **kwargs):
                from cogniverse_agents.approval.replacement_store import (
                    CanonicalReplacementRecord,
                )

                payload = dict(kwargs["candidate"])
                record_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return CanonicalReplacementRecord(
                    payload=payload,
                    json=record_json,
                    sha256=hashlib.sha256(record_json.encode()).hexdigest(),
                )

            @asynccontextmanager
            async def replacement_event_lock(self, **kwargs):
                yield

        original = ReviewItem(
            item_id="fault_original",
            data={"query": "original"},
            confidence=0.4,
        )
        replacement = ReviewItem(
            item_id="fault_replacement",
            data={"query": "replacement"},
            confidence=0.8,
            status=ApprovalStatus.REGENERATED,
            metadata={
                "original_item_id": original.item_id,
                "decision": {
                    "reviewer": "reviewer",
                    "feedback": "replace it",
                    "corrections": {"query": "replacement"},
                    "timestamp": "2026-08-05T00:00:00+00:00",
                },
            },
        )
        storage = object.__new__(ApprovalStorageImpl)
        storage.tenant_id = "acme:approval"
        storage.project_name = "synthetic_data"
        storage.full_project_name = "cogniverse-acme:approval-synthetic_data"
        storage.telemetry_manager = _TelemetryManager()
        storage.provider = SimpleNamespace(
            traces=SimpleNamespace(get_all_spans=AsyncMock(return_value=pd.DataFrame()))
        )
        storage._replacement_records = _ReplacementRecords()

        with pytest.raises(RuntimeError) as error:
            await storage.replace_item("fault_batch", original, replacement)

        assert str(error.value) == (
            "Failed to persist replacement: batch=fault_batch "
            "original=fault_original replacement=fault_replacement"
        )
        assert isinstance(error.value.__cause__, TimeoutError)
        assert str(error.value.__cause__) == "telemetry export timed out"


class TestPendingBatchesBackendFailurePropagates:
    """A telemetry-backend failure must raise, not read as an empty queue.

    get_pending_batches previously flattened every exception to [] — a
    Phoenix outage made the human approval queue silently show nothing
    pending.
    """

    @pytest.mark.asyncio
    async def test_get_pending_batches_raises_on_backend_failure(self):
        from unittest.mock import AsyncMock, MagicMock

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl

        manager = MagicMock()
        provider = MagicMock()
        provider.traces.get_all_spans = AsyncMock(
            side_effect=TimeoutError("phoenix query timed out")
        )
        manager.get_provider.return_value = provider
        manager.config.get_project_name.return_value = "cogniverse-acme:prod"

        storage = ApprovalStorageImpl(
            grpc_endpoint="http://phoenix:4317",
            http_endpoint="http://phoenix:6006",
            tenant_id="acme:prod",
            telemetry_manager=manager,
        )
        with pytest.raises(TimeoutError, match="phoenix query timed out"):
            await storage.get_pending_batches()
