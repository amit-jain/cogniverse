"""
Tests for Human-in-the-Loop Approval System

Tests approval interfaces, agents, confidence extraction, and feedback handling.
"""

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

pytestmark = [pytest.mark.unit]


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
        assert batch.approval_rate == pytest.approx(1 / 3)  # 1 auto-approved out of 3


class TestConfidenceExtractor:
    """Test SyntheticDataConfidenceExtractor"""

    def test_high_confidence_no_retries(self):
        """Test high confidence for first-attempt success"""
        extractor = SyntheticDataConfidenceExtractor()

        data = {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow"],
            "reasoning": "Including TensorFlow as primary entity",
            "_generation_metadata": {"retry_count": 0, "max_retries": 3},
        }

        confidence = extractor.extract(data)

        # Should have high confidence (0.9-1.0) for first attempt with entities
        assert confidence >= 0.9
        assert confidence <= 1.0

    def test_low_confidence_many_retries(self):
        """Test low confidence for many retries"""
        extractor = SyntheticDataConfidenceExtractor()

        data = {
            "query": "find tutorial",
            "entities": ["TensorFlow"],
            "reasoning": "",
            "_generation_metadata": {"retry_count": 3, "max_retries": 3},
        }

        confidence = extractor.extract(data)

        # Should have low confidence due to max retries
        assert confidence < 0.7

    def test_confidence_with_missing_entities(self):
        """Test confidence penalty for missing entities"""
        extractor = SyntheticDataConfidenceExtractor()

        data = {
            "query": "find tutorial",  # Missing TensorFlow
            "entities": ["TensorFlow"],
            "reasoning": "Including TensorFlow",
            "_generation_metadata": {"retry_count": 0, "max_retries": 3},
        }

        confidence = extractor.extract(data)

        # Should have reduced confidence due to missing entity
        assert confidence < 0.8

    def test_confidence_breakdown(self):
        """Test detailed confidence breakdown"""
        extractor = SyntheticDataConfidenceExtractor()

        data = {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow"],
            "reasoning": "Including TensorFlow as primary entity",
            "_generation_metadata": {"retry_count": 1, "max_retries": 3},
        }

        breakdown = extractor.get_confidence_breakdown(data)

        assert "final_confidence" in breakdown
        assert "retry_count" in breakdown
        assert "has_entity" in breakdown
        assert breakdown["retry_count"] == 1
        assert breakdown["has_entity"] is True

    def test_nested_generation_metadata_applies_retry_penalty(self):
        """RoutingGenerator stores _generation_metadata nested under the
        schema's ``metadata`` field. The retry penalty must apply there too —
        otherwise a 3-retry fallback scores high and is wrongly auto-approved."""
        extractor = SyntheticDataConfidenceExtractor()
        nested = {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow"],
            "metadata": {"_generation_metadata": {"retry_count": 3, "max_retries": 3}},
        }
        top_level = {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow"],
            "_generation_metadata": {"retry_count": 3, "max_retries": 3},
        }

        # Both shapes read the same retry_count, so they score identically and
        # land below the 0.7 auto-approve threshold.
        assert extractor.extract(nested) == 0.58
        assert extractor.extract(nested) == extractor.extract(top_level)
        assert extractor.extract(nested) < 0.7

    def test_nested_generation_metadata_breakdown_reads_retry_count(self):
        extractor = SyntheticDataConfidenceExtractor()
        data = {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow"],
            "metadata": {"_generation_metadata": {"retry_count": 3, "max_retries": 3}},
        }
        assert extractor.get_confidence_breakdown(data)["retry_count"] == 3


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
        batch_low = await agent_low.process_batch(items, "b_low", {})
        assert len(batch_low.auto_approved) == 1
        assert len(batch_low.pending_review) == 0

        agent_high = HumanApprovalAgent.from_approval_config(
            ApprovalConfig(confidence_threshold=0.80),
            confidence_extractor=_FixedConfidence(),
        )
        assert agent_high.threshold == 0.80
        batch_high = await agent_high.process_batch(items, "b_high", {})
        assert len(batch_high.auto_approved) == 0
        assert len(batch_high.pending_review) == 1

    @pytest.mark.asyncio
    async def test_submit_for_review_classifies_and_persists_prebuilt_batch(self):
        """submit_for_review re-classifies a caller-built batch against the
        threshold (>= auto-approve, else pending) using each item's own
        confidence, and persists it so the dashboard surfaces it. This is the
        path the finetuning synthetic-data flow uses."""

        class _Extractor:
            def extract(self, data):  # unused: submit_for_review uses item.confidence
                return 0.0

        class _FakeStorage:
            def __init__(self):
                self.saved = []

            async def save_batch(self, batch):
                self.saved.append(batch.batch_id)
                return batch.batch_id

        storage = _FakeStorage()
        agent = HumanApprovalAgent(
            confidence_extractor=_Extractor(),
            confidence_threshold=0.85,
            storage=storage,
        )
        batch = ApprovalBatch(
            batch_id="synthetic_b1",
            items=[
                ReviewItem(item_id="i_hi", data={}, confidence=0.9),
                ReviewItem(item_id="i_lo", data={}, confidence=0.8),
            ],
            context={},
        )

        result = await agent.submit_for_review(batch)

        assert result is batch
        assert [i.item_id for i in batch.auto_approved] == ["i_hi"]
        assert [i.item_id for i in batch.pending_review] == ["i_lo"]
        assert batch.approved_count == 1
        assert storage.saved == ["synthetic_b1"]


class TestFeedbackHandler:
    """Test SyntheticDataFeedbackHandler"""

    def test_feedback_handler_initialization(self):
        """Test initializing feedback handler"""
        handler = SyntheticDataFeedbackHandler(max_regeneration_attempts=2)

        assert handler.max_attempts == 2
        assert handler.generator is not None

    @pytest.mark.asyncio
    async def test_process_rejection_regenerates(self, reset_dspy_lm):
        """Test that rejection triggers regeneration"""
        handler = SyntheticDataFeedbackHandler(max_regeneration_attempts=2)

        item = ReviewItem(
            item_id="test_001",
            data={
                "query": "find tutorial",
                "entities": ["TensorFlow"],
                "entity_types": ["TECHNOLOGY"],
                "topics": "machine learning",
                "_generation_metadata": {"retry_count": 3, "max_retries": 3},
            },
            confidence=0.5,
        )

        decision = ReviewDecision(
            item_id="test_001",
            approved=False,
            feedback="Query doesn't include TensorFlow",
            corrections={"entities": ["TensorFlow", "Tutorial"]},
        )

        regenerated = await handler.process_rejection(item, decision)

        # Should successfully regenerate
        assert regenerated is not None
        assert regenerated.item_id.startswith("test_001_regen")
        assert regenerated.status == ApprovalStatus.REGENERATED
        assert "original_item_id" in regenerated.metadata


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


class TestApprovalStorageEventLoop:
    """Telemetry-indexing delays must not block the event loop.

    get_batch / get_pending_batches / get_item_span_id paused with a blocking
    time.sleep inside async methods, freezing every other coroutine on the
    worker for the full indexing-lag window. They must await asyncio.sleep.
    """

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
            traces=SimpleNamespace(get_spans=AsyncMock(return_value=pd.DataFrame()))
        )

        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(100):
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        # Awaits the 0.5s indexing delay, then returns [] on the empty frame.
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
                    "attributes.pending_review": 2,
                    "context.span_id": "s1",
                    "parent_id": None,
                },
                {
                    "name": "approval_batch",
                    "attributes.batch_id": "b2",
                    "attributes.pending_review": 1,
                    "context.span_id": "s2",
                    "parent_id": None,
                },
            ]
        )
        get_spans = AsyncMock(return_value=spans)
        storage = object.__new__(ApprovalStorageImpl)
        storage.full_project_name = "cogniverse-acme:acme-synthetic_data"
        storage.tenant_id = "acme:acme"
        storage.project_name = "synthetic_data"
        storage.provider = SimpleNamespace(
            traces=SimpleNamespace(get_spans=get_spans),
            annotations=SimpleNamespace(
                get_annotations=AsyncMock(return_value=pd.DataFrame())
            ),
        )

        batches = await storage.get_pending_batches()

        assert {b.batch_id for b in batches} == {"b1", "b2"}
        # One fetch total — not 1 + N (the per-batch get_batch re-fetch).
        assert get_spans.call_count == 1


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
        provider.traces.get_spans = AsyncMock(
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
