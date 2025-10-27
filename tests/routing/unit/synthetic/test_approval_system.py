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
            reviewer="test_user"
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
                status=ApprovalStatus.AUTO_APPROVED
            ),
            ReviewItem(
                item_id="test_002",
                data={"query": "query2"},
                confidence=0.7,
                status=ApprovalStatus.PENDING_REVIEW
            ),
            ReviewItem(
                item_id="test_003",
                data={"query": "query3"},
                confidence=0.6,
                status=ApprovalStatus.PENDING_REVIEW
            ),
        ]

        batch = ApprovalBatch(batch_id="batch_001", items=items, context={})

        assert len(batch.auto_approved) == 1
        assert len(batch.pending_review) == 2
        assert len(batch.approved) == 0
        assert len(batch.rejected) == 0
        assert batch.approval_rate == pytest.approx(1/3)  # 1 auto-approved out of 3


class TestConfidenceExtractor:
    """Test SyntheticDataConfidenceExtractor"""

    def test_high_confidence_no_retries(self):
        """Test high confidence for first-attempt success"""
        extractor = SyntheticDataConfidenceExtractor()

        data = {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow"],
            "reasoning": "Including TensorFlow as primary entity",
            "_generation_metadata": {
                "retry_count": 0,
                "max_retries": 3
            }
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
            "_generation_metadata": {
                "retry_count": 3,
                "max_retries": 3
            }
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
            "_generation_metadata": {
                "retry_count": 0,
                "max_retries": 3
            }
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
            "_generation_metadata": {
                "retry_count": 1,
                "max_retries": 3
            }
        }

        breakdown = extractor.get_confidence_breakdown(data)

        assert "final_confidence" in breakdown
        assert "retry_count" in breakdown
        assert "has_entity" in breakdown
        assert breakdown["retry_count"] == 1
        assert breakdown["has_entity"] is True


class TestHumanApprovalAgent:
    """Test HumanApprovalAgent"""

    def test_agent_initialization(self):
        """Test initializing approval agent"""
        extractor = SyntheticDataConfidenceExtractor()
        agent = HumanApprovalAgent(
            confidence_extractor=extractor,
            confidence_threshold=0.85,
            storage=None
        )

        assert agent.confidence_extractor is not None
        assert agent.threshold == 0.85

    def test_approval_stats(self):
        """Test get_approval_stats"""
        extractor = SyntheticDataConfidenceExtractor()
        agent = HumanApprovalAgent(
            confidence_extractor=extractor,
            confidence_threshold=0.85,
            storage=None
        )

        items = [
            ReviewItem(
                item_id="test_001",
                data={"query": "query1"},
                confidence=0.95,
                status=ApprovalStatus.AUTO_APPROVED
            ),
            ReviewItem(
                item_id="test_002",
                data={"query": "query2"},
                confidence=0.7,
                status=ApprovalStatus.PENDING_REVIEW
            ),
        ]

        batch = ApprovalBatch(batch_id="batch_001", items=items, context={})

        stats = agent.get_approval_stats(batch)

        assert stats["total_items"] == 2
        assert stats["auto_approved"] == 1
        assert stats["pending_review"] == 1
        assert stats["overall_approval_rate"] == 0.5
        assert "avg_confidence" in stats


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
                "_generation_metadata": {
                    "retry_count": 3,
                    "max_retries": 3
                }
            },
            confidence=0.5
        )

        decision = ReviewDecision(
            item_id="test_001",
            approved=False,
            feedback="Query doesn't include TensorFlow",
            corrections={
                "entities": ["TensorFlow", "Tutorial"]
            }
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
        from cogniverse_core.config.unified_config import ApprovalConfig

        config = ApprovalConfig(
            enabled=True,
            confidence_threshold=0.9,
            storage_backend="phoenix"
        )

        assert config.enabled is True
        assert config.confidence_threshold == 0.9
        assert config.storage_backend == "phoenix"

    def test_approval_config_from_dict(self):
        """Test creating ApprovalConfig from dict"""
        from cogniverse_core.config.unified_config import ApprovalConfig

        data = {
            "enabled": True,
            "confidence_threshold": 0.88,
            "storage_backend": "database",
            "phoenix_project_name": "my_project"
        }

        config = ApprovalConfig.from_dict(data)

        assert config.enabled is True
        assert config.confidence_threshold == 0.88
        assert config.storage_backend == "database"
        assert config.phoenix_project_name == "my_project"

    def test_approval_config_to_dict(self):
        """Test converting ApprovalConfig to dict"""
        from cogniverse_core.config.unified_config import ApprovalConfig

        config = ApprovalConfig(
            enabled=True,
            confidence_threshold=0.92,
            reviewer_email="test@example.com"
        )

        config_dict = config.to_dict()

        assert config_dict["enabled"] is True
        assert config_dict["confidence_threshold"] == 0.92
        assert config_dict["reviewer_email"] == "test@example.com"
