"""
Generic Approval Interfaces

Provides abstract interfaces for human-in-the-loop approval system.
Uses dependency injection pattern to support any domain (synthetic data,
optimization results, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ApprovalStatus(Enum):
    """Status of a review item"""

    AUTO_APPROVED = "auto_approved"  # High confidence, no human review needed
    PENDING_REVIEW = "pending_review"  # Low confidence, awaiting human review
    APPROVED = "approved"  # Human approved
    REJECTED = "rejected"  # Human rejected
    REGENERATED = "regenerated"  # Rejected and regenerated


@dataclass
class ReviewItem:
    """
    Generic item for human review - works for any domain

    Examples:
    - Synthetic query generation: {"query": "...", "entities": [...]}
    - Optimization results: {"module": "routing", "improvement": 0.12}
    - Agent outputs: {"agent": "search", "results": [...]}
    """

    item_id: str
    data: Dict[str, Any]  # Domain-specific content
    confidence: float  # Normalized 0-1 score
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
    created_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ReviewDecision:
    """Human decision on a review item"""

    item_id: str
    approved: bool
    feedback: Optional[str] = None
    corrections: Dict[str, Any] = field(default_factory=dict)
    reviewer: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ApprovalBatch:
    """Batch of items for approval with status tracking"""

    batch_id: str
    items: List[ReviewItem]
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def auto_approved(self) -> List[ReviewItem]:
        """Items that were automatically approved"""
        return [
            item for item in self.items if item.status == ApprovalStatus.AUTO_APPROVED
        ]

    @property
    def pending_review(self) -> List[ReviewItem]:
        """Items awaiting human review"""
        return [
            item for item in self.items if item.status == ApprovalStatus.PENDING_REVIEW
        ]

    @property
    def approved(self) -> List[ReviewItem]:
        """Items approved by human"""
        return [item for item in self.items if item.status == ApprovalStatus.APPROVED]

    @property
    def rejected(self) -> List[ReviewItem]:
        """Items rejected by human"""
        return [item for item in self.items if item.status == ApprovalStatus.REJECTED]

    @property
    def approval_rate(self) -> float:
        """Percentage of items approved (auto + human)"""
        total = len(self.items)
        if total == 0:
            return 0.0
        approved_count = len(self.auto_approved) + len(self.approved)
        return approved_count / total


class ConfidenceExtractor(ABC):
    """
    Extract confidence score from domain-specific data

    Implementations define how to calculate confidence for their domain.
    Examples:
    - SyntheticDataConfidenceExtractor: Use DSPy retry count
    - OptimizationConfidenceExtractor: Use improvement delta
    - AgentOutputConfidenceExtractor: Use agent's self-reported confidence
    """

    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> float:
        """
        Return 0-1 confidence score for the item

        Args:
            data: Domain-specific data dictionary

        Returns:
            Float between 0 and 1 (0 = no confidence, 1 = full confidence)
        """
        pass


class FeedbackHandler(ABC):
    """
    Handle rejection feedback - domain-specific regeneration/logging

    Implementations define what to do when humans reject an item.
    Examples:
    - SyntheticDataFeedbackHandler: Regenerate query with DSPy using feedback
    - OptimizationFeedbackHandler: Log issues, adjust hyperparameters
    - AgentOutputFeedbackHandler: Re-run agent with corrected inputs
    """

    @abstractmethod
    async def process_rejection(
        self, item: ReviewItem, decision: ReviewDecision
    ) -> Optional[ReviewItem]:
        """
        Process rejection, optionally return regenerated item

        Args:
            item: Original item that was rejected
            decision: Human decision with feedback and corrections

        Returns:
            New ReviewItem if regeneration succeeded, None otherwise
        """
        pass


class ApprovalStorage(ABC):
    """
    Storage backend for approval data

    Implementations define where approval data is persisted.
    Examples:
    - ApprovalStorageImpl: Store as telemetry spans with annotations
    - DatabaseApprovalStorage: Store in relational database
    - FileApprovalStorage: Store as JSON files
    """

    @abstractmethod
    async def save_batch(self, batch: ApprovalBatch) -> str:
        """
        Save approval batch

        Args:
            batch: Batch to save

        Returns:
            Batch ID for retrieval
        """
        pass

    @abstractmethod
    async def get_batch(self, batch_id: str) -> Optional[ApprovalBatch]:
        """
        Retrieve approval batch

        Args:
            batch_id: ID of batch to retrieve

        Returns:
            ApprovalBatch if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_item(self, item: ReviewItem) -> None:
        """
        Update a review item's status

        Args:
            item: Item with updated status
        """
        pass

    @abstractmethod
    async def get_pending_batches(
        self, context_filter: Optional[Dict[str, Any]] = None
    ) -> List[ApprovalBatch]:
        """
        Get batches with pending reviews

        Args:
            context_filter: Optional filter by batch context

        Returns:
            List of batches with pending items
        """
        pass
