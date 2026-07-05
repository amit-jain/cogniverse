"""
Human-in-the-Loop Approval System

Generic approval infrastructure for human review of AI-generated outputs.
Supports any domain via dependency injection pattern.
"""

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.approval.orchestrator import DecisionOrchestrator
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ApprovalStorage,
    ConfidenceExtractor,
    FeedbackHandler,
    ReviewDecision,
    ReviewItem,
)

__all__ = [
    "ApprovalStatus",
    "ReviewItem",
    "ReviewDecision",
    "ApprovalBatch",
    "ConfidenceExtractor",
    "FeedbackHandler",
    "ApprovalStorage",
    "HumanApprovalAgent",
    "ApprovalStorageImpl",
    "DecisionOrchestrator",
]
