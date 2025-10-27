"""
Synthetic Data Approval

Domain-specific approval implementations for synthetic data generation.
"""

from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)
from cogniverse_synthetic.approval.feedback_handler import (
    SyntheticDataFeedbackHandler,
)

__all__ = [
    "SyntheticDataConfidenceExtractor",
    "SyntheticDataFeedbackHandler",
]
