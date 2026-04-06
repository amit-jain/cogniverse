"""
Simplified scorers for Inspect AI integration.

These scorers are designed to work with Inspect AI's actual interface
where the scorer function receives the model output and target directly.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_configured_scorers(config: dict[str, Any]) -> list:
    """Get list of configured scorers.

    Returns empty list — simple_relevance_scorer was a stub.
    Use inspect_scorers.get_configured_scorers for real scoring.
    """
    return []


def simple_relevance_scorer():
    """Stub — not implemented."""
    raise NotImplementedError(
        "simple_relevance_scorer is a stub — implement actual relevance scoring "
        "or use inspect_scorers.get_configured_scorers instead"
    )
