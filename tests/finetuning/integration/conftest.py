"""
Integration test configuration for finetuning tests.

Provides fixtures for testing adapter registry with real Vespa.
"""

import logging

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

logger = logging.getLogger(__name__)
