"""
Integration test configuration for finetuning tests.

Provides fixtures for testing adapter registry with real Vespa.
"""

import logging

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401

# Conftest-level re-export: test modules must NOT import shared_vespa
# themselves — a module-level import defines a second FixtureDef with its
# own session cache, booting a second Vespa container mid-sweep.
from tests.conftest import shared_vespa  # noqa: F401

logger = logging.getLogger(__name__)
