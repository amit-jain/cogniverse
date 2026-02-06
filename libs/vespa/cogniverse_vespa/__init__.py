"""
Vespa backend module.

This module provides Vespa integration for Cogniverse, including:
- Document ingestion
- Vector and text search
- Schema management
- Self-registration with backend registry
"""

# Import backend to trigger self-registration
from .backend import VespaBackend
from .ingestion_client import VespaPyClient
from .vespa_search_client import VespaVideoSearchClient

# Export main classes
__all__ = [
    "VespaBackend",
    "VespaPyClient",
    "VespaVideoSearchClient",
]
