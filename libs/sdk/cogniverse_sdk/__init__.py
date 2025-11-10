"""
Cogniverse SDK - Core interfaces for backend implementations.

This package defines the interfaces that backend implementations must satisfy.
It has zero dependencies and serves as the foundation for the entire system.
"""

__version__ = "0.1.0"

from cogniverse_sdk.document import Document, SearchResult

__all__ = ["Document", "SearchResult"]
