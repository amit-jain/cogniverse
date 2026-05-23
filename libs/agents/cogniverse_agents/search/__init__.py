"""Unified search interface for video retrieval.

``SearchResult`` and ``SearchBackend`` are re-exported from
``cogniverse_sdk`` — the canonical home. The previous duplicates in
``cogniverse_agents.search.base`` and ``cogniverse_runtime.search.base``
were byte-identical dead copies and have been removed.
"""

from cogniverse_sdk.document import SearchResult
from cogniverse_sdk.interfaces.backend import SearchBackend

__all__ = [
    "SearchBackend",
    "SearchResult",
    "SearchService",  # Keep in __all__ but lazy import
]


# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == "SearchService":
        from .service import SearchService

        return SearchService
    raise AttributeError(f"module {__name__} has no attribute {name}")
