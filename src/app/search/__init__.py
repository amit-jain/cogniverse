"""Unified search interface for video retrieval."""

from .base import SearchBackend, SearchResult

__all__ = [
    'SearchBackend',
    'SearchResult',
    'SearchService'  # Keep in __all__ but lazy import
]

# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == 'SearchService':
        from .service import SearchService
        return SearchService
    raise AttributeError(f"module {__name__} has no attribute {name}")