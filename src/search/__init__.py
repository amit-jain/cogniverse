"""Unified search interface for video retrieval."""

from .search import SearchBackend, SearchResult
from .vespa_search_backend import VespaSearchBackend
from .search_service import SearchService

__all__ = [
    'SearchBackend',
    'SearchResult',
    'VespaSearchBackend',
    'SearchService'
]