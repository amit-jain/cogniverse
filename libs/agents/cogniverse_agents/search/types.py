"""Shared types for the search subsystem.

Defines the content-modality taxonomy and the reranker-facing result shape
consumed by both the live request path (reranker in
`libs/runtime/.../routers/search.py`) and the offline analytics path
(`routing/modality_*` modules, `routing/xgboost_meta_models.py`, dashboards,
`evaluation/quality_monitor.py`).

`QueryModality` is a closed taxonomy; it is used as a dict key for XGBoost
feature schemas and per-modality analytics buckets. Expanding or shrinking
it is a training-contract change for the offline optimization pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class QueryModality(Enum):
    """Content-modality taxonomy for queries and search results."""

    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TEXT = "text"
    MIXED = "mixed"


@dataclass
class RerankerSearchResult:
    """Search result shape consumed by rerankers for scoring and comparison.

    Distinct from `cogniverse_agents.search.base.SearchResult`, which uses
    Document objects for API responses. This dataclass is the reranker's
    internal view: flat fields for fast access during scoring loops.
    """

    id: str
    title: str
    content: str
    modality: str
    score: float
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None
