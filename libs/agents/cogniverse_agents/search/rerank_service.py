"""Shared reranking entry point over plain result dicts.

Both the ``/search/rerank`` HTTP endpoint and the evaluation harness need the
same thing: pick a reranker by strategy name, run it over a list of result
dicts, and get reranked dicts back. This module is the single place that does
the strategy → reranker selection and the dict ↔ ``RerankerSearchResult``
conversion, so the two callers can't drift apart (which is how the endpoint's
reranking silently broke before).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cogniverse_agents.search.temporal_query import extract_time_range
from cogniverse_agents.search.types import RerankerSearchResult


def _parse_timestamp(d: Dict[str, Any]) -> Optional[datetime]:
    """Pull creation_timestamp (epoch ms — ingestion writes int(time()*1000))
    off a search-result dict so temporal reranking has a real value instead of
    always falling back to the neutral 0.5 score (the timestamp was never set)."""
    raw = d.get("creation_timestamp")
    if raw is None:
        raw = (d.get("metadata") or {}).get("creation_timestamp")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 100_000_000_000:
        # Caller-supplied results (POST /search/rerank) can carry a
        # seconds epoch; without this guard it lands in 1970 and the doc
        # scores as year-old. Same threshold as search_agent._epoch_ms.
        value *= 1000.0
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        # OverflowError: a caller-supplied astronomically large epoch —
        # same unparseable-timestamp contract as the siblings.
        return None


def _to_rsr(d: Dict[str, Any]) -> RerankerSearchResult:
    return RerankerSearchResult(
        id=str(d.get("id") or d.get("source_id") or d.get("document_id") or ""),
        title=d.get("title", "") or "",
        content=d.get("content", "") or d.get("description", "") or "",
        modality=d.get("modality", "") or d.get("content_type", "") or "",
        score=float(d.get("score", 0.0) or 0.0),
        metadata=d.get("metadata", {}) or {},
        timestamp=_parse_timestamp(d),
    )


def _to_dict(r: RerankerSearchResult) -> Dict[str, Any]:
    out = asdict(r)
    if r.timestamp is not None:
        out["timestamp"] = r.timestamp.isoformat()
    return out


def build_reranker(strategy: str, tenant_id: str, config_manager: Optional[Any] = None):
    """Construct the live reranker for ``strategy``.

    Raises ``ValueError`` for an unknown strategy (callers surface it as 400).
    """
    if strategy == "learned":
        from cogniverse_agents.search.learned_reranker import LearnedReranker

        return LearnedReranker(tenant_id=tenant_id, config_manager=config_manager)
    if strategy == "hybrid":
        from cogniverse_agents.search.hybrid_reranker import HybridReranker

        return HybridReranker(tenant_id=tenant_id, config_manager=config_manager)
    if strategy == "multi_modal":
        from cogniverse_agents.search.multi_modal_reranker import MultiModalReranker

        return MultiModalReranker()
    raise ValueError(f"Unknown strategy: {strategy}")


async def rerank_result_dicts(
    query: str,
    results: List[Dict[str, Any]],
    strategy: str,
    tenant_id: str,
    config_manager: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Rerank a list of result dicts with the named live reranker.

    Returns the results as dicts in reranked order (empty input → empty list).
    """
    if not results:
        return []
    # Caller-supplied JSON: a non-dict element would AttributeError deep in
    # _to_rsr and surface as a 500; reject it as the 400 it is.
    bad = next((r for r in results if not isinstance(r, dict)), None)
    if bad is not None:
        raise ValueError(f"results must be objects; got {type(bad).__name__} element")
    reranker = build_reranker(strategy, tenant_id, config_manager)
    rerank_kwargs: Dict[str, Any] = {}
    # Feed the temporal scorer a query time-range only when the query carries
    # explicit temporal intent; otherwise the multi-modal reranker keeps its
    # neutral temporal score (no distortion of non-temporal queries).
    if strategy == "multi_modal":
        time_range = extract_time_range(query)
        if time_range is not None:
            rerank_kwargs["context"] = {"temporal": {"time_range": time_range}}
    reranked = await reranker.rerank(
        query=query, results=[_to_rsr(r) for r in results], **rerank_kwargs
    )
    return [_to_dict(r) for r in reranked]
