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
from typing import Any, Dict, List, Optional

from cogniverse_agents.search.types import RerankerSearchResult


def _to_rsr(d: Dict[str, Any]) -> RerankerSearchResult:
    return RerankerSearchResult(
        id=str(d.get("id") or d.get("source_id") or d.get("document_id") or ""),
        title=d.get("title", "") or "",
        content=d.get("content", "") or d.get("description", "") or "",
        modality=d.get("modality", "") or d.get("content_type", "") or "",
        score=float(d.get("score", 0.0) or 0.0),
        metadata=d.get("metadata", {}) or {},
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
    reranker = build_reranker(strategy, tenant_id, config_manager)
    reranked = await reranker.rerank(query=query, results=[_to_rsr(r) for r in results])
    return [_to_dict(r) for r in reranked]
