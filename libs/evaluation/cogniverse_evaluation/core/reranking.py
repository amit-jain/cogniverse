"""Apply the live (production) rerankers to evaluation traces.

The eval harness reranks each trace's retrieved results with the same rerankers
the search path ships (``learned`` / ``hybrid`` / ``multi_modal``) via the shared
``rerank_result_dicts`` service — so offline metrics measure what production
actually does, not a separate eval-only reranking implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cogniverse_agents.search.rerank_service import rerank_result_dicts

logger = logging.getLogger(__name__)


async def apply_reranking_to_traces(
    traces: List[Dict[str, Any]],
    strategy: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Rerank each trace's ``results`` with the live ``strategy`` reranker.

    Args:
        traces: trace dicts, each with ``query`` and a ``results`` list of dicts.
        strategy: live reranker name (``learned`` / ``hybrid`` / ``multi_modal``).
        config: optional; reads ``tenant_id`` and ``config_manager`` (needed by
            the learned/hybrid rerankers to load their tenant-scoped settings).

    Returns:
        The same traces, with each ``results`` reordered by the reranker. A
        per-trace failure is logged and leaves that trace's results unchanged.
    """
    config = config or {}
    tenant_id = config.get("tenant_id", "")
    config_manager = config.get("config_manager")

    for trace in traces:
        results = trace.get("results")
        if not isinstance(results, list) or not results:
            continue
        try:
            trace["results"] = await rerank_result_dicts(
                query=str(trace.get("query", "")),
                results=results,
                strategy=strategy,
                tenant_id=tenant_id,
                config_manager=config_manager,
            )
        except Exception as exc:
            logger.warning(
                "Reranking trace %s with '%s' failed; leaving order unchanged: %s",
                trace.get("trace_id"),
                strategy,
                exc,
            )
    return traces
