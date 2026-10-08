"""A tenant's recorded searches scored against its golden set.

Each ``search_service.search`` span whose query is a golden query is scored
under the span's ``profile`` and ``strategy``; the latest successful search per
profile, strategy and query counts. A result names its source by
``result_source_title_key``, the key golden sets name sources by, and a source
is counted once, at its best rank.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from cogniverse_evaluation.metrics.custom import calculate_metrics_suite
from cogniverse_foundation.telemetry.span_contract import (
    read_span_attributes,
    read_span_io,
)
from cogniverse_foundation.telemetry.span_metrics import span_succeeded
from cogniverse_sdk.document import result_source_title_key

SEARCH_SPAN_NAME = "search_service.search"

# The ranks a scored query lists, and the metrics it is scored on.
RETRIEVED_LIMIT = 10
_METRICS = {
    "mrr": "mrr",
    "ndcg": "ndcg",
    "recall_at_1": "recall@1",
    "recall_at_5": "recall@5",
    "precision_at_5": "precision@5",
}


def _query_scores(retrieved: List[str], expected: List[str]) -> Dict[str, float]:
    suite = calculate_metrics_suite(retrieved, expected, k_values=[1, 5])
    return {name: float(suite[key]) for name, key in _METRICS.items()}


def score_recorded_searches(
    spans: pd.DataFrame, golden_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Score ``spans`` (a tenant's ``search_service.search`` spans) against
    ``golden_rows`` (canonical rows: ``query`` and a list of
    ``expected_videos``).

    Returns:

    - ``golden_queries``: the number of golden queries;
    - ``strategies``: per profile and strategy, sorted, the number of scored
      queries, the mean of each metric and ``success_rate``, the share of
      queries whose first result is expected;
    - ``queries``: each scored query, by profile and strategy, in golden
      order, with its expected and retrieved sources, when it was searched,
      its trace and its metrics;
    - ``unsearched_queries``: golden queries no profile scored, in golden order;
    - ``failed_searches``: golden-query searches that failed, which are not
      scored;
    - ``unscored_searches``: golden-query searches with a result that carries
      no source title, which cannot be matched and are not scored.
    """
    expected_by_query: Dict[str, List[str]] = {}
    for row in golden_rows:
        expected_by_query[row["query"]] = list(row["expected_videos"])
    golden_order = {query: index for index, query in enumerate(expected_by_query)}

    latest: Dict[tuple, Dict[str, Any]] = {}
    failed = unscored = 0
    for _, span in spans.iterrows():
        attributes = read_span_attributes(span)
        query = str(attributes.get("query") or "").strip()
        if query not in expected_by_query:
            continue
        if not span_succeeded(span.get("status_code")):
            failed += 1
            continue
        rows = read_span_io(span)["output"]
        try:
            if not isinstance(rows, list):
                raise ValueError("the search recorded no result rows")
            retrieved = list(dict.fromkeys(result_source_title_key(r) for r in rows))
        except ValueError:
            unscored += 1
            continue
        key = (
            str(attributes.get("profile") or "unknown"),
            str(attributes.get("strategy") or "default"),
            query,
        )
        searched_at = pd.Timestamp(span["start_time"])
        searched_at = (
            searched_at.tz_localize("UTC")
            if searched_at.tzinfo is None
            else searched_at.tz_convert("UTC")
        )
        if key not in latest or searched_at > latest[key]["searched_at"]:
            trace_id = span.get("context.trace_id", span.get("trace_id"))
            latest[key] = {
                "searched_at": searched_at,
                "retrieved": retrieved,
                "trace_id": None if pd.isna(trace_id) else str(trace_id),
            }

    queries = []
    for (profile, strategy, query), search in sorted(
        latest.items(), key=lambda item: (*item[0][:2], golden_order[item[0][2]])
    ):
        expected = expected_by_query[query]
        queries.append(
            {
                "profile": profile,
                "strategy": strategy,
                "query": query,
                "expected": expected,
                "retrieved": search["retrieved"][:RETRIEVED_LIMIT],
                "searched_at": search["searched_at"].isoformat(),
                "trace_id": search["trace_id"],
                **_query_scores(search["retrieved"], expected),
            }
        )

    strategies = []
    for profile, strategy in sorted({(q["profile"], q["strategy"]) for q in queries}):
        scored = [
            q for q in queries if (q["profile"], q["strategy"]) == (profile, strategy)
        ]
        strategies.append(
            {
                "profile": profile,
                "strategy": strategy,
                "queries": len(scored),
                **{
                    name: sum(q[name] for q in scored) / len(scored)
                    for name in _METRICS
                },
                "success_rate": sum(q["mrr"] == 1.0 for q in scored) / len(scored),
            }
        )

    searched = {q["query"] for q in queries}
    return {
        "golden_queries": len(expected_by_query),
        "strategies": strategies,
        "queries": queries,
        "unsearched_queries": [q for q in expected_by_query if q not in searched],
        "failed_searches": failed,
        "unscored_searches": unscored,
    }


def dataset_golden_rows(examples: pd.DataFrame) -> List[Dict[str, Any]]:
    """Canonical golden rows of an evaluation dataset's examples, as
    ``DatasetManager`` writes them: ``input.query`` and a comma-joined
    ``output.expected_videos``.

    An example without a query or an expected source is left out; a query
    listed again keeps its first position and its last expectation.
    """
    expected: Dict[str, List[str]] = {}
    for _, example in examples.iterrows():
        query = str((example["input"] or {}).get("query") or "").strip()
        raw = (example["output"] or {}).get("expected_videos") or ""
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        sources = [str(item).strip() for item in items if str(item).strip()]
        if query and sources:
            expected[query] = sources
    return [
        {"query": query, "expected_videos": sources}
        for query, sources in expected.items()
    ]
