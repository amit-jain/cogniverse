"""Real-Phoenix happy path for the Evaluation tab's data loaders.

The outage side of the contract (dead endpoint / error status raises
PhoenixUnavailableError) is pinned by unit tests. These tests pin the success
side against a real Phoenix container: the GraphQL dataset listing returns the
dataset a writer created, a golden dataset nobody searched loads as an
explicit empty result (distinct from an outage), and search spans the
production writers record are scored into the exact per-query structure the
tab renders, keyed by source title, each tenant from its own spans even when
two tenants are scored at once.
"""

from __future__ import annotations

import threading
import time
from urllib.parse import quote
from uuid import uuid4

import pytest
import requests
import streamlit as st

from cogniverse_dashboard.tabs import evaluation
from cogniverse_dashboard.utils import tenant_project_name
from cogniverse_foundation.telemetry.context import (
    add_search_results_to_span,
    search_span,
)
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


@pytest.fixture
def phoenix_tab(phoenix_container, telemetry_manager_with_phoenix):
    """The tab over the Phoenix container, with the dashboard's telemetry
    manager pointed at it: the tab derives each tenant's project from it."""
    st.session_state["phoenix_url"] = phoenix_container["http_endpoint"]
    st.cache_data.clear()
    yield evaluation
    st.cache_data.clear()
    st.session_state.pop("phoenix_url", None)


def _create_dataset(client, name):
    return client.datasets.create_dataset(
        name=name,
        inputs=[{"query": "find the red car"}],
        outputs=[{"expected_videos": "v1,v2"}],
    )


def _dataset_id(tab, name):
    return next(d["id"] for d in tab.get_phoenix_datasets() if d["name"] == name)


def _record_search(tenant, query, profile, strategy, rows):
    with search_span(
        tenant_id=tenant, query=query, ranking_strategy=strategy, profile=profile
    ) as span:
        add_search_results_to_span(span, rows)


def _row(document_id, source_title):
    return {
        "id": document_id,
        "source_id": f"sha-of-{document_id}",
        "source_title": source_title,
        "score": 1.0,
    }


def _wait_for_searches(endpoint, tenant, expected):
    project = quote(tenant_project_name(get_telemetry_manager(), tenant), safe="")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = requests.get(
            f"{endpoint}/v1/projects/{project}/spans", params={"limit": 100}, timeout=10
        )
        if response.status_code == 200:
            names = [span["name"] for span in response.json()["data"]]
            if names.count("search_service.search") == expected:
                return
        time.sleep(0.5)
    raise AssertionError(f"{expected} search spans never reached Phoenix")


def test_dataset_listing_returns_created_dataset(phoenix_tab, phoenix_client):
    name = f"eval-tab-ds-{uuid4().hex[:8]}"
    _create_dataset(phoenix_client, name)

    listed = phoenix_tab.get_phoenix_datasets()

    match = [d for d in listed if d["name"] == name]
    assert len(match) == 1
    assert match[0]["example_count"] == 1
    assert match[0]["id"]  # GraphQL global id feeds the examples loader


def test_a_golden_dataset_nobody_searched_loads_as_empty(phoenix_tab, phoenix_client):
    name = f"eval-tab-empty-{uuid4().hex[:8]}"
    _create_dataset(phoenix_client, name)

    data = phoenix_tab.get_golden_search_results(
        _dataset_id(phoenix_tab, name), f"evaltab{uuid4().hex[:8]}:empty", 1
    )

    # No recorded search is a VALID empty result; an unreachable Phoenix raises
    # PhoenixUnavailableError instead (pinned in the unit tests).
    assert data == evaluation.GoldenSearchResults(results={}, unscored_searches=0)


def test_recorded_searches_score_into_exact_tab_structure(
    phoenix_tab, phoenix_client, phoenix_container, telemetry_manager_with_phoenix
):
    tenant = f"evaltab{uuid4().hex[:8]}:prod"
    name = f"eval-tab-spans-{uuid4().hex[:8]}"
    _create_dataset(phoenix_client, name)

    # An earlier search of the same profile, strategy and query: the latest
    # one is what the tab scores.
    _record_search(
        tenant, "find the red car", "profile_a", "strategy_b", [_row("d2", "v2.mp4")]
    )
    time.sleep(0.05)
    # Content-hash ids throughout; only the titles name the golden ids.
    # Duplicate v1 on purpose: the loader must dedupe retrieved sources.
    _record_search(
        tenant,
        "find the red car",
        "profile_a",
        "strategy_b",
        [_row("d1", "v1.mp4"), _row("d3", "v3.mkv"), _row("d1b", "v1.mp4")],
    )
    # A query outside the dataset is not scored.
    _record_search(
        tenant,
        "a query nobody asked",
        "profile_a",
        "strategy_b",
        [_row("d9", "v9.mp4")],
    )
    # A search whose result carries no title cannot be scored.
    _record_search(
        tenant,
        "find the red car",
        "profile_a",
        "strategy_untitled",
        [{"id": "d4", "source_id": "sha-of-d4", "score": 1.0}],
    )
    telemetry_manager_with_phoenix.force_flush(timeout_millis=10000)
    _wait_for_searches(phoenix_container["http_endpoint"], tenant, 4)
    st.cache_data.clear()

    data = phoenix_tab.get_golden_search_results(
        _dataset_id(phoenix_tab, name), tenant, 1
    )

    # Exact structure the tab renders: expected_videos csv split, retrieved
    # sources keyed by title and deduped in rank order, metrics computed from
    # that ranking (v1 at rank 1 -> mrr 1.0, recall@1 1.0; 1 of 2 expected in
    # the top 5 -> recall@5 0.5), aggregates equal to the single query's.
    assert data == evaluation.GoldenSearchResults(
        results={
            "profile_a": {
                "strategy_b": {
                    "queries": [
                        {
                            "query": "find the red car",
                            "expected": ["v1", "v2"],
                            "results": ["v1", "v3"],
                            "metrics": {
                                "mrr": 1.0,
                                "recall@1": 1.0,
                                "recall@5": 0.5,
                            },
                        }
                    ],
                    "aggregate_metrics": {
                        "mrr": {"mean": 1.0},
                        "recall@1": {"mean": 1.0},
                        "recall@5": {"mean": 0.5},
                    },
                }
            }
        },
        unscored_searches=1,
    )


def _scored(results, mrr, recall_1, recall_5):
    metrics = {"mrr": mrr, "recall@1": recall_1, "recall@5": recall_5}
    return evaluation.GoldenSearchResults(
        results={
            "profile_a": {
                "strategy_b": {
                    "queries": [
                        {
                            "query": "find the red car",
                            "expected": ["v1", "v2"],
                            "results": results,
                            "metrics": metrics,
                        }
                    ],
                    "aggregate_metrics": {
                        name: {"mean": value} for name, value in metrics.items()
                    },
                }
            }
        },
        unscored_searches=0,
    )


def test_concurrent_tenants_score_only_their_own_searches(
    phoenix_tab, phoenix_client, phoenix_container, telemetry_manager_with_phoenix
):
    name = f"eval-tab-tenants-{uuid4().hex[:8]}"
    _create_dataset(phoenix_client, name)
    dataset_id = _dataset_id(phoenix_tab, name)
    first, second = (f"evaltab{uuid4().hex[:8]}:{t}" for t in ("one", "two"))
    # The same golden query, ranked differently in each tenant's corpus.
    _record_search(
        first,
        "find the red car",
        "profile_a",
        "strategy_b",
        [_row("a1", "v1.mp4"), _row("a3", "v3.mp4")],
    )
    _record_search(
        second,
        "find the red car",
        "profile_a",
        "strategy_b",
        [_row("b9", "v9.mp4"), _row("b2", "v2.mp4")],
    )
    telemetry_manager_with_phoenix.force_flush(timeout_millis=10000)
    for tenant in (first, second):
        _wait_for_searches(phoenix_container["http_endpoint"], tenant, 1)
    st.cache_data.clear()

    barrier = threading.Barrier(2)
    scored = {}

    def score(tenant):
        barrier.wait(timeout=30)
        scored[tenant] = phoenix_tab.get_golden_search_results(dataset_id, tenant, 1)

    threads = [threading.Thread(target=score, args=(t,)) for t in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert scored == {
        first: _scored(["v1", "v3"], 1.0, 1.0, 0.5),
        second: _scored(["v9", "v2"], 0.5, 0.0, 0.5),
    }
