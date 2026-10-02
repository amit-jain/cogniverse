"""The Optimization tab's golden dataset builder reads what the tab writes.

Real Phoenix: search spans recorded by the production span writers, ratings
saved by the Search Annotations tab's own writer, and the builder reading
both back. Re-rating a span replaces its rating (Phoenix keys an annotation
by name and span), the threshold filters on the rating that stands, results
key by source title, an annotated search with no titled result contributes
nothing, and two tenants built at once each read only their own spans and
ratings.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
import pytest
import requests
import streamlit as st

from cogniverse_dashboard.tabs import optimization
from cogniverse_dashboard.utils import tenant_project_name
from cogniverse_foundation.telemetry.context import (
    add_search_results_to_span,
    search_span,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


def _row(document_id, source_title=None):
    row = {"id": document_id, "source_id": f"sha-of-{document_id}", "score": 1.0}
    if source_title is not None:
        row["source_title"] = source_title
    return row


SEARCHES = {
    "find the red car": [
        _row("d1", "v1.mp4"),
        _row("d2", "v2.mkv"),
        _row("d1b", "v1.mp4"),
    ],
    "find the blue boat": [_row("d3", "v3.mp4")],
    "find the green tree": [_row("d4", "v4.mp4")],
    "find the untitled clip": [_row("d5")],
}
# Saved in order; the last rating of each span stands.
RATINGS = {
    "find the red car": [0.5, 0.9],
    "find the blue boat": [0.9, 0.5],
    "find the untitled clip": [1.0],
}


def _span_ids_by_query(endpoint, project, expected):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = requests.get(
            f"{endpoint}/v1/projects/{quote(project, safe='')}/spans",
            params={"limit": 100},
            timeout=10,
        )
        if response.status_code == 200:
            found = {
                span["attributes"]["query"]: span["context"]["span_id"]
                for span in response.json()["data"]
                if span["name"] == "search_service.search"
            }
            if len(found) == expected:
                return found
        time.sleep(0.5)
    raise AssertionError(f"{expected} search spans never reached Phoenix")


def _wait_for_ratings(client, project, expected):
    """Phoenix inserts annotations in the background; wait until the rating
    that stands on each span is the last one saved."""
    deadline = time.monotonic() + 60
    standing = {}
    while time.monotonic() < deadline:
        annotations = client.spans.get_span_annotations(
            span_ids=list(expected),
            project_identifier=project,
            include_annotation_names=["search_quality_annotation"],
        )
        standing = {
            annotation["span_id"]: annotation["result"]["score"]
            for annotation in annotations
        }
        if standing == expected:
            return
        time.sleep(0.5)
    raise AssertionError(f"ratings standing {standing}, expected {expected}")


def test_builder_reads_the_annotation_tab_writes(
    telemetry_manager_with_phoenix, phoenix_container, phoenix_client
):
    tenant = f"goldbuild{uuid4().hex[:8]}:prod"
    project = tenant_project_name(telemetry_manager_with_phoenix, tenant)
    st.session_state["current_tenant"] = tenant
    started = datetime.now(timezone.utc)
    try:
        for query, rows in SEARCHES.items():
            with search_span(
                tenant_id=tenant,
                query=query,
                ranking_strategy="strategy_b",
                profile="profile_a",
            ) as span:
                add_search_results_to_span(span, rows)
        telemetry_manager_with_phoenix.force_flush(timeout_millis=10000)
        span_ids = _span_ids_by_query(
            phoenix_container["http_endpoint"], project, len(SEARCHES)
        )

        for query, ratings in RATINGS.items():
            for rating in ratings:
                optimization._save_search_annotation(
                    span_ids[query], rating, "Star Rating (1-5)", "", tenant
                )
        _wait_for_ratings(
            phoenix_client,
            project,
            {span_ids[query]: ratings[-1] for query, ratings in RATINGS.items()},
        )

        dataset = asyncio.run(
            optimization._build_golden_dataset_from_phoenix(
                tenant, min_rating=0.8, lookback_days=1
            )
        )
    finally:
        st.session_state.pop("current_tenant", None)

    assert list(dataset) == ["find the red car"]
    recorded_at = pd.Timestamp(dataset["find the red car"].pop("timestamp"))
    assert started - timedelta(seconds=5) <= recorded_at <= datetime.now(timezone.utc)
    assert dataset["find the red car"] == {
        "expected_videos": ["v1", "v2"],
        "relevance_scores": {"v1": 1.0, "v2": 0.5},
        "avg_relevance": 0.9,
        "profile": "profile_a",
    }


def test_concurrent_builds_read_only_their_own_tenant(
    telemetry_manager_with_phoenix, phoenix_container, phoenix_client
):
    # The same query, rated above the threshold in both tenants, retrieves
    # different sources in each.
    searches = {
        f"goldbuild{uuid4().hex[:8]}:one": [_row("a1", "v1.mp4")],
        f"goldbuild{uuid4().hex[:8]}:two": [_row("b7", "v7.mkv"), _row("b8", "v8.mp4")],
    }
    span_ids = {}
    for tenant, rows in searches.items():
        with search_span(
            tenant_id=tenant,
            query="find the red car",
            ranking_strategy="strategy_b",
            profile="profile_a",
        ) as span:
            add_search_results_to_span(span, rows)
    telemetry_manager_with_phoenix.force_flush(timeout_millis=10000)
    for tenant in searches:
        project = tenant_project_name(telemetry_manager_with_phoenix, tenant)
        span_ids[tenant] = _span_ids_by_query(
            phoenix_container["http_endpoint"], project, 1
        )["find the red car"]
        optimization._save_search_annotation(
            span_ids[tenant], 0.9, "Star Rating (1-5)", "", tenant
        )
        _wait_for_ratings(phoenix_client, project, {span_ids[tenant]: 0.9})

    async def build_both():
        return await asyncio.gather(
            *(
                optimization._build_golden_dataset_from_phoenix(
                    tenant, min_rating=0.8, lookback_days=1
                )
                for tenant in searches
            )
        )

    built = dict(zip(searches, asyncio.run(build_both())))

    for dataset in built.values():
        dataset["find the red car"].pop("timestamp")
    first, second = searches
    assert built == {
        first: {
            "find the red car": {
                "expected_videos": ["v1"],
                "relevance_scores": {"v1": 1.0},
                "avg_relevance": 0.9,
                "profile": "profile_a",
            }
        },
        second: {
            "find the red car": {
                "expected_videos": ["v7", "v8"],
                "relevance_scores": {"v7": 1.0, "v8": 0.5},
                "avg_relevance": 0.9,
                "profile": "profile_a",
            }
        },
    }
