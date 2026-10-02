#!/usr/bin/env python3
"""
Golden-set evaluation tab: the tenant's recorded searches scored against a
Phoenix golden dataset, in the tabbed format of generate_tabbed_html_report.py.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import quote

import plotly.graph_objects as go
import requests
import streamlit as st

from cogniverse_sdk.document import result_source_title_key

logger = logging.getLogger(__name__)


_PHOENIX_REQUEST_TIMEOUT_S = 15


class PhoenixUnavailableError(RuntimeError):
    """Phoenix could not be reached or answered with an error status."""


def _phoenix_base_url() -> str:
    """Phoenix base URL from the dashboard's configured telemetry URL.

    The app shell sets ``st.session_state["phoenix_url"]`` from the system
    config's ``telemetry_url``.
    """
    url = st.session_state.get("phoenix_url")
    if not url:
        raise PhoenixUnavailableError(
            "Phoenix is not configured for this dashboard: the session has no "
            "phoenix_url (SystemConfig.telemetry_url)"
        )
    return url


def query_phoenix_graphql(query: str) -> Dict[str, Any]:
    """Execute a GraphQL query against Phoenix.

    Raises :class:`PhoenixUnavailableError` on connection failure or a
    non-200 — a Phoenix outage must render as an error, not as an empty
    dataset list indistinguishable from a fresh project. The timeout bounds
    a hung Phoenix so the tab never freezes.
    """
    url = f"{_phoenix_base_url()}/graphql"
    try:
        response = requests.post(
            url,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=_PHOENIX_REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise PhoenixUnavailableError(f"Phoenix unreachable at {url}: {exc}") from exc
    if response.status_code != 200:
        raise PhoenixUnavailableError(
            f"Phoenix returned HTTP {response.status_code} for {url}"
        )
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def get_phoenix_datasets() -> List[Dict[str, Any]]:
    """Get all datasets from Phoenix GraphQL API"""
    query = """
    query {
        datasets {
            edges {
                node {
                    id
                    name
                    exampleCount
                    createdAt
                    description
                    metadata
                }
            }
        }
    }
    """

    result = query_phoenix_graphql(query)
    datasets = []

    if result and "data" in result and result["data"]:
        for edge in result.get("data", {}).get("datasets", {}).get("edges", []):
            if edge and "node" in edge:
                node = edge["node"]
                datasets.append(
                    {
                        "id": node["id"],
                        "name": node["name"],
                        "example_count": node["exampleCount"],
                        "created_at": node["createdAt"],
                        "description": node.get("description", ""),
                        "metadata": node.get("metadata", {}),
                    }
                )

    return datasets


def calculate_metrics(results: List[str], expected: List[str]) -> Dict[str, float]:
    """Calculate retrieval metrics"""
    if not expected:
        return {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0}

    mrr = 0.0
    for i, video in enumerate(results[:10]):
        if video in expected:
            mrr = 1.0 / (i + 1)
            break

    recall_1 = 1.0 if results and results[0] in expected else 0.0
    recall_5 = (
        len(set(results[:5]) & set(expected)) / len(expected) if expected else 0.0
    )

    return {"mrr": mrr, "recall@1": recall_1, "recall@5": recall_5}


def format_video_result(
    video_id: str, expected_videos: List[str], position: int
) -> str:
    """Format video result with icon"""
    if video_id in expected_videos:
        return f'<span style="color: green;">✓ {video_id}</span>'
    else:
        return f'<span style="color: red;">✗ {video_id}</span>'


def _aggregate_experiment_metrics(experiment_data: Dict[str, Any]) -> None:
    """Compute mean MRR / recall@1 / recall@5 per profile/strategy from the
    collected per-query metrics. Mutates ``experiment_data`` in place. Run once
    after all experiments are loaded — recomputing it inside the per-experiment
    loop was O(experiments^2) and redundant."""
    for prof in experiment_data:
        for strat in experiment_data[prof]:
            queries = experiment_data[prof][strat]["queries"]
            if not queries:
                continue
            mrr_values = [q["metrics"]["mrr"] for q in queries]
            recall1_values = [q["metrics"]["recall@1"] for q in queries]
            recall5_values = [q["metrics"]["recall@5"] for q in queries]
            experiment_data[prof][strat]["aggregate_metrics"] = {
                "mrr": {"mean": sum(mrr_values) / len(mrr_values)},
                "recall@1": {"mean": sum(recall1_values) / len(recall1_values)},
                "recall@5": {"mean": sum(recall5_values) / len(recall5_values)},
            }


SEARCH_SPAN_NAME = "search_service.search"
_SPAN_PAGE_LIMIT = 1000


@dataclass
class GoldenSearchResults:
    """Per-profile, per-strategy scores of the recorded searches of one golden
    dataset, and how many matching searches could not be scored."""

    results: Dict[str, Any] = field(default_factory=dict)
    unscored_searches: int = 0


def _phoenix_get(
    path: str,
    params: Dict[str, Any] | None = None,
    *,
    missing_project: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """GET a Phoenix REST path. ``missing_project`` is returned when Phoenix
    answers that the path's project does not exist (no span was ever
    recorded for it); every other non-200 answer raises."""
    url = f"{_phoenix_base_url()}{path}"
    try:
        response = requests.get(url, params=params, timeout=_PHOENIX_REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        raise PhoenixUnavailableError(f"Phoenix unreachable at {url}: {exc}") from exc
    if (
        missing_project is not None
        and response.status_code == 404
        and response.text.startswith("Project with name ")
        and response.text.rstrip().endswith(" not found")
    ):
        return missing_project
    if response.status_code != 200:
        raise PhoenixUnavailableError(
            f"Phoenix returned HTTP {response.status_code} for {url}"
        )
    return response.json()


def _golden_expectations(dataset_id: str) -> Dict[str, List[str]]:
    """Each example's query and expected source keys, as DatasetManager writes
    them (``query`` in, comma-joined ``expected_videos`` out)."""
    expected: Dict[str, List[str]] = {}
    payload = _phoenix_get(f"/v1/datasets/{dataset_id}/examples")
    for example in payload["data"]["examples"]:
        query = str(example["input"].get("query", "")).strip()
        raw = example["output"].get("expected_videos", "")
        items = raw.split(",") if isinstance(raw, str) else list(raw or [])
        ids = [str(item).strip() for item in items if str(item).strip()]
        if query and ids:
            expected[query] = ids
    return expected


def _search_spans(tenant_id: str, lookback_hours: int) -> List[Dict[str, Any]]:
    """The tenant's search spans in the window, every page of them."""
    from cogniverse_dashboard.utils import tenant_project_name
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    project = tenant_project_name(get_telemetry_manager(), tenant_id)
    start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    params: Dict[str, Any] = {
        "limit": _SPAN_PAGE_LIMIT,
        "start_time": start.isoformat(),
    }
    spans: List[Dict[str, Any]] = []
    while True:
        page = _phoenix_get(
            f"/v1/projects/{quote(project, safe='')}/spans",
            params,
            missing_project={"data": [], "next_cursor": None},
        )
        spans.extend(
            span for span in page.get("data", []) if span["name"] == SEARCH_SPAN_NAME
        )
        cursor = page.get("next_cursor")
        if not cursor:
            return spans
        params = {**params, "cursor": cursor}


@st.cache_data(ttl=60, show_spinner="Scoring recorded searches...")
def get_golden_search_results(
    dataset_id: str, tenant_id: str, lookback_hours: int = 168
) -> GoldenSearchResults:
    """Score the tenant's recorded searches of a golden dataset's queries.

    Every ``search_service.search`` span in the window whose query is one of
    the dataset's is scored under its ``profile`` and ``strategy``: its result
    rows are keyed by ``result_source_title_key`` — the key golden sets name a
    source by — and the latest search per profile, strategy and query counts.
    A search with a row that carries no ``source_title`` is not scored and is
    counted in ``unscored_searches``. Raises ``PhoenixUnavailableError`` when
    Phoenix does not answer.
    """
    import json

    expected_by_query = _golden_expectations(dataset_id)
    latest: Dict[tuple, Dict[str, Any]] = {}
    unscored = 0
    for span in _search_spans(tenant_id, lookback_hours):
        attributes = span.get("attributes") or {}
        query = str(attributes.get("query", "")).strip()
        if query not in expected_by_query:
            continue
        try:
            rows = json.loads(attributes.get("output.value") or "[]")
            retrieved = list(
                dict.fromkeys(result_source_title_key(row) for row in rows)
            )
        except ValueError:
            unscored += 1
            continue
        key = (
            str(attributes.get("profile") or "unknown"),
            str(attributes.get("strategy") or "default"),
            query,
        )
        if key not in latest or span["start_time"] > latest[key]["start_time"]:
            latest[key] = {"start_time": span["start_time"], "retrieved": retrieved}

    results: Dict[str, Any] = {}
    for (profile, strategy, query), searched in sorted(latest.items()):
        expected = expected_by_query[query]
        results.setdefault(profile, {}).setdefault(
            strategy,
            {
                "queries": [],
                "aggregate_metrics": {
                    "mrr": {"mean": 0},
                    "recall@1": {"mean": 0},
                    "recall@5": {"mean": 0},
                },
            },
        )["queries"].append(
            {
                "query": query,
                "expected": expected,
                "results": searched["retrieved"],
                "metrics": calculate_metrics(searched["retrieved"], expected),
            }
        )
    _aggregate_experiment_metrics(results)
    return GoldenSearchResults(results=results, unscored_searches=unscored)


def render_evaluation_tab():
    """Render the evaluation tab with EXACT tabbed format"""
    st.subheader("🧪 Golden Set Evaluation")

    # Get datasets — a Phoenix outage renders as an error, never as the
    # same empty state a fresh project shows.
    try:
        datasets = get_phoenix_datasets()
    except PhoenixUnavailableError as exc:
        st.error(f"Cannot load datasets: {exc}")
        return
    if not datasets:
        st.warning("No datasets found in Phoenix.")
        return

    # Sort datasets by creation date
    datasets.sort(key=lambda x: x["created_at"], reverse=True)

    # Dataset selector
    dataset_names = [ds["name"] for ds in datasets]
    selected_dataset_name = st.selectbox(
        "Select Dataset",
        dataset_names,
        index=0,
        format_func=lambda x: (
            f"{x} ({next(ds['example_count'] for ds in datasets if ds['name'] == x)} examples)"
        ),
    )

    selected_dataset = next(
        ds for ds in datasets if ds["name"] == selected_dataset_name
    )

    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Examples", selected_dataset["example_count"])
    with col2:
        created_date = (
            selected_dataset["created_at"].split("T")[0]
            if "T" in selected_dataset["created_at"]
            else selected_dataset["created_at"]
        )
        st.metric("Created", created_date)
    with col3:
        st.markdown(
            f"[View in Phoenix]({_phoenix_base_url()}/datasets/{selected_dataset['id']})"
        )

    tenant_id = st.session_state["current_tenant"]
    lookback_hours = st.number_input(
        "Lookback (hours)", min_value=1, max_value=24 * 90, value=168
    )
    with st.spinner("Scoring recorded searches..."):
        try:
            scored = get_golden_search_results(
                selected_dataset["id"], tenant_id, int(lookback_hours)
            )
        except PhoenixUnavailableError as exc:
            st.error(f"Cannot load recorded searches: {exc}")
            return
    experiment_data = scored.results
    if scored.unscored_searches:
        st.caption(
            f"{scored.unscored_searches} recorded searches of these queries "
            "carry a result with no source title and are not scored."
        )

    st.markdown("---")

    # Only show profiles that have data
    profiles_with_data = []
    for profile_key in experiment_data:
        # Use the profile key as the display name directly
        profiles_with_data.append((profile_key, profile_key))

    if not profiles_with_data:
        st.warning(
            f"No searches of this dataset's queries recorded for tenant "
            f"{tenant_id} in the last {int(lookback_hours)} hours."
        )
        return

    # Create profile tabs (MAIN TABS) - only for profiles with data
    profile_tabs = st.tabs([profile_name for _, profile_name in profiles_with_data])

    # Use actual experiment data from Phoenix
    experiment_results = experiment_data

    # For each profile tab
    for prof_idx, ((profile_key, profile_name), profile_tab) in enumerate(
        zip(profiles_with_data, profile_tabs)
    ):
        with profile_tab:
            # Get strategies that actually have data for this profile
            strategies_with_data = []
            if profile_key in experiment_results:
                for strat_key in experiment_results[profile_key]:
                    # Use the strategy key as the display name directly
                    strategies_with_data.append((strat_key, strat_key))

            if not strategies_with_data:
                st.info("No strategies found for this profile")
                continue

            # Create strategy tabs (NESTED TABS) - only for strategies with data
            strategy_tabs = st.tabs(
                [strat_name for _, strat_name in strategies_with_data]
            )

            # For each strategy tab
            for strat_idx, ((strat_key, strat_name), strategy_tab) in enumerate(
                zip(strategies_with_data, strategy_tabs)
            ):
                with strategy_tab:
                    # Get experiment data from loaded results
                    profile_data = experiment_results.get(profile_key, {})
                    strategy_data = profile_data.get(strat_key, {})

                    # We know there's data because we only created tabs for strategies with data
                    if not strategy_data or not strategy_data.get("queries"):
                        st.warning("No query results found")
                        continue

                    # Summary metrics (like HTML report)
                    metrics = strategy_data.get("aggregate_metrics", {})
                    mrr = metrics.get("mrr", {}).get("mean", 0)
                    recall1 = metrics.get("recall@1", {}).get("mean", 0)
                    recall5 = metrics.get("recall@5", {}).get("mean", 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MRR Score", f"{mrr * 100:.1f}%")
                    with col2:
                        st.metric("Recall@1", f"{recall1 * 100:.1f}%")
                    with col3:
                        st.metric("Recall@5", f"{recall5 * 100:.1f}%")
                    with col4:
                        st.metric("Queries", len(strategy_data.get("queries", [])))

                    # Query results table
                    st.markdown("#### Query Results")

                    queries_data = strategy_data.get("queries", [])
                    if queries_data:
                        # Styled headers
                        header_html = """
                        <div style="display: flex; padding: 10px 0; background-color: #f0f2f6; border-radius: 4px; margin-bottom: 10px;">
                            <div style="flex: 3; padding: 0 10px;"><strong style="color: #1f2937; font-size: 14px;">QUERY</strong></div>
                            <div style="flex: 3; padding: 0 10px;"><strong style="color: #1f2937; font-size: 14px;">EXPECTED</strong></div>
                            <div style="flex: 6; padding: 0 10px;"><strong style="color: #1f2937; font-size: 14px;">RETRIEVED RESULTS</strong></div>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)

                        # Create compact display for each query
                        for idx, q in enumerate(queries_data):
                            query = q["query"]
                            expected = q.get("expected", [])
                            results = q.get("results", [])[:3]  # Top 3
                            metrics = q.get("metrics", {})
                            query_mrr = metrics.get("mrr", 0)
                            recall_1 = metrics.get("recall@1", 0)
                            recall_5 = metrics.get("recall@5", 0)

                            # Single row layout
                            col1, col2, col3 = st.columns([3, 3, 6])

                            with col1:
                                st.markdown(f"**{query}**")

                            with col2:
                                expected_str = (
                                    ", ".join([f"`{v}`" for v in expected])
                                    if expected
                                    else "`None`"
                                )
                                st.markdown(expected_str)

                            with col3:
                                # Format retrieved with marks
                                retrieved_parts = []
                                for vid in results:
                                    if vid in expected:
                                        retrieved_parts.append(f"✅ `{vid}`")
                                    else:
                                        retrieved_parts.append(f"❌ `{vid}`")
                                retrieved_str = (
                                    " | ".join(retrieved_parts)
                                    if retrieved_parts
                                    else "No results"
                                )

                                # Create styled badges
                                mrr_style = (
                                    "background-color: #d4edda; color: #155724;"
                                    if query_mrr >= 0.7
                                    else (
                                        "background-color: #fff3cd; color: #856404;"
                                        if query_mrr >= 0.3
                                        else "background-color: #f8d7da; color: #721c24;"
                                    )
                                )
                                r1_style = (
                                    "background-color: #d4edda; color: #155724;"
                                    if recall_1 >= 0.7
                                    else (
                                        "background-color: #fff3cd; color: #856404;"
                                        if recall_1 >= 0.3
                                        else "background-color: #f8d7da; color: #721c24;"
                                    )
                                )
                                r5_style = (
                                    "background-color: #d4edda; color: #155724;"
                                    if recall_5 >= 0.7
                                    else (
                                        "background-color: #fff3cd; color: #856404;"
                                        if recall_5 >= 0.3
                                        else "background-color: #f8d7da; color: #721c24;"
                                    )
                                )

                                badges_html = f"""
                                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 8px; {mrr_style}">MRR: {query_mrr:.3f}</span>
                                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 4px; {r1_style}">R@1: {recall_1:.3f}</span>
                                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 4px; {r5_style}">R@5: {recall_5:.3f}</span>
                                """

                                st.markdown(
                                    retrieved_str + badges_html, unsafe_allow_html=True
                                )

                            # Subtle separator
                            st.markdown("---")
                    else:
                        st.info("No query results available")

    # Success Matrix Heatmap (at the bottom)
    st.markdown("---")
    st.markdown("### 📊 Experiment Success Matrix")

    # Build matrix data from actual experiment results
    # Collect all unique strategies that were actually run
    all_strategies_run = set()
    for profile_key in experiment_results:
        for strat_key in experiment_results[profile_key]:
            all_strategies_run.add(strat_key)

    if all_strategies_run:
        # Build matrix only for profiles and strategies that have data
        matrix_profiles = []
        matrix_strategies = sorted(list(all_strategies_run))
        matrix_data = []

        for profile_key, profile_name in profiles_with_data:
            if profile_key in experiment_results:
                matrix_profiles.append(profile_name)
                row = []
                for strat_key in matrix_strategies:
                    if strat_key in experiment_results[profile_key]:
                        # Calculate success (1 if experiment ran successfully, 0 if not)
                        queries = experiment_results[profile_key][strat_key].get(
                            "queries", []
                        )
                        if queries:
                            # Calculate average recall@1 as success metric
                            recall_sum = sum(
                                1
                                for q in queries
                                if q.get("metrics", {}).get("mrr", 0) == 1.0
                            )
                            success_rate = recall_sum / len(queries) if queries else 0
                            row.append(success_rate)
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                if row:  # Only add if we have data
                    matrix_data.append(row)

        # Use strategy keys directly as display names
        matrix_strategies_display = matrix_strategies

        if matrix_data and matrix_profiles and matrix_strategies_display:
            # Format text as percentages
            text_data = [[f"{val * 100:.0f}%" for val in row] for row in matrix_data]

            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix_data,
                    x=matrix_strategies_display,
                    y=matrix_profiles,
                    colorscale=[[0, "#e74c3c"], [1, "#27ae60"]],
                    showscale=False,
                    text=text_data,
                    texttemplate="%{text}",
                    hovertemplate="Profile: %{y}<br>Strategy: %{x}<br>Success Rate: %{text}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Experiment Success by Profile and Strategy (Recall@1)",
                xaxis_title="Strategy",
                yaxis_title="Profile",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No experiment data available to display success matrix.")
    else:
        st.info(
            "No experiments found. Run experiments first to see the success matrix."
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Phoenix Tabbed Evaluation", layout="wide")
    render_evaluation_tab()
