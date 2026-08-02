"""Real-Phoenix happy path for the Evaluation tab's data loaders.

The outage side of the contract (dead endpoint / error status raises
PhoenixUnavailableError) is pinned by unit tests. These tests pin the success
side against a real Phoenix container: the GraphQL dataset listing returns the
dataset a writer created, a dataset with zero experiments loads as an explicit
empty result (distinct from an outage), and a real experiment's runs parse
into the exact per-query structure the tab renders.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import streamlit as st

from cogniverse_dashboard.tabs import evaluation

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


@pytest.fixture
def phoenix_tab(phoenix_container):
    st.session_state["phoenix_url"] = phoenix_container["http_endpoint"]
    st.cache_data.clear()
    yield evaluation
    st.cache_data.clear()
    st.session_state.pop("phoenix_url", None)


@pytest.fixture
def phoenix_client(phoenix_container):
    from phoenix.client import Client

    return Client(base_url=phoenix_container["http_endpoint"])


def _create_dataset(client, name):
    return client.datasets.create_dataset(
        name=name,
        inputs=[{"query": "find the red car"}],
        outputs=[{"expected_videos": "v1,v2"}],
    )


def test_dataset_listing_returns_created_dataset(phoenix_tab, phoenix_client):
    name = f"eval-tab-ds-{uuid4().hex[:8]}"
    _create_dataset(phoenix_client, name)

    listed = phoenix_tab.get_phoenix_datasets()

    match = [d for d in listed if d["name"] == name]
    assert len(match) == 1
    assert match[0]["example_count"] == 1
    assert match[0]["id"]  # GraphQL global id feeds the experiment loader


def test_dataset_with_no_experiments_loads_as_empty(phoenix_tab, phoenix_client):
    name = f"eval-tab-empty-{uuid4().hex[:8]}"
    _create_dataset(phoenix_client, name)
    listed = phoenix_tab.get_phoenix_datasets()
    dataset_id = next(d["id"] for d in listed if d["name"] == name)

    data = phoenix_tab.get_all_experiment_data_for_dataset(dataset_id)

    # Zero experiments is a VALID empty result; an unreachable Phoenix raises
    # PhoenixUnavailableError instead (pinned in the unit tests).
    assert data == {}


def test_experiment_runs_parse_into_exact_tab_structure(phoenix_tab, phoenix_client):
    name = f"eval-tab-exp-{uuid4().hex[:8]}"
    dataset = _create_dataset(phoenix_client, name)

    def task(example):
        return {
            "profile": "profile_a",
            "ranking_strategy": "strategy_b",
            # Duplicate v1 on purpose: the loader must dedupe retrieved videos.
            "results": [
                {"video_id": "v1"},
                {"video_id": "v3"},
                {"video_id": "v1"},
            ],
        }

    phoenix_client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        experiment_name=f"exp-{uuid4().hex[:8]}",
        experiment_metadata={"profile": "profile_a", "ranking_strategy": "strategy_b"},
        print_summary=False,
    )

    listed = phoenix_tab.get_phoenix_datasets()
    dataset_id = next(d["id"] for d in listed if d["name"] == name)
    st.cache_data.clear()
    data = phoenix_tab.get_all_experiment_data_for_dataset(dataset_id)

    # Exact structure the tab renders: expected_videos csv split, retrieved
    # videos deduped in rank order, metrics computed from that ranking
    # (v1 hits at rank 1 -> mrr 1.0, recall@1 1.0; 1 of 2 expected in the
    # top 5 -> recall@5 0.5), aggregates equal to the single query's metrics.
    assert data == {
        "profile_a": {
            "strategy_b": {
                "queries": [
                    {
                        "query": "find the red car",
                        "expected": ["v1", "v2"],
                        "results": ["v1", "v3"],
                        "metrics": {"mrr": 1.0, "recall@1": 1.0, "recall@5": 0.5},
                    }
                ],
                "aggregate_metrics": {
                    "mrr": {"mean": 1.0},
                    "recall@1": {"mean": 1.0},
                    "recall@5": {"mean": 0.5},
                },
            }
        }
    }
