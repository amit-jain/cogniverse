"""Interaction tests for the optimization tab's three main submit flows.

`_filter_search_spans` has direct logic tests; these drive the interactive
flows the smoke test never clicks — golden-dataset build (telemetry
boundary), synthetic-data generation (runtime HTTP boundary + approval
split), and the Argo workflow submit (subprocess boundary) — pinning the
exact payload each flow hands to its boundary and the exact status text it
renders back.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _restore_patched_boundaries():
    """The AppTest scripts patch ``requests.post``, ``subprocess.run`` and
    the telemetry manager factory in-process; restore all three so the
    fakes don't leak into later test files."""
    import subprocess

    import requests

    import cogniverse_foundation.telemetry.manager as tm

    originals = (requests.post, subprocess.run, tm.get_telemetry_manager)
    yield
    requests.post, subprocess.run, tm.get_telemetry_manager = originals


def _golden_dataset_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        from datetime import datetime, timezone

        import pandas as pd
        import streamlit as st

        st.session_state["current_tenant"] = "acme"

        import cogniverse_foundation.telemetry.manager as tm

        spans_df = pd.DataFrame(
            [
                {
                    "name": "video_search.query",
                    "attributes.annotation.score": 0.9,
                    "attributes.query": "cats playing piano",
                    "attributes.results": [{"id": "video_1"}, {"video_id": "video_2"}],
                    "attributes.profile": "video_colpali",
                    "start_time": datetime(2026, 6, 1, tzinfo=timezone.utc),
                },
                {
                    "name": "video_search.query",
                    "attributes.annotation.score": 0.3,
                    "attributes.query": "dogs surfing",
                    "attributes.results": [{"id": "video_9"}],
                    "attributes.profile": "video_colpali",
                    "start_time": datetime(2026, 6, 2, tzinfo=timezone.utc),
                },
                {
                    "name": "cogniverse.routing",
                    "attributes.annotation.score": 0.95,
                    "attributes.query": "not a search span",
                    "attributes.results": [{"id": "video_x"}],
                    "attributes.profile": "video_colpali",
                    "start_time": datetime(2026, 6, 3, tzinfo=timezone.utc),
                },
            ]
        )

        class _Traces:
            async def get_spans(self, **kwargs):
                st.session_state.setdefault("_get_spans_calls", []).append(
                    (
                        kwargs.get("project"),
                        (kwargs["end_time"] - kwargs["start_time"]).days,
                    )
                )
                return spans_df

        class _Provider:
            traces = _Traces()

        class _Manager:
            def get_provider(self, tenant_id=None):
                st.session_state.setdefault("_provider_tenants", []).append(tenant_id)
                return _Provider()

        tm.get_telemetry_manager = lambda: _Manager()

        import cogniverse_dashboard.tabs.optimization as opt

        opt._render_golden_dataset_tab()
        """
    ).strip()
    path = tmp_path / "app_golden_dataset.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_golden_dataset_build_filters_by_rating_and_span_name(tmp_path: Path) -> None:
    at = _golden_dataset_app(tmp_path)
    at.run()
    at.button[0].click().run()

    assert at.exception == []
    assert at.session_state["_get_spans_calls"] == [("cogniverse-acme", 30)]
    assert at.session_state["_provider_tenants"] == ["acme"]
    assert "Built golden dataset with 1 queries" in [s.value for s in at.success]

    # Only the search span above the 0.8 rating threshold survives; the
    # low-rated search span and the routing span are dropped.
    assert at.session_state["golden_dataset"] == {
        "cats playing piano": {
            "expected_videos": ["video_1", "video_2"],
            "relevance_scores": {"video_1": 1.0, "video_2": 0.5},
            "avg_relevance": 0.9,
            "profile": "video_colpali",
            "timestamp": "2026-06-01T00:00:00+00:00",
        }
    }
    assert at.session_state["golden_dataset_size"] == 1


def _synthetic_data_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import streamlit as st

        st.session_state["current_tenant"] = "acme"
        st.session_state["runtime_url"] = "http://runtime.test:8000"

        import requests

        _RESULT = {
            "optimizer": "profile",
            "count": 2,
            "selected_profiles": ["video_colpali", "frame_based_colpali"],
            "profile_selection_reasoning": "Two profiles cover the sampled content",
            "schema_name": "ProfileSelectionExampleSchema",
            "metadata": {"generation_time_ms": 1234},
            "tenant_id": "acme",
            "data": [
                {
                    "query": "find TensorFlow tutorial videos",
                    "entities": ["TensorFlow"],
                    "reasoning": "TensorFlow is the primary entity to include",
                    "_generation_metadata": {"retry_count": 0},
                },
                {
                    "query": "cat video",
                    "entities": ["dog"],
                    "reasoning": "",
                    "_generation_metadata": {"retry_count": 2},
                },
            ],
        }

        class _Response:
            status_code = 200

            def json(self):
                return _RESULT

        def _fake_post(url, json=None, timeout=None):
            st.session_state.setdefault("_post_calls", []).append(
                (url, json, timeout)
            )
            return _Response()

        requests.post = _fake_post

        import cogniverse_dashboard.tabs.optimization as opt

        opt._render_synthetic_data_tab()
        """
    ).strip()
    path = tmp_path / "app_synthetic_data.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_synthetic_generation_posts_exact_payload_and_splits_approval(
    tmp_path: Path,
) -> None:
    at = _synthetic_data_app(tmp_path)
    at.run()
    at.button[0].click().run()

    assert at.exception == []
    assert at.session_state["_post_calls"] == [
        (
            "http://runtime.test:8000/synthetic/generate",
            {
                "optimizer": "profile",
                "count": 100,
                "vespa_sample_size": 200,
                "strategies": ["diverse"],
                "max_profiles": 3,
                "tenant_id": "acme",
            },
            300,
        )
    ]

    # Confidence extraction is deterministic: the clean item scores 1.0
    # (auto-approved at the default 0.85 threshold), the retried item with a
    # missing entity and a short query scores 0.39 (pending review).
    batch = at.session_state["last_generated_batch"]
    assert [item.confidence for item in batch.items] == [1.0, 0.39]
    assert [item.status.value for item in batch.items] == [
        "auto_approved",
        "pending_review",
    ]
    assert [item.data["query"] for item in batch.pending_review] == ["cat video"]

    successes = [s.value for s in at.success]
    assert "Generated 2 examples: 1 auto-approved, 1 awaiting review" in successes
    infos = [i.value for i in at.info]
    assert "**Profile Selection**: Two profiles cover the sampled content" in infos
    assert (
        "**1 items need your review**. "
        "Navigate to the **Approval Queue** tab to review them." in infos
    )

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Schema"] == "ProfileSelectionExampleSchema"
    assert metrics["Generation Time"] == "1234ms"
    assert metrics["Profiles Used"] == "2"
    assert metrics["Auto-Approved"] == "1"
    assert metrics["Pending Review"] == "1"
    assert metrics["Avg Confidence"] == f"{(1.0 + 0.39) / 2:.2f}"

    code_blocks = [c.value for c in at.code]
    assert "video_colpali" in code_blocks
    assert "frame_based_colpali" in code_blocks

    # The pending item is offered for inline review.
    assert "🔍 Review Low-Confidence Items" in [h.value for h in at.subheader]


def _workflow_submit_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import subprocess
        from pathlib import Path
        from types import SimpleNamespace

        import streamlit as st

        st.session_state["current_tenant"] = "acme"

        def _fake_run(args, capture_output=None, text=None, timeout=None):
            yaml_text = Path(args[2]).read_text()
            st.session_state.setdefault("_run_calls", []).append(
                (args[:2] + args[3:], yaml_text)
            )
            return SimpleNamespace(
                returncode=0,
                stdout="workflow.argoproj.io/routing-opt-routing-x7k2p created",
                stderr="",
            )

        subprocess.run = _fake_run

        import cogniverse_dashboard.tabs.optimization as opt

        opt._render_routing_optimization_tab()
        """
    ).strip()
    path = tmp_path / "app_workflow_submit.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_workflow_submit_builds_exact_argo_spec(tmp_path: Path) -> None:
    at = _workflow_submit_app(tmp_path)
    at.run()
    at.button[0].click().run()

    assert at.exception == []
    run_calls = at.session_state["_run_calls"]
    assert len(run_calls) == 1
    argv, yaml_text = run_calls[0]
    assert argv == ["argo", "submit", "-n", "cogniverse"]

    workflow = yaml.safe_load(yaml_text)
    assert workflow["apiVersion"] == "argoproj.io/v1alpha1"
    assert workflow["kind"] == "Workflow"
    assert workflow["metadata"] == {
        "generateName": "routing-opt-routing-",
        "namespace": "cogniverse",
    }
    assert workflow["spec"]["workflowTemplateRef"] == {"name": "batch-optimization"}
    assert workflow["spec"]["arguments"]["parameters"] == [
        {"name": "tenant-id", "value": "acme"},
        {"name": "optimizer-category", "value": "routing"},
        {"name": "optimizer-type", "value": "routing"},
        {"name": "max-iterations", "value": "100"},
        {"name": "use-synthetic-data", "value": "true"},
    ]

    assert "Workflow submitted successfully!" in [s.value for s in at.success]
    assert "workflow.argoproj.io/routing-opt-routing-x7k2p created" in [
        c.value for c in at.code
    ]


def test_golden_dataset_excludes_nan_annotation_scores(monkeypatch) -> None:
    """pandas yields NaN (not None) for a missing score when the column
    exists — NaN < min_rating is False, which let unannotated spans into
    the dataset with avg_relevance NaN and broke the JSON export."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    import pandas as pd

    from cogniverse_dashboard.tabs import optimization as opt

    spans_df = pd.DataFrame(
        [
            {
                "name": "search",
                "attributes.annotation.score": 0.9,
                "attributes.query": "annotated query",
                "attributes.results": [
                    {"video_id": "v1", "relevance": 1.0},
                ],
                "attributes.profile": "video_colpali",
                "start_time": "2026-06-01T00:00:00+00:00",
            },
            {
                "name": "search",
                "attributes.annotation.score": float("nan"),
                "attributes.query": "unannotated query",
                "attributes.results": [
                    {"video_id": "v2", "relevance": 1.0},
                ],
                "attributes.profile": "video_colpali",
                "start_time": "2026-06-01T00:00:00+00:00",
            },
        ]
    )
    provider = MagicMock()
    provider.traces.get_spans = AsyncMock(return_value=spans_df)
    manager = MagicMock()
    manager.get_provider.return_value = provider
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: manager,
    )

    dataset = asyncio.run(
        opt._build_golden_dataset_from_phoenix("acme", min_rating=0.8, lookback_days=7)
    )

    assert list(dataset.keys()) == ["annotated query"]
    assert dataset["annotated query"]["avg_relevance"] == 0.9
