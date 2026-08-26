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
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from streamlit.testing.v1 import AppTest


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


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

        st.session_state["current_tenant"] = "acme:acme"
        st.session_state["runtime_url"] = "http://runtime.test:8000"

        class _ApprovalAgent:
            threshold = 0.85

            async def submit_for_review(self, batch):
                from cogniverse_agents.approval import ApprovalStatus

                for item in batch.items:
                    item.status = (
                        ApprovalStatus.AUTO_APPROVED
                        if item.confidence >= self.threshold
                        else ApprovalStatus.PENDING_REVIEW
                    )
                st.session_state["_submitted_batch"] = {
                    "batch_id": batch.batch_id,
                    "context": batch.context,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "status": item.status.value,
                            "metadata": item.metadata,
                        }
                        for item in batch.items
                    ],
                }
                return batch

            def get_approval_stats(self, batch):
                return {
                    "auto_approved": len(batch.auto_approved),
                    "pending_review": len(batch.pending_review),
                    "avg_confidence": sum(
                        item.confidence for item in batch.items
                    ) / len(batch.items),
                }

        st.session_state["approval_agent"] = _ApprovalAgent()
        st.session_state["approval_agent_tenant_id"] = "acme:acme"

        import requests

        _RESULT = {
            "optimizer": "profile",
            "count": 2,
            "selected_profiles": ["video_colpali", "frame_based_colpali"],
            "profile_selection_reasoning": "Two profiles cover the sampled content",
            "schema_name": "ProfileSelectionExampleSchema",
            "metadata": {"generation_time_ms": 1234},
            "data": [
                {
                    "query": "find TensorFlow tutorial videos",
                    "available_profiles": "video_colpali,frame_based_colpali",
                    "selected_profile": "video_colpali",
                    "reasoning": "Video retrieval matches the requested tutorial.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "medium",
                },
                {
                    "query": "cat video",
                    "available_profiles": "video_colpali,frame_based_colpali",
                    "selected_profile": "frame_based_colpali",
                    "reasoning": "Frame retrieval matches the short visual query.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "simple",
                },
            ],
        }

        class _Response:
            status_code = 200

            def json(self):
                return _RESULT

        def _fake_post(url, json=None, timeout=None):
            from cogniverse_synthetic.schemas import SyntheticDataRequest

            validated = SyntheticDataRequest.model_validate(json)
            st.session_state.setdefault("_post_calls", []).append(
                (url, validated.model_dump(), timeout)
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
    assert at.selectbox[0].options == [
        "query_enhancement",
        "profile",
        "routing",
        "entity_extraction",
    ]
    at.button[0].click().run()

    assert at.exception == []
    assert at.session_state["_post_calls"] == [
        (
            "http://runtime.test:8000/synthetic/generate",
            {
                "optimizer": "profile",
                "count": 100,
                "vespa_sample_size": 200,
                "strategy": "diverse",
                "max_profiles": 3,
                "tenant_id": "acme:acme",
            },
            300,
        )
    ]

    batch = at.session_state["last_generated_batch"]
    assert [item.confidence for item in batch.items] == [0.0, 0.0]
    assert [item.status.value for item in batch.items] == [
        "pending_review",
        "pending_review",
    ]
    assert [item.data["query"] for item in batch.pending_review] == [
        "find TensorFlow tutorial videos",
        "cat video",
    ]
    assert at.session_state["_submitted_batch"] == {
        "batch_id": batch.batch_id,
        "context": {
            "optimizer": "profile",
            "agent_type": "profile_selection",
            "tenant_id": "acme:acme",
            "profiles": ["video_colpali", "frame_based_colpali"],
        },
        "items": [
            {
                "item_id": f"{batch.batch_id}_0",
                "status": "pending_review",
                "metadata": {
                    "approval_batch_id": batch.batch_id,
                    "agent_type": "profile_selection",
                },
            },
            {
                "item_id": f"{batch.batch_id}_1",
                "status": "pending_review",
                "metadata": {
                    "approval_batch_id": batch.batch_id,
                    "agent_type": "profile_selection",
                },
            },
        ],
    }

    successes = [s.value for s in at.success]
    assert "Generated 2 examples: 0 auto-approved, 2 awaiting review" in successes
    infos = [i.value for i in at.info]
    assert "**Profile Selection**: Two profiles cover the sampled content" in infos
    assert (
        "**2 items need your review**. "
        "Navigate to the **Approval Queue** tab to review them." in infos
    )

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Schema"] == "ProfileSelectionExampleSchema"
    assert metrics["Generation Time"] == "1234ms"
    assert metrics["Profiles Used"] == "2"
    assert metrics["Auto-Approved"] == "0"
    assert metrics["Pending Review"] == "2"
    assert metrics["Avg Confidence"] == "0.00"
    assert metrics["Retries"] == "0"

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


def test_create_dataset_from_upload_threads_tenant() -> None:
    """The CSV-upload path scopes the dataset store to the session tenant —
    the zero-arg DatasetManager() construction this replaced raised TypeError
    on every upload."""
    from unittest.mock import patch

    from cogniverse_dashboard.tabs.optimization import create_dataset_from_upload

    with patch("cogniverse_evaluation.data.DatasetManager") as manager_cls:
        manager_cls.return_value.create_from_csv.return_value = "ds-up"

        result = create_dataset_from_upload("acme:dash", "/tmp/q.csv", "uploaded")

    assert result == "ds-up"
    manager_cls.assert_called_once_with(tenant_id="acme:dash")
    manager_cls.return_value.create_from_csv.assert_called_once_with(
        csv_path="/tmp/q.csv",
        dataset_name="uploaded",
        description="Uploaded via optimization dashboard",
    )


def test_inline_approval_persists_exact_decision_before_session_mutation(
    monkeypatch,
) -> None:
    from cogniverse_agents.approval import (
        ApprovalBatch,
        ApprovalStatus,
        ReviewItem,
    )
    from cogniverse_dashboard.tabs import optimization

    item = ReviewItem(
        item_id="item-1",
        data={"query": "find Curie"},
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    batch = ApprovalBatch(batch_id="batch-17", items=[item])
    persisted = deepcopy(item)
    persisted.status = ApprovalStatus.APPROVED
    agent = MagicMock()
    agent.apply_decision = AsyncMock(return_value=persisted)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        last_generated_batch=batch,
        pending_items=[item],
        approved_items=[],
        user_email="reviewer@example.test",
    )
    monkeypatch.setattr(optimization, "st", fake_st)

    optimization._handle_inline_approval(item, 0)

    agent.apply_decision.assert_awaited_once()
    batch_id, decision = agent.apply_decision.await_args.args
    assert batch_id == "batch-17"
    assert (
        decision.item_id,
        decision.approved,
        decision.feedback,
        decision.corrections,
        decision.reviewer,
    ) == ("item-1", True, None, {}, "reviewer@example.test")
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert batch.items == [persisted]
    assert batch.items[0] is persisted
    assert fake_st.session_state["pending_items"] == []
    assert fake_st.session_state["approved_items"] == [persisted]
    assert fake_st.session_state["approved_items"][0] is persisted
    fake_st.rerun.assert_called_once_with()


def test_inline_rejection_persists_exact_canonical_corrections(monkeypatch) -> None:
    from cogniverse_agents.approval import (
        ApprovalBatch,
        ApprovalStatus,
        ReviewItem,
    )
    from cogniverse_dashboard.tabs import optimization

    item = ReviewItem(
        item_id="item-1",
        data={
            "query": "PyTorch was created by Meta AI",
            "entities": [
                {"text": "PyTorch", "type": "PRODUCT"},
                {"text": "Meta AI", "type": "ORG"},
            ],
            "relationships": [
                {"source": "Meta AI", "target": "PyTorch", "type": "created"}
            ],
        },
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    batch = ApprovalBatch(batch_id="batch-17", items=[item])
    regenerated = ReviewItem(
        item_id="item-1_regen_0",
        data={"query": "JAX was created by Google"},
        confidence=0.8,
        status=ApprovalStatus.REGENERATED,
    )
    agent = MagicMock()
    agent.apply_decision = AsyncMock(return_value=regenerated)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        last_generated_batch=batch,
        pending_items=[item],
        rejected_items=[],
        user_email="reviewer@example.test",
    )
    monkeypatch.setattr(optimization, "st", fake_st)
    corrections = {
        "entities": [
            {"text": "JAX", "type": "PRODUCT"},
            {"text": "Google", "type": "ORG"},
        ],
        "relationships": [{"source": "Google", "target": "JAX", "type": "created"}],
    }

    optimization._handle_inline_rejection(
        item,
        0,
        "Use the corrected product and organization.",
        corrections,
    )

    agent.apply_decision.assert_awaited_once()
    batch_id, decision = agent.apply_decision.await_args.args
    assert batch_id == "batch-17"
    assert (
        decision.item_id,
        decision.approved,
        decision.feedback,
        decision.corrections,
        decision.reviewer,
    ) == (
        "item-1",
        False,
        "Use the corrected product and organization.",
        corrections,
        "reviewer@example.test",
    )
    assert item.status is ApprovalStatus.REJECTED
    assert batch.items == [regenerated]
    assert batch.items[0] is regenerated
    assert fake_st.session_state["pending_items"] == [regenerated]
    assert fake_st.session_state["pending_items"][0] is regenerated
    assert fake_st.session_state["rejected_items"] == [(item, decision)]
    fake_st.rerun.assert_called_once_with()


def test_inline_approval_rejects_non_approved_persistence_result(monkeypatch) -> None:
    from cogniverse_agents.approval import ApprovalBatch, ApprovalStatus, ReviewItem
    from cogniverse_dashboard.tabs import optimization

    item = ReviewItem(
        item_id="item-1",
        data={"query": "find Curie"},
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    batch = ApprovalBatch(batch_id="batch-17", items=[item])
    agent = MagicMock()
    agent.apply_decision = AsyncMock(return_value=item)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        last_generated_batch=batch,
        pending_items=[item],
        approved_items=[],
    )
    monkeypatch.setattr(optimization, "st", fake_st)

    optimization._handle_inline_approval(item, 0)

    agent.apply_decision.assert_awaited_once()
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert batch.items == [item]
    assert fake_st.session_state["pending_items"] == [item]
    assert fake_st.session_state["approved_items"] == []
    fake_st.error.assert_called_once_with(
        "Failed to approve item: decision persistence returned pending_review; "
        "expected approved"
    )
    fake_st.rerun.assert_not_called()


def test_inline_approval_failure_leaves_session_and_item_pending(monkeypatch) -> None:
    from cogniverse_agents.approval import ApprovalBatch, ApprovalStatus, ReviewItem
    from cogniverse_dashboard.tabs import optimization

    item = ReviewItem(
        item_id="item-1",
        data={"query": "find Curie"},
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    batch = ApprovalBatch(batch_id="batch-17", items=[item])
    agent = MagicMock()
    agent.apply_decision = AsyncMock(side_effect=TimeoutError("Phoenix timed out"))
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        last_generated_batch=batch,
        pending_items=[item],
        approved_items=[],
    )
    monkeypatch.setattr(optimization, "st", fake_st)

    optimization._handle_inline_approval(item, 0)

    agent.apply_decision.assert_awaited_once()
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert fake_st.session_state["pending_items"] == [item]
    assert fake_st.session_state["approved_items"] == []
    fake_st.error.assert_called_once_with("Failed to approve item: Phoenix timed out")
    fake_st.rerun.assert_not_called()


def test_synthetic_batch_persistence_failure_does_not_publish_session_batch(
    monkeypatch,
) -> None:
    from cogniverse_dashboard.tabs import optimization

    agent = MagicMock()
    agent.submit_for_review = AsyncMock(side_effect=TimeoutError("Phoenix timed out"))
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        current_tenant="acme",
    )
    monkeypatch.setattr(optimization, "st", fake_st)
    result = {
        "optimizer": "profile",
        "schema_name": "ProfileSelectionExampleSchema",
        "selected_profiles": ["video_colpali"],
        "data": [
            {
                "query": "find transformer lectures",
                "available_profiles": "video_colpali,text_bm25",
                "selected_profile": "video_colpali",
                "reasoning": "Video retrieval matches the requested lectures.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            }
        ],
    }

    optimization._process_approval_workflow(result, tenant_id="acme")

    agent.submit_for_review.assert_awaited_once()
    assert "last_generated_batch" not in fake_st.session_state
    assert "pending_items" not in fake_st.session_state
    assert "approved_items" not in fake_st.session_state
    fake_st.error.assert_called_once_with(
        "❌ Approval workflow failed: Phoenix timed out"
    )


@pytest.mark.parametrize(
    ("optimizer", "schema_name", "item_data", "agent_type"),
    [
        (
            "query_enhancement",
            "QueryEnhancementExampleSchema",
            {
                "query": "find transformer lectures",
                "enhanced_query": "find transformer video lectures",
                "expansion_terms": ["video"],
                "synonyms": ["presentation"],
                "context": "video_colpali",
                "reasoning": "The production enhancer grounded the extra term.",
            },
            "query_enhancement",
        ),
        (
            "profile",
            "ProfileSelectionExampleSchema",
            {
                "query": "find transformer lectures",
                "available_profiles": "video_colpali,text_bm25",
                "selected_profile": "video_colpali",
                "reasoning": "Video retrieval matches the requested lectures.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            },
            "profile_selection",
        ),
        (
            "routing",
            "RoutingExperienceSchema",
            {
                "query": "find Marie Curie biographies",
                "entities": [{"text": "Marie Curie", "type": "PERSON"}],
                "relationships": [],
                "enhanced_query": "find Marie Curie(PERSON) biographies",
                "chosen_agent": "document_agent",
                "routing_confidence": 0.84,
                "search_quality": 0.0,
                "agent_success": False,
                "user_satisfaction": None,
                "processing_time": 0.0,
                "reward": None,
                "timestamp": "2026-08-05T00:00:00+00:00",
                "metadata": {
                    "_outcome_metadata": {
                        "observed": True,
                        "required_field_semantics": {
                            "routing_confidence": "observed_gateway_confidence",
                            "search_quality": "unobserved_zero_sentinel",
                            "agent_success": "unobserved_false_sentinel",
                            "processing_time": "unobserved_zero_sentinel",
                        },
                    }
                },
            },
            "routing",
        ),
        (
            "entity_extraction",
            "EntityExtractionExampleSchema",
            {
                "query": "Marie Curie discovered radium",
                "entities": [
                    {"text": "Marie Curie", "type": "PERSON"},
                    {"text": "radium", "type": "CONCEPT"},
                ],
                "relationships": [
                    {
                        "source": "Marie Curie",
                        "target": "radium",
                        "type": "discovered",
                    }
                ],
            },
            "entity_extraction",
        ),
    ],
)
def test_synthetic_batch_maps_optimizer_to_exact_training_consumer(
    monkeypatch,
    optimizer,
    schema_name,
    item_data,
    agent_type,
) -> None:
    from contextlib import nullcontext

    from cogniverse_agents.approval import ApprovalStatus
    from cogniverse_dashboard.tabs import optimization

    submitted = []

    class PersistedAgent:
        async def submit_for_review(self, batch):
            submitted.append(batch)
            for item in batch.items:
                item.status = ApprovalStatus.PENDING_REVIEW
            return batch

        def get_approval_stats(self, batch):
            return {
                "auto_approved": 0,
                "pending_review": len(batch.items),
                "avg_confidence": batch.items[0].confidence,
            }

    fake_st = MagicMock()
    fake_st.session_state = _SessionState(approval_agent=PersistedAgent())
    fake_st.columns.return_value = [nullcontext(), nullcontext(), nullcontext()]
    monkeypatch.setattr(optimization, "st", fake_st)

    optimization._process_approval_workflow(
        {
            "optimizer": optimizer,
            "schema_name": schema_name,
            "selected_profiles": ["video_colpali"],
            "data": [item_data],
        },
        tenant_id="acme:training",
    )

    assert len(submitted) == 1
    batch = submitted[0]
    assert batch.context == {
        "optimizer": optimizer,
        "agent_type": agent_type,
        "tenant_id": "acme:training",
        "profiles": ["video_colpali"],
    }
    assert len(batch.items) == 1
    assert batch.items[0].metadata == {
        "approval_batch_id": batch.batch_id,
        "agent_type": agent_type,
    }
    assert fake_st.session_state["pending_items"] == batch.items
    fake_st.error.assert_not_called()


def test_synthetic_batch_rejects_optimizer_without_training_consumer(
    monkeypatch,
) -> None:
    from cogniverse_dashboard.tabs import optimization

    agent = MagicMock()
    agent.submit_for_review = AsyncMock()
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(approval_agent=agent)
    monkeypatch.setattr(optimization, "st", fake_st)

    optimization._process_approval_workflow(
        {
            "optimizer": "workflow",
            "schema_name": "WorkflowExecutionSchema",
            "selected_profiles": [],
            "data": [],
        },
        tenant_id="acme:training",
    )

    agent.submit_for_review.assert_not_awaited()
    assert "last_generated_batch" not in fake_st.session_state
    fake_st.error.assert_called_once_with(
        "❌ Approval workflow failed: optimizer 'workflow' has no finetuning "
        "training-data consumer"
    )


def test_concurrent_batch_submissions_keep_exact_batch_identity() -> None:
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cogniverse_agents.approval import ApprovalBatch
    from cogniverse_dashboard.tabs import optimization

    barrier = threading.Barrier(2)

    class PersistedAgent:
        async def submit_for_review(self, batch):
            await asyncio.to_thread(barrier.wait)
            return batch

    batches = [
        ApprovalBatch(batch_id="batch-a", items=[], context={"tenant_id": "acme:a"}),
        ApprovalBatch(batch_id="batch-b", items=[], context={"tenant_id": "acme:b"}),
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        persisted = list(
            pool.map(
                lambda batch: optimization._submit_persisted_batch(
                    PersistedAgent(), batch
                ),
                batches,
            )
        )

    assert [(batch.batch_id, batch.context["tenant_id"]) for batch in persisted] == [
        ("batch-a", "acme:a"),
        ("batch-b", "acme:b"),
    ]


def test_concurrent_generated_batch_ids_are_unique_and_canonical() -> None:
    import re
    from concurrent.futures import ThreadPoolExecutor

    from cogniverse_dashboard.tabs import optimization

    with ThreadPoolExecutor(max_workers=16) as pool:
        batch_ids = list(
            pool.map(
                lambda _: optimization._new_approval_batch_id("profile"), range(64)
            )
        )

    assert len(batch_ids) == 64
    assert len(set(batch_ids)) == 64
    assert all(
        re.fullmatch(r"synthetic_profile_[0-9a-f]{32}", batch_id)
        for batch_id in batch_ids
    )
