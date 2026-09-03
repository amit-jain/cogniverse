"""Dashboard rejection regeneration against the configured production LM path."""

import asyncio
import os
import re
import socket
import subprocess
import time
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.approval import ApprovalStorageImpl, HumanApprovalAgent
from cogniverse_core.approval.interfaces import (
    ApprovalStatus,
    ReviewDecision,
    ReviewItem,
)
from cogniverse_dashboard.tabs import approval_queue, optimization
from cogniverse_foundation.config.utils import (
    create_default_config_manager,
    get_config,
)
from cogniverse_synthetic.approval import SyntheticDataConfidenceExtractor
from cogniverse_synthetic.schemas import RoutingExperienceSchema

pytestmark = [pytest.mark.integration, pytest.mark.local_only]


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def dashboard_approval_redis_url():
    port = _free_port()
    container_name = f"cogniverse-dashboard-approval-{os.getpid()}-{time.time_ns()}"
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start dashboard approval Redis: {result.stderr}")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", container_name, "redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
            check=False,
        )
        pytest.fail("Dashboard approval Redis did not become ready within 30 seconds")

    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
            check=False,
        )


@pytest.mark.asyncio
@pytest.mark.requires_lm
async def test_dashboard_handler_regenerates_with_tenant_primary_lm(
    ensure_host_ollama,
) -> None:
    _ = ensure_host_ollama
    tenant_id = "acme:dashboard-reviewer"
    config_manager = create_default_config_manager()
    primary = (
        get_config(
            tenant_id=tenant_id,
            config_manager=config_manager,
        )
        .get_llm_config()
        .primary
    )

    handler = approval_queue._build_feedback_handler(config_manager, tenant_id)

    assert handler.generator.lm.model == primary.model
    assert handler.generator.lm.kwargs["api_base"] == primary.api_base

    outcome_metadata = {
        "observed": True,
        "required_field_semantics": {
            "routing_confidence": "observed_gateway_confidence",
            "search_quality": "unobserved_zero_sentinel",
            "agent_success": "unobserved_false_sentinel",
            "processing_time": "unobserved_zero_sentinel",
        },
    }
    item = ReviewItem(
        item_id="dashboard-routing-1",
        data={
            "query": "find a physicist biography",
            "entities": [{"text": "Curie", "type": "PERSON"}],
            "relationships": [],
            "enhanced_query": "find a physicist biography",
            "chosen_agent": "document_agent",
            "routing_confidence": 0.84,
            "search_quality": 0.0,
            "agent_success": False,
            "user_satisfaction": None,
            "processing_time": 0.0,
            "reward": None,
            "metadata": {"_outcome_metadata": outcome_metadata},
        },
        confidence=0.84,
    )
    decision = ReviewDecision(
        item_id=item.item_id,
        approved=False,
        feedback="Use the scientist's complete name in the query.",
        corrections={
            "entities": [{"text": "Marie Curie", "type": "PERSON"}],
            "relationships": [],
            "topics": ["radioactivity research"],
        },
        reviewer="dashboard-reviewer@example.test",
    )

    regenerated = await handler.process_rejection(item, decision)

    assert regenerated.item_id == "dashboard-routing-1_regen_0"
    assert regenerated.status is ApprovalStatus.REGENERATED
    assert regenerated.confidence == 0.0
    assert regenerated.data["entities"] == [{"text": "Marie Curie", "type": "PERSON"}]
    assert regenerated.data["relationships"] == []
    assert re.search(
        r"(?<!\w)Marie Curie(?!\w)",
        regenerated.data["query"],
        flags=re.IGNORECASE,
    )
    assert re.search(
        r"(?<!\w)Marie Curie\(PERSON\)(?!\w)",
        regenerated.data["enhanced_query"],
        flags=re.IGNORECASE,
    )
    assert RoutingExperienceSchema.model_validate(regenerated.data)
    assert set(regenerated.data["metadata"]) == {
        "_outcome_metadata",
        "_generation_metadata",
    }
    assert regenerated.data["metadata"]["_outcome_metadata"] == {
        "observed": False,
        "required_field_semantics": {
            "routing_confidence": "unobserved_zero_sentinel",
            "search_quality": "unobserved_zero_sentinel",
            "agent_success": "unobserved_false_sentinel",
            "processing_time": "unobserved_zero_sentinel",
        },
    }
    assert outcome_metadata == {
        "observed": True,
        "required_field_semantics": {
            "routing_confidence": "observed_gateway_confidence",
            "search_quality": "unobserved_zero_sentinel",
            "agent_success": "unobserved_false_sentinel",
            "processing_time": "unobserved_zero_sentinel",
        },
    }
    generation = regenerated.data["metadata"]["_generation_metadata"]
    assert set(generation) == {
        "retry_count",
        "max_retries",
        "regeneration_attempt",
        "max_regeneration_attempts",
        "regeneration",
        "original_query",
        "human_feedback",
        "corrections_applied",
        "reasoning",
    }
    assert generation["retry_count"] in {0, 1, 2}
    assert generation["max_retries"] == 3
    assert generation["regeneration_attempt"] == 1
    assert generation["max_regeneration_attempts"] == 2
    assert generation["regeneration"] is True
    assert generation["original_query"] == "find a physicist biography"
    assert generation["human_feedback"] == decision.feedback
    assert generation["corrections_applied"] == decision.corrections
    assert isinstance(generation["reasoning"], str)
    assert len(generation["reasoning"]) >= 10
    assert generation["reasoning"] == generation["reasoning"].strip()


def test_dashboard_submit_approve_reloads_exact_item_from_real_phoenix(
    monkeypatch,
    phoenix_container,
    telemetry_manager_with_phoenix,
    dashboard_approval_redis_url,
) -> None:
    tenant_id = f"acme:dashboard-{time.time_ns()}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=dashboard_approval_redis_url,
    )
    agent = HumanApprovalAgent(
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        confidence_threshold=0.85,
        storage=storage,
    )
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        approved_items=[],
        rejected_items=[],
        user_email="dashboard-reviewer@example.test",
    )
    fake_st.columns.return_value = [nullcontext(), nullcontext(), nullcontext()]
    monkeypatch.setattr(optimization, "st", fake_st)
    monkeypatch.setattr(approval_queue, "st", fake_st)

    record = {
        "query": "find transformer lectures",
        "available_profiles": "video_colpali,text_bm25",
        "selected_profile": "video_colpali",
        "reasoning": "Video retrieval matches the requested lectures.",
        "query_intent": "video_search",
        "modality": "video",
        "complexity": "medium",
    }
    optimization._process_approval_workflow(
        {
            "optimizer": "profile",
            "schema_name": "ProfileSelectionExampleSchema",
            "selected_profiles": ["video_colpali"],
            "data": [record],
        },
        tenant_id=tenant_id,
    )

    submitted = fake_st.session_state["last_generated_batch"]
    assert submitted.context == {
        "optimizer": "profile",
        "agent_type": "profile_selection",
        "tenant_id": tenant_id,
        "profiles": ["video_colpali"],
    }
    assert len(submitted.pending_review) == 1
    pending = submitted.pending_review[0]
    assert pending.data == record
    assert pending.confidence == 0.0
    assert pending.metadata == {
        "approval_batch_id": submitted.batch_id,
        "agent_type": "profile_selection",
    }

    approval_queue._handle_approval(pending, 0)
    fake_st.error.assert_not_called()

    reloaded_storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=dashboard_approval_redis_url,
    )
    deadline = time.monotonic() + 15
    while True:
        reloaded = asyncio.run(reloaded_storage.get_batch(submitted.batch_id))
        if reloaded.items[0].status is ApprovalStatus.APPROVED:
            break
        if time.monotonic() >= deadline:
            pytest.fail(
                "Phoenix did not expose the approved dashboard item within 15 seconds"
            )
        time.sleep(0.25)

    assert reloaded.context == submitted.context
    assert len(reloaded.items) == 1
    approved = reloaded.items[0]
    assert approved.item_id == pending.item_id
    assert approved.data == record
    assert approved.confidence == 0.0
    assert approved.status is ApprovalStatus.APPROVED
    assert approved.reviewed_at is not None
    assert approved.metadata == {
        "approval_batch_id": submitted.batch_id,
        "agent_type": "profile_selection",
        "decision": {
            "reviewer": "dashboard-reviewer@example.test",
            "feedback": None,
            "corrections": {},
            "timestamp": approved.reviewed_at.isoformat(),
        },
    }
    assert fake_st.session_state["pending_items"] == []
    assert fake_st.session_state["approved_items"] == [approved]
    fake_st.error.assert_not_called()


def test_rejected_tab_regenerate_uses_real_phoenix_and_redis(
    monkeypatch,
    phoenix_container,
    telemetry_manager_with_phoenix,
    dashboard_approval_redis_url,
) -> None:
    tenant_id = f"acme:dashboard-regenerate-{time.time_ns()}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=telemetry_manager_with_phoenix,
        redis_url=dashboard_approval_redis_url,
    )
    agent = HumanApprovalAgent(
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        feedback_handler=approval_queue._build_feedback_handler(
            create_default_config_manager(), tenant_id
        ),
        confidence_threshold=0.85,
        storage=storage,
    )
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        approved_items=[],
        rejected_items=[],
        user_email="dashboard-reviewer@example.test",
    )
    fake_st.columns.return_value = [nullcontext(), nullcontext(), nullcontext()]
    fake_st.expander.return_value = nullcontext()
    monkeypatch.setattr(optimization, "st", fake_st)
    monkeypatch.setattr(approval_queue, "st", fake_st)

    record = {
        "query": "find transformer lectures",
        "available_profiles": "video_colpali,text_bm25",
        "selected_profile": "video_colpali",
        "reasoning": "Video retrieval matches the requested lectures.",
        "query_intent": "video_search",
        "modality": "video",
        "complexity": "medium",
    }
    optimization._process_approval_workflow(
        {
            "optimizer": "profile",
            "schema_name": "ProfileSelectionExampleSchema",
            "selected_profiles": ["video_colpali", "text_bm25"],
            "data": [record],
        },
        tenant_id=tenant_id,
    )
    pending = fake_st.session_state["pending_items"][0]
    approval_queue._handle_rejection(
        pending,
        0,
        "Use the lexical profile for the title terms.",
        {
            "selected_profile": "text_bm25",
            "reasoning": "Exact title terms make lexical retrieval the target.",
        },
    )
    first_replacement = fake_st.session_state["pending_items"][0]
    fake_st.button.side_effect = lambda label, **kwargs: label == "🔄 Regenerate"

    approval_queue._render_rejected_items_tab()

    assert len(fake_st.session_state["pending_items"]) == 1
    replacement = fake_st.session_state["pending_items"][0]
    assert replacement.item_id == first_replacement.item_id
    assert replacement.data == {
        **record,
        "selected_profile": "text_bm25",
        "reasoning": "Exact title terms make lexical retrieval the target.",
    }
    persisted = asyncio.run(
        storage.get_batch(
            fake_st.session_state.rejected_items[0][0].metadata["approval_batch_id"]
        )
    )
    regenerated = [
        item for item in persisted.items if item.status is ApprovalStatus.REGENERATED
    ]
    assert len(regenerated) == 1
    assert regenerated[0].item_id == replacement.item_id
    assert regenerated[0].data == replacement.data
    fake_st.success.assert_called_with(
        f"Regenerated {pending.item_id} as {replacement.item_id}"
    )
