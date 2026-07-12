"""Real-Phoenix round-trip for TrainingMethodSelector.analyze_data().

Emits one routing span and attaches one approved + one rejected annotation on
the SAME span. On the real Phoenix annotations frame span_id is the INDEX (no
column), so the old ``drop_duplicates(subset=['span_id'])`` raised
KeyError('span_id'). This pins the reset_index normalization.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_finetuning.dataset.method_selector import TrainingMethodSelector

pytestmark = pytest.mark.integration


@pytest.fixture
def telemetry_manager(phoenix_container):
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    yield manager
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_analyze_data_counts_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = "ms_rt"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )

    with telemetry_manager.span(
        name="routing_agent",
        tenant_id=tenant_id,
        project_name=project_name,
        attributes={
            "input.query": "find sunset videos",
            "output.response": "default route",
        },
    ):
        pass
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )

    span_id = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=full_project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=1000,
        )
        if spans is not None and not spans.empty and "name" in spans.columns:
            match = spans[spans["name"] == "routing_agent"]
            if not match.empty:
                span_id = match.iloc[0]["context.span_id"]
                break
        await asyncio.sleep(2)
    assert span_id is not None, f"routing_agent span not found in {full_project}"

    # Approved + rejected on the SAME span (distinct annotation names so Phoenix
    # keeps both) => exactly one preference pair.
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_review_approved",
        label="approved",
        score=1.0,
        metadata={},
        project=full_project,
    )
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="human_review_rejected",
        label="rejected",
        score=0.0,
        metadata={},
        project=full_project,
    )

    selector = TrainingMethodSelector()

    # analyze_data does not raise on empty annotations (returns zero counts), so
    # poll on preference_pairs. On the OLD code, once annotations land it raises
    # KeyError('span_id') here, which propagates and fails the test.
    analysis = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        analysis = await selector.analyze_data(
            provider=provider,
            project=full_project,
            agent_type="routing",
            min_sft_examples=50,
            min_dpo_pairs=1,
        )
        if analysis.preference_pairs >= 1:
            break
        await asyncio.sleep(2)

    assert analysis is not None
    assert analysis.total_spans == 1
    assert analysis.approved_count == 1
    assert analysis.rejected_count == 1
    assert analysis.preference_pairs == 1
    assert analysis.recommended_method == "dpo"
    assert analysis.needs_synthetic is False
