"""Real-Phoenix round-trip for PhoenixAnnotationStore.log_evaluations.

The quality-monitor cycle uploads bulk evaluations via
``provider.annotations.log_evaluations``, but tests/evaluation/unit/
test_span_evaluator.py mocks the whole provider, so the actual Phoenix
upload-and-read was never exercised. Here a real span is emitted, log_evaluations
annotates it against a real Phoenix, and get_annotations reads the score and
label back on that span.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture(scope="module")
def eval_telemetry(phoenix_container):
    """Real TelemetryManager (sync export) backed by the Phoenix container."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv(
            "TELEMETRY_OTLP_ENDPOINT", phoenix_container["otlp_endpoint"]
        ),
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.mark.asyncio
async def test_log_evaluations_round_trips_through_phoenix(eval_telemetry):
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    tenant = canonical_tenant_id(f"acme:evalrt{uuid4().hex[:6]}")
    project = eval_telemetry.config.get_project_name(tenant)

    with eval_telemetry.span("evaluated_op", tenant_id=tenant) as span:
        span_id = format(span.get_span_context().span_id, "016x")
    eval_telemetry.force_flush(timeout_millis=10000)

    provider = eval_telemetry.get_provider(tenant_id=tenant, project_name=project)

    # Wait until the span is queryable; keep its row for the annotation read.
    spans_df = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        df = await provider.traces.get_spans(
            project=project,
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=1000,
        )
        if df is not None and not df.empty and "context.span_id" in df.columns:
            hit = df[df["context.span_id"] == span_id]
            if not hit.empty:
                spans_df = hit
                break
        await asyncio.sleep(2)
    assert spans_df is not None, f"span {span_id} not indexed in {project}"

    eval_df = pd.DataFrame(
        [
            {
                "span_id": span_id,
                "score": 0.75,
                "label": "relevant",
                "explanation": "top hit matched the query",
            }
        ]
    )
    await provider.annotations.log_evaluations("search_relevance", eval_df, project)

    ann = pd.DataFrame()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        ann = await provider.annotations.get_annotations(
            spans_df, project, annotation_names=["search_relevance"]
        )
        if ann is not None and not ann.empty:
            break
        await asyncio.sleep(2)

    assert ann is not None and not ann.empty, (
        "log_evaluations annotation was never read back from Phoenix"
    )
    assert len(ann) == 1, ann.to_dict("records")
    row = ann.iloc[0]
    # Column names surfaced to check the real Phoenix annotation shape.
    assert row["result.score"] == 0.75, ann.columns.tolist()
    assert row["result.label"] == "relevant", ann.columns.tolist()
