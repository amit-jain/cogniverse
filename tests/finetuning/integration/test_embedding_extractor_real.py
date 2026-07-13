"""Real-Phoenix round-trip for TripletExtractor.extract().

Emits a search span with two results, attaches a result_click annotation whose
metadata carries the clicked result_id, then extracts a triplet. On the real
Phoenix annotations frame span_id is the INDEX, the name is in an
``annotation_name`` column, and the clicked id lives in ``metadata.result_id``
(never ``result.result_id``) — the old reads produced zero triplets.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogniverse_finetuning.dataset.embedding_extractor import TripletExtractor

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
async def test_extract_builds_triplet_from_real_phoenix(
    phoenix_container, telemetry_manager
):
    tenant_id = f"trip{uuid4().hex[:8]}"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"
    query = "cat playing fetch"

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )

    results = [
        {"video_id": "vA", "score": 0.9, "content": "a cat chasing a ball"},
        {"video_id": "vB", "score": 0.8, "content": "a dog running in a park"},
    ]
    with telemetry_manager.span(
        name="video_search",
        tenant_id=tenant_id,
        project_name=project_name,
        attributes={
            "input.query": query,
            "input.modality": "video",
            "output.value": json.dumps(results),
        },
    ) as span:
        span_id = format(span.get_span_context().span_id, "016x")
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )

    # Wait for the span to index.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=full_project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=1000,
        )
        if (
            spans is not None
            and not spans.empty
            and "context.span_id" in spans.columns
            and (spans["context.span_id"] == span_id).any()
        ):
            break
        await asyncio.sleep(2)

    # The click annotation carries the clicked result_id in its metadata.
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="result_click",
        label="clicked",
        score=1.0,
        metadata={"result_id": "vA"},
        project=full_project,
    )

    extractor = TripletExtractor(provider=provider)

    triplets = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        triplets = await extractor.extract(
            project=full_project,
            modality="video",
            strategy="top_k",
            min_triplets=1,
        )
        if triplets:
            break
        await asyncio.sleep(2)

    assert len(triplets) == 1
    t = triplets[0]
    assert t.anchor == query
    assert t.positive == "a cat chasing a ball"
    assert t.negative == "a dog running in a park"
    assert t.modality == "video"
    assert t.metadata["span_id"] == span_id
