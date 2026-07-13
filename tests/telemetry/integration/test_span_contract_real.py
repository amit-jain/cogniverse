"""One span contract, round-tripped through real Phoenix for two operations.

record_span_io writes input.value / output.value / operation on a real span; a
search span (list output) and a query_enhancement span (dict output) both go
through the SAME helper, and read_span_io reads both back off the flattened
Phoenix frame with identical code. This pins that a single reader serves every
operation type — the property the canonical contract exists to guarantee.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogniverse_foundation.telemetry.span_contract import (
    OP_QUERY_ENHANCEMENT,
    OP_SEARCH,
    read_span_io,
    record_span_io,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture
def contract_telemetry(phoenix_container):
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


SEARCH_QUERY = "cat playing fetch"
SEARCH_OUTPUT = [
    {
        "document_id": "vid_pos",
        "video_id": "vid_pos",
        "source_id": "vid_pos",
        "id": "vid_pos",
        "score": 0.91,
        "content": "a cat chasing a ball",
    },
    {
        "document_id": "vid_neg",
        "video_id": "vid_neg",
        "source_id": "vid_neg",
        "id": "vid_neg",
        "score": 0.42,
        "content": "a dog asleep",
    },
]
QE_INPUT = "ml tutorials"
QE_OUTPUT = {
    "original_query": "ml tutorials",
    "enhanced_query": "machine learning tutorials and guides",
    "expansion_terms": ["deep learning", "neural networks"],
    "confidence": 0.85,
}


@pytest.mark.asyncio
async def test_one_helper_round_trips_search_and_domain(contract_telemetry):
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    tenant = canonical_tenant_id(f"acme:contract{uuid4().hex[:8]}")

    with contract_telemetry.span("search_service.search", tenant_id=tenant) as span:
        record_span_io(
            span,
            input_value=SEARCH_QUERY,
            output=SEARCH_OUTPUT,
            operation=OP_SEARCH,
            modality="video",
        )
    with contract_telemetry.span(
        "cogniverse.query_enhancement", tenant_id=tenant
    ) as span:
        record_span_io(
            span,
            input_value=QE_INPUT,
            output=QE_OUTPUT,
            operation=OP_QUERY_ENHANCEMENT,
        )
    contract_telemetry.force_flush()

    project = contract_telemetry.config.get_project_name(tenant)
    provider = contract_telemetry.get_provider(tenant_id=tenant, project_name=project)

    rows_by_name = {}
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=1000,
        )
        if spans is not None and not spans.empty:
            for name in ("search_service.search", "cogniverse.query_enhancement"):
                hit = spans[spans["name"] == name]
                if not hit.empty:
                    rows_by_name[name] = hit.iloc[0]
        if len(rows_by_name) == 2:
            break
        await asyncio.sleep(2)

    assert set(rows_by_name) == {
        "search_service.search",
        "cogniverse.query_enhancement",
    }

    # Search: same reader yields the query, the list output, the type, modality.
    search_io = read_span_io(rows_by_name["search_service.search"])
    assert search_io["input"] == SEARCH_QUERY
    assert search_io["operation"] == OP_SEARCH
    assert search_io["modality"] == "video"
    assert isinstance(search_io["output"], list)
    assert [r["document_id"] for r in search_io["output"]] == ["vid_pos", "vid_neg"]
    assert search_io["output"][0]["content"] == "a cat chasing a ball"
    assert search_io["output"][0]["score"] == 0.91

    # query_enhancement: identical reader yields the dict output — no bespoke path.
    qe_io = read_span_io(rows_by_name["cogniverse.query_enhancement"])
    assert qe_io["input"] == QE_INPUT
    assert qe_io["operation"] == OP_QUERY_ENHANCEMENT
    assert isinstance(qe_io["output"], dict)
    assert qe_io["output"]["original_query"] == "ml tutorials"
    assert qe_io["output"]["enhanced_query"] == "machine learning tutorials and guides"
    assert qe_io["output"]["expansion_terms"] == ["deep learning", "neural networks"]
    assert qe_io["output"]["confidence"] == 0.85
