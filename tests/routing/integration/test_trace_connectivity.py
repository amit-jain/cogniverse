"""
Test trace connectivity across unified tenant project.

Verifies that parent-child span relationships are maintained when
all user operations use the same project (cogniverse-{tenant}).
"""

import asyncio
import logging

import pytest
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_trace_connectivity_unified_project():
    """
    Verify trace connectivity when multiple operations use unified tenant project.

    This test ensures that:
    1. Parent request span and child routing span share same trace_id
    2. Parent span_id is properly set on child span
    3. All spans appear in the same Phoenix project
    """
    # Setup telemetry manager
    TelemetryManager.reset()

    config = TelemetryConfig(
        enabled=True,
        otlp_endpoint="http://localhost:4317",
        provider_config={
            "http_endpoint": "http://localhost:6006",
            "grpc_endpoint": "http://localhost:4317"
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )

    manager = TelemetryManager(config=config)
    tenant_id = "test-trace-connectivity"

    # Simulate a request flow: request -> routing -> agent
    parent_span_id = None
    parent_trace_id = None
    child_span_id = None
    child_trace_id = None
    child_parent_id = None

    # STEP 1: Create parent request span (unified tenant project)
    with manager.span(
        "cogniverse.request",
        tenant_id=tenant_id,
        attributes={"request.query": "test query"}
    ) as parent_span:
        # Capture parent span info
        if parent_span and hasattr(parent_span, 'context'):
            parent_span_id = parent_span.context.span_id
            parent_trace_id = parent_span.context.trace_id
            logger.info(f"Parent span created: trace_id={parent_trace_id}, span_id={parent_span_id}")

        # STEP 2: Create child routing span (same unified tenant project)
        with manager.span(
            "cogniverse.routing",
            tenant_id=tenant_id,
            attributes={"routing.agent": "video_search"}
        ) as child_span:
            # Capture child span info
            if child_span and hasattr(child_span, 'context'):
                child_span_id = child_span.context.span_id
                child_trace_id = child_span.context.trace_id
                logger.info(f"Child span created: trace_id={child_trace_id}, span_id={child_span_id}")

                # Get parent span ID from child
                if hasattr(child_span, 'parent') and child_span.parent:
                    child_parent_id = child_span.parent.span_id
                    logger.info(f"Child parent_id: {child_parent_id}")

                # Simulate work
                await asyncio.sleep(0.01)

    # Force flush
    manager.force_flush(timeout_millis=5000)

    # ASSERTIONS
    assert parent_span_id is not None, "Parent span should be created"
    assert child_span_id is not None, "Child span should be created"
    assert parent_trace_id is not None, "Parent should have trace_id"
    assert child_trace_id is not None, "Child should have trace_id"

    # KEY ASSERTION: Same trace ID
    assert parent_trace_id == child_trace_id, (
        f"Parent and child should share same trace_id. "
        f"Parent: {parent_trace_id}, Child: {child_trace_id}"
    )

    # KEY ASSERTION: Parent-child relationship
    assert child_parent_id == parent_span_id, (
        f"Child's parent_id should match parent's span_id. "
        f"Expected: {parent_span_id}, Got: {child_parent_id}"
    )

    logger.info("✅ Trace connectivity verified!")
    logger.info(f"   Trace ID: {parent_trace_id}")
    logger.info(f"   Parent span ID: {parent_span_id}")
    logger.info(f"   Child span ID: {child_span_id}")
    logger.info(f"   Child parent ID: {child_parent_id}")


@pytest.mark.asyncio
async def test_cross_operation_trace_connectivity():
    """
    Test trace connectivity across different operation types in same tenant.

    Simulates: Request -> Routing -> Search -> Agent
    All should share same trace_id and proper parent-child relationships.
    """
    TelemetryManager.reset()

    config = TelemetryConfig(
        enabled=True,
        otlp_endpoint="http://localhost:4317",
        provider_config={
            "http_endpoint": "http://localhost:6006",
            "grpc_endpoint": "http://localhost:4317"
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )

    manager = TelemetryManager(config=config)
    tenant_id = "test-cross-op-trace"

    trace_ids = []
    span_infos = []

    # STEP 1: Request span
    with manager.span(
        "cogniverse.request",
        tenant_id=tenant_id
    ) as req_span:
        if req_span and hasattr(req_span, 'context'):
            trace_ids.append(req_span.context.trace_id)
            span_infos.append({
                "name": "request",
                "span_id": req_span.context.span_id,
                "trace_id": req_span.context.trace_id,
                "parent_id": req_span.parent.span_id if hasattr(req_span, 'parent') and req_span.parent else None
            })

        # STEP 2: Routing span (child of request)
        with manager.span(
            "cogniverse.routing",
            tenant_id=tenant_id
        ) as routing_span:
            if routing_span and hasattr(routing_span, 'context'):
                trace_ids.append(routing_span.context.trace_id)
                span_infos.append({
                    "name": "routing",
                    "span_id": routing_span.context.span_id,
                    "trace_id": routing_span.context.trace_id,
                    "parent_id": routing_span.parent.span_id if hasattr(routing_span, 'parent') and routing_span.parent else None
                })

            # STEP 3: Search span (child of routing)
            with manager.span(
                "cogniverse.search",
                tenant_id=tenant_id
            ) as search_span:
                if search_span and hasattr(search_span, 'context'):
                    trace_ids.append(search_span.context.trace_id)
                    span_infos.append({
                        "name": "search",
                        "span_id": search_span.context.span_id,
                        "trace_id": search_span.context.trace_id,
                        "parent_id": search_span.parent.span_id if hasattr(search_span, 'parent') and search_span.parent else None
                    })

                await asyncio.sleep(0.01)

    manager.force_flush(timeout_millis=5000)

    # ASSERTIONS
    assert len(trace_ids) == 3, "Should have 3 spans"
    assert len(set(trace_ids)) == 1, f"All spans should share same trace_id, got: {trace_ids}"

    # Verify parent-child chain
    assert span_infos[0]["parent_id"] is None, "Request span should be root (no parent)"
    assert span_infos[1]["parent_id"] == span_infos[0]["span_id"], "Routing should be child of Request"
    assert span_infos[2]["parent_id"] == span_infos[1]["span_id"], "Search should be child of Routing"

    logger.info("✅ Cross-operation trace connectivity verified!")
    for info in span_infos:
        logger.info(f"   {info['name']}: span_id={info['span_id']}, parent_id={info['parent_id']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
