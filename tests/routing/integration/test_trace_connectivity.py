"""
Test trace connectivity across unified tenant project.

Verifies that parent-child span relationships are maintained when
all user operations use the same project (cogniverse-{tenant}).

These are integration tests that require:
1. cogniverse-telemetry-phoenix package installed
2. Phoenix server running on localhost:4317 (gRPC) and localhost:6006 (HTTP)

Run with: pytest tests/routing/integration/test_trace_connectivity.py -v -m integration
"""

import asyncio
import logging

import pytest
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager, NoOpSpan

logger = logging.getLogger(__name__)


def skip_if_no_provider(span):
    """Skip test if telemetry provider is not available."""
    if span is None or isinstance(span, NoOpSpan):
        pytest.skip("Telemetry provider not available (cogniverse-telemetry-phoenix not installed)")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.telemetry
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
        # Skip if provider not available
        skip_if_no_provider(parent_span)

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
@pytest.mark.integration
@pytest.mark.telemetry
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
        skip_if_no_provider(req_span)
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


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.telemetry
async def test_session_based_trace_linking():
    """
    Test that multiple requests with same session_id share the session context.

    This verifies that:
    1. Multiple separate requests (different trace_ids) can share a session_id
    2. Each request creates its own trace with proper parent-child spans
    3. session_span() properly wraps requests in session context

    Note: The actual session.id attribute propagation is handled by OpenInference
    (using_session context manager). This test verifies the TelemetryManager
    API works correctly - Phoenix integration tests verify the attribute appears.
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
    tenant_id = "test-session-linking"
    session_id = "user-session-abc123"  # Same session across requests

    request_traces = []

    # Simulate 3 separate API requests from same user session
    for request_num in range(3):
        # Each request gets its own trace, but shares session_id
        with manager.session_span(
            f"api.search.request_{request_num}",
            tenant_id=tenant_id,
            session_id=session_id,
            attributes={"request_number": request_num}
        ) as request_span:
            skip_if_no_provider(request_span)
            if request_span and hasattr(request_span, 'context'):
                trace_id = request_span.context.trace_id
                span_id = request_span.context.span_id

                request_traces.append({
                    "request_num": request_num,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "session_id": session_id,
                })

                logger.info(
                    f"Request {request_num}: trace_id={trace_id}, "
                    f"span_id={span_id}, session_id={session_id}"
                )

                # Create child span within request
                with manager.span(
                    "cogniverse.routing",
                    tenant_id=tenant_id,
                    attributes={"routing.agent": "video_search"}
                ) as child_span:
                    if child_span and hasattr(child_span, 'context'):
                        # Child should have same trace_id as parent request
                        assert child_span.context.trace_id == trace_id, (
                            "Child span should share trace_id with parent request span"
                        )

                await asyncio.sleep(0.01)

    manager.force_flush(timeout_millis=5000)

    # ASSERTIONS
    assert len(request_traces) == 3, "Should have 3 request traces"

    # Each request should have a DIFFERENT trace_id (separate requests)
    trace_ids = [t["trace_id"] for t in request_traces]
    assert len(set(trace_ids)) == 3, (
        f"Each request should have unique trace_id. Got: {trace_ids}"
    )

    # All requests should share the SAME session_id
    session_ids = [t["session_id"] for t in request_traces]
    assert len(set(session_ids)) == 1, (
        f"All requests should share same session_id. Got: {session_ids}"
    )
    assert session_ids[0] == session_id, "Session ID should match"

    logger.info("✅ Session-based trace linking verified!")
    logger.info(f"   Session ID: {session_id}")
    logger.info(f"   Trace IDs (should be different): {trace_ids}")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.telemetry
async def test_session_span_with_nested_hierarchy():
    """
    Test that session_span properly maintains parent-child hierarchy within a request.

    Verifies:
    1. session_span creates root span for the request
    2. Nested spans are children of the session_span
    3. All spans in hierarchy share same trace_id
    4. Parent-child relationships are correct
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
    tenant_id = "test-session-hierarchy"
    session_id = "hierarchy-test-session"

    span_infos = []

    # Request with session tracking
    with manager.session_span(
        "api.search.request",
        tenant_id=tenant_id,
        session_id=session_id,
        attributes={"query": "test query"}
    ) as root_span:
        skip_if_no_provider(root_span)
        if root_span and hasattr(root_span, 'context'):
            span_infos.append({
                "name": "root",
                "span_id": root_span.context.span_id,
                "trace_id": root_span.context.trace_id,
                "parent_id": root_span.parent.span_id if hasattr(root_span, 'parent') and root_span.parent else None
            })

            # Child: Routing
            with manager.span(
                "cogniverse.routing",
                tenant_id=tenant_id
            ) as routing_span:
                if routing_span and hasattr(routing_span, 'context'):
                    span_infos.append({
                        "name": "routing",
                        "span_id": routing_span.context.span_id,
                        "trace_id": routing_span.context.trace_id,
                        "parent_id": routing_span.parent.span_id if hasattr(routing_span, 'parent') and routing_span.parent else None
                    })

                    # Grandchild: Search
                    with manager.span(
                        "cogniverse.search",
                        tenant_id=tenant_id
                    ) as search_span:
                        if search_span and hasattr(search_span, 'context'):
                            span_infos.append({
                                "name": "search",
                                "span_id": search_span.context.span_id,
                                "trace_id": search_span.context.trace_id,
                                "parent_id": search_span.parent.span_id if hasattr(search_span, 'parent') and search_span.parent else None
                            })

                        await asyncio.sleep(0.01)

    manager.force_flush(timeout_millis=5000)

    # ASSERTIONS
    assert len(span_infos) == 3, "Should have 3 spans (root, routing, search)"

    # All spans should share same trace_id
    trace_ids = [s["trace_id"] for s in span_infos]
    assert len(set(trace_ids)) == 1, f"All spans should share trace_id. Got: {trace_ids}"

    # Verify parent-child chain
    root = span_infos[0]
    routing = span_infos[1]
    search = span_infos[2]

    assert root["parent_id"] is None, "Root span (session_span) should have no parent"
    assert routing["parent_id"] == root["span_id"], "Routing should be child of root"
    assert search["parent_id"] == routing["span_id"], "Search should be child of routing"

    logger.info("✅ Session span hierarchy verified!")
    logger.info(f"   Trace ID: {trace_ids[0]}")
    for info in span_infos:
        logger.info(f"   {info['name']}: span_id={info['span_id']}, parent_id={info['parent_id']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
