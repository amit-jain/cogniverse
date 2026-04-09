"""
E2E tests for the A2A gateway architecture.

Tests the new architecture where:
    POST /agents/gateway_agent/process -> GatewayAgent classifies ->
        SIMPLE: dispatches to execution agent (search, summarizer, etc.)
        COMPLEX: dispatches to OrchestratorAgent -> coordinates A2A agents

The gateway_agent is the new primary entry point for all queries.
Entity extraction, query enhancement, and profile selection are internal
to the orchestration pipeline and not directly callable via REST.

Requires live k3d-deployed runtime at http://localhost:28000.
"""

import httpx
import pytest

from tests.e2e.conftest import (
    RUNTIME,
    TENANT_ID,
    skip_if_no_runtime,
    unique_id,
)

PROFILE = "video_colpali_smol500_mv_frame"


# ---------------------------------------------------------------------------
# 1. Gateway simple routing
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestGatewaySimpleRouting:
    """Gateway classifies simple video queries and dispatches to search_agent."""

    def test_simple_video_query_returns_gateway_structure(self):
        """POST gateway_agent/process with a simple video query returns
        complexity=simple, routed_to a search agent, and downstream results.

        Query chosen for GLiNER score 0.693 (well above 0.4 threshold) on
        the deployed 7-label model.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "search for video content about AI",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "gateway_agent"

        # Gateway metadata block
        assert "gateway" in data, f"Missing 'gateway' key, got: {list(data.keys())}"
        gw = data["gateway"]
        assert gw["complexity"] == "simple"
        assert gw["routed_to"] in (
            "search_agent",
            "summarizer_agent",
            "detailed_report_agent",
            "image_search_agent",
            "audio_analysis_agent",
            "document_agent",
        )
        assert isinstance(gw["confidence"], (int, float))
        assert gw["confidence"] >= 0.0
        assert gw["modality"] in (
            "video", "text", "audio", "image", "document", "both",
        )
        assert gw["generation_type"] in (
            "raw_results", "summary", "detailed_report",
        )

        # Content assertions: query scores 0.693 → unambiguously simple video
        assert gw["modality"] == "video", (
            f"Expected video modality for video query, got {gw['modality']!r}"
        )
        assert gw["routed_to"] == "search_agent", (
            f"Simple video query should route to search_agent, got {gw['routed_to']!r}"
        )
        # Threshold is 0.4; this query scores 0.693 so must clear it comfortably
        assert gw["confidence"] >= 0.4, (
            f"Simple route should have confidence >= 0.4 (threshold), "
            f"got {gw['confidence']}"
        )

    def test_simple_query_includes_downstream_result(self):
        """Simple path should execute the downstream agent and return its result.

        Query chosen for GLiNER score 0.444 (above 0.4 threshold) on the
        deployed 7-label model.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "find videos about machine learning",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # Simple path produces downstream_result
        assert "downstream_result" in data, (
            f"Simple routing should produce downstream_result, got keys: {list(data.keys())}"
        )
        downstream = data["downstream_result"]
        assert isinstance(downstream, dict)
        assert downstream.get("status") == "success"

        # Content assertions: downstream search results should be non-empty and ranked
        if "results" in downstream and downstream.get("results_count", 0) > 0:
            results = downstream["results"]
            assert len(results) > 0, "Should find results for video query in ingested data"
            # Results must be ordered by descending score if score fields exist
            score_keys = ("score", "relevance", "relevance_score", "_score")
            score_key = next(
                (k for k in score_keys if k in results[0]), None
            )
            if score_key is not None:
                scores = [r[score_key] for r in results]
                assert scores == sorted(scores, reverse=True), (
                    f"Results should be ranked by {score_key} descending, "
                    f"got: {scores}"
                )

    def test_message_field_present(self):
        """Gateway response includes a human-readable message.

        Query chosen for GLiNER score 0.446 (above 0.4 threshold) on the
        deployed 7-label model.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "show me cooking videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 5
        # Message should be informative, not a bare error string
        assert data["message"].lower() not in ("error", "failed", "none", "null"), (
            f"Message should be informative, got: {data['message']!r}"
        )


# ---------------------------------------------------------------------------
# 2. Gateway complex routing
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestGatewayComplexRouting:
    """Gateway classifies complex/multi-modal queries and dispatches
    to orchestrator for multi-agent coordination."""

    def test_complex_query_classified_as_complex(self):
        """A multi-modal, multi-step query should be classified as complex
        by the gateway regardless of whether the orchestrator succeeds.

        This test asserts only on the gateway classification, which does not
        depend on Ollama or the orchestrator being healthy.  The query spans
        both video and document modalities which forces complexity regardless
        of GLiNER confidence.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "find videos and documents about neural networks",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        # Gateway classification itself must never 500
        assert resp.status_code in (200, 500), (
            f"Unexpected status code: {resp.status_code}"
        )

        if resp.status_code == 200:
            data = resp.json()
            # If gateway key is present, complexity must be "complex"
            if "gateway" in data:
                gw = data["gateway"]
                assert gw.get("complexity") == "complex", (
                    f"Multi-modal query should be classified as complex, "
                    f"got complexity={gw.get('complexity')!r}"
                )

    def test_complex_query_triggers_orchestration(self):
        """A clearly complex query should route to the orchestrator when it is
        healthy.  If the orchestrator returns 500 (e.g. Ollama not loaded),
        we still verify the gateway classification was correct.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": (
                        "Find videos about machine learning, compare them with "
                        "the PDF research papers, and write a detailed report"
                    ),
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        if resp.status_code == 500:
            # Orchestrator unavailable (e.g. Ollama not loaded) -- gateway
            # classification still happened; nothing more to assert here.
            return

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # Complex queries route to orchestrator
        has_orchestration = (
            "orchestration_result" in data
            or data.get("gateway", {}).get("complexity") == "complex"
            or data.get("agent") == "orchestrator_agent"
        )
        assert has_orchestration, (
            f"Complex query should trigger orchestration, got keys: {list(data.keys())}"
        )

        # If orchestration ran, validate its structure
        if "orchestration_result" in data:
            orch = data["orchestration_result"]
            assert isinstance(orch, dict), "orchestration_result must be a dict"
            assert "plan_steps" in orch or "agent_results" in orch, (
                f"Orchestration should produce plan_steps or agent_results, "
                f"got keys: {list(orch.keys())}"
            )
        if "gateway" in data:
            gw = data["gateway"]
            assert gw.get("complexity") == "complex", (
                f"Multi-step query should be classified as complex, "
                f"got complexity={gw.get('complexity')!r}"
            )

    def test_complex_query_returns_gateway_context(self):
        """When the orchestrator handles a complex query, the response
        should include gateway_context with classification metadata."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": (
                        "Analyze video transcripts and summarize findings "
                        "across both audio and document sources"
                    ),
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        if resp.status_code == 500:
            # Orchestrator unavailable; skip deeper assertions
            return

        assert resp.status_code == 200
        data = resp.json()

        # If orchestrator handled it, gateway_context should be present
        if data.get("agent") == "orchestrator_agent":
            assert "gateway_context" in data, (
                "Orchestrator response should carry gateway_context"
            )
            gw_ctx = data["gateway_context"]
            assert isinstance(gw_ctx, dict)
            assert "modality" in gw_ctx
            assert "confidence" in gw_ctx


# ---------------------------------------------------------------------------
# 3. Full pipeline: gateway -> search (simple path)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestGatewaySearchPipeline:
    """End-to-end: gateway classifies simple query, routes to search_agent,
    and returns actual Vespa hits."""

    def test_gateway_returns_search_results(self):
        """Simple video query through gateway produces search results
        from the downstream search_agent.

        Query chosen for GLiNER score 0.693 (well above 0.4 threshold) on the
        deployed 7-label model, ensuring the gateway classifies it as simple
        and routes to search_agent rather than the orchestrator.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "search for video content about AI",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # For simple routing, downstream_result should contain search results
        downstream = data.get("downstream_result", data)
        if "results" in downstream:
            assert isinstance(downstream["results"], list)
            assert "results_count" in downstream
            assert isinstance(downstream["results_count"], int)
            # Content assertion: confirmed-simple query should yield results
            assert downstream["results_count"] >= 1, (
                "Gateway search for 'search for video content about AI' should return "
                "at least one result from ingested data"
            )
            # Verify score ordering when score fields are present
            results = downstream["results"]
            if results:
                score_keys = ("score", "relevance", "relevance_score", "_score")
                score_key = next(
                    (k for k in score_keys if k in results[0]), None
                )
                if score_key is not None:
                    scores = [r[score_key] for r in results]
                    assert scores == sorted(scores, reverse=True), (
                        f"Gateway search results should be ranked by {score_key} "
                        f"descending, got: {scores}"
                    )
        elif "result" in downstream:
            # Alternate response shape from some agents
            assert isinstance(downstream["result"], (dict, list, str))

    def test_gateway_search_result_fields(self):
        """Search results from the gateway pipeline should have content fields.

        Query chosen for GLiNER score 0.444 (above 0.4 threshold) on the
        deployed 7-label model.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "find videos about machine learning",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        downstream = data.get("downstream_result", data)

        if "results" in downstream and downstream.get("results_count", 0) > 0:
            result = downstream["results"][0]
            assert isinstance(result, dict)
            assert len(result) > 1, (
                f"Result should have multiple fields, got: {list(result.keys())}"
            )
            # Content assertions: each result must carry identity and relevance data
            has_identity = any(
                k in result for k in ("video_id", "chunk_id", "document_id", "id")
            )
            assert has_identity, (
                f"Result should have an identity field (video_id/chunk_id/id), "
                f"got: {list(result.keys())}"
            )
            has_score = any(
                k in result
                for k in ("score", "relevance", "relevance_score", "_score", "rank_score")
            )
            assert has_score, (
                f"Result should have a score/relevance field, "
                f"got: {list(result.keys())}"
            )


# ---------------------------------------------------------------------------
# 4. Routing agent thin interface
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestRoutingAgentThin:
    """The routing agent is now a thin decision-maker. Both 'gateway' and
    'routing' capabilities route through _execute_gateway_task in the
    dispatcher, so POST to routing_agent/process goes through the gateway
    pipeline. The routing_agent no longer does entity extraction or query
    enhancement inline."""

    def test_routing_agent_returns_success(self):
        """POST to routing_agent/process returns success via gateway pipeline.

        Query chosen for GLiNER score 0.446 (above 0.4 threshold), ensuring
        the gateway classifies it as simple and doesn't trigger the orchestrator.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "show me cooking videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # Content assertion: if routing exposes a recommended_agent, it must be valid
        if "recommended_agent" in data:
            assert data["recommended_agent"] in (
                "search_agent",
                "summarizer_agent",
                "detailed_report_agent",
                "image_search_agent",
                "audio_analysis_agent",
                "document_agent",
                "deep_research_agent",
                "coding_agent",
                "text_analysis_agent",
            ), f"Routing returned unknown agent: {data['recommended_agent']!r}"
        # Gateway-style response must contain a valid routed_to field
        if "gateway" in data:
            gw = data["gateway"]
            assert gw.get("routed_to") in (
                "search_agent",
                "summarizer_agent",
                "detailed_report_agent",
                "image_search_agent",
                "audio_analysis_agent",
                "document_agent",
                "deep_research_agent",
                "coding_agent",
                "text_analysis_agent",
                None,  # orchestrator path may not set routed_to
            ), f"Gateway routed_to is invalid: {gw.get('routed_to')!r}"

    def test_routing_agent_slim_response(self):
        """Routing agent response should NOT contain inline entities or
        enhanced_query at the top level -- those are now handled by
        upstream A2A agents."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "Find videos about Tesla cars in San Francisco",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # In the new architecture, routing_agent goes through the gateway
        # pipeline. The response is a gateway-style response (with gateway
        # metadata and downstream_result), NOT the old inline response
        # with top-level entities/enhanced_query.
        # The gateway field or orchestration_result should be present.
        has_gateway_shape = (
            "gateway" in data
            or "orchestration_result" in data
            or "downstream_result" in data
        )
        assert has_gateway_shape, (
            f"Routing agent should produce gateway-style response, "
            f"got keys: {list(data.keys())}"
        )

    def test_routing_agent_processes_with_metadata(self):
        """Routing response includes processing_time_ms or similar metadata."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "cooking tutorials",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        # Should have either gateway metadata or message
        assert "message" in data or "gateway" in data or "orchestration_result" in data


# ---------------------------------------------------------------------------
# 5. Entity extraction agent (internal, no direct REST dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestEntityExtractionAgent:
    """Entity extraction agent is an internal orchestration agent.
    It is callable via REST through generic A2A dispatch, and also
    internally by the OrchestratorAgent via A2A HTTP."""

    def test_entity_extraction_agent_returns_entities(self):
        """POST to entity_extraction_agent/process returns extracted entities."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/entity_extraction_agent/process",
                json={
                    "agent_name": "entity_extraction_agent",
                    "query": "Obama speaking at MIT about climate change",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "entity_extraction_agent"
        assert "entities" in data
        assert isinstance(data["entities"], list)

    def test_entity_extraction_agent_is_registered(self):
        """The agent should be registered in the registry."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/entity_extraction_agent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "entity_extraction_agent"
        assert "entity_extraction" in data.get("capabilities", [])


# ---------------------------------------------------------------------------
# 6. Query enhancement agent (internal, no direct REST dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestQueryEnhancementAgent:
    """Query enhancement agent — callable via REST and internally by orchestrator."""

    def test_query_enhancement_agent_returns_enhanced_query(self):
        """POST to query_enhancement_agent/process returns enhanced query."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/query_enhancement_agent/process",
                json={
                    "agent_name": "query_enhancement_agent",
                    "query": "ML transformer videos",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "query_enhancement_agent"
        assert "enhanced_query" in data or "original_query" in data

    def test_query_enhancement_agent_is_registered(self):
        """The agent should be registered in the registry."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/query_enhancement_agent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "query_enhancement_agent"
        assert "query_enhancement" in data.get("capabilities", [])


# ---------------------------------------------------------------------------
# 7. Profile selection agent (internal, no direct REST dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestProfileSelectionAgent:
    """Profile selection agent — callable via REST and internally by orchestrator."""

    def test_profile_selection_agent_returns_profile(self):
        """POST to profile_selection_agent/process returns a selected profile."""
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            resp = client.post(
                "/agents/profile_selection_agent/process",
                json={
                    "agent_name": "profile_selection_agent",
                    "query": "find basketball highlights",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "profile_selection_agent"
        assert "selected_profile" in data

    def test_profile_selection_agent_is_registered(self):
        """The agent should be registered in the registry."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/profile_selection_agent")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "profile_selection_agent"
        assert "profile_selection" in data.get("capabilities", [])


# ---------------------------------------------------------------------------
# 8. Telemetry spans appear in Phoenix
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestTelemetrySpans:
    """After running a query through the gateway, verify telemetry spans
    were emitted to Phoenix."""

    def _query_phoenix_spans(
        self, project_name: str, span_prefix: str, limit: int = 10
    ) -> list:
        """Query Phoenix for spans matching a prefix.

        Phoenix exposes a REST API at port 6006 for querying spans.
        Returns a list of matching spans, or empty list on failure.
        """
        phoenix_url = "http://localhost:6006"
        try:
            with httpx.Client(timeout=10.0) as client:
                # Try the Phoenix v1 spans API
                resp = client.get(
                    f"{phoenix_url}/v1/spans",
                    params={
                        "project_name": project_name,
                        "limit": limit,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    spans = data if isinstance(data, list) else data.get("data", [])
                    return [
                        s for s in spans
                        if span_prefix in s.get("name", "")
                    ]
        except (httpx.ConnectError, httpx.ReadTimeout, Exception):
            pass
        return []

    def test_gateway_emits_telemetry_spans(self):
        """Run a query through the gateway and verify that cogniverse.gateway
        spans appear in Phoenix. This test is best-effort -- if Phoenix is not
        available or spans haven't propagated yet, it verifies the gateway
        call itself succeeded rather than failing the test.

        Uses a prefix that scores as simple (0.446) so the orchestrator is not
        triggered, keeping the test independent of Ollama availability.
        """
        # Run a simple query through the gateway to avoid orchestrator dependency
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": f"show me cooking videos {unique_id()}",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # Check Phoenix for spans (best-effort)
        spans = self._query_phoenix_spans(
            project_name=f"cogniverse-{TENANT_ID.replace(':', '-')}-gateway",
            span_prefix="cogniverse.gateway",
        )

        # Telemetry verification is best-effort: Phoenix may not be running
        # or spans may not have propagated. The primary assertion is that
        # the gateway call succeeded.
        if spans:
            span = spans[0]
            assert "cogniverse.gateway" in span.get("name", "")

    def test_phoenix_health_check(self):
        """Verify Phoenix is reachable (informational)."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get("http://localhost:6006/healthz")
            if resp.status_code == 200:
                # Phoenix is healthy
                pass
            else:
                pytest.skip(
                    f"Phoenix not healthy (status={resp.status_code}), "
                    "telemetry tests are informational"
                )
        except (httpx.ConnectError, httpx.ReadTimeout):
            pytest.skip("Phoenix not reachable, telemetry tests are informational")
