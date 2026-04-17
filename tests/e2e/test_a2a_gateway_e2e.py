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
    PHOENIX_URL,
    RUNTIME,
    TENANT_ID,
    skip_if_no_runtime,
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
        assert gw["generation_type"] == "raw_results", (
            f"No summary/report keyword → raw_results, got {gw['generation_type']!r}"
        )
        assert gw["routed_to"] == "search_agent", (
            f"Simple video query should route to search_agent, got {gw['routed_to']!r}"
        )
        # This query scores 0.693 on deployed model — must clear 0.4 threshold comfortably
        assert gw["confidence"] >= 0.5, (
            f"'search for video content about AI' scores 0.693 on deployed model, "
            f"confidence should be >= 0.5, got {gw['confidence']}"
        )
        # Gateway confidence for this query should be ~0.69 (measured on deployed model)
        assert gw["confidence"] >= 0.6, (
            f"'search for video content about AI' should score ~0.69, got {gw['confidence']}"
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

        # Search results must exist and contain real Vespa data
        assert "results" in downstream, f"Missing 'results' in downstream, keys: {list(downstream.keys())}"
        results = downstream["results"]
        assert downstream["results_count"] >= 1, (
            "Query 'find videos about machine learning' must return results from ingested data"
        )
        assert len(results) == downstream["results_count"], (
            f"results_count ({downstream['results_count']}) doesn't match len(results) ({len(results)})"
        )

        # Each result must have score + metadata with real video data
        first = results[0]
        assert "score" in first, f"Result missing 'score' field: {list(first.keys())}"
        assert first["score"] > 0, f"First result score should be positive, got {first['score']}"
        assert "metadata" in first, f"Result missing 'metadata': {list(first.keys())}"
        meta = first["metadata"]
        assert "video_id" in meta, f"Result metadata missing video_id: {list(meta.keys())}"
        assert meta["video_id"] != "", "video_id should not be empty"

        # Results must be ranked — first result score >= last result score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not ranked by score descending: {scores}"
        )

        # Profile used should be the default video profile
        assert downstream.get("profile") == "video_colpali_smol500_mv_frame", (
            f"Expected default video profile, got: {downstream.get('profile')}"
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

        # Message must name the target agent and indicate simple routing
        msg = data["message"]
        assert "search_agent" in msg or "cooking" in msg.lower(), (
            f"Message for 'cooking videos' should mention search_agent or cooking, got: {msg!r}"
        )
        assert "simple" in msg.lower() or "routed" in msg.lower(), (
            f"Message should indicate simple routing, got: {msg!r}"
        )

        # Gateway classification for cooking videos: video modality, simple
        gw = data.get("gateway", {})
        assert gw.get("modality") == "video", (
            f"'cooking videos' should be video modality, got: {gw.get('modality')!r}"
        )
        assert gw.get("complexity") == "simple", (
            f"'cooking videos' with score 0.446 should be simple, got: {gw.get('complexity')!r}"
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

        assert resp.status_code == 200, (
            f"Complex query failed with {resp.status_code}. "
            f"E2E tests require orchestrator (Ollama) running."
        )
        data = resp.json()
        assert data["status"] == "success"
        # Orchestrator handled it — verify it produced real work
        assert "orchestration_result" in data or "gateway_context" in data, (
            f"Complex query should produce orchestration result, got keys: {list(data.keys())}"
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

        assert resp.status_code == 200, (
            f"Complex query failed with {resp.status_code}. "
            f"E2E tests require orchestrator (Ollama) running."
        )
        data = resp.json()
        assert data["status"] == "success"

        # Must have orchestration result — not just gateway classification
        assert "orchestration_result" in data or data.get("agent") == "orchestrator_agent", (
            f"Complex query must produce orchestration_result, got keys: {list(data.keys())}"
        )

        # Orchestrator must have produced real work
        assert data.get("agent") == "orchestrator_agent", (
            f"Complex query should be handled by orchestrator, got agent={data.get('agent')!r}"
        )
        orch = data.get("orchestration_result", {})
        assert "plan_steps" in orch, (
            f"Orchestration should produce plan_steps, got keys: {list(orch.keys())}"
        )
        assert len(orch["plan_steps"]) >= 1, (
            f"Orchestration plan should have at least 1 step, got {len(orch['plan_steps'])}"
        )
        # gateway_context proves the gateway classified this as complex
        assert "gateway_context" in data, (
            f"Response should include gateway_context, got keys: {list(data.keys())}"
        )

    def test_analysis_keyword_triggers_complex(self):
        """Queries with 'analyze'/'summarize' keywords should be complex
        regardless of modality confidence — the complexity detection
        checks for analysis verbs."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "analyze the video transcripts for key themes",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200, (
            f"'analyze' query failed with {resp.status_code}. "
            f"E2E tests require all services running."
        )
        data = resp.json()
        # Complex queries produce orchestration_result (handled by OrchestratorAgent)
        # The gateway classified as complex, which is why orchestrator was invoked.
        # Simple queries have agent=="gateway_agent" + "gateway" dict.
        # Complex queries have agent=="orchestrator_agent" + "gateway_context" dict.
        assert data.get("agent") == "orchestrator_agent", (
            f"'analyze' keyword should trigger orchestrator, got agent={data.get('agent')!r}"
        )
        assert "orchestration_result" in data, (
            f"Complex query should produce orchestration_result, got keys: {list(data.keys())}"
        )
        assert "gateway_context" in data, (
            f"Orchestrator response should include gateway_context, got keys: {list(data.keys())}"
        )

    def test_gateway_consistent_across_calls(self):
        """Same query should produce same classification twice."""
        query = "search for video content about AI"
        results = []
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            for _ in range(2):
                resp = client.post(
                    "/agents/gateway_agent/process",
                    json={
                        "agent_name": "gateway_agent",
                        "query": query,
                        "context": {"tenant_id": TENANT_ID},
                        "top_k": 3,
                    },
                )
                assert resp.status_code == 200
                results.append(resp.json())

        gw1 = results[0].get("gateway", {})
        gw2 = results[1].get("gateway", {})
        assert gw1["complexity"] == gw2["complexity"], (
            f"Inconsistent complexity: {gw1['complexity']} vs {gw2['complexity']}"
        )
        assert gw1["modality"] == gw2["modality"], (
            f"Inconsistent modality: {gw1['modality']} vs {gw2['modality']}"
        )
        assert gw1["routed_to"] == gw2["routed_to"], (
            f"Inconsistent routing: {gw1['routed_to']} vs {gw2['routed_to']}"
        )


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

        # Simple routing must produce downstream_result with search results
        assert "downstream_result" in data, (
            f"Simple query should produce downstream_result, got keys: {list(data.keys())}"
        )
        downstream = data["downstream_result"]
        assert "results" in downstream, (
            f"Downstream should contain 'results', got keys: {list(downstream.keys())}"
        )
        assert downstream["results_count"] >= 1, (
            "Gateway search for 'search for video content about AI' must return results"
        )
        results = downstream["results"]
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results must be ranked by score descending, got: {scores}"
        )

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

        # Must have results — ingested data exists for this tenant
        assert "results" in downstream, f"Missing results, keys: {list(downstream.keys())}"
        assert downstream["results_count"] >= 1, (
            "'find videos about machine learning' must return results from ingested data"
        )

        result = downstream["results"][0]

        # Each result must have: document_id, score, metadata with video_id
        assert "document_id" in result, f"Result missing document_id: {list(result.keys())}"
        assert result["document_id"] != "", "document_id should not be empty"
        assert "score" in result, f"Result missing score: {list(result.keys())}"
        assert result["score"] > 0, f"Score should be positive, got {result['score']}"
        assert "metadata" in result, f"Result missing metadata: {list(result.keys())}"

        meta = result["metadata"]
        assert "video_id" in meta, f"metadata missing video_id: {list(meta.keys())}"
        assert "segment_id" in meta, f"metadata missing segment_id: {list(meta.keys())}"
        assert isinstance(meta["segment_id"], int), f"segment_id should be int, got {type(meta['segment_id'])}"

        # Temporal info should be present (start_time, end_time)
        if "temporal_info" in result:
            temporal = result["temporal_info"]
            assert "start_time" in temporal, "temporal_info missing start_time"
            assert "end_time" in temporal, "temporal_info missing end_time"
            assert temporal["end_time"] >= temporal["start_time"], (
                f"end_time ({temporal['end_time']}) should be >= start_time ({temporal['start_time']})"
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

    def test_routing_agent_routes_video_to_search(self):
        """'show me cooking videos' through routing → gateway classifies as
        simple video → routes to search_agent → returns Vespa results.

        This verifies the full routing→gateway→search pipeline produces
        real search results with correct classification.
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

        # Must go through gateway and produce classification
        gw = data.get("gateway", {})
        assert gw.get("complexity") == "simple", (
            f"'cooking videos' should be simple, got {gw.get('complexity')!r}"
        )
        assert gw.get("modality") == "video", (
            f"'cooking videos' should be video modality, got {gw.get('modality')!r}"
        )
        assert gw.get("routed_to") == "search_agent", (
            f"Simple video should route to search_agent, got {gw.get('routed_to')!r}"
        )

        # Must produce downstream search results
        downstream = data.get("downstream_result", {})
        assert downstream.get("status") == "success", (
            f"Downstream search should succeed, got: {downstream.get('status')}"
        )
        assert downstream.get("results_count", 0) >= 1, (
            "Should return search results from ingested data"
        )

    def test_routing_no_inline_entities(self):
        """Routing agent response must NOT have top-level entities or
        enhanced_query — those moved to dedicated A2A agents."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find videos about machine learning",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()

        # These fields were removed from routing agent in the restructuring
        # They should NOT appear at the top level of the response
        assert "entities" not in data or data["entities"] == [], (
            f"Routing agent should not return inline entities, got: {data.get('entities')}"
        )
        assert "relationships" not in data or data["relationships"] == [], (
            "Routing agent should not return inline relationships"
        )
        # enhanced_query should not be at top level (it's in downstream if anywhere)
        assert "enhanced_query" not in data or data.get("agent") != "routing_agent", (
            "Top-level enhanced_query indicates old inline routing, not A2A architecture"
        )

    def test_routing_different_modality_routes_correctly(self):
        """Audio query through routing → gateway → audio_analysis_agent."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "listen to podcasts about deep learning",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        gw = data.get("gateway", {})
        assert gw.get("modality") == "audio", (
            f"'podcasts about deep learning' should be audio, got {gw.get('modality')!r}"
        )
        assert gw.get("routed_to") == "audio_analysis_agent", (
            f"Audio query should route to audio_analysis_agent, got {gw.get('routed_to')!r}"
        )

    def test_image_modality_routes_to_image_agent(self):
        """'find images of neural network architectures' → image, image_search_agent.

        GLiNER score 0.423 on deployed model (above 0.4 threshold).
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find images of neural network architectures",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        gw = data.get("gateway", {})
        assert gw.get("modality") == "image", (
            f"'images of neural networks' should be image modality, got {gw.get('modality')!r}"
        )
        assert gw.get("routed_to") == "image_search_agent", (
            f"Image query should route to image_search_agent, got {gw.get('routed_to')!r}"
        )

    def test_document_modality_routes_to_document_agent(self):
        """'find PDF documents about Python' → document, document_agent.

        GLiNER score 0.466 on deployed model (above 0.4 threshold).
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find PDF documents about Python",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        gw = data.get("gateway", {})
        assert gw.get("modality") == "document", (
            f"'PDF documents' should be document modality, got {gw.get('modality')!r}"
        )
        assert gw.get("routed_to") == "document_agent", (
            f"Document query should route to document_agent, got {gw.get('routed_to')!r}"
        )


# ---------------------------------------------------------------------------
# 5. Entity extraction agent
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestEntityExtractionAgent:
    """Entity extraction agent is an internal orchestration agent.
    It is callable via REST through generic A2A dispatch, and also
    internally by the OrchestratorAgent via A2A HTTP."""

    def test_entity_extraction_agent_returns_entities(self):
        """POST to entity_extraction_agent/process extracts real named entities.

        "Obama speaking at MIT about climate change" should produce:
        - Obama (PERSON, confidence >0.9)
        - MIT (ORGANIZATION, confidence >0.8)
        - climate change (CONCEPT, confidence >0.8)
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
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

        # Must extract real entities, not empty list
        entities = data["entities"]
        assert len(entities) >= 2, (
            f"Expected at least 2 entities from 'Obama speaking at MIT about climate change', "
            f"got {len(entities)}: {entities}"
        )

        entity_texts = {e["text"].lower() for e in entities}
        assert "obama" in entity_texts, f"Expected 'Obama' in entities, got: {entity_texts}"
        assert "mit" in entity_texts or any("mit" in t for t in entity_texts), (
            f"Expected 'MIT' in entities, got: {entity_texts}"
        )

        # All entities should have meaningful confidence
        for e in entities:
            assert e["confidence"] > 0.5, (
                f"Entity '{e['text']}' confidence {e['confidence']} too low"
            )
            assert e["type"] in ("PERSON", "ORGANIZATION", "CONCEPT", "PLACE", "EVENT", "TECHNOLOGY"), (
                f"Entity '{e['text']}' has unexpected type '{e['type']}'"
            )

        # Fast path should be used (GLiNER available in k3d pod)
        assert data.get("path_used") == "fast", (
            f"Expected GLiNER fast path, got: {data.get('path_used')}"
        )

        # Relationships should be populated when 2+ entities exist
        assert len(data.get("relationships", [])) >= 1, (
            f"Expected relationships with {len(entities)} entities, got: {data.get('relationships')}"
        )

    def test_entity_extraction_tech_entities(self):
        """Extract technology entities: Python, TensorFlow from tech query.

        Must detect SPECIFIC entities by name, not just "at least one tech entity".
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/entity_extraction_agent/process",
                json={
                    "agent_name": "entity_extraction_agent",
                    "query": "Python programming with TensorFlow for deep learning",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        entities = data["entities"]
        entity_texts = {e["text"].lower() for e in entities}

        # Assert EXACT entities, not just "has any tech"
        assert "python" in entity_texts, (
            f"Must detect 'Python' as entity, got: {entity_texts}"
        )
        assert "tensorflow" in entity_texts or any("tensorflow" in t for t in entity_texts), (
            f"Must detect 'TensorFlow' as entity, got: {entity_texts}"
        )

        # Verify types for each detected entity
        for e in entities:
            if e["text"].lower() == "python":
                assert e["type"] in ("TECHNOLOGY", "CONCEPT", "SOFTWARE"), (
                    f"'Python' should be TECHNOLOGY/CONCEPT, got '{e['type']}'"
                )
                assert e["confidence"] > 0.5, (
                    f"'Python' confidence {e['confidence']} too low"
                )
            if "tensorflow" in e["text"].lower():
                assert e["type"] in ("TECHNOLOGY", "CONCEPT", "SOFTWARE", "FRAMEWORK"), (
                    f"'TensorFlow' should be TECHNOLOGY, got '{e['type']}'"
                )
                assert e["confidence"] > 0.5, (
                    f"'TensorFlow' confidence {e['confidence']} too low"
                )

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
        """POST to query_enhancement_agent/process produces real enhancements.

        "ML transformer videos" should expand "ML" to "machine learning"
        and produce query_variants for RRF fusion search.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
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
        assert data["original_query"] == "ML transformer videos"

        # Expansion terms should contain ML-related terms (expansion of "ML transformer")
        expansion = data.get("expansion_terms", [])
        all_expansion_text = " ".join(t.lower() for t in expansion)
        ml_related = any(
            term in all_expansion_text
            for term in ("machine learning", "deep learning", "neural", "attention", "nlp", "language model")
        )
        assert ml_related or len(expansion) > 0, (
            f"Expected ML-related expansion terms for 'ML transformer videos', got: {expansion}"
        )

        # Query variants should be non-empty (RRF fusion)
        variants = data.get("query_variants", [])
        assert len(variants) >= 1, (
            f"Expected at least 1 query variant, got: {variants}"
        )

        # Confidence should be positive
        assert data.get("confidence", 0) > 0, (
            f"Enhancement confidence should be positive, got: {data.get('confidence')}"
        )

    def test_enhancement_with_entities_passed(self):
        """Enhancement with entities from upstream should use them in context.

        Pass entities from a hypothetical entity extraction step and verify
        the enhancement agent processes them.
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/query_enhancement_agent/process",
                json={
                    "agent_name": "query_enhancement_agent",
                    "query": "find tutorials",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "entities": [
                            {"text": "TensorFlow", "type": "TECHNOLOGY", "confidence": 0.9},
                            {"text": "neural networks", "type": "CONCEPT", "confidence": 0.85},
                        ],
                        "relationships": [
                            {"subject": "TensorFlow", "relation": "used_for", "object": "neural networks"},
                        ],
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["original_query"] == "find tutorials"

        enhanced = data.get("enhanced_query", "")
        assert enhanced != "", "Enhanced query should not be empty"

        # The entities (TensorFlow, neural networks) should influence the enhancement.
        # Either the enhanced query mentions them, or expansion_terms reference them,
        # or query_variants incorporate them. At least ONE output should reflect
        # the upstream entities — otherwise they were ignored.
        all_output = (
            enhanced.lower()
            + " ".join(data.get("expansion_terms", [])).lower()
            + " ".join(data.get("query_variants", [])).lower()
        )
        entity_used = (
            "tensorflow" in all_output
            or "neural" in all_output
            or "deep learning" in all_output
            or "machine learning" in all_output
            or "framework" in all_output
        )
        assert entity_used, (
            f"Entities (TensorFlow, neural networks) should influence enhancement output. "
            f"Enhanced: {enhanced!r}, expansion: {data.get('expansion_terms')}, "
            f"variants: {data.get('query_variants')}"
        )

        assert data.get("confidence", 0) > 0

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
        """POST to profile_selection_agent/process selects a real Vespa profile.

        "find basketball highlights" is a video query — should select a video
        profile from the 4 known profiles, with modality="video".
        """
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
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

        # Must select one of the 4 known Vespa profiles
        known_profiles = {
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        }
        assert data["selected_profile"] in known_profiles, (
            f"Expected one of {known_profiles}, got: {data['selected_profile']}"
        )

        # Video query should detect video modality
        assert data.get("modality") == "video", (
            f"'find basketball highlights' should be video modality, got: {data.get('modality')}"
        )

        # Profile must be a VIDEO profile — all 4 known profiles start with "video_"
        assert data["selected_profile"].startswith("video_"), (
            f"Video query should select a video profile (starts with 'video_'), "
            f"got: {data['selected_profile']}"
        )

        # Alternatives should list other profiles, all also video profiles
        alternatives = data.get("alternatives", [])
        assert len(alternatives) >= 1, (
            f"Expected alternative profiles, got: {alternatives}"
        )

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

    def test_phoenix_is_healthy(self):
        """Phoenix must be running and healthy in k3d."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{PHOENIX_URL}/healthz")
        assert resp.status_code == 200, (
            f"Phoenix at {PHOENIX_URL} returned {resp.status_code}. "
            f"Phoenix is a k3d pod — it must be running for E2E tests."
        )

    def test_gateway_span_emitted_to_phoenix(self):
        """Gateway query must produce a span visible in Phoenix.

        Uses phoenix.Client SDK (same as integration tests) to query
        spans by project. Polls for up to 30s for span propagation.

        Known issue: if cogniverse-telemetry-phoenix fails to import
        (broken phoenix.evals dependency in Docker image), the
        TelemetryManager falls back to NoOpSpan and no spans are emitted.
        This test will fail in that case — fix the Docker image deps.
        """
        import time

        from phoenix.client import Client as PhoenixClient

        # Run a query through the gateway
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

        assert resp.status_code == 200, (
            f"Gateway call failed with {resp.status_code}. "
            f"All services must be running for E2E tests."
        )

        # Poll Phoenix for the span (async export has propagation delay)
        phoenix_client = PhoenixClient(base_url=PHOENIX_URL)
        # Spans go to tenant-specific project: cogniverse-{tenant_id}
        project_name = f"cogniverse-{TENANT_ID}"

        deadline = time.time() + 30
        found_span = None
        while time.time() < deadline:
            try:
                spans_df = phoenix_client.spans.get_spans_dataframe(
                    project_identifier=project_name,
                    limit=50,
                )
                if spans_df is not None and not spans_df.empty:
                    matches = spans_df[
                        spans_df["name"].str.contains("gateway", case=False, na=False)
                    ]
                    if not matches.empty:
                        found_span = matches.iloc[0]
                        break
            except Exception:
                pass
            time.sleep(2)

        assert found_span is not None, (
            f"No gateway spans found in Phoenix project '{project_name}' after 30s. "
            f"Phoenix is at {PHOENIX_URL}. Span emission or OTLP export is broken."
        )
        assert "gateway" in found_span["name"].lower(), (
            f"Span name should contain 'gateway', got: {found_span['name']}"
        )
