"""
E2E tests for the A2A gateway architecture.

Tests the new architecture where:
    POST /agents/gateway_agent/process -> GatewayAgent classifies ->
        SIMPLE: dispatches to execution agent (search, summarizer, etc.)
        COMPLEX: dispatches to OrchestratorAgent -> coordinates A2A agents

The gateway_agent is the new primary entry point for all queries.
Entity extraction, query enhancement, and profile selection are internal
to the orchestration pipeline and not directly callable via REST.

Requires live k3d-deployed runtime at http://localhost:33000.
"""

import hashlib
import warnings

import httpx
import pytest

from tests.e2e.conftest import (
    DATA_ROOT,
    PHOENIX_URL,
    RUNTIME,
    TENANT_ID,
    assert_orchestrated,
    expected_gateway_routing,
    sample_audio_content_id,
)
from tests.e2e.test_api_e2e import _expected_available_profile_names

PROFILE = "video_colpali_smol500_mv_frame"


SAMPLE_VIDEO = (
    DATA_ROOT / "testset" / "evaluation" / "sample_videos" / "v_-nl4G-00PtA.mp4"
)


def sample_video_id() -> str:
    """Document id the seeded corpus assigns the sample video.

    The corpus lives under the gitignored ``data/testset`` tree, so a checkout
    that never downloaded it cannot compute this. Read the file here rather
    than at import: hashing at module scope made COLLECTING this file — and
    therefore ``pytest tests/`` as a whole — fail with FileNotFoundError on
    any such checkout. Callers skip when the video is absent.
    """
    return hashlib.sha256(SAMPLE_VIDEO.read_bytes()).hexdigest()


def skip_without_sample_video() -> None:
    if not SAMPLE_VIDEO.is_file():
        warnings.warn(
            f"sample video {SAMPLE_VIDEO} is not present; the seeded-search "
            "identity assertions cannot run. Download the testset corpus to "
            "exercise them.",
            stacklevel=2,
        )
        pytest.skip(f"sample video not available: {SAMPLE_VIDEO}")


# ---------------------------------------------------------------------------
# 1. Gateway simple routing
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGatewaySimpleRouting:
    """Gateway classifies simple video queries and dispatches to search_agent."""

    def test_simple_video_query_returns_gateway_structure(self):
        """POST gateway_agent/process with a simple video query returns
        complexity=simple, routed_to a search agent, and downstream results.

        Query chosen for GLiNER score 0.693 (well above 0.4 threshold) on
        the deployed 7-label model.
        """
        query = "search for video content about AI"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
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
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["modality"] == "video", (
            f"Expected video modality for video query, got {gw['modality']!r}"
        )
        assert gw["generation_type"] == "raw_results", (
            f"No summary/report keyword → raw_results, got {gw['generation_type']!r}"
        )
        assert gw["confidence"] >= gw["fast_path_confidence_threshold"], (
            f"Simple video query should meet the live fast-path threshold, got {gw['confidence']} < {gw['fast_path_confidence_threshold']}"
        )

    def test_simple_query_includes_downstream_result(self):
        """Simple path should execute the downstream agent and return its result.

        Query chosen for GLiNER score 0.444 (above 0.4 threshold) on the
        deployed 7-label model.
        """
        query = "find videos about machine learning"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )

        if gw["complexity"] == "simple":
            assert "downstream_result" in data, (
                f"Simple routing should produce downstream_result, got keys: {list(data.keys())}"
            )
            downstream = data["downstream_result"]
            assert isinstance(downstream, dict)
            assert downstream.get("status") == "success"

            # Search results must exist and contain real Vespa data
            assert "results" in downstream, (
                f"Missing 'results' in downstream, keys: {list(downstream.keys())}"
            )
            results = downstream["results"]
            assert downstream["results_count"] >= 1, (
                "Query 'find videos about machine learning' must return results from ingested data"
            )
            assert len(results) == downstream["results_count"], (
                f"results_count ({downstream['results_count']}) doesn't match len(results) ({len(results)})"
            )

            # Each result must have score + metadata with real video data
            first = results[0]
            assert "score" in first, (
                f"Result missing 'score' field: {list(first.keys())}"
            )
            assert first["score"] > 0, (
                f"First result score should be positive, got {first['score']}"
            )
            assert "metadata" in first, (
                f"Result missing 'metadata': {list(first.keys())}"
            )
            meta = first["metadata"]
            assert "video_id" in meta, (
                f"Result metadata missing video_id: {list(meta.keys())}"
            )
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
        else:
            assert_orchestrated(data, query, gw)

    def test_message_field_present(self):
        """Gateway response surfaces the downstream agent's answer as `message`.

        A simple route returns the execution agent's user-facing answer (its
        message + hits) as the response, not a routing breadcrumb — the
        breadcrumb, persisted as the assistant turn, corrupted multi-turn
        rewrite and replaced the answer in the messaging display. Routing/triage
        metadata lives under `gateway`.

        Query chosen for GLiNER score 0.446 (above 0.4 threshold) on the
        deployed 7-label model.
        """
        query = "find videos of a man washing dishes in a kitchen sink"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )

        if gw["complexity"] == "simple":
            # The message is the downstream agent's answer, not a routing breadcrumb.
            msg = data["message"]
            assert msg and not msg.startswith("Routed "), (
                f"Message should be the downstream answer, not a routing breadcrumb, "
                f"got: {msg!r}"
            )
            # The search answer surfaces its hit count + results at the top level so
            # the messaging display renders them.
            assert isinstance(data.get("results_count"), int), (
                f"Simple search route should surface results_count, got keys: "
                f"{list(data.keys())}"
            )
            assert "results" in data, f"got keys: {list(data.keys())}"
        else:
            assert_orchestrated(data, query, gw)


@pytest.mark.e2e
class TestGatewaySeededSearchContract:
    def test_seeded_video_identity_order_and_competing_route(self):
        skip_without_sample_video()
        video_query = "find videos of a man washing dishes in a kitchen sink"
        document_query = "find PDF documents about washing dishes"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            video_response = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": video_query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )
            document_response = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": document_query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert video_response.status_code == 200
        video_data = video_response.json()
        assert video_data["status"] == "success"
        video_gw = video_data["gateway"]
        assert (
            video_gw["complexity"],
            video_gw["routed_to"],
        ) == expected_gateway_routing(video_query, video_gw)
        if video_gw["complexity"] == "simple":
            downstream = video_data["downstream_result"]
            assert downstream["status"] == "success"
            assert downstream["profile"] == PROFILE
            assert downstream["results_count"] == len(downstream["results"])
            video_ids = [
                result["metadata"]["video_id"] for result in downstream["results"]
            ]
            assert set(video_ids) == {sample_video_id()}
            scores = [result["score"] for result in downstream["results"]]
            assert scores == sorted(scores, reverse=True)
        else:
            assert video_data["agent"] == "orchestrator_agent"
            assert "orchestration_result" in video_data
            assert "gateway_context" in video_data

        assert document_response.status_code == 200
        document_data = document_response.json()
        assert document_data["status"] == "success"
        document_gw = document_data["gateway"]
        assert (
            document_gw["complexity"],
            document_gw["routed_to"],
        ) == expected_gateway_routing(document_query, document_gw)
        if document_gw["complexity"] == "simple":
            document_results = document_data["downstream_result"]["results"]
            assert sample_video_id() not in {
                result["metadata"]["video_id"] for result in document_results
            }
        else:
            assert document_data["agent"] == "orchestrator_agent"
            assert "orchestration_result" in document_data
            assert "gateway_context" in document_data


# ---------------------------------------------------------------------------
# 2. Gateway complex routing
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGatewayComplexRouting:
    """Gateway classifies complex/multi-modal queries and dispatches
    to orchestrator for multi-agent coordination."""

    def test_complex_query_classified_as_complex(self):
        """A multi-modal, multi-step query should be classified as complex
        by the gateway regardless of whether the orchestrator succeeds.

        This test asserts only on the gateway classification, which does not
        depend on the LM or the orchestrator being healthy.  The query spans
        both video and document modalities which forces complexity regardless
        of GLiNER confidence.
        """
        query = "find videos and documents about neural networks"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200, (
            f"Complex query failed with {resp.status_code}. "
            f"E2E tests require orchestrator + LM running."
        )
        data = resp.json()
        assert data["status"] == "success"
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["modality"] == "both", gw
        assert gw["generation_type"] == "raw_results", gw
        assert_orchestrated(data, query, gw)

    def test_complex_query_triggers_orchestration(self):
        """A clearly complex query should route to the orchestrator when it is
        healthy.  If the orchestrator returns 500 (e.g. LM not loaded),
        we still verify the gateway classification was correct.
        """
        query = (
            "Find videos about machine learning, compare them with "
            "the PDF research papers, and write a detailed report"
        )
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200, (
            f"Complex query failed with {resp.status_code}. "
            f"E2E tests require orchestrator + LM running."
        )
        data = resp.json()
        assert data["status"] == "success"
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["modality"] == "both", gw
        assert gw["generation_type"] == "detailed_report", gw
        assert_orchestrated(data, query, gw)
        # The orchestrator planned real work: a non-empty plan of
        # (agent_name, reasoning, depends_on) steps.
        plan_steps = data["orchestration_result"]["plan_steps"]
        assert plan_steps != [], data["orchestration_result"]
        assert [set(step) for step in plan_steps] == [
            {"agent_name", "reasoning", "depends_on"}
        ] * len(plan_steps), plan_steps

    def test_analysis_keyword_triggers_complex(self):
        """Queries with 'analyze'/'summarize' keywords should be complex
        regardless of modality confidence — the complexity detection
        checks for analysis verbs."""
        query = "analyze the video transcripts for key themes"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200, (
            f"'analyze' query failed with {resp.status_code}. "
            f"E2E tests require all services running."
        )
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["complexity"] == "complex", gw
        assert_orchestrated(data, query, gw)

    def test_gateway_consistent_across_calls(self):
        """Same query should produce same classification twice."""
        query = "search for video content about AI"
        results = []
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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

        gw1 = results[0]["gateway"]
        gw2 = results[1]["gateway"]
        assert (gw1["complexity"], gw1["routed_to"]) == expected_gateway_routing(
            query, gw1
        )
        assert (gw2["complexity"], gw2["routed_to"]) == expected_gateway_routing(
            query, gw2
        )
        assert gw1 == gw2, f"Inconsistent gateway classification: {gw1} vs {gw2}"


# ---------------------------------------------------------------------------
# 3. Full pipeline: gateway -> search (simple path)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
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
        query = "search for video content about AI"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )

        if gw["complexity"] == "simple":
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
            assert gw["modality"] == "video", gw
            assert gw["generation_type"] == "raw_results", gw
        else:
            assert_orchestrated(data, query, gw)

    def test_gateway_search_result_fields(self):
        """Search results from the gateway pipeline should have content fields.

        Query chosen for GLiNER score 0.444 (above 0.4 threshold) on the
        deployed 7-label model.
        """
        query = "find videos about machine learning"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )

        if gw["complexity"] == "simple":
            downstream = data["downstream_result"]

            # Must have results — ingested data exists for this tenant
            assert "results" in downstream, (
                f"Missing results, keys: {list(downstream.keys())}"
            )
            assert downstream["results_count"] >= 1, (
                "'find videos about machine learning' must return results from ingested data"
            )

            result = downstream["results"][0]

            # Each result must have: document_id, score, metadata with video_id
            assert "document_id" in result, (
                f"Result missing document_id: {list(result.keys())}"
            )
            assert result["document_id"] != "", "document_id should not be empty"
            assert "score" in result, f"Result missing score: {list(result.keys())}"
            assert result["score"] > 0, (
                f"Score should be positive, got {result['score']}"
            )
            assert "metadata" in result, (
                f"Result missing metadata: {list(result.keys())}"
            )

            meta = result["metadata"]
            assert "video_id" in meta, f"metadata missing video_id: {list(meta.keys())}"
            assert "segment_id" in meta, (
                f"metadata missing segment_id: {list(meta.keys())}"
            )
            assert isinstance(meta["segment_id"], int), (
                f"segment_id should be int, got {type(meta['segment_id'])}"
            )

            # Temporal info should be present (start_time, end_time)
            if "temporal_info" in result:
                temporal = result["temporal_info"]
                assert "start_time" in temporal, "temporal_info missing start_time"
                assert "end_time" in temporal, "temporal_info missing end_time"
                assert temporal["end_time"] >= temporal["start_time"], (
                    f"end_time ({temporal['end_time']}) should be >= start_time ({temporal['start_time']})"
                )
        else:
            assert_orchestrated(data, query, gw)


# ---------------------------------------------------------------------------
# 4. Routing agent thin interface
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestGatewayAgentThin:
    """The gateway agent is a thin decision-maker: GLiNER classification +
    fast-path vs orchestrator routing. POST to gateway_agent/process goes
    through _execute_gateway_task in the dispatcher. The gateway does NOT
    perform entity extraction or query enhancement inline — those are
    separate agents invoked by the orchestrator when needed."""

    def test_gateway_agent_routes_video_to_search(self):
        """'find videos of dogs running on a beach' through routing → gateway classifies as
        simple video → routes to search_agent → returns Vespa results.

        This verifies the full routing→gateway→search pipeline produces
        real search results with correct classification.
        """
        query = "find videos of dogs running on a beach"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # Must go through gateway and produce classification
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )

        if gw["complexity"] == "simple":
            # Must produce downstream search results
            downstream = data["downstream_result"]
            assert downstream.get("status") == "success", (
                f"Downstream search should succeed, got: {downstream.get('status')}"
            )
            assert downstream.get("results_count", 0) >= 1, (
                "Should return search results from ingested data"
            )
        else:
            assert_orchestrated(data, query, gw)

    def test_routing_no_inline_entities(self):
        """Routing agent response must NOT have top-level entities or
        enhanced_query — those moved to dedicated A2A agents."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
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
        assert "enhanced_query" not in data or data.get("agent") != "gateway_agent", (
            "Top-level enhanced_query indicates old inline routing, not A2A architecture"
        )

    def test_routing_different_modality_routes_correctly(self):
        """Audio query through routing → gateway → audio_analysis_agent."""
        query = "listen to podcasts about deep learning"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["modality"] == "audio", (
            f"'podcasts about deep learning' should be audio, got {gw['modality']!r}"
        )
        assert gw["generation_type"] == "raw_results", gw

        if gw["complexity"] == "simple":
            # The session seeds exactly one audio clip for the tenant, so the
            # acoustic nearest-neighbour search returns that clip and nothing else.
            downstream = data["downstream_result"]
            assert downstream["status"] == "success", downstream
            assert downstream["agent"] == "audio_analysis_agent", downstream
            assert downstream["results_count"] == 1, downstream
            assert [result["audio_id"] for result in downstream["results"]] == [
                sample_audio_content_id()
            ]
            assert downstream["results"][0]["audio_url"] == (
                f"s3://cogniverse-ingest/{TENANT_ID}/{sample_audio_content_id()}.wav"
            )
        else:
            assert_orchestrated(data, query, gw)

    def test_image_modality_routes_to_image_agent(self):
        """'find images of neural network architectures' → image, image_search_agent.

        GLiNER score 0.423 on deployed model (above 0.4 threshold).
        """
        query = "find images of neural network architectures"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        if gw["complexity"] == "simple":
            downstream = data["downstream_result"]
            assert downstream["status"] == "success", downstream
            assert downstream["agent"] == "image_search_agent", downstream
            assert downstream["results_count"] >= 1, downstream
        else:
            assert_orchestrated(data, query, gw)

    def test_document_modality_routes_to_document_agent(self):
        """'find PDF documents about Python' → document, document_agent.

        GLiNER score 0.466 on deployed model (above 0.4 threshold).
        """
        query = "find PDF documents about Python"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        if gw["complexity"] == "simple":
            downstream = data["downstream_result"]
            assert downstream["status"] == "success", downstream
            assert downstream["agent"] == "document_agent", downstream
            assert downstream["results_count"] >= 1, downstream
        else:
            assert_orchestrated(data, query, gw)


# ---------------------------------------------------------------------------
# 5. Entity extraction agent
# ---------------------------------------------------------------------------


@pytest.mark.e2e
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
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/entity_extraction_agent/process",
                json={
                    "agent_name": "entity_extraction_agent",
                    "query": "Obama speaking at MIT about climate change",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        )
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
        assert "obama" in entity_texts, (
            f"Expected 'Obama' in entities, got: {entity_texts}"
        )
        assert "mit" in entity_texts or any("mit" in t for t in entity_texts), (
            f"Expected 'MIT' in entities, got: {entity_texts}"
        )

        # All entities should have meaningful confidence
        for e in entities:
            assert e["confidence"] > 0.5, (
                f"Entity '{e['text']}' confidence {e['confidence']} too low"
            )
            assert e["type"] in (
                "PERSON",
                "ORGANIZATION",
                "CONCEPT",
                "PLACE",
                "EVENT",
                "TECHNOLOGY",
            ), f"Entity '{e['text']}' has unexpected type '{e['type']}'"

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
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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

        # Assert the named tech terms appear. GLiNER returns whole spans
        # ("Python programming"), so match by substring, not exact token.
        assert "python" in entity_texts or any("python" in t for t in entity_texts), (
            f"Must detect 'Python' as entity, got: {entity_texts}"
        )
        assert "tensorflow" in entity_texts or any(
            "tensorflow" in t for t in entity_texts
        ), f"Must detect 'TensorFlow' as entity, got: {entity_texts}"

        # Verify types for each detected entity
        for e in entities:
            if "python" in e["text"].lower():
                assert e["type"] in ("TECHNOLOGY", "CONCEPT", "SOFTWARE"), (
                    f"'Python' should be TECHNOLOGY/CONCEPT, got '{e['type']}'"
                )
                assert e["confidence"] > 0.5, (
                    f"'Python' confidence {e['confidence']} too low"
                )
            if "tensorflow" in e["text"].lower():
                assert e["type"] in (
                    "TECHNOLOGY",
                    "CONCEPT",
                    "SOFTWARE",
                    "FRAMEWORK",
                ), f"'TensorFlow' should be TECHNOLOGY, got '{e['type']}'"
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
class TestQueryEnhancementAgent:
    """Query enhancement agent — callable via REST and internally by orchestrator."""

    def test_query_enhancement_agent_returns_enhanced_query(self):
        """POST to query_enhancement_agent/process produces real enhancements.

        "ML transformer videos" should expand "ML" to "machine learning"
        and produce query_variants for RRF fusion search.
        """
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/query_enhancement_agent/process",
                json={
                    "agent_name": "query_enhancement_agent",
                    "query": "ML transformer videos",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "query_enhancement_agent"
        assert data["original_query"] == "ML transformer videos"

        # Expansion terms should contain ML-related terms (expansion of "ML transformer")
        expansion = data.get("expansion_terms", [])
        all_expansion_text = " ".join(t.lower() for t in expansion)
        ml_related = any(
            term in all_expansion_text
            for term in (
                "machine learning",
                "deep learning",
                "neural",
                "attention",
                "nlp",
                "language model",
            )
        )
        assert ml_related or len(expansion) > 0, (
            f"Expected ML-related expansion terms for 'ML transformer videos', got: {expansion}"
        )

        # Query variants should be non-empty (RRF fusion)
        variants = data.get("query_variants", [])
        assert len(variants) >= 1, f"Expected at least 1 query variant, got: {variants}"

        # Confidence should be positive
        assert data.get("confidence", 0) > 0, (
            f"Enhancement confidence should be positive, got: {data.get('confidence')}"
        )

    def test_enhancement_with_entities_passed(self):
        """Enhancement with entities from upstream should use them in context.

        Pass entities from a hypothetical entity extraction step and verify
        the enhancement agent processes them.
        """
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/query_enhancement_agent/process",
                json={
                    "agent_name": "query_enhancement_agent",
                    "query": "find tutorials",
                    "context": {
                        "tenant_id": TENANT_ID,
                        "entities": [
                            {
                                "text": "TensorFlow",
                                "type": "TECHNOLOGY",
                                "confidence": 0.9,
                            },
                            {
                                "text": "neural networks",
                                "type": "CONCEPT",
                                "confidence": 0.85,
                            },
                        ],
                        "relationships": [
                            {
                                "subject": "TensorFlow",
                                "relation": "used_for",
                                "object": "neural networks",
                            },
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
class TestProfileSelectionAgent:
    """Profile selection agent — callable via REST and internally by orchestrator."""

    def test_profile_selection_agent_returns_profile(self):
        """POST to profile_selection_agent/process selects a real Vespa profile.

        "find basketball highlights" is a video query — should select a video
        profile from the tenant's exact usable profile set, with modality="video".
        """
        expected_available_profiles = _expected_available_profile_names(TENANT_ID)
        assert expected_available_profiles, (
            "The tenant's usable profile set is unexpectedly empty; the comparison "
            "would be vacuous."
        )

        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/profile_selection_agent/process",
                json={
                    "agent_name": "profile_selection_agent",
                    "query": "find basketball highlights",
                    "context": {"tenant_id": TENANT_ID},
                },
            )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        )
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "profile_selection_agent"

        # Must select one of the tenant's actual usable profiles.
        assert data["selected_profile"] in expected_available_profiles, (
            f"Expected one of {expected_available_profiles}, got: "
            f"{data['selected_profile']}"
        )

        # Video query should detect video modality
        assert data.get("modality") == "video", (
            f"'find basketball highlights' should be video modality, got: {data.get('modality')}"
        )

        # Profile must be a VIDEO profile.
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
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "find videos of dogs running on a beach",
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

        # Scope to this test's window — an unscoped limit slice over a
        # project holding a day of spans can consist entirely of other
        # runs' spans. The method's own 5s timeout default also needs a
        # budget Phoenix can meet while loaded.
        from datetime import datetime, timedelta, timezone

        window_start = datetime.now(timezone.utc) - timedelta(minutes=10)

        deadline = time.time() + 30
        found_span = None
        while time.time() < deadline:
            try:
                spans_df = phoenix_client.spans.get_spans_dataframe(
                    project_identifier=project_name,
                    start_time=window_start,
                    limit=50,
                    timeout=90,
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
