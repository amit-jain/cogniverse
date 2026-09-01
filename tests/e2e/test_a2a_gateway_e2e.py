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

import httpx
import pytest

from tests.e2e.conftest import (
    DATA_ROOT,
    PHOENIX_URL,
    RUNTIME,
    TENANT_ID,
    _ensure_sample_content_ingested,
    _ingest_sample_documents,
    assert_orchestrated,
    expected_gateway_routing,
    register_tenant_and_wait,
    sample_audio_content_id,
    unique_id,
)
from tests.e2e.test_api_e2e import (
    DOCUMENT_PROFILE,
    PROFILE,
    _deploy_profile_for_tenant,
    _expected_available_profile_names,
)

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


def require_sample_video() -> None:
    """Refuse to run without the corpus the identity assertions are built on.

    Skipping here would silently disarm every seeded-search identity
    assertion in this module, so a missing corpus fails the same way
    ``_profile_selection_corpus`` handles it.
    """
    if not SAMPLE_VIDEO.is_file():
        pytest.fail(
            f"sample video {SAMPLE_VIDEO} is not present, so the seeded-search "
            "identity assertions cannot run. Download the testset corpus."
        )


@pytest.fixture
def seeded_search_tenant():
    require_sample_video()

    org_id = unique_id("a2a_seeded")
    tenant_id = f"{org_id}:t1"

    with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
        try:
            resp = client.post(
                "/admin/organizations",
                json={
                    "org_id": org_id,
                    "org_name": org_id.replace("_", "-"),
                    "created_by": "e2e",
                },
            )
            assert resp.status_code in (200, 201, 409), resp.text

            register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)
            _deploy_profile_for_tenant(client, PROFILE, tenant_id)
            _deploy_profile_for_tenant(client, DOCUMENT_PROFILE, tenant_id)

            _ensure_sample_content_ingested(
                SAMPLE_VIDEO,
                profile=PROFILE,
                media_type="video/mp4",
                tenant_id=tenant_id,
            )
            seeded_documents = _ingest_sample_documents(tenant_id=tenant_id)
            yield tenant_id, seeded_documents
        finally:
            try:
                client.delete(f"/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


# ---------------------------------------------------------------------------
# 1. Gateway simple routing
# ---------------------------------------------------------------------------


def _tenant_document_count(tenant_id: str) -> int:
    """Number of documents the tenant's document profile currently serves."""
    with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
        response = client.post(
            "/search/",
            json={
                "query": "washing dishes",
                "tenant_id": tenant_id,
                "profile": DOCUMENT_PROFILE,
                "top_k": 1000,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results_count"] == len(body["results"])
    return body["results_count"]


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
            assert downstream.get("profile") == PROFILE, (
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
    def test_seeded_video_identity_order_and_competing_route(
        self, seeded_search_tenant
    ):
        tenant_id, seeded_documents = seeded_search_tenant
        video_query = "find videos of a man washing dishes in a kitchen sink"
        document_query = "find PDF documents about washing dishes"
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            video_response = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": video_query,
                    "context": {"tenant_id": tenant_id},
                    "top_k": 5,
                },
            )
            document_response = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": document_query,
                    "context": {"tenant_id": tenant_id},
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
            # This tenant owns only the seeded video and the two seeded
            # captions, so the document route must return those captions in
            # rank order.
            document_downstream = document_data["downstream_result"]
            document_results = document_downstream["results"]
            assert document_downstream["results_count"] == len(document_results)
            expected_titles = ("v_0BtHd6dvm78.txt", "v_-nl4G-00PtA.txt")
            assert len(document_results) == len(expected_titles)
            assert [result["title"] for result in document_results] == list(
                expected_titles
            )
            assert [result["document_id"] for result in document_results] == [
                seeded_documents[title] for title in expected_titles
            ]
            assert [result["document_type"] for result in document_results] == [
                "txt",
                "txt",
            ]
            # Search agents expose relevance_score (document_agent.py:44,
            # audio_analysis_agent.py:50, image_search_agent.py:36); only the
            # video route emits "score".
            scores = [result["relevance_score"] for result in document_results]
            assert scores == sorted(scores, reverse=True)
            assert [result["strategy_used"] for result in document_results] == [
                "text",
                "text",
            ]
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

    def test_complex_query_records_both_detected_modalities(self):
        """The recorded orchestration evidence must keep every detected
        modality for a multi-modal query."""
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
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["modality"] == "both", gw
        assert_orchestrated(data, query, gw)

        iterative_loop = data["orchestration_result"]["metadata"]["iterative_loop"]
        evidence_modalities = {
            hit["modality"]
            for hit in iterative_loop["accumulated_evidence"]
            if isinstance(hit, dict) and hit.get("modality")
        }
        assert evidence_modalities == {"video", "document"}, iterative_loop

    def test_planner_repeating_an_agent_still_orchestrates(self):
        """The planner LM listed deep_research_agent twice for this exact
        query and the request came back 400; the plan must instead
        normalize to one step per agent and execute."""
        query = "find cooking recipe demonstrations run 15"
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

        assert resp.status_code == 200, f"HTTP {resp.status_code} {resp.text[:500]}"
        data = resp.json()
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        assert gw["routed_to"] == "orchestrator_agent", gw
        assert_orchestrated(data, query, gw)
        plan_agents = [
            step["agent_name"] for step in data["orchestration_result"]["plan_steps"]
        ]
        assert plan_agents != [], data["orchestration_result"]
        assert plan_agents == list(dict.fromkeys(plan_agents)), plan_agents
        assert set(data["orchestration_result"]["agent_results"]) == set(plan_agents), (
            data["orchestration_result"]["agent_results"]
        )
        n = len(plan_agents)
        assert data["orchestration_result"]["execution_summary"].startswith(
            f"Executed {n}/{n} steps ({n} successful). Plan: "
        ), data["orchestration_result"]["execution_summary"]

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
            results = downstream["results"]
            # document_agent filters by relevance, so a query about Python
            # legitimately returns nothing from a corpus of dish-washing
            # captions. What this routing test owns is that the document
            # search ran for this exact query and reported consistently;
            # exact retrieval results are pinned against a seeded corpus in
            # TestGatewaySeededSearchContract.
            assert downstream["results_count"] == len(results), downstream
            assert downstream["message"] == (
                f"Found {len(results)} documents for '{query}'"
            ), downstream
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

        # DSPy is the primary extraction path; GLiNER + SpaCy is the fallback
        # taken only when the LM call fails (entity_extraction_agent.py:270,302).
        assert data.get("path_used") == "dspy", (
            f"Expected the DSPy primary path, got: {data.get('path_used')}"
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

        # Alternatives are exactly the tenant's other registered video-typed
        # profiles (top 3, registry order); the agent reads the same registry.
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            registry = client.get("/admin/profiles", params={"tenant_id": TENANT_ID})
        assert registry.status_code == 200, registry.text
        expected_alternatives = [
            profile["profile_name"]
            for profile in registry.json()["profiles"]
            if profile["type"] == "video"
            and profile["profile_name"] != data["selected_profile"]
        ][:3]
        assert [
            candidate["profile_name"] for candidate in data["alternatives"]
        ] == expected_alternatives

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

        # Scope to this test's window — the server-side gateway predicate
        # keeps the query narrow, but Phoenix still needs a real timeout
        # budget while the project is under load.
        from datetime import datetime, timedelta, timezone

        from phoenix.client.types.spans import SpanQuery

        from cogniverse_foundation.telemetry.config import SPAN_NAME_GATEWAY

        window_start = datetime.now(timezone.utc) - timedelta(minutes=10)
        predicate = f"name == '{SPAN_NAME_GATEWAY}'"
        query = SpanQuery().where(predicate)

        deadline = time.time() + 30
        found_span = None
        last_rows = 0
        while time.time() < deadline:
            try:
                spans_df = phoenix_client.spans.get_spans_dataframe(
                    project_identifier=project_name,
                    start_time=window_start,
                    query=query,
                    timeout=90,
                )
                last_rows = 0 if spans_df is None else len(spans_df)
                if spans_df is not None and not spans_df.empty:
                    found_span = spans_df.iloc[0]
                    break
            except Exception:
                pass
            time.sleep(2)

        assert found_span is not None, (
            f"No spans matched Phoenix predicate {predicate!r} in project "
            f"'{project_name}' after 30s; last query returned {last_rows} rows "
            f"for window_start={window_start.isoformat()} against {PHOENIX_URL}. "
            "Span emission or OTLP export is broken."
        )
        assert found_span["name"] == SPAN_NAME_GATEWAY, (
            f"Span name should be {SPAN_NAME_GATEWAY!r}, got: {found_span['name']}"
        )
