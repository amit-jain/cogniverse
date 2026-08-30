"""
E2E API tests exercising routing, search, tenant CRUD, agent registry,
A2A protocol, profile CRUD, ingestion, synthetic data, and event streaming.

Requires live runtime at http://localhost:33000 with LM + Vespa + Phoenix.
Uses flywheel_org:production tenant which has ingested data.

Architecture note (A2A gateway):
    The primary entry point is now ``gateway_agent``, which classifies queries
    and dispatches simple ones directly to execution agents (search_agent, etc.)
    and complex ones to the OrchestratorAgent for multi-agent coordination.

    Entity extraction, query enhancement, and profile selection are handled by
    dedicated A2A agents that are invoked internally by the orchestrator --
    they are NOT called inline by the gateway_agent anymore.

    The gateway_agent is now a thin DSPy-powered decision-maker. Both
    ``gateway`` and ``routing`` capabilities route through the gateway
    pipeline in the dispatcher.

    See ``test_a2a_gateway_e2e.py`` for gateway-specific E2E tests.
"""

import json
import re
import time
import uuid
from pathlib import Path

import httpx
import pytest

from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names
from cogniverse_foundation.config.unified_config import (
    SyntheticGeneratorConfig,
)
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_synthetic.profile_selector import (
    score_profile_with_configured_rules,
)
from cogniverse_synthetic.schemas import ProfileSelectionExampleSchema
from cogniverse_synthetic.topics import (
    TopicSaliency,
    extract_topic,
    topic_source_text,
)
from cogniverse_synthetic.utils.agent_inference import AgentInferrer
from tests.e2e.conftest import (
    KUBECTL_CONTEXT,
    RUNTIME,
    SAMPLE_VIDEO_PATH,
    TENANT_ID,
    _content_sha256,
    _ensure_sample_content_ingested,
    _ingest_sample_documents,
    _matching_sample_results,
    assert_orchestrated,
    expected_gateway_routing,
    register_tenant_and_wait,
    unique_id,
)
from tests.e2e.conftest import (
    _configured_profile_name as _configured_profile_name_from_config,
)

CAPTION_CORPUS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "testset"
    / "Test_Human_Annotated_Captions"
)
CAPTION_CORPUS_LIMIT = 50
SECOND_SAMPLE_VIDEO_PATH = (
    Path(__file__).resolve().parent.parent
    / "system"
    / "resources"
    / "videos"
    / "v_-D1gdv_gQyw.mp4"
)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"


def _default_video_profile_name() -> str:
    config = json.loads(CONFIG_PATH.read_text())
    active = config.get("active_video_profile")
    if isinstance(active, dict):
        name = active.get("name")
        if isinstance(name, str) and name.strip():
            return name
    if isinstance(active, str) and active.strip():
        return active
    return _configured_profile_name("video")


PROFILE = _default_video_profile_name()


def _workflow_video_profile_name() -> str:
    config = json.loads(CONFIG_PATH.read_text())
    synthetic = dict(config["synthetic"])
    synthetic["tenant_id"] = TENANT_ID
    generator_config = SyntheticGeneratorConfig.from_dict(synthetic)
    workflow_rules = generator_config.get_optimizer_config(
        "workflow"
    ).profile_scoring_rules
    scored_profiles: list[tuple[float, str]] = []
    for profile_name, profile_config in (
        config.get("backend", {}).get("profiles", {}).items()
    ):
        if not isinstance(profile_config, dict):
            continue
        if profile_config.get("type") != "video":
            continue
        score, _ = score_profile_with_configured_rules(
            workflow_rules, profile_name, profile_config
        )
        scored_profiles.append((score, profile_name))
    if not scored_profiles:
        raise RuntimeError("workflow video profile scoring found no video profiles")
    scored_profiles.sort(key=lambda item: item[0], reverse=True)
    return scored_profiles[0][1]


WORKFLOW_PROFILE = _workflow_video_profile_name()
_PROFILE_TYPE_ORDER = {
    "video": 0,
    "document": 1,
    "image": 2,
    "audio": 3,
    "code": 4,
    "wiki": 5,
}


def _configured_profile_name(
    profile_type: str, *, schema_name: str | None = None
) -> str:
    config = json.loads(CONFIG_PATH.read_text())
    return _configured_profile_name_from_config(
        config, profile_type=profile_type, schema_name=schema_name
    )


IMAGE_PROFILE = _configured_profile_name("image")
DOCUMENT_PROFILE = _configured_profile_name("document", schema_name="document_text")
AUDIO_PROFILE = _configured_profile_name("audio")


def _shipped_agent_inferrer() -> AgentInferrer:
    """The AgentInferrer the synthetic service builds from the shipped config."""
    config = json.loads(CONFIG_PATH.read_text())
    synthetic = dict(config["synthetic"])
    synthetic["tenant_id"] = TENANT_ID
    generator_config = SyntheticGeneratorConfig.from_dict(synthetic)
    return AgentInferrer(
        agents_config=config["agents"],
        agent_mappings=generator_config.get_optimizer_config("modality").agent_mappings,
    )


_SHIPPED_AGENT_INFERRER = _shipped_agent_inferrer()
# Every agent a workflow sequence can name: the primary agent per modality plus
# the summarizer / detailed_report roles infer_workflow_sequence appends.
PRIMARY_AGENT_BY_QUERY_TYPE = dict(_SHIPPED_AGENT_INFERRER.MODALITY_TO_AGENT)
CANONICAL_WORKFLOW_AGENTS = set(PRIMARY_AGENT_BY_QUERY_TYPE.values()) | {
    _SHIPPED_AGENT_INFERRER.ROLE_AGENTS["summarizer"],
    _SHIPPED_AGENT_INFERRER.ROLE_AGENTS["detailed_report"],
}


def _deploy_profile_for_tenant(
    client: httpx.Client, profile_name: str, tenant_id: str
) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    profile_def = config.get("backend", {}).get("profiles", {}).get(profile_name, {})
    assert profile_def, f"missing profile definition for {profile_name!r}"

    resp = client.post(
        "/admin/profiles",
        json={
            "profile_name": profile_name,
            "tenant_id": tenant_id,
            "type": profile_def.get("type", "video"),
            "description": profile_def.get("description", ""),
            "schema_name": profile_def.get("schema_name", profile_name),
            "embedding_model": profile_def.get("embedding_model", ""),
            "pipeline_config": profile_def.get("pipeline_config", {}),
            "strategies": profile_def.get("strategies", {}),
            "embedding_type": profile_def.get("embedding_type", "multi_vector"),
            "schema_config": profile_def.get("schema_config", {}),
            "model_specific": profile_def.get("model_specific"),
            "deploy_schema": True,
        },
        timeout=60,
    )
    assert resp.status_code in (200, 201, 409), resp.text

    resp = client.post(
        f"/admin/profiles/{profile_name}/deploy",
        json={"tenant_id": tenant_id, "force": False},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text


@pytest.mark.e2e
class TestRoutingPipeline:
    """Scenario 1: Routing agent routes query via the gateway pipeline.

    In the A2A architecture, both 'gateway' and 'routing' capabilities route
    through _execute_gateway_task in the dispatcher. The gateway_agent no
    longer does entity extraction or query enhancement inline -- those are
    handled by dedicated upstream A2A agents via the orchestrator.

    The gateway_agent is the new entry point. See test_a2a_gateway_e2e.py
    for comprehensive gateway tests.
    """

    def test_routing_decision_structure(self):
        """Routing agent returns success via the gateway pipeline."""
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
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        # Across the calibrator's whole GLiNER range the only entity is
        # video_content ("videos", 0.339 at 0.15; none at 0.5) and the keyword
        # cue is video, so the modality and generation type never move.
        assert gw["modality"] == "video", gw
        assert gw["generation_type"] == "raw_results", gw
        if gw["complexity"] == "simple":
            # In the new architecture, the response comes from the gateway
            # pipeline. The agent field will be gateway_agent on simple routes.
            assert data["agent"] == "gateway_agent"
            assert "downstream_result" in data
            assert gw["confidence"] >= gw["fast_path_confidence_threshold"]
        else:
            assert_orchestrated(data, query, gw)

    def test_routing_no_longer_returns_inline_entities(self):
        """Routing agent no longer returns entities/enhanced_query at top level.

        In the A2A architecture, entity extraction and query enhancement are
        handled by dedicated agents invoked by the orchestrator. The routing
        agent response now has a gateway-style structure with downstream_result
        or orchestration_result.
        """
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "Find videos about Tesla cars in San Francisco",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

        # The response should have gateway-style keys, not old inline keys
        has_new_shape = (
            "gateway" in data
            or "downstream_result" in data
            or "orchestration_result" in data
        )
        assert has_new_shape, (
            f"Routing should produce gateway-style response, "
            f"got keys: {list(data.keys())}"
        )
        # Old inline entity/enhancement fields must NOT appear at the top level
        assert "entities" not in data, (
            "entities should not be at the top level in the A2A architecture"
        )
        assert "enhanced_query" not in data, (
            "enhanced_query should not be at the top level in the A2A architecture"
        )
        # If recommended_agent is surfaced, it must be a known valid agent
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

    def test_routing_executes_downstream(self):
        query = "search for animal videos"
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
        assert data["status"] == "success"
        gw = data["gateway"]
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        # GLiNER tags "animal videos" video_content at 0.709 across the whole
        # 0.15-0.5 range and nothing else, so modality and confidence are fixed.
        assert gw["modality"] == "video", gw
        assert gw["generation_type"] == "raw_results", gw
        assert gw["confidence"] == pytest.approx(0.709, abs=0.001), gw
        if gw["complexity"] == "simple":
            # Content assertion: animal video query should produce search results
            downstream = data["downstream_result"]
            assert downstream["status"] == "success", downstream
            assert "results" in downstream, (
                f"Missing results, keys: {list(downstream.keys())}"
            )
            results = downstream["results"]
            assert len(results) > 0, (
                "Search for 'animal videos' should return results from ingested data"
            )
            score_keys = ("score", "relevance", "relevance_score", "_score")
            score_key = next((k for k in score_keys if k in results[0]), None)
            if score_key is not None:
                scores = [r[score_key] for r in results]
                assert scores == sorted(scores, reverse=True), (
                    f"Results should be ranked by {score_key} descending, got: {scores}"
                )
        else:
            assert_orchestrated(data, query, gw)


@pytest.mark.e2e
class TestQueryEnhancementViaGateway:
    """Scenarios 2-3: Query enhancement and entity extraction are now handled
    by dedicated A2A agents via the orchestrator pipeline.

    The gateway_agent no longer returns enhanced_query or entities at the
    top level. These tests verify the gateway pipeline works end-to-end.
    """

    def test_gateway_processes_query_successfully(self):
        """Gateway pipeline processes queries without errors."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "ML transformer videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_gateway_classifies_entity_rich_queries(self):
        """Entity-rich queries are classified and routed by the gateway."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "Obama speaking at MIT about climate change",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        # Gateway should classify and route the query
        has_routing = (
            "gateway" in data
            or "orchestration_result" in data
            or "downstream_result" in data
        )
        assert has_routing, (
            f"Gateway should classify and route, got keys: {list(data.keys())}"
        )

    def test_gateway_confidence_in_range(self):
        """Complex queries stay on the orchestrator with the live thresholds."""
        query = "find me detailed analysis of deep learning architectures"
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
        # "analysis" is a complexity keyword, so this is complex at any threshold.
        assert gw["complexity"] == "complex", gw
        assert_orchestrated(data, query, gw)


@pytest.mark.e2e
class TestOrchestration:
    """Scenarios 4-5: Gateway triggers orchestration for complex queries."""

    def test_complex_query_triggers_orchestration_or_downstream(self):
        """Complex queries route through the gateway to orchestration or
        direct downstream."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "Find videos about machine learning and write a detailed report",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] in ("gateway_agent", "orchestrator_agent")

    def test_multi_turn_routing(self):
        """Multi-turn routing preserves conversation context."""
        session_id = str(uuid.uuid4())

        transport = httpx.HTTPTransport(retries=2)
        with httpx.Client(
            base_url=RUNTIME, timeout=900.0, transport=transport
        ) as client:
            resp1 = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "search for cat videos",
                    "context": {"tenant_id": TENANT_ID, "session_id": session_id},
                    "top_k": 3,
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["status"] == "success"

            resp2 = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": "show me longer ones",
                    "context": {"tenant_id": TENANT_ID, "session_id": session_id},
                    "top_k": 3,
                    "conversation_history": [
                        {"role": "user", "content": "search for cat videos"},
                        {"role": "agent", "content": "Found cat video results"},
                    ],
                },
            )

        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == "success"


@pytest.mark.e2e
class TestSearchAPI:
    """Scenario 7: Search with profile/strategy selection and result validation."""

    def test_list_strategies(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get("/search/strategies", params={"tenant_id": TENANT_ID})

        assert resp.status_code == 200
        data = resp.json()
        strategies = data["strategies"]
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        # Strategies must be the real per-profile set POST /search accepts,
        # not the old hardcoded advertisement.
        assert "default" in strategies

    def test_list_profiles(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(
                "/search/profiles",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        profiles = data.get("profiles", data) if isinstance(data, dict) else data
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        profile_names = [p["name"] if isinstance(p, dict) else p for p in profiles]
        assert PROFILE in profile_names, (
            f"Expected {PROFILE} in profiles, got: {profile_names}"
        )

    def test_search_with_explicit_profile(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "animals in nature",
                    "profile": PROFILE,
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "results_count" in data
        assert isinstance(data["results_count"], int)
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_result_fields(self):
        """Verify search results contain expected content fields."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "sports activities",
                    "profile": PROFILE,
                    "top_k": 5,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["results_count"] >= 1, (
            "Search for 'sports activities' should return results from ingested data"
        )
        result = data["results"][0]
        assert isinstance(result, dict)
        assert len(result) > 1, (
            f"Result should have multiple fields, got: {list(result.keys())}"
        )

    def test_search_response_echoes_params(self):
        """Verify response includes the query, profile, and strategy sent."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "cooking video",
                    "profile": PROFILE,
                    "strategy": "default",
                    "top_k": 3,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "cooking video"
        assert data["profile"] == PROFILE

    def test_search_with_different_strategy(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/search/",
                json={
                    "query": "outdoor activities",
                    "profile": PROFILE,
                    "strategy": "float_float",
                    "top_k": 3,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "results_count" in data

    def test_search_rerank_validation(self):
        """POST /search/rerank without query returns 400."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/search/rerank",
                json={"results": [], "strategy": "learned"},
            )
        assert resp.status_code == 400, (
            f"Rerank without query should return 400, got {resp.status_code}"
        )

    def test_search_rerank_unknown_strategy(self):
        """POST /search/rerank with unknown strategy returns 400."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/search/rerank",
                json={
                    "query": "test",
                    "results": [{"id": "1", "score": 0.5}],
                    "strategy": "nonexistent_strategy",
                },
            )
        assert resp.status_code == 400


@pytest.mark.e2e
class TestProfileCRUD:
    """Profile management: create, list, get, update, delete."""

    def test_list_profiles_for_tenant(self):
        """GET /admin/profiles returns profile list structure for tenant."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(
                "/admin/profiles",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "profiles" in data
        assert "total_count" in data
        assert "tenant_id" in data
        assert isinstance(data["profiles"], list)
        assert isinstance(data["total_count"], int)
        assert data["total_count"] >= 0

    def test_create_then_get_profile(self):
        """Create a profile via admin API, then GET it by name."""
        profile_name = unique_id("e2e_get")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            try:
                create_resp = client.post(
                    "/admin/profiles",
                    json={
                        "profile_name": profile_name,
                        "tenant_id": TENANT_ID,
                        "type": "video",
                        "description": "E2E get test profile",
                        "schema_name": PROFILE,
                        "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                        "embedding_type": "multi_vector",
                        "deploy_schema": False,
                    },
                )
                assert create_resp.status_code == 201, (
                    f"Create profile failed: {create_resp.text}"
                )

                resp = client.get(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["profile_name"] == profile_name
                assert data["tenant_id"] == TENANT_ID
                assert "schema_name" in data
                assert "embedding_model" in data
                assert "type" in data
                assert "version" in data
                assert isinstance(data["version"], int)
            finally:
                client.delete(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )

    def test_get_nonexistent_profile_returns_404(self):
        """GET /admin/profiles/{name} returns 404 for missing profile."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(
                "/admin/profiles/nonexistent_profile_xyz",
                params={"tenant_id": TENANT_ID},
            )
        assert resp.status_code == 404

    def test_profile_create_update_delete_lifecycle(self):
        """Full lifecycle: create profile → get → update → delete."""
        profile_name = unique_id("e2e_profile")

        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            try:
                resp = client.post(
                    "/admin/profiles",
                    json={
                        "profile_name": profile_name,
                        "tenant_id": TENANT_ID,
                        "type": "video",
                        "description": "E2E test profile",
                        "schema_name": PROFILE,
                        "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                        "embedding_type": "multi_vector",
                        "deploy_schema": False,
                    },
                )
                assert resp.status_code == 201, f"Create profile failed: {resp.text}"
                data = resp.json()
                assert data["profile_name"] == profile_name
                assert data["tenant_id"] == TENANT_ID
                assert "version" in data

                resp = client.get(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )
                assert resp.status_code == 200
                assert resp.json()["profile_name"] == profile_name

                resp = client.put(
                    f"/admin/profiles/{profile_name}",
                    json={
                        "tenant_id": TENANT_ID,
                        "description": "Updated E2E test profile",
                    },
                )
                assert resp.status_code == 200
                update_data = resp.json()
                assert "description" in update_data["updated_fields"]

            finally:
                client.delete(
                    f"/admin/profiles/{profile_name}",
                    params={"tenant_id": TENANT_ID},
                )


@pytest.mark.e2e
class TestSystemStats:
    """GET /admin/system/stats returns system statistics."""

    def test_system_stats(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/admin/system/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "registered_backends" in data
        assert isinstance(data["registered_backends"], list)
        assert "timestamp" in data


@pytest.mark.e2e
class TestAgentOperations:
    """Agent registration, capability search, unregistration, and process."""

    def test_capability_based_discovery(self):
        """GET /agents/by-capability/{cap} returns matching agents."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/by-capability/search")

        assert resp.status_code == 200
        data = resp.json()
        assert data["capability"] == "search"
        assert "count" in data
        assert isinstance(data["agents"], list)

    def test_agent_upload_endpoint_removed(self):
        """POST /agents/{name}/upload was deleted.

        The endpoint was a 501 stub with no implementation path. File
        uploads have a real home at POST /ingestion/upload, so the stub
        is gone. Hitting the old URL must now produce 404 / 405."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/agents/gateway_agent/upload",
                files={"file": ("test.txt", b"test content", "text/plain")},
            )
        assert resp.status_code in (404, 405)

    def test_unregister_nonexistent_agent_returns_404(self):
        """DELETE /agents/{name} returns 404 for unknown agent."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.delete("/agents/nonexistent_agent_xyz")
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "agent_name,query",
        [
            ("text_analysis_agent", "analyze this text about video processing"),
            ("summarizer_agent", "summarize the key findings"),
            ("detailed_report_agent", "write a report on search results"),
        ],
    )
    def test_agent_process_response_structure(self, agent_name, query):
        """Each agent returns status=success with agent name."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                f"/agents/{agent_name}/process",
                json={
                    "agent_name": agent_name,
                    "query": query,
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200, f"{agent_name} process failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == agent_name

    def test_process_nonexistent_agent_returns_404(self):
        """POST /agents/{name}/process returns 404 for unknown agent."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/agents/nonexistent_agent_xyz/process",
                json={
                    "agent_name": "nonexistent_agent_xyz",
                    "query": "test",
                    "context": {"tenant_id": TENANT_ID},
                },
            )
        assert resp.status_code == 404


@pytest.mark.e2e
class TestSyntheticDataAPI:
    """Synthetic data generation endpoints."""

    @pytest.fixture(scope="class", autouse=True)
    def _seed_document_corpus(self):
        caption_paths = sorted(CAPTION_CORPUS_DIR.glob("*.txt"))[:CAPTION_CORPUS_LIMIT]
        assert len(caption_paths) >= CAPTION_CORPUS_LIMIT, (
            f"expected at least {CAPTION_CORPUS_LIMIT} caption fixtures in "
            f"{CAPTION_CORPUS_DIR}, found {len(caption_paths)}"
        )

        for caption_path in caption_paths:
            _ensure_sample_content_ingested(
                caption_path,
                profile=DOCUMENT_PROFILE,
                media_type="text/plain",
            )

    @staticmethod
    def _seeded_video_fixture_results(client: httpx.Client) -> list[dict]:
        """Exact seeded video docs the synthetic sampler should ground on."""
        seeded_video_paths = (
            SAMPLE_VIDEO_PATH,
            SECOND_SAMPLE_VIDEO_PATH,
        )
        results = []
        for video_path in seeded_video_paths:
            _ensure_sample_content_ingested(
                video_path,
                profile=PROFILE,
                media_type="video/mp4",
            )
            content_id = _content_sha256(video_path)
            response = client.post(
                "/search/",
                json={
                    "query": content_id,
                    "profile": PROFILE,
                    "strategy": "default",
                    "top_k": 1000,
                    "tenant_id": TENANT_ID,
                },
            )
            assert response.status_code == 200, response.text
            matches = _matching_sample_results(
                response.json(),
                content_id=content_id,
                tenant_id=TENANT_ID,
                profile=PROFILE,
                suffix=video_path.suffix,
                media_type="video",
            )
            assert matches, f"expected exact seeded video results for {video_path.name}"
            results.extend(matches)
        return results

    @pytest.fixture
    def workflow_seeded_tenant(self):
        org_id = unique_id("workflow_seeded")
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
                _deploy_profile_for_tenant(client, WORKFLOW_PROFILE, tenant_id)
                _deploy_profile_for_tenant(client, DOCUMENT_PROFILE, tenant_id)

                seeded_video_content_id = _ensure_sample_content_ingested(
                    SAMPLE_VIDEO_PATH,
                    profile=WORKFLOW_PROFILE,
                    media_type="video/mp4",
                    tenant_id=tenant_id,
                )
                seeded_documents = _ingest_sample_documents(tenant_id=tenant_id)
                yield tenant_id, seeded_video_content_id, seeded_documents
            finally:
                try:
                    client.delete(f"/admin/tenants/{tenant_id}")
                except httpx.HTTPError:
                    pass
                try:
                    client.delete(f"/admin/organizations/{org_id}")
                except httpx.HTTPError:
                    pass

    def test_synthetic_health(self):
        """GET /synthetic/health returns healthy status."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/health")

        assert resp.status_code == 200
        data = resp.json()
        assert set(data) == {"status", "service", "generators", "optimizers"}
        assert data == {
            "status": "healthy",
            "service": "synthetic-data-generation",
            "generators": data["generators"],
            "optimizers": 7,
        }
        assert isinstance(data["generators"], int)
        assert 0 <= data["generators"] <= 7

    def test_list_optimizers(self):
        """GET /synthetic/optimizers returns optimizer registry."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/optimizers")

        assert resp.status_code == 200
        data = resp.json()
        assert data == {
            "entity_extraction": (
                "Entity extraction. Learns to extract typed entities (and "
                "relationships) from query text."
            ),
            "query_enhancement": (
                "Query enhancement. Learns to broaden a user query with expansion "
                "terms and synonyms drawn from related content."
            ),
            "routing": (
                "Advanced routing with entity extraction. Learns to route queries "
                "based on extracted entities, relationships, and semantic "
                "understanding."
            ),
            "workflow": (
                "Workflow execution pattern optimization. Learns optimal agent "
                "sequences and parallel execution strategies for complex multi-step "
                "tasks."
            ),
            "profile": (
                "ProfileSelectionAgent optimization. Learns which Vespa backend "
                "profile best matches a query's modality, complexity, and intent."
            ),
            "unified": (
                "Unified routing and orchestration optimization. Combines routing "
                "decisions with workflow planning for end-to-end optimization."
            ),
            "cross_modal": (
                "Cross-modal fusion optimization. Generates queries that span video "
                "+ audio + text modalities so retrieval profiles with multi-vector "
                "fusion get exercised together."
            ),
        }

    def test_get_optimizer_detail(self):
        """GET /synthetic/optimizers/{name} returns optimizer info."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/optimizers/profile")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "profile"
        assert data["schema"] == "ProfileSelectionExampleSchema"
        assert data["generator"] == "ProfileGenerator"
        assert data["backend_strategy"] == "diverse"
        assert data["requires_agent_mapping"] is False
        assert data["defaults"] == {
            "sample_size": 200,
            "generation_count": 100,
        }

    def test_get_nonexistent_optimizer_returns_404(self):
        """GET /synthetic/optimizers/{name} returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/synthetic/optimizers/nonexistent_xyz")
        assert resp.status_code == 404

    def test_profile_generation_uses_ingested_tenant_content(self):
        expected_available_profiles = _expected_available_profile_names(TENANT_ID)
        profile_types = _profile_type_map()
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            self._seeded_video_fixture_results(client)
            resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "profile",
                    "count": 2,
                    "vespa_sample_size": 2,
                    "strategy": "diverse",
                    "max_profiles": 1,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200, f"Profile generation failed: {resp.text}"
        data = resp.json()
        assert set(data) == {
            "optimizer",
            "schema_name",
            "count",
            "selected_profiles",
            "profile_selection_reasoning",
            "data",
            "metadata",
        }
        assert data["optimizer"] == "profile"
        assert data["schema_name"] == "ProfileSelectionExampleSchema"
        assert data["count"] == 2
        assert len(data["selected_profiles"]) == 1
        selected_profile = data["selected_profiles"][0]
        assert selected_profile in expected_available_profiles
        assert profile_types[selected_profile] == "video"
        _assert_synthetic_metadata(
            data["metadata"],
            backend_query_strategy="diverse",
            sampled_content_count=2,
            target_count=2,
            vespa_sample_size=2,
        )
        assert len(data["data"]) == 2
        profile_fields = {
            "query",
            "available_profiles",
            "selected_profile",
            "reasoning",
            "query_intent",
            "modality",
            "complexity",
        }
        assert all(set(example) == profile_fields for example in data["data"])
        sampled_content = data["metadata"]["sampled_content"]
        saliency = TopicSaliency.from_records(sampled_content)
        topics = [
            extract_topic(record, saliency=saliency) for record in sampled_content
        ]
        assert all(topic is not None for topic in topics), sampled_content
        assert [example["query"] for example in data["data"]] == [
            f"find a video frame showing {topic}" for topic in topics
        ], [
            (example["query"], record["description"])
            for example, record in zip(data["data"], sampled_content, strict=True)
        ]

        actual_available_profiles = [
            example["available_profiles"].split(",") for example in data["data"]
        ]
        assert actual_available_profiles == [expected_available_profiles] * len(
            data["data"]
        )
        assert {example["selected_profile"] for example in data["data"]} == {
            selected_profile
        }
        assert {example["modality"] for example in data["data"]} == {"video"}
        # complexity is an LM judgment, so no single value is pinnable. The
        # vocabulary contract is enforced at the HTTP boundary: the schema
        # forbids extras and types complexity as a Literal, so a value outside
        # {simple, medium, complex} or any field drift raises here.
        assert [
            ProfileSelectionExampleSchema.model_validate(example).complexity
            for example in data["data"]
        ] == [example["complexity"] for example in data["data"]]
        # query_intent is the labeler's LM judgment, so no single value is
        # pinnable; the vocabulary contract is enforced at the HTTP boundary
        # by the Literal-typed schema, exactly as for complexity above.
        assert [
            ProfileSelectionExampleSchema.model_validate(example).query_intent
            for example in data["data"]
        ] == [example["query_intent"] for example in data["data"]]

    def test_generate_synthetic_data(self):
        """POST /synthetic/generate creates real synthetic training examples."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            self._seeded_video_fixture_results(client)
            resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "routing",
                    "count": 5,
                    "vespa_sample_size": 5,
                    "strategy": "entity_rich",
                    "max_profiles": 1,
                    "tenant_id": TENANT_ID,
                },
            )
            assert resp.status_code == 200, f"Synthetic generation failed: {resp.text}"
            data = resp.json()
            sampled_content = data["metadata"]["sampled_content"]
            source_texts = []
            missing_source_texts = []
            for result in sampled_content:
                assert isinstance(result, dict), result
                source_text = topic_source_text(result)
                if source_text is None:
                    missing_source_texts.append(result)
                    continue
                source_texts.append(source_text)
            assert missing_source_texts == [], missing_source_texts
            assert len(set(source_texts)) == len(sampled_content)
            saliency = TopicSaliency.from_records(sampled_content)
            topics = [
                extract_topic(record, saliency=saliency) for record in sampled_content
            ]
            # Every sampled record yields a topic; name any that did not.
            assert [
                record
                for record, topic in zip(sampled_content, topics)
                if topic is None
            ] == []
            # Distinct sources never collapse onto one topic.
            assert sorted(set(topics)) == sorted(topics)

            expected_extractions = []
            for record, topic in zip(sampled_content, topics):
                source_text = topic_source_text(record)
                assert source_text is not None, record
                # The topic is a contiguous span of its own source text ...
                assert topic in source_text, (
                    f"topic {topic!r} must be a span of source text {source_text!r}"
                )
                # ... and never that source's leading words, compared over the
                # same width as the topic itself.
                assert topic != " ".join(source_text.split()[: len(topic.split())]), (
                    f"topic is the source prefix: {topic!r} from {source_text!r}"
                )
                extraction_response = client.post(
                    "/agents/entity_extraction_agent/process",
                    json={
                        "agent_name": "entity_extraction_agent",
                        "query": topic,
                        "context": {"tenant_id": TENANT_ID},
                    },
                )
                assert extraction_response.status_code == 200, extraction_response.text
                extraction = extraction_response.json()
                assert extraction["query"] == topic
                assert extraction["path_used"] == "dspy"
                assert all(entity["text"] in topic for entity in extraction["entities"])
                seen_exact: dict[str, str] = {}
                seen_casefold: set[str] = set()
                canonical_entities = []
                for entity in extraction["entities"]:
                    if entity["text"] in seen_exact:
                        assert seen_exact[entity["text"]] == entity["type"]
                        continue
                    seen_exact[entity["text"]] = entity["type"]
                    if entity["text"].casefold() in seen_casefold:
                        continue
                    seen_casefold.add(entity["text"].casefold())
                    canonical_entities.append(
                        {"text": entity["text"], "type": entity["type"]}
                    )
                expected_extractions.append(
                    {
                        "query": topic,
                        "source_text": source_text,
                        "entities": canonical_entities,
                        "relationships": [
                            {
                                "source": relationship["subject"],
                                "target": relationship["object"],
                                "type": relationship["relation"],
                            }
                            for relationship in extraction["relationships"]
                        ],
                    }
                )
            expected_grounded_examples = [
                {
                    "query": item["query"],
                    "entities": item["entities"],
                    "relationships": item["relationships"],
                }
                for item in expected_extractions
                if item["entities"]
            ]

        assert resp.status_code == 200, f"Synthetic generation failed: {resp.text}"
        data = resp.json()
        assert set(data) == {
            "optimizer",
            "schema_name",
            "count",
            "selected_profiles",
            "profile_selection_reasoning",
            "data",
            "metadata",
        }
        expected_available_profiles = _expected_available_profile_names(TENANT_ID)
        profile_types = _profile_type_map()
        assert data["optimizer"] == "routing"
        assert data["schema_name"] == "RoutingExperienceSchema"
        assert len(data["selected_profiles"]) == 1
        selected_profile = data["selected_profiles"][0]
        assert selected_profile in expected_available_profiles
        assert profile_types[selected_profile] == "video"
        generation = _assert_synthetic_metadata_fields(
            data["metadata"],
            backend_query_strategy="entity_rich",
            sampled_content_count=5,
            target_count=5,
            vespa_sample_size=5,
        )
        # The fixture is five frames of one clip; adjacent captions can share a
        # topic, so the quota is filled only when the content supports it and
        # every shortfall is a recorded content drop, never a hidden failure.
        assert data["count"] == len(data["data"]) == generation["returned_count"]
        assert generation["returned_count"] + generation["shortfall_count"] == 5
        assert generation["surplus_exhausted"] is (generation["shortfall_count"] > 0)
        content_drop_prefixes = (
            "RoutingGenerator generated duplicate canonical label (",
            "Failed to generate valid entity query after ",
            "EntityExtractionGenerator generated 0 unique grounded examples",
        )
        assert [
            drop["reason"].startswith(content_drop_prefixes)
            for drop in generation["dropped_examples"]
        ] == [True] * generation["dropped_count"]
        assert generation["returned_count"] == len(
            {(example["query"], example["chosen_agent"]) for example in data["data"]}
        )
        actual_grounded_examples = [
            {
                "query": example["query"],
                "entities": example["entities"],
                "relationships": example["relationships"],
            }
            for example in data["data"]
        ]
        assert actual_grounded_examples == expected_grounded_examples
        sampled_corpus = " ".join(
            " ".join(str(value).split())
            for result in sampled_content
            for value in result["metadata"].values()
            if isinstance(value, str)
        ).casefold()
        routing_fields = {
            "query",
            "entities",
            "relationships",
            "enhanced_query",
            "chosen_agent",
            "routing_confidence",
            "search_quality",
            "agent_success",
            "user_satisfaction",
            "processing_time",
            "reward",
            "timestamp",
            "metadata",
        }
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            for example in data["data"]:
                assert set(example) == routing_fields
                gateway_response = client.post(
                    "/agents/gateway_agent/process",
                    json={
                        "agent_name": "gateway_agent",
                        "query": example["query"],
                        "context": {"tenant_id": TENANT_ID},
                    },
                )
                assert gateway_response.status_code == 200, gateway_response.text
                gateway = gateway_response.json()["gateway"]
                assert example["chosen_agent"] == gateway["routed_to"]
                assert example["routing_confidence"] == gateway["confidence"]
                assert {
                    "entities": example["entities"],
                    "relationships": example["relationships"],
                } in expected_extractions
                assert all(
                    set(entity) == {"text", "type"} for entity in example["entities"]
                )
                assert len(example["entities"]) == len(
                    {
                        (entity["text"].casefold(), entity["type"])
                        for entity in example["entities"]
                    }
                )
                assert all(
                    entity["text"].casefold() in sampled_corpus
                    for entity in example["entities"]
                )
                entity_words = [
                    word
                    for entity in example["entities"]
                    for word in re.findall(r"\w+", entity["text"])
                    if len(word) > 3
                ] or [entity["text"] for entity in example["entities"]]
                assert any(
                    re.search(
                        rf"(?<!\w){re.escape(word)}(?!\w)",
                        example["query"],
                        flags=re.IGNORECASE,
                    )
                    for word in entity_words
                )
                assert all(
                    f"{entity['text']}({entity['type']})".casefold()
                    in example["enhanced_query"].casefold()
                    for entity in example["entities"]
                )
                generation_metadata = example["metadata"]["_generation_metadata"]
                assert set(generation_metadata) == {
                    "retry_count",
                    "max_retries",
                    "reasoning",
                }
                assert type(generation_metadata["retry_count"]) is int
                assert generation_metadata["retry_count"] in {0, 1, 2}
                assert type(generation_metadata["max_retries"]) is int
                assert generation_metadata["max_retries"] == 3
                assert 20 <= len(generation_metadata["reasoning"]) <= 2_000
                assert "available_profiles" not in example
                assert "workflow_id" not in example

    def test_generate_synthetic_data_cross_modal(self):
        """POST /synthetic/generate with cross_modal optimizer."""
        expected_available_profiles = _expected_available_profile_names(TENANT_ID)
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "cross_modal",
                    "count": 2,
                    "vespa_sample_size": 2,
                    "strategy": "multi_modal_sequences",
                    "max_profiles": 2,
                    "tenant_id": TENANT_ID,
                },
            )

        assert resp.status_code == 200, f"Cross-modal generation failed: {resp.text}"
        data = resp.json()
        assert set(data) == {
            "optimizer",
            "schema_name",
            "count",
            "selected_profiles",
            "profile_selection_reasoning",
            "data",
            "metadata",
        }
        assert data["optimizer"] == "cross_modal"
        assert data["schema_name"] == "ProfileSelectionExampleSchema"
        assert data["count"] == 2
        assert len(data["data"]) == 2
        profile_types = _profile_type_map()
        selected_profiles = data["selected_profiles"]
        assert len(selected_profiles) == 2
        assert len(set(selected_profiles)) == 2
        video_profiles = [
            profile
            for profile in selected_profiles
            if profile_types[profile] == "video"
        ]
        assert len(video_profiles) == 1
        video_profile = video_profiles[0]
        assert all(
            profile in expected_available_profiles for profile in selected_profiles
        )
        assert len({profile_types[profile] for profile in selected_profiles}) == 2
        _assert_synthetic_metadata(
            data["metadata"],
            backend_query_strategy="multi_modal_sequences",
            sampled_content_count=2,
            target_count=2,
            vespa_sample_size=2,
        )
        video_modality = profile_types[video_profile]
        other_profile = next(
            profile for profile in selected_profiles if profile != video_profile
        )
        other_modality = profile_types[other_profile]
        video_first_pattern = re.compile(
            rf"^find (?P<video_topic>.+) in {re.escape(video_modality)} content together with "
            rf"(?P<other_topic>.+) in {re.escape(other_modality)} content$"
        )
        other_first_pattern = re.compile(
            rf"^find (?P<other_topic>.+) in {re.escape(other_modality)} content together with "
            rf"(?P<video_topic>.+) in {re.escape(video_modality)} content$"
        )
        orderings = {}
        for example in data["data"]:
            video_first = video_first_pattern.match(example["query"])
            other_first = other_first_pattern.match(example["query"])
            assert (video_first is None) != (other_first is None), example["query"]
            match = video_first or other_first
            ordering = "video_first" if video_first else "other_first"
            assert ordering not in orderings, example["query"]
            orderings[ordering] = (match["video_topic"], match["other_topic"])
        assert set(orderings) == {"video_first", "other_first"}
        video_topic, other_topic = orderings["video_first"]
        assert orderings["other_first"] == (video_topic, other_topic)
        assert 1 <= len(video_topic.split()) <= 20, video_topic
        assert 1 <= len(other_topic.split()) <= 20, other_topic
        profile_fields = {
            "query",
            "available_profiles",
            "selected_profile",
            "reasoning",
            "query_intent",
            "modality",
            "complexity",
        }
        for example in data["data"]:
            assert set(example) == profile_fields
            available_profiles = example["available_profiles"].split(",")
            assert available_profiles == expected_available_profiles
            assert example["selected_profile"] in selected_profiles
            assert example["modality"] == profile_types[example["selected_profile"]]
            # query_intent is the labeler's LM judgment; the vocabulary is
            # enforced at the boundary by the Literal-typed schema.
            assert (
                ProfileSelectionExampleSchema.model_validate(example).query_intent
                == example["query_intent"]
            )
            assert "chosen_agent" not in example
            assert "workflow_id" not in example
        assert len({example["query"] for example in data["data"]}) == 2
        assert {example["modality"] for example in data["data"]} == {video_modality}
        # complexity is an LM judgment, so no single value is pinnable. The
        # vocabulary contract is enforced at the HTTP boundary: the schema
        # forbids extras and types complexity as a Literal, so a value outside
        # {simple, medium, complex} or any field drift raises here.
        assert [
            ProfileSelectionExampleSchema.model_validate(example).complexity
            for example in data["data"]
        ] == [example["complexity"] for example in data["data"]]

    def test_generate_workflow_ids_are_unique_and_schema_specific(
        self, workflow_seeded_tenant
    ):
        tenant_id, seeded_video_content_id, seeded_documents = workflow_seeded_tenant
        expected_available_profiles = _expected_available_profile_names(tenant_id)
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            agents_response = client.get("/agents/")
            resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "workflow",
                    "count": 4,
                    "vespa_sample_size": 2,
                    "strategy": "multi_modal_sequences",
                    "max_profiles": 2,
                    "tenant_id": tenant_id,
                },
            )

        assert agents_response.status_code == 200, agents_response.text
        registered_agents = set(agents_response.json()["agents"])
        assert CANONICAL_WORKFLOW_AGENTS <= registered_agents
        assert resp.status_code == 200, f"Workflow generation failed: {resp.text}"
        data = resp.json()
        assert set(data) == {
            "optimizer",
            "schema_name",
            "count",
            "selected_profiles",
            "profile_selection_reasoning",
            "data",
            "metadata",
        }
        assert data["optimizer"] == "workflow"
        assert data["schema_name"] == "WorkflowExecutionSchema"
        assert data["count"] == 4
        assert len(data["data"]) == 4
        selected_profiles = data["selected_profiles"]
        assert len(selected_profiles) == 2
        assert len(set(selected_profiles)) == 2
        profile_types = _profile_type_map()
        assert (
            sum(1 for profile in selected_profiles if profile_types[profile] == "video")
            == 1
        )
        assert all(
            profile in expected_available_profiles for profile in selected_profiles
        )
        assert len({_profile_family(profile) for profile in selected_profiles}) == 2
        _assert_synthetic_metadata(
            data["metadata"],
            backend_query_strategy="multi_modal_sequences",
            sampled_content_count=2,
            target_count=4,
            vespa_sample_size=2,
        )
        sampled_content = data["metadata"]["sampled_content"]
        profile_types = _profile_type_map()
        video_records = [
            record
            for record in sampled_content
            if profile_types[record["profile_name"]] == "video"
        ]
        document_records = [
            record
            for record in sampled_content
            if profile_types[record["profile_name"]] == "document"
        ]
        assert len(video_records) == 1, sampled_content
        assert len(document_records) == 1, sampled_content
        video_record = video_records[0]
        document_record = document_records[0]
        seeded_document_titles = set(seeded_documents)
        seeded_document_ids = set(seeded_documents.values())
        assert video_record["source_id"] == seeded_video_content_id
        assert document_record["source_id"] in seeded_document_ids
        document_title = next(
            title
            for title, content_id in seeded_documents.items()
            if content_id == document_record["source_id"]
        )
        assert document_title in seeded_document_titles
        for record in sampled_content:
            assert topic_source_text(record) is not None, sampled_content
        saliency = TopicSaliency.from_records(sampled_content)
        video_topic = extract_topic(video_record, saliency=saliency)
        document_topic = extract_topic(document_record, saliency=saliency)
        assert video_topic is not None, sampled_content
        assert document_topic is not None, sampled_content
        assert video_topic != document_topic
        queries = [example["query"] for example in data["data"]]
        assert queries == [
            f"find {video_topic}",
            f"summarize {video_topic}",
            f"analyze {video_topic} and generate report",
            f"find {document_topic}",
        ]
        workflow_ids = [example["workflow_id"] for example in data["data"]]
        assert len(set(workflow_ids)) == 4
        assert all(
            len(workflow_id) == 51
            and workflow_id.startswith("synthetic_workflow_")
            and workflow_id.removeprefix("synthetic_workflow_").isalnum()
            for workflow_id in workflow_ids
        )
        workflow_fields = {
            "workflow_id",
            "query",
            "query_type",
            "execution_time",
            "success",
            "agent_sequence",
            "task_count",
            "parallel_efficiency",
            "confidence_score",
            "user_satisfaction",
            "error_details",
            "timestamp",
            "metadata",
        }
        for example in data["data"]:
            assert set(example) == workflow_fields
            assert example["task_count"] == len(example["agent_sequence"])
            assert set(example["agent_sequence"]) <= CANONICAL_WORKFLOW_AGENTS
            assert (
                example["agent_sequence"][0]
                == PRIMARY_AGENT_BY_QUERY_TYPE[example["query_type"]]
            )
            assert "selected_profile" not in example
            assert "chosen_agent" not in example

    def test_generate_rejects_unknown_optimizer_and_strategy(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            optimizer_resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "missing_optimizer",
                    "count": 1,
                    "tenant_id": TENANT_ID,
                },
            )
            strategy_resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "profile",
                    "count": 1,
                    "strategy": "tenant_ambiguous_wildcard",
                    "tenant_id": TENANT_ID,
                },
            )
            plural_strategy_resp = client.post(
                "/synthetic/generate",
                json={
                    "optimizer": "profile",
                    "count": 1,
                    "strategies": ["diverse"],
                    "tenant_id": TENANT_ID,
                },
            )

        assert optimizer_resp.status_code == 400
        assert optimizer_resp.json()["detail"].startswith(
            "Unknown optimizer: 'missing_optimizer'. Available:"
        )
        assert strategy_resp.status_code == 422
        strategy_error = strategy_resp.json()["detail"][0]
        assert strategy_error["loc"] == ["body", "strategy"]
        assert "Unsupported sampling strategy" in strategy_error["msg"]
        assert plural_strategy_resp.status_code == 422
        plural_error = plural_strategy_resp.json()["detail"][0]
        assert plural_error["loc"] == ["body", "strategies"]
        assert plural_error["type"] == "extra_forbidden"


@pytest.mark.e2e
class TestEventOperations:
    """Event queue cancel and offset endpoints."""

    def test_cancel_nonexistent_workflow_returns_404(self):
        """POST /events/workflows/{id}/cancel returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/events/workflows/nonexistent_wf_xyz/cancel",
                json={"reason": "test"},
            )
        assert resp.status_code == 404

    def test_cancel_nonexistent_ingestion_returns_404(self):
        """POST /events/ingestion/{id}/cancel returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post(
                "/events/ingestion/nonexistent_job_xyz/cancel",
                json={"reason": "test"},
            )
        assert resp.status_code == 404

    def test_queue_offset_not_found(self):
        """GET /events/queues/{id}/offset returns 404 for unknown."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/events/queues/nonexistent_q_xyz/offset")
        assert resp.status_code == 404


@pytest.mark.e2e
class TestTenantCRUD:
    """Scenario 15: Full tenant lifecycle create -> list -> delete via API."""

    def test_tenant_lifecycle(self):
        org_id = unique_id("apiorg")
        tenant_name = "test_tenant"
        tenant_full_id = f"{org_id}:{tenant_name}"

        # Tenant create triggers a Vespa app-package redeploy whose latency
        # scales with the number of tenant schemas in the cluster (the
        # whole package is recompiled). Under sweep load with 100+
        # schemas it routinely takes 60-90 s; the original 60 s timeout
        # produced a flaky ReadTimeout. 180 s covers the worst case
        # observed without weakening the assertions.
        with httpx.Client(base_url=RUNTIME, timeout=180.0) as client:
            try:
                resp = client.post(
                    "/admin/organizations",
                    json={
                        "org_id": org_id,
                        "org_name": f"E2E Test Org {org_id}",
                        "created_by": "e2e_test",
                    },
                )
                assert resp.status_code == 200, f"Create org failed: {resp.text}"
                org_data = resp.json()
                assert org_data["org_id"] == org_id

                resp = client.get("/admin/organizations")
                assert resp.status_code == 200
                orgs = resp.json()
                org_ids = [o["org_id"] for o in orgs["organizations"]]
                assert org_id in org_ids

                resp = client.post(
                    "/admin/tenants",
                    json={
                        "tenant_id": tenant_full_id,
                        "created_by": "e2e_test",
                    },
                )
                assert resp.status_code == 200, f"Create tenant failed: {resp.text}"
                tenant_data = resp.json()
                assert tenant_data["tenant_full_id"] == tenant_full_id

                resp = client.get(f"/admin/organizations/{org_id}/tenants")
                assert resp.status_code == 200
                tenants = resp.json()
                tenant_ids = [t["tenant_full_id"] for t in tenants["tenants"]]
                assert tenant_full_id in tenant_ids

                resp = client.get(f"/admin/tenants/{tenant_full_id}")
                assert resp.status_code == 200
                assert resp.json()["tenant_full_id"] == tenant_full_id

            finally:
                client.delete(f"/admin/tenants/{tenant_full_id}")
                client.delete(f"/admin/organizations/{org_id}")

    def test_org_not_found_returns_404(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/admin/organizations/nonexistent_org_xyz")
        assert resp.status_code == 404, (
            f"Non-existent org should return 404, got {resp.status_code}"
        )

    def test_simple_form_tenant_id_normalizes_for_get_and_delete(self):
        """POST/GET/DELETE accept the simple form ``tid`` and resolve to ``tid:tid``.

        Regression: POST has always normalized simple form to ``org:tenant``
        (storing doc_id ``acme:acme`` for input ``acme``). GET and
        assert_tenant_exists previously did NOT normalize, so a simple-form
        input that POST happily wrote could not be read back. Tests like
        graph_cli that minted simple-form tenants would 404 forever.
        Both forms must now resolve identically.
        """
        from tests.e2e.conftest import unique_id

        tid_simple = unique_id("apinorm")
        tid_canonical = f"{tid_simple}:{tid_simple}"
        with httpx.Client(base_url=RUNTIME, timeout=180.0) as client:
            try:
                resp = client.post(
                    "/admin/tenants",
                    json={"tenant_id": tid_simple, "created_by": "e2e_norm"},
                )
                assert resp.status_code == 200, resp.text
                created = resp.json()
                # Runtime normalized to colon form on storage.
                assert created["tenant_full_id"] == tid_canonical, created

                # GET via simple form must succeed (this is what was broken).
                resp = client.get(f"/admin/tenants/{tid_simple}")
                assert resp.status_code == 200, (
                    f"GET with simple form {tid_simple!r} returned "
                    f"{resp.status_code}; runtime should have canonicalized "
                    f"to {tid_canonical!r}"
                )
                assert resp.json()["tenant_full_id"] == tid_canonical

                # GET via canonical form must also succeed (no regression).
                resp = client.get(f"/admin/tenants/{tid_canonical}")
                assert resp.status_code == 200
                assert resp.json()["tenant_full_id"] == tid_canonical
            finally:
                # DELETE via simple form must succeed too.
                resp = client.delete(f"/admin/tenants/{tid_simple}")
                assert resp.status_code in (200, 204, 404), (
                    f"DELETE with simple form {tid_simple!r} returned "
                    f"{resp.status_code}"
                )
                # And the tenant should now be gone.
                resp = client.get(f"/admin/tenants/{tid_simple}")
                assert resp.status_code == 404


@pytest.mark.e2e
class TestAgentRegistryAndHealth:
    """Scenario 18: Health endpoints and agent registry queries."""

    def test_health_check(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cogniverse-runtime"
        assert "backends" in data
        assert "agents" in data

    def test_liveness_probe(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health/live")

        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_probe(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "not_ready")

    def test_list_agents(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/")

        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "count" in data
        assert data["count"] >= 1

    def test_agent_stats(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "total_agents" in data, (
            f"Stats missing total_agents, got: {list(data.keys())}"
        )
        assert isinstance(data["total_agents"], int)
        assert data["total_agents"] >= 1

    def test_root_endpoint(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/")

        assert resp.status_code == 200

    def test_get_agent_info(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/gateway_agent")

        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data or "name" in data

    def test_agent_card(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/agents/gateway_agent/card")

        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card or "agent_name" in card

    @pytest.mark.parametrize(
        "agent_name",
        [
            "gateway_agent",
            "search_agent",
            "text_analysis_agent",
            "summarizer_agent",
            "detailed_report_agent",
            "entity_extraction_agent",
            "query_enhancement_agent",
            "profile_selection_agent",
            "orchestrator_agent",
        ],
    )
    def test_registered_agents_accessible(self, agent_name):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get(f"/agents/{agent_name}")
        assert resp.status_code == 200, f"Agent {agent_name} not accessible"

    def test_search_agent_process(self):
        """Scenario 18 sub-test: Direct search agent process."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "nature documentary",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "search_agent"


@pytest.mark.e2e
class TestA2AProtocol:
    """Scenario 19: A2A protocol agent card, tasks/send, and streaming."""

    def test_runtime_agent_card(self):
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/a2a/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "skills" in card or "capabilities" in card

    def test_agent_card_advertises_streaming(self):
        """AgentCard capabilities should include streaming=True."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/a2a/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        capabilities = card.get("capabilities", {})
        assert capabilities.get("streaming") is True, (
            f"AgentCard should advertise streaming=True, got: {capabilities}"
        )

    def test_a2a_single_turn(self):
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-api-1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": "search for nature videos"}
                            ],
                            "messageId": str(uuid.uuid4()),
                        },
                        "configuration": {
                            "acceptedOutputModes": ["text"],
                        },
                        "metadata": {
                            "tenant_id": TENANT_ID,
                        },
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data, f"Expected result, got: {data}"
        result = data["result"]
        assert result["id"]
        assert result["contextId"]

    def test_a2a_streaming_produces_sse_events(self):
        """message/stream returns SSE events with progress + final result."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with client.stream(
                "POST",
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-stream-1",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "summarize what machine learning is",
                                }
                            ],
                            "messageId": str(uuid.uuid4()),
                        },
                        "metadata": {
                            "agent_name": "summarizer_agent",
                            "tenant_id": TENANT_ID,
                            "stream": True,
                        },
                    },
                },
            ) as resp:
                assert resp.status_code == 200
                events = []
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if data_str:
                            events.append(json.loads(data_str))

        assert len(events) >= 2, (
            f"Streaming should produce ≥2 SSE events (progress + final), got {len(events)}"
        )

        # Parse agent events from A2A wrapper
        parsed = []
        for event in events:
            for part in (
                event.get("result", {})
                .get("status", {})
                .get("message", {})
                .get("parts", [])
            ):
                text = part.get("text", "")
                if text:
                    try:
                        parsed.append(json.loads(text))
                    except json.JSONDecodeError:
                        pass

        # Should have status events (from emit_progress) and a final event
        types = [e.get("type") for e in parsed]
        assert "status" in types, f"Should have progress events, got types: {types}"

        # Final event should contain summary
        finals = [e for e in parsed if e.get("type") == "final"]
        assert len(finals) == 1, f"Should have exactly 1 final event, got: {parsed}"
        assert "data" in finals[0]
        assert "summary" in finals[0]["data"]
        summary = finals[0]["data"]["summary"]
        assert len(summary) > 20, f"Summary too short: '{summary}'"
        summary_lower = summary.lower()
        assert any(
            term in summary_lower
            for term in ["machine learning", "ml", "learn", "algorithm", "data"]
        ), f"Summary should reference ML, got: '{summary}'"


@pytest.mark.e2e
class TestStreamingAllAgents:
    """Verify A2A streaming works for multiple agent types."""

    @pytest.mark.parametrize(
        "agent_name,query,expect_streaming",
        [
            ("summarizer_agent", "summarize AI trends briefly", True),
            ("detailed_report_agent", "write a report on search results", True),
            ("gateway_agent", "find videos about cats", True),
        ],
    )
    def test_streaming_agent_returns_events(self, agent_name, query, expect_streaming):
        """message/stream returns SSE events for streaming-capable agents."""
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with client.stream(
                "POST",
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-stream-all",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "role": "user",
                            "messageId": str(uuid.uuid4()),
                            "contextId": str(uuid.uuid4()),
                            "parts": [{"kind": "text", "text": query}],
                        },
                        "metadata": {
                            "agent_name": agent_name,
                            "tenant_id": TENANT_ID,
                            "stream": True,
                        },
                    },
                },
            ) as resp:
                assert resp.status_code == 200
                events = []
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if data_str:
                            raw = json.loads(data_str)
                            for part in (
                                raw.get("result", {})
                                .get("status", {})
                                .get("message", {})
                                .get("parts", [])
                            ):
                                text = part.get("text", "")
                                if text:
                                    try:
                                        events.append(json.loads(text))
                                    except json.JSONDecodeError:
                                        pass

        assert len(events) >= 1, (
            f"{agent_name}: should return ≥1 event, got {len(events)}"
        )

        if expect_streaming:
            types = [e.get("type") for e in events]
            assert "status" in types, (
                f"{agent_name}: streaming agent should emit progress events, got: {types}"
            )


@pytest.mark.e2e
class TestIngestionAPI:
    """Ingestion endpoints: start, status, upload validation."""

    def test_start_ingestion_invalid_dir_returns_error(self):
        """POST /ingestion/start with non-existent directory returns error."""
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": "/nonexistent/path/e2e_fake_dir",
                    "profile": PROFILE,
                    "backend": "vespa",
                    "tenant_id": TENANT_ID,
                },
            )
        assert resp.status_code in (400, 422, 500), (
            f"Non-existent video_dir should fail, got {resp.status_code}"
        )

    def test_ingestion_status_not_found(self):
        """GET /ingestion/status/{fake_id} returns 404."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/ingestion/status/nonexistent-job-id-xyz")
        assert resp.status_code == 404, (
            f"Non-existent job_id should return 404, got {resp.status_code}"
        )
        data = resp.json()
        assert "not found" in data.get("detail", "").lower()

    def test_upload_requires_file(self):
        """POST /ingestion/upload without file returns 422 (validation error)."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.post("/ingestion/upload")
        assert resp.status_code == 422, (
            f"Upload without file should return 422, got {resp.status_code}"
        )


# Real artifact ingestion + search round-trip tests
#
# Each test uploads a REAL file, ingests it through the actual ML pipeline,
# waits for Vespa indexing, then searches and verifies results.
#
# Artifacts:
# - Video: tracked 18-second 1280x720 clip v_-D1gdv_gQyw.mp4
# - Image: generated JPEG of that tracked video's first frame
# - Audio: generated 10-second, 16 kHz mono WAV from that video's audio stream
# - PDF: deterministic one-page PDF generated from repository evaluation text
# - Document: dataset_summary.md (real markdown about the evaluation set)


_GENERATION_METADATA_FIELDS = frozenset(
    {
        "requested_count",
        "returned_count",
        "shortfall_count",
        "floor_count",
        "surplus_exhausted",
        "dropped_count",
        "dropped_examples",
    }
)
_SAMPLED_CONTENT_METADATA_FIELDS = frozenset(
    {
        "profile_name",
        "schema_name",
        "source_id",
        "segment_id",
        "description",
    }
)


def _assert_synthetic_metadata(
    metadata: dict,
    *,
    backend_query_strategy: str,
    sampled_content_count: int,
    target_count: int,
    vespa_sample_size: int,
) -> None:
    """Pin response metadata exactly, except the fields a live LM varies.

    Drop counts depend on how many examples the model returned ungrounded on
    this run, so they carry an invariant instead of a fixed value.
    """
    generation = _assert_synthetic_metadata_fields(
        metadata,
        backend_query_strategy=backend_query_strategy,
        sampled_content_count=sampled_content_count,
        target_count=target_count,
        vespa_sample_size=vespa_sample_size,
    )
    assert generation["returned_count"] == target_count
    assert generation["shortfall_count"] == 0
    assert generation["surplus_exhausted"] is False


def _assert_synthetic_metadata_fields(
    metadata: dict,
    *,
    backend_query_strategy: str,
    sampled_content_count: int,
    target_count: int,
    vespa_sample_size: int,
) -> dict:
    """Pin every metadata field except the quota outcome; return the block."""
    generation = metadata.get("generation")
    sampled_content = metadata.get("sampled_content")
    assert isinstance(generation, dict), f"metadata has no generation block: {metadata}"
    assert isinstance(sampled_content, list), (
        f"metadata has no sampled_content block: {metadata}"
    )
    assert {
        key: value
        for key, value in metadata.items()
        if key not in {"generation", "sampled_content"}
    } == {
        "backend_query_strategy": backend_query_strategy,
        "sampled_content_count": sampled_content_count,
        "target_count": target_count,
        "vespa_sample_size": vespa_sample_size,
    }
    assert len(sampled_content) == sampled_content_count
    for record in sampled_content:
        assert isinstance(record, dict), record
        assert set(record) == _SAMPLED_CONTENT_METADATA_FIELDS, record
        assert len(record) == len(_SAMPLED_CONTENT_METADATA_FIELDS), record
        assert record["profile_name"].strip() != ""
        assert record["schema_name"].strip() != ""
        assert record["source_id"].strip() != ""
        assert isinstance(record["segment_id"], int), record
    assert set(generation) == set(_GENERATION_METADATA_FIELDS)
    assert generation["requested_count"] == target_count
    assert generation["floor_count"] == 1
    assert generation["dropped_count"] == len(generation["dropped_examples"])
    for drop in generation["dropped_examples"]:
        assert set(drop) == {"candidate", "reason"}
        assert drop["reason"].strip() != ""
    return generation


def _expected_artifact_source_url(path: Path, tenant_id: str = TENANT_ID) -> str:
    """Return the exact content-addressed MinIO URL used by upload.

    Upload partitions the object key by the canonical tenant id, so a
    caller passing the simple form still reads back the ``org:tenant``
    partition.
    """
    import hashlib

    from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    partition = canonical_tenant_id(tenant_id)
    return f"s3://cogniverse-ingest/{partition}/{digest}{path.suffix}"


def _expected_available_profile_names(tenant_id: str) -> list[str]:
    """Return the tenant-usable backend profiles offered at serving time."""
    names = tenant_usable_profile_names(create_default_config_manager(), tenant_id)
    if not names:
        pytest.fail(
            "tenant_usable_profile_names returned nothing; the expected profile "
            "list would be empty and every comparison against it would pass "
            f"vacuously. tenant_id={tenant_id!r}"
        )
    return names


def _profile_type_map() -> dict[str, str]:
    config = json.loads(CONFIG_PATH.read_text())
    profiles = config.get("backend", {}).get("profiles", {})
    return {
        profile_name: profile_config["type"]
        for profile_name, profile_config in profiles.items()
        if isinstance(profile_config, dict)
        and isinstance(profile_config.get("type"), str)
    }


def _profile_family(profile_name: str) -> str:
    normalized = profile_name.lower()
    for family in ("colpali", "colqwen", "videoprism"):
        if re.search(rf"(?<![a-z0-9]){re.escape(family)}(?![a-z0-9])", normalized):
            return family
    return profile_name


def _assert_artifact_search_hit(
    response: httpx.Response,
    *,
    query: str,
    profile: str,
    video_id: str,
    expected_metadata: dict,
) -> None:
    """Pin a top-1 search response to the artifact that was just uploaded."""
    assert response.status_code == 200, (
        f"Search failed for profile={profile!r}: HTTP {response.status_code} "
        f"body={response.text[:500]}"
    )
    payload = response.json()
    assert payload["query"] == query
    assert payload["profile"] == profile
    assert payload["results_count"] == 1, payload
    assert len(payload["results"]) == 1, payload
    hit = payload["results"][0]
    assert hit["document_id"].startswith(f"{video_id}_"), hit
    assert isinstance(hit["score"], (int, float)) and not isinstance(
        hit["score"], bool
    ), hit
    metadata = hit["metadata"]
    for key, expected in expected_metadata.items():
        assert metadata.get(key) == expected, (
            f"top hit metadata[{key!r}] did not identify the uploaded artifact: "
            f"expected={expected!r} actual={metadata.get(key)!r} hit={hit!r}"
        )


@pytest.mark.e2e
class TestVideoIngestionAndSearch:
    """Upload the tracked real video, verify ingestion, then retrieve it."""

    def test_upload_video_and_search(self, real_video_path):
        """The tracked fire-lighting clip produces and retrieves 10 frames."""
        assert real_video_path.name == "v_-D1gdv_gQyw.mp4"
        assert real_video_path.stat().st_size == 5_524_837
        tenant_id = unique_id("ingest_e2e")
        expected_source_url = _expected_artifact_source_url(real_video_path, tenant_id)
        with httpx.Client(base_url=RUNTIME, timeout=1800.0) as client:
            resp = client.post(
                "/admin/tenants",
                json={"tenant_id": tenant_id, "created_by": "e2e-test"},
                timeout=30,
            )
            assert resp.status_code in (200, 201, 409), resp.text
            _deploy_profile_for_tenant(client, PROFILE, tenant_id)
            with open(real_video_path, "rb") as f:
                # wait=true keeps the synchronous response shape
                # (status, chunks_created, ...); the default async
                # path returns only ingest_id + state=queued.
                resp = client.post(
                    "/ingestion/upload?wait=true&wait_timeout=540",
                    files={"file": (real_video_path.name, f, "video/mp4")},
                    data={
                        "profile": PROFILE,
                        "tenant_id": tenant_id,
                    },
                )

            assert resp.status_code == 200, f"Video upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["state"] == "complete"
            assert upload_data["filename"] == real_video_path.name
            assert upload_data["source_url"] == expected_source_url
            assert upload_data["chunks_created"] == 10, upload_data
            assert upload_data["documents_fed"] == 10, upload_data
            assert isinstance(upload_data["video_id"], str) and upload_data["video_id"]

            time.sleep(5)

            query = "man lighting a fire outdoors"
            search_resp = client.post(
                "/search/",
                json={
                    "query": query,
                    "profile": PROFILE,
                    "top_k": 1,
                    "tenant_id": tenant_id,
                    "filters": {"source_url": expected_source_url},
                },
            )
            _assert_artifact_search_hit(
                search_resp,
                query=query,
                profile=PROFILE,
                video_id=upload_data["video_id"],
                expected_metadata={
                    "source_url": expected_source_url,
                    "video_id": upload_data["video_id"],
                },
            )


@pytest.mark.e2e
class TestImageIngestionAndSearch:
    """Upload the tracked video's generated first-frame JPEG and retrieve it."""

    def _deploy_schema_if_needed(self, client, profile_name):
        """Deploy schema for profile if not already deployed."""
        resp = client.post(
            f"/admin/profiles/{profile_name}/deploy",
            json={"tenant_id": TENANT_ID, "force": False},
        )
        assert resp.status_code == 200, (
            f"Schema deployment request failed: HTTP {resp.status_code} {resp.text}"
        )
        body = resp.json()
        assert body["profile_name"] == profile_name, body
        assert body["tenant_id"] == TENANT_ID, body
        assert body["deployment_status"] in {"success", "already_deployed"}, body

    def test_upload_image_and_search(self, real_image_path):
        """The 1280x720 first frame is embedded once and returned as top hit."""
        from PIL import Image

        assert real_image_path.name == "tracked_video_frame.jpg"
        with Image.open(real_image_path) as image:
            assert image.format == "JPEG"
            assert image.size == (1280, 720)
        expected_source_url = _expected_artifact_source_url(real_image_path)
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            self._deploy_schema_if_needed(client, IMAGE_PROFILE)

            with open(real_image_path, "rb") as f:
                # force=true: this test proves the processing path; a prior
                # run against the same cluster already ingested these exact
                # bytes (a plain upload would be a dedup echo with 0 chunks).
                resp = client.post(
                    "/ingestion/upload?wait=true&wait_timeout=540&force=true",
                    files={"file": (real_image_path.name, f, "image/jpeg")},
                    data={
                        "profile": IMAGE_PROFILE,
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Image upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["state"] == "complete"
            assert upload_data["existing"] is False, upload_data
            assert upload_data["filename"] == real_image_path.name
            assert upload_data["source_url"] == expected_source_url
            assert upload_data["chunks_created"] == 1, upload_data
            assert upload_data["documents_fed"] == 1, upload_data
            assert upload_data["video_id"] == _content_sha256(real_image_path)

            time.sleep(5)

            query = "man in a yellow shirt outdoors beside a red chair"
            search_resp = client.post(
                "/search/",
                json={
                    "query": query,
                    "profile": IMAGE_PROFILE,
                    "top_k": 1,
                    "tenant_id": TENANT_ID,
                    "filters": {"source_url": expected_source_url},
                },
            )
            _assert_artifact_search_hit(
                search_resp,
                query=query,
                profile=IMAGE_PROFILE,
                video_id=upload_data["video_id"],
                expected_metadata={"source_url": expected_source_url},
            )


@pytest.mark.e2e
class TestAudioIngestionAndSearch:
    """Upload the tracked video's generated audio and retrieve it."""

    def test_upload_extracted_audio_processing(self, extracted_audio_path):
        """The exact 10-second mono fixture is embedded once and retrieved.

        It contains the tracked video's real speech and ambient sound, not a
        synthetic tone.
        """
        import wave

        assert extracted_audio_path.name == "tracked_video_audio.wav"
        with wave.open(str(extracted_audio_path), "rb") as audio:
            assert (
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getframerate(),
                audio.getnframes(),
            ) == (1, 2, 16_000, 160_000)
        expected_source_url = _expected_artifact_source_url(extracted_audio_path)
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with open(extracted_audio_path, "rb") as f:
                # force=true: this test proves the processing path, and the
                # session fixture may already have ingested these exact bytes
                # (a plain upload would then be a dedup echo with 0 chunks).
                resp = client.post(
                    "/ingestion/upload?wait=true&wait_timeout=540&force=true",
                    files={"file": (extracted_audio_path.name, f, "audio/wav")},
                    data={
                        "profile": AUDIO_PROFILE,
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Audio upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["state"] == "complete"
            assert upload_data["existing"] is False, upload_data
            assert upload_data["filename"] == extracted_audio_path.name
            assert upload_data["source_url"] == expected_source_url
            assert upload_data["chunks_created"] == 1, upload_data
            assert upload_data["documents_fed"] == 1, upload_data
            assert upload_data["video_id"] == _content_sha256(extracted_audio_path)

            time.sleep(3)

            query = "man speaking outdoors"
            search_resp = client.post(
                "/search/",
                json={
                    "query": query,
                    "profile": AUDIO_PROFILE,
                    "top_k": 1,
                    "tenant_id": TENANT_ID,
                    "filters": {"source_url": expected_source_url},
                },
            )
            _assert_artifact_search_hit(
                search_resp,
                query=query,
                profile=AUDIO_PROFILE,
                video_id=upload_data["video_id"],
                expected_metadata={
                    "source_url": expected_source_url,
                    "audio_id": upload_data["video_id"],
                },
            )


@pytest.mark.e2e
class TestPDFIngestionAndSearch:
    """Upload the generated evaluation-text PDF and retrieve its exact text."""

    def test_upload_pdf_processing(self, real_pdf_path):
        """Upload deterministic one-page PDF → text extraction → embedding.

        Uploads async and polls ``/ingestion/{id}/status`` rather than
        holding ``wait=true`` open: a full PDF ingest can stay silent
        past the k3d serverlb's ``proxy_timeout 600``, which then cuts
        the TCP stream mid-wait ("Server disconnected without sending a
        response"). Polling exercises the same worker path without a
        long-silent connection.
        """
        from PyPDF2 import PdfReader

        assert real_pdf_path.name == "evaluation_dataset.pdf"
        reader = PdfReader(str(real_pdf_path))
        assert len(reader.pages) == 1
        pdf_text = (reader.pages[0].extract_text() or "").strip()
        assert pdf_text.splitlines() == [
            "Evaluation Dataset",
            "Video-ChatGPT Benchmark",
            "Provides 500 test videos from ActivityNet-200.",
        ]
        expected_source_url = _expected_artifact_source_url(real_pdf_path)
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with open(real_pdf_path, "rb") as f:
                resp = client.post(
                    "/ingestion/upload?force=true",
                    files={"file": (real_pdf_path.name, f, "application/pdf")},
                    data={
                        "profile": DOCUMENT_PROFILE,
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code in (200, 202), f"PDF upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["filename"] == real_pdf_path.name
            assert upload_data["source_url"] == expected_source_url
            ingest_id = upload_data["ingest_id"]

            deadline = time.time() + 1200
            state = upload_data["state"]
            latest: dict = {}
            while time.time() < deadline:
                status_resp = client.get(f"/ingestion/{ingest_id}/status")
                assert status_resp.status_code == 200, (
                    f"PDF ingest status failed: HTTP {status_resp.status_code} "
                    f"body={status_resp.text[:500]}"
                )
                status_data = status_resp.json()
                state = status_data["state"]
                latest = status_data["latest"]
                if state in ("complete", "failed", "error"):
                    break
                time.sleep(5)

            assert state == "complete", (
                f"PDF ingest {ingest_id} did not complete within 1200s; "
                f"state={state!r} latest={latest!r}"
            )
            pipeline_result = latest.get("result", {}) or {}
            assert pipeline_result["source_url"] == expected_source_url
            assert pipeline_result["chunks"] == 1, pipeline_result
            assert pipeline_result["documents_fed"] == 1, pipeline_result
            assert pipeline_result["keyframes"] == 0, pipeline_result
            assert (
                isinstance(pipeline_result["video_id"], str)
                and pipeline_result["video_id"]
            )

            time.sleep(3)

            query = "Provides 500 test videos from ActivityNet-200"
            search_resp = client.post(
                "/search/",
                json={
                    "query": query,
                    "profile": DOCUMENT_PROFILE,
                    "top_k": 1,
                    "tenant_id": TENANT_ID,
                    "filters": {"document_id": pipeline_result["video_id"]},
                },
            )
            _assert_artifact_search_hit(
                search_resp,
                query=query,
                profile=DOCUMENT_PROFILE,
                video_id=pipeline_result["video_id"],
                expected_metadata={
                    "document_id": pipeline_result["video_id"],
                    "full_text": pdf_text,
                },
            )


@pytest.mark.e2e
class TestDocumentIngestionAndSearch:
    """Upload tracked dataset_summary.md and retrieve its exact content."""

    def test_upload_markdown_processing(self, real_document_path):
        """Upload tracked markdown → LateOn embedding → filtered retrieval."""
        assert real_document_path.name == "dataset_summary.md"
        document_text = real_document_path.read_text(encoding="utf-8").strip()
        assert "# Evaluation Dataset" in document_text
        assert "Provides:\n- **500 test videos** from ActivityNet-200" in document_text
        expected_source_url = _expected_artifact_source_url(real_document_path)
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            with open(real_document_path, "rb") as f:
                # force=true: this test proves the processing path; a prior
                # run against the same cluster already ingested these exact
                # bytes (a plain upload would be a dedup echo with 0 chunks).
                resp = client.post(
                    "/ingestion/upload?wait=true&wait_timeout=540&force=true",
                    files={"file": (real_document_path.name, f, "text/markdown")},
                    data={
                        "profile": DOCUMENT_PROFILE,
                        "tenant_id": TENANT_ID,
                    },
                )

            assert resp.status_code == 200, f"Document upload failed: {resp.text}"
            upload_data = resp.json()
            assert upload_data["status"] == "success"
            assert upload_data["state"] == "complete"
            assert upload_data["existing"] is False, upload_data
            assert upload_data["filename"] == real_document_path.name
            assert upload_data["source_url"] == expected_source_url
            assert upload_data["chunks_created"] == 1, upload_data
            assert upload_data["documents_fed"] == 1, upload_data
            assert upload_data["video_id"] == _content_sha256(real_document_path)

            time.sleep(3)

            query = "125 extracted sample video retrieval queries"
            search_resp = client.post(
                "/search/",
                json={
                    "query": query,
                    "profile": DOCUMENT_PROFILE,
                    "top_k": 1,
                    "tenant_id": TENANT_ID,
                    "filters": {"document_id": upload_data["video_id"]},
                },
            )
            _assert_artifact_search_hit(
                search_resp,
                query=query,
                profile=DOCUMENT_PROFILE,
                video_id=upload_data["video_id"],
                expected_metadata={
                    "document_id": upload_data["video_id"],
                    "full_text": document_text,
                },
            )


# Scenario 20 (API portion): Event queue listing
# Placed before batch ingestion because batch starts CPU-bound ColPali
# inference that blocks the async event loop for minutes.


@pytest.mark.e2e
class TestEventEndpoints:
    """Scenario 20 API: Event queue listing."""

    def test_list_queues(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get(
                "/events/queues",
                params={"tenant_id": TENANT_ID},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_queue_not_found(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.get("/events/queues/nonexistent_fake_id")

        assert resp.status_code == 404, (
            f"Non-existent queue should return 404, got {resp.status_code}"
        )


@pytest.mark.e2e
class TestBatchVideoIngestion:
    """Directory-based batch ingestion of the tracked clip in its own tenant.

    ``/ingestion/start`` reads a directory on the runtime pod's filesystem,
    so the tracked clip is copied into the pod first. Frame embedding runs
    off the event loop, so ``/health/live`` must keep answering while the
    job is processing.
    """

    BATCH_VIDEO = "v_-6dz6tBH77I.mp4"
    POD_BATCH_ROOT = "/app/outputs/temp/e2e_batch"

    def _copy_video_into_pod(self, tenant_id: str) -> str:
        from tests.e2e.conftest import _kubectl_e2e, _require_kubectl_success

        host_dir = Path(__file__).parent.parent / "system" / "resources" / "videos"
        tracked_videos = sorted(path.name for path in host_dir.glob("*.mp4"))
        assert tracked_videos == ["v_-6dz6tBH77I.mp4", "v_-D1gdv_gQyw.mp4"], (
            f"tracked E2E video set is incomplete: {tracked_videos!r}"
        )
        pod_dir = f"{self.POD_BATCH_ROOT}/{tenant_id}"
        # ``kubectl cp`` addresses a pod, not a deployment; the exec calls
        # below resolve ``deploy/`` themselves, so both target the same pod
        # only while the runtime runs a single replica (as it does here).
        pods = _kubectl_e2e(
            "-n",
            "cogniverse",
            "get",
            "pods",
            "-l",
            "app.kubernetes.io/component=runtime",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        _require_kubectl_success(
            pods, ["kubectl", "--context", KUBECTL_CONTEXT, "get", "pods", "runtime"]
        )
        runtime_pods = pods.stdout.split()
        assert len(runtime_pods) == 1, (
            f"expected exactly one running runtime pod, got {runtime_pods!r}"
        )
        runtime_pod = runtime_pods[0]
        mkdir = _kubectl_e2e(
            "-n",
            "cogniverse",
            "exec",
            "deploy/cogniverse-runtime",
            "-c",
            "runtime",
            "--",
            "mkdir",
            "-p",
            pod_dir,
        )
        _require_kubectl_success(
            mkdir,
            ["kubectl", "--context", KUBECTL_CONTEXT, "exec", "mkdir", pod_dir],
        )
        copy = _kubectl_e2e(
            "-n",
            "cogniverse",
            "cp",
            "-c",
            "runtime",
            str(host_dir / self.BATCH_VIDEO),
            f"{runtime_pod}:{pod_dir}/{self.BATCH_VIDEO}",
            timeout=120,
        )
        _require_kubectl_success(
            copy,
            ["kubectl", "--context", KUBECTL_CONTEXT, "cp", self.BATCH_VIDEO, pod_dir],
        )
        listing = _kubectl_e2e(
            "-n",
            "cogniverse",
            "exec",
            "deploy/cogniverse-runtime",
            "-c",
            "runtime",
            "--",
            "ls",
            pod_dir,
        )
        _require_kubectl_success(
            listing,
            ["kubectl", "--context", KUBECTL_CONTEXT, "exec", "ls", pod_dir],
        )
        assert listing.stdout.split() == [self.BATCH_VIDEO], listing.stdout
        return pod_dir

    def test_batch_ingestion_start(self):
        """Start batch ingestion → poll to completion → the clip is retrievable."""
        tenant_id = unique_id("batch_e2e")
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/admin/tenants",
                json={"tenant_id": tenant_id, "created_by": "e2e-test"},
                timeout=30,
            )
            assert resp.status_code in (200, 201), resp.text
            _deploy_profile_for_tenant(client, PROFILE, tenant_id)
            pod_dir = self._copy_video_into_pod(tenant_id)

            resp = client.post(
                "/ingestion/start",
                json={
                    "video_dir": pod_dir,
                    "profile": PROFILE,
                    "backend": "vespa",
                    "tenant_id": tenant_id,
                    "max_videos": 1,
                    "batch_size": 1,
                },
            )
            assert resp.status_code == 200, f"Batch ingestion start failed: {resp.text}"
            data = resp.json()
            assert data == {
                "job_id": data["job_id"],
                "status": "started",
                "message": "Ingestion job started successfully",
            }
            job_id = data["job_id"]

        status_data = None
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            time.sleep(5)
            with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
                health_resp = client.get("/health/live")
                assert health_resp.status_code == 200, (
                    "the event loop stalled during batch ingestion: "
                    f"/health/live returned {health_resp.status_code}"
                )
                status_resp = client.get(f"/ingestion/status/{job_id}")
                assert status_resp.status_code == 200, status_resp.text
                status_data = status_resp.json()
                if status_data["status"] not in ("started", "processing"):
                    break

        assert status_data == {
            "job_id": job_id,
            "status": "completed",
            "videos_processed": 1,
            "videos_total": 1,
            "errors": [],
        }, status_data

        video_id = Path(self.BATCH_VIDEO).stem
        query = "a person on camera"
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            search_resp = client.post(
                "/search/",
                json={
                    "query": query,
                    "profile": PROFILE,
                    "top_k": 1,
                    "tenant_id": tenant_id,
                    "filters": {"video_id": video_id},
                },
            )
        _assert_artifact_search_hit(
            search_resp,
            query=query,
            profile=PROFILE,
            video_id=video_id,
            expected_metadata={"video_id": video_id},
        )

    def test_runtime_responsive_after_batch(self):
        """Verify runtime is fully responsive after batch ingestion completes."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"

            # Also verify agent processing still works
            resp = client.get("/agents/")
            assert resp.status_code == 200
