"""``context["max_iterations"]`` reaches ``DeepResearchInput`` through the
runtime dispatch path.

``AgentDispatcher._execute_deep_research_task`` builds the ``DeepResearchInput``
the deep research agent runs on. The forwarding tests mount the real agents
router on the real dispatcher and record the typed input the agent receives;
the real-LM tests let the run continue through the real LM, encoder and
Vespa and pin what the research loop guarantees for the forwarded bound.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from vespa.application import Vespa

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchInput,
    DeepResearchOutput,
)
from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader
from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents
from tests.runtime.integration.conftest import SCHEMAS_DIR, skip_if_no_lm
from tests.utils.async_polling import wait_for_condition_sync
from tests.utils.vespa_test_helpers import deploy_tenant_schema

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration

TENANT_ID = "test:unit"
MAX_ITERATIONS_DEFAULT = DeepResearchInput.model_fields["max_iterations"].default

# The live tenant serves exactly one shipped profile, so the grounding search
# resolves its query encoder and reads a schema that holds one indexed frame.
LIVE_TENANT_ID = "deep_research:live"
_SHIPPED_CONFIG = json.loads((SCHEMAS_DIR.parent / "config.json").read_text())
LIVE_PROFILE = _SHIPPED_CONFIG["active_video_profile"]
LIVE_VIDEO_ID = "outdoor_hiking_trail"


@pytest.fixture(scope="module")
def deep_research_dispatcher(config_manager, schema_loader):
    registry = AgentRegistry(tenant_id=TENANT_ID, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="deep_research_agent",
            url="http://localhost:8010",
            capabilities=["deep_research", "analysis"],
            health_endpoint="/health",
        )
    )
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.fixture
def route_client(deep_research_dispatcher):
    agents._dispatcher = deep_research_dispatcher
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    agents._dispatcher = None


@pytest.fixture
def recorded_inputs(monkeypatch):
    """Replace the research loop with a recorder of the typed input it was
    handed; the dispatcher, router and agent construction stay real."""
    captured: list[DeepResearchInput] = []

    async def _record(self, input_data):
        captured.append(input_data)
        return DeepResearchOutput(summary="recorded")

    monkeypatch.setattr(DeepResearchAgent, "_process_impl", _record)
    return captured


@pytest.fixture
def recorded_live_inputs(monkeypatch):
    """Record the typed input and call through to the real research loop."""
    captured: list[DeepResearchInput] = []
    original = DeepResearchAgent.process

    async def _record_then_run(self, input, stream=False):
        captured.append(input)
        return await original(self, input, stream)

    monkeypatch.setattr(DeepResearchAgent, "process", _record_then_run)
    return captured


@pytest.fixture(scope="module")
def live_corpus(config_manager, vespa_instance, tomoro_search_url):
    """One Tomoro-embedded outdoor frame indexed in the live tenant's schema."""
    profile = BackendProfileConfig.from_dict(
        LIVE_PROFILE, _SHIPPED_CONFIG["backend"]["profiles"][LIVE_PROFILE]
    )
    config_manager.add_backend_profile(profile, tenant_id=LIVE_TENANT_ID)
    schema_name = deploy_tenant_schema(
        vespa_instance,
        tenant_id=LIVE_TENANT_ID,
        base_schema_name=profile.schema_name,
        config_manager=config_manager,
    )

    client, _ = RemoteColPaliLoader(
        model_name=profile.embedding_model,
        config={"remote_inference_url": tomoro_search_url},
        logger=logger,
    ).load_model()
    encoded = client.process_images(
        [Image.new("RGB", (224, 224), color=(34, 139, 34))],
        model_name=profile.embedding_model,
    )
    embeddings = np.asarray(
        encoded.get("embeddings") if isinstance(encoded, dict) else encoded
    ).astype(np.float32)
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)

    statuses = {}
    app = Vespa(url=f"http://localhost:{vespa_instance['http_port']}")
    app.feed_iterable(
        iter=[
            {
                "id": LIVE_VIDEO_ID,
                "fields": {
                    "video_id": LIVE_VIDEO_ID,
                    "video_title": "Hikers on an outdoor mountain trail",
                    "segment_id": 0,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "segment_description": (
                        "Hikers walk an outdoor mountain trail through a green "
                        "forest, carrying backpacks and trekking poles"
                    ),
                    "audio_transcript": "",
                    "embedding": {
                        str(i): vector.tolist() for i, vector in enumerate(embeddings)
                    },
                    "embedding_binary": {
                        str(i): vector.tolist() for i, vector in enumerate(binarized)
                    },
                },
            }
        ],
        schema=schema_name,
        namespace="video",
        callback=lambda response, doc_id: statuses.update(
            {doc_id: response.status_code}
        ),
    )
    assert statuses == {LIVE_VIDEO_ID: 200}

    def indexed_ids():
        response = app.query(
            yql=f"select video_id from {schema_name} where true", hits=10
        )
        return {hit["fields"]["video_id"] for hit in response.hits}

    wait_for_condition_sync(
        lambda: indexed_ids() == {LIVE_VIDEO_ID},
        timeout=60,
        description=f"indexed frame for {LIVE_TENANT_ID}",
    )
    return LIVE_VIDEO_ID


def _evidence_video_ids(research: dict) -> list[list[str]]:
    return [
        [hit["video_id"] for hit in entry["results"]] for entry in research["evidence"]
    ]


def _post(client: TestClient, query: str, context: dict):
    return client.post(
        "/agents/deep_research_agent/process",
        json={"agent_name": "deep_research_agent", "query": query, "context": context},
    )


class TestMaxIterationsForwarding:
    def test_request_context_bound_reaches_input(self, route_client, recorded_inputs):
        query = "what changed in checkout latency after the outage?"
        resp = _post(route_client, query, {"tenant_id": TENANT_ID, "max_iterations": 1})

        assert resp.status_code == 200, resp.text
        assert len(recorded_inputs) == 1
        received = recorded_inputs[0]
        assert received.max_iterations == 1
        assert received.query == query
        assert received.tenant_id == TENANT_ID
        assert received.rlm is None

        body = resp.json()
        assert body["status"] == "success"
        assert body["agent"] == "deep_research_agent"
        assert body["message"] == f"Research complete for '{query}'"
        assert body["result"]["summary"] == "recorded"

    def test_absent_bound_uses_input_default(self, route_client, recorded_inputs):
        resp = _post(route_client, "no bound supplied", {"tenant_id": TENANT_ID})

        assert resp.status_code == 200, resp.text
        assert len(recorded_inputs) == 1
        assert recorded_inputs[0].max_iterations == MAX_ITERATIONS_DEFAULT

    @pytest.mark.asyncio
    async def test_direct_dispatch_forwards_bound(
        self, deep_research_dispatcher, recorded_inputs
    ):
        result = await deep_research_dispatcher.dispatch(
            agent_name="deep_research_agent",
            query="direct dispatch",
            context={"tenant_id": TENANT_ID, "max_iterations": 2},
        )

        assert len(recorded_inputs) == 1
        assert recorded_inputs[0].max_iterations == 2
        assert result["status"] == "success"
        assert result["result"]["summary"] == "recorded"


@skip_if_no_lm
class TestMaxIterationsLive:
    """Real LM, real Tomoro query encoder, real Vespa behind the dispatcher's
    own ``_execute_search_task`` search leg, over a tenant whose one servable
    profile holds one indexed frame."""

    @pytest.mark.asyncio
    async def test_bound_of_one_runs_exactly_one_iteration(
        self,
        deep_research_dispatcher,
        live_corpus,
        dspy_lm_planning,
        recorded_live_inputs,
    ):
        # The research loop advances ``iteration`` before it can stop, and a
        # run that never iterates raises instead of returning, so a returned
        # result with bound 1 always reports exactly one iteration.
        result = await deep_research_dispatcher.dispatch(
            agent_name="deep_research_agent",
            query="What visual patterns appear in outdoor activity videos?",
            context={"tenant_id": LIVE_TENANT_ID, "max_iterations": 1},
        )

        assert len(recorded_live_inputs) == 1
        assert recorded_live_inputs[0].max_iterations == 1
        assert result["status"] == "success"
        assert result["agent"] == "deep_research_agent"
        research = result["result"]
        assert research["iterations_used"] == 1
        # One iteration searches each sub-question once, and every search
        # returns the one indexed frame.
        assert [entry["question"] for entry in research["evidence"]] == research[
            "sub_questions"
        ]
        assert _evidence_video_ids(research) == [[live_corpus]] * len(
            research["sub_questions"]
        )

    @pytest.mark.asyncio
    async def test_absent_bound_runs_on_input_default(
        self,
        deep_research_dispatcher,
        live_corpus,
        dspy_lm_planning,
        recorded_live_inputs,
    ):
        # ``iterations_used`` is not pinned here: with the default bound the
        # evaluator LM decides whether the loop stops early.
        result = await deep_research_dispatcher.dispatch(
            agent_name="deep_research_agent",
            query="What visual patterns appear in outdoor activity videos?",
            context={"tenant_id": LIVE_TENANT_ID},
        )

        assert len(recorded_live_inputs) == 1
        assert recorded_live_inputs[0].max_iterations == MAX_ITERATIONS_DEFAULT
        assert result["status"] == "success"
        assert result["agent"] == "deep_research_agent"
        research = result["result"]
        # The first iteration searches every sub-question in order; later
        # iterations search the remaining gaps, each against the same frame.
        sub_questions = research["sub_questions"]
        assert [
            entry["question"] for entry in research["evidence"][: len(sub_questions)]
        ] == sub_questions
        assert _evidence_video_ids(research) == [[live_corpus]] * len(
            research["evidence"]
        )
