"""SearchAgent builds its query encoder only when a search needs embeddings.

Real Vespa, real registry, real agent objects. The profile names the shipped
ColPali inference service; each scenario leaves that service with no URL or
points it at a closed port, so the outcome shows whether the agent built or
called the encoder and how an encoder failure surfaces.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import uuid
from pathlib import Path

import pytest
import requests

from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_core.query.encoders import (
    EncoderNotConfiguredError,
    EncoderUnavailableError,
    QueryEncoderFactory,
)
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
SHIPPED_PROFILE = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
][BASE_SCHEMA]
ENCODER_SERVICE = SHIPPED_PROFILE["inference_services"]["embedding"]
PROFILE = "agentenc_frames"
TENANT = f"agentenc{uuid.uuid4().hex[:8]}:unit"
QUERY = "red kayak river"
SEGMENTS = {
    "vid_described": {
        "video_title": "clip one",
        "segment_description": "a red kayak drifts down the river",
        "audio_transcript": "hello there",
    },
    "vid_spoken": {
        "video_title": "clip two",
        "segment_description": "a person talks to the camera",
        "audio_transcript": "we packed the kayak for the trip",
    },
}


@pytest.fixture(scope="module")
def corpus(shared_vespa):
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()
    config_manager = make_config_manager(shared_vespa)
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(PROFILE, SHIPPED_PROFILE), tenant_id=TENANT
    )
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT,
        base_schema_name=BASE_SCHEMA,
        config_manager=config_manager,
    )
    for video_id, fields in SEGMENTS.items():
        response = requests.post(
            f"http://localhost:{shared_vespa['http_port']}/document/v1/"
            f"content/{schema}/docid/{video_id}_seg_0",
            json={"fields": {"video_id": video_id, "segment_id": 0, **fields}},
            timeout=30,
        )
        assert response.status_code == 200, response.text
    yield {"vespa": shared_vespa, "config_manager": config_manager}
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture
def agent(corpus):
    vespa = corpus["vespa"]
    return SearchAgent(
        deps=SearchAgentDeps(
            profile=PROFILE,
            tenant_id=TENANT,
            backend_url="http://localhost",
            backend_port=vespa["http_port"],
            backend_config_port=vespa["config_port"],
            auto_create_memory_schema=False,
        ),
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        config_manager=corpus["config_manager"],
        port=8061,
    )


@pytest.fixture
def encoder_service_down(corpus):
    config_manager = corpus["config_manager"]
    unconfigured = config_manager.get_system_config()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{probe.getsockname()[1]}"
    config_manager.set_system_config(
        dataclasses.replace(
            unconfigured, inference_service_urls={ENCODER_SERVICE: closed}
        )
    )
    yield
    config_manager.set_system_config(unconfigured)
    QueryEncoderFactory._encoder_cache.clear()


def _video_ids(results):
    return [result["video_id"] for result in results]


def test_construction_and_a_bm25_search_build_no_encoder(agent):
    results = agent._search_by_text(
        QUERY, tenant_id=TENANT, top_k=5, ranking="bm25_only"
    )

    assert _video_ids(results) == ["vid_described", "vid_spoken"]
    assert agent.__dict__.get("_query_encoder") is None
    assert dict(QueryEncoderFactory._encoder_cache) == {}


def test_an_embedding_strategy_without_an_encoder_service_is_a_config_gap(agent):
    with pytest.raises(EncoderNotConfiguredError) as caught:
        agent._search_by_text(QUERY, tenant_id=TENANT, top_k=5, ranking="float_float")

    assert caught.value.profile == PROFILE
    assert str(caught.value).startswith(
        f"Profile '{PROFILE}' declares a query encoder that could not be built: "
        f"ValueError: Profile '{PROFILE}' specifies inference_services.embedding="
        f"'{ENCODER_SERVICE}' but no URL is configured."
    )


def test_an_embedding_strategy_with_its_encoder_down_is_an_outage(
    encoder_service_down, agent
):
    with pytest.raises(EncoderUnavailableError) as caught:
        agent._search_by_text(QUERY, tenant_id=TENANT, top_k=5, ranking="float_float")

    assert (caught.value.profile, caught.value.service) == (PROFILE, ENCODER_SERVICE)
    assert type(caught.value.__cause__).__name__ == "ConnectionError"
