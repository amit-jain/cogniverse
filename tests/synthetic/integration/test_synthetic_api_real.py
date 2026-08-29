"""Real Vespa coverage for ``POST /synthetic/batch/generate``."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
)
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
)
from cogniverse_synthetic import api as synthetic_api
from cogniverse_synthetic.service import SyntheticDataService
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
from tests.utils.synthetic_config import video_synthetic_generator_config
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = [pytest.mark.integration, pytest.mark.requires_lm]


@pytest.fixture(scope="module")
def real_service(shared_vespa):
    tenant_id = f"synapi{uuid.uuid4().hex[:8]}:media"
    profile_name = "video_colpali_smol500_mv_frame"
    records = [
        {
            "data_id": "api-curie-radium-segment",
            "video_id": "api-curie-radium",
            "title": "Marie Curie discovered radium",
            "description": "Marie Curie and Pierre Curie isolated radium in Paris.",
            "start_time": 0.0,
            "end_time": 12.0,
        },
        {
            "data_id": "api-tesla-current-segment",
            "video_id": "api-tesla-current",
            "title": "Nikola Tesla developed alternating current",
            "description": "Tesla demonstrated an alternating-current motor in 1888.",
            "start_time": 12.0,
            "end_time": 24.0,
        },
    ]
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        tenant_id=tenant_id,
        profiles={
            profile_name: BackendProfileConfig(
                profile_name=profile_name,
                type="video",
                schema_name=profile_name,
                embedding_type="multi_vector",
                pipeline_config={"extract_keyframes": True},
            )
        },
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant_id})
    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    schema = registry.deploy_schema(
        tenant_id=tenant_id,
        base_schema_name=profile_name,
    )
    vespa_app = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    )
    for segment_id, record in enumerate(records):
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id=record["data_id"],
            fields={
                "video_id": record["video_id"],
                "video_title": record["title"],
                "source_url": f"http://example.test/{record['video_id']}",
                "segment_id": segment_id,
                "segment_description": record["description"],
                "start_time": record["start_time"],
                "end_time": record["end_time"],
            },
        )
        assert feed.is_successful(), feed.json

    for _ in range(20):
        indexed = backend.query_metadata_documents(
            schema=schema,
            yql=f"select * from sources {schema} where true limit 2",
            hits=2,
        )
        if len(indexed) == 2:
            assert {item["video_title"] for item in indexed} == {
                record["title"] for record in records
            }
            break
        time.sleep(0.5)
    else:
        pytest.fail("API source document was not indexed by Vespa")

    config_manager.set_backend_config(backend_config)

    profile_agent = ProfileSelectionAgent(
        deps=ProfileSelectionDeps(available_profiles=[profile_name])
    )
    profile_agent._config_manager = config_manager
    profile_agent.telemetry_manager = RecordingTelemetryManager()
    profile_agent.dspy_module.selector = lambda **_: dspy.Prediction(
        selected_profile=profile_name,
        confidence="0.98",
        reasoning="The production selector chose the live video profile.",
        query_intent="video_search",
        modality="text",
        complexity="complex",
    )

    async def label_profile(query, available_profiles, request_tenant_id):
        return await profile_agent.process(
            ProfileSelectionInput(
                query=query,
                available_profiles=available_profiles,
                tenant_id=request_tenant_id,
            )
        )

    service = SyntheticDataService(
        backend=backend,
        generator_config=video_synthetic_generator_config(tenant_id),
        backend_config=backend_config,
        agents_config=json.loads(Path("configs/config.json").read_text())["agents"],
        profile_labeler=label_profile,
        config_manager=config_manager,
    )
    try:
        yield SimpleNamespace(
            service=service,
            tenant_id=tenant_id,
            profile_name=profile_name,
            expected_queries={
                "find a video frame showing Marie Curie and Pierre Curie isolated",
                "find a video frame showing Tesla demonstrated an alternating-current motor",
            },
        )
    finally:
        backend.close()


@pytest.fixture
def client(real_service, monkeypatch):
    app = FastAPI()
    app.include_router(synthetic_api.router)
    monkeypatch.setattr(synthetic_api, "_service", real_service.service)
    with TestClient(app) as test_client:
        yield test_client


def test_batch_generate_reads_persisted_content_through_service(client, real_service):
    response = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "profile",
            "count_per_batch": 1,
            "num_batches": 2,
            "vespa_sample_size": 2,
            "max_profiles": 1,
            "tenant_id": real_service.tenant_id,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["optimizer"] == "profile"
    assert body["num_batches"] == 2
    assert body["examples_per_batch"] == 1
    assert body["total_examples"] == 2
    assert body["batches"] == [
        {
            "batch_index": 0,
            "count": 1,
            "profiles": [real_service.profile_name],
        },
        {
            "batch_index": 1,
            "count": 1,
            "profiles": [real_service.profile_name],
        },
    ]
    assert {example["query"] for example in body["data"]} == (
        real_service.expected_queries
    )
    assert len(body["data"]) == len(real_service.expected_queries) == 2
    assert body["data"] == [
        {
            "query": example["query"],
            "available_profiles": real_service.profile_name,
            "selected_profile": real_service.profile_name,
            "modality": "video",
            "complexity": "complex",
            "query_intent": "video_search",
            "reasoning": "The production selector chose the live video profile.",
        }
        for example in body["data"]
    ]
