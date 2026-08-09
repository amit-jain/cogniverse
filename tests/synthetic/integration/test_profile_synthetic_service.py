"""Profile examples use the production selector against real Vespa content."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest

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
from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.backend import VespaBackend
from tests.utils.synthetic_config import video_synthetic_generator_config
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def profile_service(shared_vespa):
    tenant_id = f"synprofile{uuid.uuid4().hex[:8]}:media"
    profile_name = "video_colpali_smol500_mv_frame"
    title = "Marie Curie discovered radium"
    description = "Marie Curie isolated radium in a Paris laboratory."
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    profiles = {
        profile_name: BackendProfileConfig(
            profile_name=profile_name,
            type="video",
            schema_name=profile_name,
            embedding_type="multi_vector",
            pipeline_config={"extract_keyframes": True},
        )
    }
    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        profiles=profiles,
        tenant_id=tenant_id,
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
    feed = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    ).feed_data_point(
        schema=schema,
        data_id="curie-radium-segment",
        fields={
            "video_id": "curie-radium",
            "video_title": title,
            "source_url": "http://example.test/curie-radium",
            "segment_id": 0,
            "segment_description": description,
            "start_time": 0.0,
            "end_time": 12.0,
        },
    )
    assert feed.is_successful(), feed.json

    for _ in range(20):
        indexed = backend.query_metadata_documents(
            schema=schema,
            yql=f"select * from sources {schema} where true limit 1",
            hits=1,
        )
        if indexed:
            assert indexed[0]["video_id"] == "curie-radium"
            assert indexed[0]["video_title"] == title
            assert indexed[0]["segment_description"] == description
            break
        time.sleep(0.5)
    else:
        pytest.fail("Curie source document was not indexed by Vespa")

    profile_agent = ProfileSelectionAgent(
        deps=ProfileSelectionDeps(available_profiles=[profile_name])
    )
    profile_agent.dspy_module.selector = lambda **_: dspy.Prediction(
        selected_profile=profile_name,
        confidence="0.96",
        reasoning="The deployed selector chose the indexed video profile.",
        query_intent="historical_fact_lookup",
        modality="video",
        complexity="medium",
    )

    async def label_profile(
        query: str, available_profiles: list[str], request_tenant_id: str
    ):
        assert request_tenant_id == tenant_id
        return await profile_agent.process(
            ProfileSelectionInput(
                query=query,
                available_profiles=available_profiles,
                tenant_id=request_tenant_id,
            )
        )

    agents_config = json.loads(Path("configs/config.json").read_text())["agents"]
    service = SyntheticDataService(
        backend=backend,
        generator_config=video_synthetic_generator_config(tenant_id),
        backend_config=backend_config,
        agents_config=agents_config,
        profile_labeler=label_profile,
    )
    try:
        yield SimpleNamespace(
            service=service,
            tenant_id=tenant_id,
            profile_name=profile_name,
            expected_query=title,
            agents_config=agents_config,
        )
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_service_generates_profile_examples(profile_service):
    request = SyntheticDataRequest(
        tenant_id=profile_service.tenant_id,
        optimizer="profile",
        count=1,
        vespa_sample_size=1,
        max_profiles=1,
    )
    response = await profile_service.service.generate(request)

    assert response.optimizer == "profile"
    assert response.schema_name == ProfileSelectionExampleSchema.__name__
    assert response.count == 1
    assert len(response.data) == 1
    assert response.selected_profiles == [profile_service.profile_name]
    assert response.metadata["sampled_content_count"] == 1

    for item in response.data:
        assert item["query"] == profile_service.expected_query
        assert item["available_profiles"] == profile_service.profile_name
        assert item["selected_profile"] == profile_service.profile_name
        assert item["modality"] == "video"
        assert item["complexity"] == "medium"
        assert item["query_intent"] == "historical_fact_lookup"
        assert "confidence" not in item
        assert item["reasoning"] == (
            "The deployed selector chose the indexed video profile."
        )


@pytest.mark.asyncio
async def test_service_rejects_profile_count_above_unique_indexed_topics(
    profile_service,
):
    with pytest.raises(ValueError) as error:
        await profile_service.service.generate(
            SyntheticDataRequest(
                tenant_id=profile_service.tenant_id,
                optimizer="profile",
                count=2,
                vespa_sample_size=1,
                max_profiles=1,
            )
        )

    assert str(error.value) == (
        "ProfileGenerator generated 1 unique grounded examples but "
        "target_count=2; source_context=1 unique source topics"
    )


@pytest.mark.requires_lm
@pytest.mark.asyncio
async def test_real_lm_profile_agent_labels_indexed_source_without_module_patch(
    profile_service,
    ensure_host_ollama,
    dspy_test_lm,
):
    _ = ensure_host_ollama
    agent = ProfileSelectionAgent(
        deps=ProfileSelectionDeps(available_profiles=[profile_service.profile_name])
    )

    async def label_profile(query, available_profiles, tenant_id):
        return await agent.process(
            ProfileSelectionInput(
                query=query,
                available_profiles=available_profiles,
                tenant_id=tenant_id,
            )
        )

    service = SyntheticDataService(
        backend=profile_service.service.backend,
        generator_config=profile_service.service.generator_config,
        backend_config=profile_service.service.backend_config,
        agents_config=profile_service.agents_config,
        profile_labeler=label_profile,
    )
    with dspy.context(lm=dspy_test_lm):
        response = await service.generate(
            SyntheticDataRequest(
                tenant_id=profile_service.tenant_id,
                optimizer="profile",
                count=1,
                vespa_sample_size=1,
                max_profiles=1,
            )
        )

    assert response.count == 1
    assert response.selected_profiles == [profile_service.profile_name]
    item = response.data[0]
    assert item["query"] == profile_service.expected_query
    assert item["available_profiles"] == profile_service.profile_name
    assert item["selected_profile"] == profile_service.profile_name
    assert item["modality"] in {"video", "image", "text", "audio", "document"}
    assert item["complexity"] in {"simple", "medium", "complex"}
    assert item["query_intent"].strip() == item["query_intent"]
    assert 3 <= len(item["query_intent"]) <= 100
    assert item["reasoning"].strip() == item["reasoning"]
    assert 10 <= len(item["reasoning"]) <= 1000
    assert "confidence" not in item


@pytest.mark.asyncio
async def test_service_response_serializes_to_optimizer_demo_shape(profile_service):
    """The optimizer's ``_load_approved_synthetic_data`` consumer reads
    each demo as ``{"input": <json string>}`` and re-instantiates a
    ``dspy.Example`` from the parsed dict. This test asserts the
    service response can be rendered into that shape and round-tripped
    back into a usable dict — i.e. the contract between
    ``run_synthetic_generation`` and ``run_profile_optimization`` holds
    for the new generator.
    """
    request = SyntheticDataRequest(
        tenant_id=profile_service.tenant_id,
        optimizer="profile",
        count=1,
        vespa_sample_size=1,
        max_profiles=1,
    )
    response = await profile_service.service.generate(request)

    for item in response.data:
        encoded = json.dumps(item, default=str)
        decoded = json.loads(encoded)
        assert isinstance(decoded, dict)
        assert decoded == {
            "query": profile_service.expected_query,
            "available_profiles": profile_service.profile_name,
            "selected_profile": profile_service.profile_name,
            "modality": "video",
            "complexity": "medium",
            "query_intent": "historical_fact_lookup",
            "reasoning": "The deployed selector chose the indexed video profile.",
        }
