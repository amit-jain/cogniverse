"""Integration test: full SyntheticDataService dispatch for the profile
optimizer.

ProfileGenerator is pattern-based (no DSPy LM dependency), so this test
exercises the request -> registry -> service -> generator -> response
flow end-to-end without external services. Confirms profile examples
flow through the same pipeline as the other optimizer types.
"""

import json

import pytest

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService


@pytest.fixture
def profile_service():
    return SyntheticDataService(
        generator_config=SyntheticGeneratorConfig(tenant_id="test:profile"),
        backend_config=BackendConfig(profiles={}, tenant_id="test:profile"),
    )


@pytest.mark.asyncio
async def test_service_generates_profile_examples(profile_service):
    request = SyntheticDataRequest(
        tenant_id="test:profile", optimizer="profile", count=8
    )
    response = await profile_service.generate(request)

    assert response.optimizer == "profile"
    assert response.schema_name == ProfileSelectionExampleSchema.__name__
    assert response.count == 8
    assert len(response.data) == 8

    for item in response.data:
        for field in (
            "query",
            "available_profiles",
            "selected_profile",
            "modality",
            "complexity",
            "query_intent",
            "confidence",
            "reasoning",
        ):
            assert field in item, f"missing {field} in {item}"
        available = [p.strip() for p in item["available_profiles"].split(",")]
        assert item["selected_profile"] in available


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
        tenant_id="test:profile", optimizer="profile", count=3
    )
    response = await profile_service.generate(request)

    for item in response.data:
        encoded = json.dumps(item, default=str)
        decoded = json.loads(encoded)
        assert isinstance(decoded, dict)
        assert decoded["selected_profile"]
        assert decoded["query"]
        assert decoded["modality"] in {
            "video",
            "image",
            "audio",
            "document",
            "text",
        }
        assert decoded["complexity"] in {"simple", "medium", "complex"}
