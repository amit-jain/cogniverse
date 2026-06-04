"""Real-service coverage for POST /synthetic/batch/generate.

Drives the batch route through ``httpx`` ASGI against a real
``SyntheticDataService`` backed by the live LLM (routing optimizer needs no
Vespa profiles), so the per-batch generation loop and the aggregated envelope
are proven against real generation rather than a stubbed service.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic import api as synthetic_api
from cogniverse_synthetic.service import SyntheticDataService

pytestmark = pytest.mark.integration


@pytest.fixture
def real_service():
    generator_config = SyntheticGeneratorConfig(
        tenant_id="test:unit",
        optimizer_configs={
            "routing": OptimizerGenerationConfig(
                optimizer_type="routing",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                        module_type="Predict",
                    )
                },
            ),
        },
    )
    return SyntheticDataService(
        generator_config=generator_config,
        backend_config=BackendConfig(profiles={}, tenant_id="test:unit"),
    )


@pytest.fixture
def client(real_service, monkeypatch):
    app = FastAPI()
    app.include_router(synthetic_api.router)
    monkeypatch.setattr(synthetic_api, "_service", real_service)
    return TestClient(app)


def test_batch_generate_produces_real_examples(client, dspy_test_lm):
    resp = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "routing",
            "count_per_batch": 1,
            "num_batches": 2,
            "tenant_id": "test:unit",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["optimizer"] == "routing"
    assert body["num_batches"] == 2
    assert body["examples_per_batch"] == 1
    assert body["total_examples"] == 2
    assert [b["batch_index"] for b in body["batches"]] == [0, 1]
    assert all(b["count"] == 1 for b in body["batches"])

    # The aggregated payload carries the real generated routing examples.
    assert len(body["data"]) == 2
    for example in body["data"]:
        assert example["query"]
        assert "entities" in example
        assert "enhanced_query" in example
