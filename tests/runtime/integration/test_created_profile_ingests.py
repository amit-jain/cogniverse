"""A profile created through ``POST /admin/profiles`` can be ingested into.

The profile starts from a shipped template read off ``/admin/profile-
templates``, is created with its schema deployed to real Vespa, and a text
document runs through the production ingestion pipeline under it. The
embedding sidecar is a stand-in serving the PyLate routes the ColBERT loader
calls; everything else (config store, schema deployment, segmentation, feed)
is real. The fed document is read back from Vespa.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vespa.application import Vespa

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_runtime.routers import admin
from tests.utils.profile_payload import profile_create_payload
from tests.utils.pylate_stub import serve_pylate_stub, token_vector

pytestmark = [pytest.mark.integration]

TEMPLATE = "document_text_semantic"
TEXT = "Glaciers carve valleys over thousands of years"


@pytest.fixture
def admin_client(config_manager, schema_loader):
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def pylate_service(config_manager):
    """The stand-in, registered under the service name the template names."""
    with serve_pylate_stub() as url:
        system = config_manager.get_system_config()
        previous = dict(system.inference_service_urls)
        system.inference_service_urls["colbert_pylate"] = url
        config_manager.set_system_config(system)
        try:
            yield url
        finally:
            system = config_manager.get_system_config()
            system.inference_service_urls = previous
            config_manager.set_system_config(system)


def test_a_document_ingests_through_a_profile_made_from_a_template(
    admin_client,
    pylate_service,
    config_manager,
    schema_loader,
    vespa_instance,
    tmp_path,
):
    tenant = f"profingest{uuid.uuid4().hex[:8]}:main"
    templates = admin_client.get(
        "/admin/profile-templates", params={"tenant_id": tenant}
    )
    assert templates.status_code == 200, templates.text
    template = next(
        t["config"]
        for t in templates.json()["templates"]
        if t["profile_name"] == TEMPLATE
    )
    assert template["model_loader"] == "colbert"
    assert template["inference_services"] == {"embedding": "colbert_pylate"}

    name = f"notes_{uuid.uuid4().hex[:6]}"
    created = admin_client.post(
        "/admin/profiles", json=profile_create_payload(name, template, tenant)
    )
    assert created.status_code == 201, created.text
    tenant_schema = created.json()["tenant_schema_name"]
    assert tenant_schema == f"document_text_{tenant.replace(':', '_')}"

    source = tmp_path / "glaciers.txt"
    source.write_text(TEXT)
    pipeline = VideoIngestionPipeline(
        tenant_id=tenant,
        config_manager=config_manager,
        schema_loader=schema_loader,
        schema_name=name,
    )
    result = asyncio.run(pipeline.process_video_async(source))
    assert result["status"] == "completed", result

    app = Vespa(url="http://localhost", port=vespa_instance["http_port"])
    tokens = TEXT.split()
    response = app.query(
        body={
            "yql": f"select document_title, full_text from {tenant_schema} where true",
            # Sibling schemas share the rank profile with other tensor dims.
            "model.restrict": tenant_schema,
            "ranking.profile": "float_float",
            "input.query(qt)": {str(i): token_vector(t) for i, t in enumerate(tokens)},
            "hits": 10,
        }
    )
    assert response.status_code == 200
    hits = [
        (hit["fields"]["document_title"], hit["fields"]["full_text"])
        for hit in response.hits
    ]
    assert hits == [("glaciers.txt", TEXT)]
    # MaxSim of the query against the stored token vectors: each query token
    # meets its own vector, so the stored embedding is the sidecar's.
    expected = sum(sum(x * x for x in token_vector(token)) for token in tokens)
    assert response.hits[0]["relevance"] == pytest.approx(expected, rel=2e-2)
