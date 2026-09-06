"""Real Vespa exports preserve schema routing, filter escaping, and failures."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from cogniverse_foundation.config.utils import get_config
from cogniverse_vespa._vespa_factory import make_vespa_app
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.vespa_test_helpers import deploy_tenant_schema

pytestmark = pytest.mark.integration


def test_export_embeddings_filter_is_applied_and_escaped(
    shared_vespa, config_manager, schema_loader
):
    tenant = f"ex{uuid.uuid4().hex[:6]}"
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant,
        base_schema_name="video_colpali_smol500_mv_frame",
        config_manager=config_manager,
    )

    http_port = shared_vespa["http_port"]
    vespa_app = make_vespa_app(url="http://localhost", port=http_port)
    target_id = 'v"a'  # embedded quote must be escaped in the selection
    for data_id, video_id, title in [
        ("doc_target", target_id, "Target"),
        ("doc_other", "other", "Other"),
    ]:
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id=data_id,
            namespace="content",
            fields={
                "video_id": video_id,
                "video_title": title,
                "source_url": f"http://example.test/{data_id}",
                "segment_id": 0,
            },
        )
        assert feed.is_successful(), feed.json

    cfg = get_config(tenant_id=tenant, config_manager=config_manager)
    backend = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": http_port,
            "profiles": cfg.get("backend", {}).get("profiles", {}),
            "default_profiles": cfg.get("backend", {}).get("default_profiles", {}),
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    # Poll until the visit returns the filtered doc.
    docs = []
    for _ in range(20):
        docs = backend.export_embeddings(
            schema=schema, filters={"video_id": target_id}, include_embeddings=False
        )
        if docs:
            break
        time.sleep(0.5)

    ids = [d.get("video_id") for d in docs]
    assert ids == [target_id], f"filter not applied/escaped; got {ids}"


@pytest.fixture(scope="module")
def configured_export_backend(shared_vespa, config_manager, schema_loader):
    tenant_suffix = uuid.uuid4().hex[:6]
    schemas = {}
    expected = {}
    vespa_app = make_vespa_app(url="http://localhost", port=shared_vespa["http_port"])
    for label in ("default", "override"):
        schema = deploy_tenant_schema(
            shared_vespa,
            tenant_id=f"export{label}{tenant_suffix}",
            base_schema_name="video_colpali_smol500_mv_frame",
            config_manager=config_manager,
        )
        schemas[label] = schema
        fields = {
            "video_id": f"{label}_video",
            "video_title": f"{label} title",
            "source_url": f"http://example.test/{label}",
            "segment_id": 3,
        }
        feed = vespa_app.feed_data_point(
            schema=schema,
            data_id="export_document",
            namespace="content",
            fields=fields,
        )
        assert feed.status_code == 200, feed.json
        expected[label] = [{"id": f"id:content:{schema}::export_document", **fields}]

    backend = VespaSearchBackend(
        backend_url="http://localhost",
        backend_port=shared_vespa["http_port"],
        schema_name=schemas["default"],
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    try:
        for label, schema in schemas.items():
            docs = []
            for _ in range(20):
                docs = backend.export_embeddings(schema=schema)
                if docs == expected[label]:
                    break
                time.sleep(0.5)
            assert docs == expected[label]
        yield backend, schemas, expected
    finally:
        backend.close()


def test_export_embeddings_uses_configured_schema_and_explicit_override(
    configured_export_backend,
):
    backend, schemas, expected = configured_export_backend

    assert backend.export_embeddings() == expected["default"]
    assert backend.export_embeddings(schema=schemas["override"]) == expected["override"]
    assert backend.export_embeddings() == expected["default"]


def test_export_embeddings_concurrent_override_preserves_configured_default(
    configured_export_backend,
):
    backend, schemas, expected = configured_export_backend
    labels = ["default", "override", "default", "override"]
    barrier = Barrier(len(labels))

    def export(label):
        barrier.wait(timeout=10)
        if label == "default":
            return backend.export_embeddings()
        return backend.export_embeddings(schema=schemas[label])

    with ThreadPoolExecutor(max_workers=len(labels)) as executor:
        results = list(executor.map(export, labels))

    assert results == [expected[label] for label in labels]
    assert backend.export_embeddings() == expected["default"]


def test_export_embeddings_configured_missing_schema_raises_with_visit_route(
    shared_vespa,
    configured_export_backend,
):
    _, schemas, _ = configured_export_backend
    missing_schema = "missing_" + schemas["default"]
    backend = VespaSearchBackend(
        backend_url="http://localhost",
        backend_port=shared_vespa["http_port"],
        schema_name=missing_schema,
    )
    try:
        with pytest.raises(RuntimeError) as raised:
            backend.export_embeddings()
        assert str(raised.value) == (
            "Vespa embedding export returned HTTP 400 from "
            f"http://localhost:{shared_vespa['http_port']}/document/v1/"
            f"content/{missing_schema}/docid"
        )
    finally:
        backend.close()
