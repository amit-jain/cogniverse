"""The Embedding Atlas route over documents ingested into real Vespa.

Documents go through the production ingestion pipeline under a profile made
from the shipped ``document_text_semantic`` template, with a stand-in PyLate
sidecar encoding each token as a known vector. The map is checked against an
independent PCA of those vectors.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.decomposition import PCA

from cogniverse_runtime.routers import admin, embedding_atlas
from tests.utils.document_ingest import ingest_texts
from tests.utils.http_fault_proxy import HTTPFaultProxy
from tests.utils.profile_payload import profile_create_payload
from tests.utils.pylate_stub import serve_pylate_stub, token_vector

pytestmark = [pytest.mark.integration]

TEMPLATE = "document_text_semantic"
TEXTS = {
    "rivers.txt": "Rivers carve canyons over thousands of years",
    "glaciers.txt": "Glaciers grind valleys into wide troughs",
    "rivers_copy.txt": "Rivers carve canyons over thousands of years",
    "volcanoes.txt": "Volcanoes build islands from cooling lava",
}


@pytest.fixture(scope="module")
def client(config_manager, schema_loader):
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.include_router(embedding_atlas.router, prefix="/admin/tenant")
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def pylate_service(config_manager):
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


def _ingested_tenant(client, config_manager, schema_loader, tmp_path, texts) -> str:
    return ingest_texts(client, config_manager, schema_loader, tmp_path, texts)


def _atlas(client, tenant: str, **params):
    return client.get(
        f"/admin/tenant/{tenant}/embeddings/atlas",
        params={"profile": "notes", **params},
    )


def _pooled(text: str) -> np.ndarray:
    return np.mean([token_vector(token) for token in text.split()], axis=0)


@pytest.fixture(scope="module")
def tenant(client, config_manager, schema_loader, pylate_service, tmp_path_factory):
    return _ingested_tenant(
        client, config_manager, schema_loader, tmp_path_factory.mktemp("atlas"), TEXTS
    )


class TestAtlas:
    def test_each_document_is_placed_where_its_embedding_puts_it(self, client, tenant):
        response = _atlas(client, tenant)
        assert response.status_code == 200, response.text
        body = response.json()
        assert (
            body["tenant_id"],
            body["profile"],
            body["schema_name"],
            body["embedding_field"],
            body["dimensions"],
            body["without_embedding"],
        ) == (
            tenant,
            "notes",
            f"document_text_{tenant.replace(':', '_')}",
            "embedding",
            128,
            0,
        )
        points = {p["title"]: p for p in body["points"]}
        assert sorted(points) == sorted(TEXTS)
        assert {title: p["text"] for title, p in points.items()} == TEXTS

        # The same text lands on the same spot.
        rivers, copy = points["rivers.txt"], points["rivers_copy.txt"]
        assert (rivers["x"], rivers["y"]) == (copy["x"], copy["y"])

        # The map matches an independent PCA of the stand-in's vectors, up to
        # each axis's sign and the bfloat16 the schema stores.
        titles = [p["title"] for p in body["points"]]
        expected = PCA(n_components=2).fit(
            np.vstack([_pooled(TEXTS[t]) for t in titles])
        )
        reference = expected.transform(np.vstack([_pooled(TEXTS[t]) for t in titles]))
        mapped = np.array([[p["x"], p["y"]] for p in body["points"]])
        assert np.allclose(np.abs(mapped), np.abs(reference), atol=0.02)
        assert body["explained_variance"] == pytest.approx(
            expected.explained_variance_ratio_.tolist(), abs=0.01
        )

    def test_the_limit_caps_the_documents_read(self, client, tenant):
        body = _atlas(client, tenant, limit=2).json()
        assert len(body["points"]) == 2
        assert {p["title"] for p in body["points"]} <= set(TEXTS)

    def test_an_unknown_profile_or_undeployed_schema_is_named(self, client, tenant):
        missing = client.get(
            f"/admin/tenant/{tenant}/embeddings/atlas", params={"profile": "nope"}
        )
        assert (missing.status_code, missing.json()["detail"]) == (
            404,
            f"No profile 'nope' for tenant '{tenant}'",
        )
        undeployed = client.get(
            f"/admin/tenant/{tenant}/embeddings/atlas",
            params={"profile": "image_colpali_mv"},
        )
        assert (undeployed.status_code, undeployed.json()["detail"]) == (
            404,
            f"Schema 'image_colpali_mv' of profile 'image_colpali_mv' is not "
            f"deployed for tenant '{tenant}'",
        )


class TestConcurrency:
    def test_concurrent_maps_of_two_tenants_keep_each_tenants_documents(
        self, client, tenant, config_manager, schema_loader, pylate_service, tmp_path
    ):
        other_texts = {"deserts.txt": "Deserts spread where rain rarely falls"}
        other = _ingested_tenant(
            client, config_manager, schema_loader, tmp_path, other_texts
        )
        barrier = threading.Barrier(6)

        def read(name: str):
            barrier.wait()
            response = _atlas(client, name)
            assert response.status_code == 200, response.text
            return name, sorted(p["title"] for p in response.json()["points"])

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(read, [tenant, other] * 3))
        expected = {tenant: sorted(TEXTS), other: sorted(other_texts)}
        assert [titles == expected[name] for name, titles in results] == [True] * 6


class TestFaultContract:
    def test_a_refused_document_read_is_an_error_not_an_empty_map(
        self, client, config_manager, vespa_instance
    ):
        """With the tenant's schema deployed, its Vespa traffic is moved onto
        a proxy that refuses the document read."""
        from cogniverse_core.registries.backend_registry import BackendRegistry

        tenant = f"atlasdown{uuid.uuid4().hex[:6]}:main"
        templates = client.get(
            "/admin/profile-templates", params={"tenant_id": tenant}
        ).json()["templates"]
        template = next(t["config"] for t in templates if t["profile_name"] == TEMPLATE)
        created = client.post(
            "/admin/profiles", json=profile_create_payload("notes", template, tenant)
        )
        assert created.status_code == 201, created.text

        http_port = vespa_instance["http_port"]
        registry = BackendRegistry.get_instance()
        with HTTPFaultProxy(lambda path: f"http://localhost:{http_port}") as proxy:
            system = config_manager.get_system_config()
            system.backend_port = proxy.port
            config_manager.set_system_config(system)
            registry.clear_instances()
            try:
                proxy.arm(
                    lambda method, path, body: (
                        method == "GET" and path.startswith("/document/v1/")
                    ),
                    failure=True,
                )
                proxy.release.set()
                response = _atlas(client, tenant)
            finally:
                system = config_manager.get_system_config()
                system.backend_port = http_port
                config_manager.set_system_config(system)
                registry.clear_instances()
        assert proxy.entered.is_set()
        assert response.status_code == 502, response.text
        assert response.json()["detail"] == {
            "error": "embedding_export_failed",
            "message": "Reading the documents of profile 'notes' failed; the "
            "runtime log names the cause.",
            "failure": "RuntimeError",
            "tenant_id": tenant,
            "profile": "notes",
        }
