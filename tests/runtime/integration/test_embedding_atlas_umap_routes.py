"""The UMAP Embedding Atlas routes over documents ingested into real Vespa.

Documents go through the production ingestion pipeline under a profile made
from the shipped ``document_text_semantic`` template, with a stand-in PyLate
sidecar encoding each token as a known vector, so documents sharing words
share most of their pooled vector. Maps are cached in a real Redis with their
generation, and worker processes are real processes sharing it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import multiprocessing
import pickle
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from cogniverse_runtime import atlas_projection
from cogniverse_runtime.atlas_projection import (
    LAYOUT_KEY,
    AtlasCacheUnavailableError,
    ProjectionCache,
    set_projection_cache,
)
from cogniverse_runtime.routers import admin, embedding_atlas
from cogniverse_runtime.shared_state import connect_shared_state_redis
from tests.utils.document_ingest import ingest_texts
from tests.utils.http_fault_proxy import HTTPFaultProxy
from tests.utils.pylate_stub import serve_pylate_stub, token_vector

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.usefixtures("profile_change_events"),
]

RIVERS = {
    "rivers-1.txt": "rivers carve deep canyons over thousands of years",
    "rivers-2.txt": "rivers carve canyons through soft rock over years",
    "rivers-3.txt": "old rivers carve wide canyons over many years",
    "rivers-4.txt": "rivers carve canyons and valleys over years",
}
VOLCANOES = {
    "volcanoes-1.txt": "volcanoes build islands from cooling lava flows",
    "volcanoes-2.txt": "volcanoes build new islands from lava",
    "volcanoes-3.txt": "lava from volcanoes build islands slowly",
    "volcanoes-4.txt": "volcanoes erupt lava that build islands",
}
TEXTS = {**RIVERS, **VOLCANOES}

_SPAWN = multiprocessing.get_context("spawn")
# How long a worker's build of the test map takes at least.
WORKER_BUILD_S = 1.0
# A query each worker places on the test map with its fitted reducer.
WORKER_QUERY = np.array([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def _worker_map(redis_url, tenant, generation, builds_key, barrier, results):
    """One runtime worker process: its own cache on the shared Redis, reading
    the test map once every worker is ready; puts its coords, computed_at,
    generation and the place of ``WORKER_QUERY`` on ``results``."""

    async def read():
        redis = await connect_shared_state_redis(redis_url)
        cache = ProjectionCache(redis)
        documents = [{"id": str(i), "title": None, "text": None} for i in range(5)]

        def build():
            SyncRedis.from_url(redis_url).incr(builds_key)
            time.sleep(WORKER_BUILD_S)
            return atlas_projection.build_map(documents, np.eye(5, 8))

        try:
            barrier.wait(60)
            built = await cache.get(tenant, "p", 5, generation, build)
            return (
                built.coords.tolist(),
                built.computed_at.isoformat(),
                built.generation,
                atlas_projection.place_queries(built, WORKER_QUERY).tolist(),
            )
        finally:
            await redis.aclose()

    results.put(asyncio.run(read()))


def _in_workers(count, redis_url, tenant, generation, builds_key):
    """The test map read by ``count`` worker processes at once."""
    barrier = _SPAWN.Barrier(count)
    results = _SPAWN.Queue()
    workers = [
        _SPAWN.Process(
            target=_worker_map,
            args=(redis_url, tenant, generation, builds_key, barrier, results),
        )
        for _ in range(count)
    ]
    for worker in workers:
        worker.start()
    read = [results.get(timeout=240) for _ in workers]
    for worker in workers:
        worker.join(60)
    assert [worker.exitcode for worker in workers] == [0] * count
    return read


def _pooled(text: str) -> np.ndarray:
    return np.mean([token_vector(token) for token in text.split()], axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


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


def _app(config_manager, schema_loader, redis_url, *, connect=True):
    """The atlas and admin routes with the atlas cache on ``redis_url``,
    connected (and pinged) at startup unless ``connect`` is false."""

    @asynccontextmanager
    async def cache(_app):
        redis = (
            await connect_shared_state_redis(redis_url)
            if connect
            else Redis.from_url(redis_url, socket_connect_timeout=2)
        )
        previous = atlas_projection.projection_cache()
        set_projection_cache(ProjectionCache(redis))
        try:
            yield
        finally:
            set_projection_cache(previous)
            await redis.aclose()

    app = FastAPI(lifespan=cache)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(embedding_atlas.router, prefix="/admin/tenant")
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    return app


@pytest.fixture(scope="module")
def client(config_manager, schema_loader, workflow_state_redis_url):
    with TestClient(
        _app(config_manager, schema_loader, workflow_state_redis_url)
    ) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def tenant(client, config_manager, schema_loader, pylate_service, tmp_path_factory):
    return ingest_texts(
        client, config_manager, schema_loader, tmp_path_factory.mktemp("umap"), TEXTS
    )


def _umap(client, tenant, **body):
    return client.post(
        f"/admin/tenant/{tenant}/embeddings/atlas/umap",
        json={"profile": "notes", **body},
    )


def _titles(body, cluster):
    return {p["title"] for p in body["points"] if p["cluster"] == cluster}


class TestUmapAtlas:
    def test_the_layout_clusters_each_topic_and_names_it(self, client, tenant):
        response = _umap(client, tenant)
        assert response.status_code == 200, response.text
        body = response.json()
        assert (
            body["tenant_id"],
            body["profile"],
            body["schema_name"],
            body["embedding_field"],
            body["dimensions"],
            body["without_embedding"],
            body["queries"],
        ) == (
            tenant,
            "notes",
            f"document_text_{tenant.replace(':', '_')}",
            "embedding",
            128,
            0,
            [],
        )
        assert {p["title"]: p["text"] for p in body["points"]} == TEXTS
        clusters = {c["id"]: c for c in body["clusters"]}
        assert sorted(
            (frozenset(_titles(body, cluster)), c["size"])
            for cluster, c in clusters.items()
        ) == sorted([(frozenset(RIVERS), 4), (frozenset(VOLCANOES), 4)])
        names = {
            frozenset(_titles(body, cluster)): c["label"]
            for cluster, c in clusters.items()
        }
        assert names == {
            frozenset(RIVERS): "rivers, canyons, carve",
            frozenset(VOLCANOES): "volcanoes, build, islands",
        }

    def test_a_query_is_placed_by_its_topic_with_its_nearest_documents(
        self, client, tenant
    ):
        query = "rivers carve canyons"
        body = _umap(client, tenant, queries=[query, "volcanoes build islands"]).json()
        rivers, volcanoes = body["queries"]
        assert (rivers["label"], rivers["text"]) == ("Query 1", query)
        assert volcanoes["label"] == "Query 2"

        by_title = {p["title"]: p for p in body["points"]}
        expected = sorted(
            TEXTS, key=lambda t: (-_cosine(_pooled(query), _pooled(TEXTS[t])), t)
        )[:3]
        assert [d["title"] for d in rivers["similar"]] == expected
        assert {d["title"] for d in rivers["similar"]} <= set(RIVERS)
        for document in rivers["similar"]:
            assert document["id"] == by_title[document["title"]]["id"]
            assert document["similarity"] == pytest.approx(
                _cosine(_pooled(query), _pooled(TEXTS[document["title"]])), abs=0.02
            )
        assert {d["title"] for d in volcanoes["similar"]} <= set(VOLCANOES)

        def centre(titles):
            return np.mean([[by_title[t]["x"], by_title[t]["y"]] for t in titles], 0)

        place = np.array([rivers["x"], rivers["y"]])
        assert np.linalg.norm(place - centre(RIVERS)) < np.linalg.norm(
            place - centre(VOLCANOES)
        )

    def test_a_layout_is_kept_until_it_is_invalidated(self, client, tenant):
        first = _umap(client, tenant, limit=100).json()
        again = _umap(client, tenant, limit=100).json()
        assert (again["computed_at"], again["generation"], again["points"]) == (
            first["computed_at"],
            first["generation"],
            first["points"],
        )
        retired = client.delete(
            f"/admin/tenant/{tenant}/embeddings/atlas/umap", params={"profile": "notes"}
        )
        assert retired.json() == {
            "tenant_id": tenant,
            "profile": "notes",
            "generation": first["generation"] + 1,
        }
        rebuilt = _umap(client, tenant, limit=100).json()
        assert rebuilt["generation"] == first["generation"] + 1
        assert rebuilt["computed_at"] > first["computed_at"]
        # The same set lays out the same way.
        assert rebuilt["points"] == first["points"]

    def test_fewer_documents_than_umap_needs_are_refused(
        self, client, config_manager, schema_loader, pylate_service, tmp_path
    ):
        few = ingest_texts(
            client,
            config_manager,
            schema_loader,
            tmp_path,
            {"a.txt": "one text", "b.txt": "two texts", "c.txt": "three texts"},
        )
        response = _umap(client, few)
        assert (response.status_code, response.json()["detail"]) == (
            422,
            "A UMAP map needs at least 4 documents with an embedding; profile "
            f"'notes' of tenant '{few}' has 3.",
        )

    def test_unknown_profiles_and_undeployed_schemas_are_named(self, client, tenant):
        missing = client.post(
            f"/admin/tenant/{tenant}/embeddings/atlas/umap", json={"profile": "nope"}
        )
        assert (missing.status_code, missing.json()["detail"]) == (
            404,
            f"No profile 'nope' for tenant '{tenant}'",
        )


class TestConcurrency:
    def test_concurrent_cold_reads_share_one_layout(
        self, client, config_manager, schema_loader, pylate_service, tmp_path
    ):
        fresh = ingest_texts(client, config_manager, schema_loader, tmp_path, RIVERS)
        barrier = threading.Barrier(6)

        def read(_):
            barrier.wait()
            response = _umap(client, fresh)
            assert response.status_code == 200, response.text
            return response.json()["computed_at"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            built = set(pool.map(read, range(6)))
        assert len(built) == 1

    async def test_one_build_serves_every_waiter_and_a_failed_build_is_not_kept(
        self, workflow_state_redis_url
    ):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        cache = ProjectionCache(redis)
        tenant = f"umapflight{uuid.uuid4().hex[:6]}:main"
        builds = []
        release = threading.Event()
        documents = [{"id": str(i), "title": None, "text": None} for i in range(5)]
        vectors = np.eye(5, 8)

        def build():
            builds.append(threading.get_ident())
            release.wait(10)
            return atlas_projection.build_map(documents, vectors)

        try:
            generation = await cache.generation(tenant, "p")
            waiters = [
                asyncio.create_task(cache.get(tenant, "p", 5, generation, build))
                for _ in range(5)
            ]
            await asyncio.sleep(0.2)
            release.set()
            maps = await asyncio.gather(*waiters)
            assert len(builds) == 1
            assert {id(m) for m in maps} == {id(maps[0])}

            def failing():
                builds.append("failed")
                raise ConnectionError("vespa down")

            other = f"umapfail{uuid.uuid4().hex[:6]}:main"
            results = await asyncio.gather(
                *(cache.get(other, "p", 5, 0, failing) for _ in range(3)),
                return_exceptions=True,
            )
            assert [type(r).__name__ for r in results] == ["ConnectionError"] * 3
            assert builds.count("failed") == 1
            recovered = await cache.get(other, "p", 5, 0, build)
            assert recovered.coords.shape == (5, 2)
        finally:
            await redis.aclose()

    async def test_an_invalidation_on_one_replica_retires_the_map_on_another(
        self, workflow_state_redis_url
    ):
        first_redis = await connect_shared_state_redis(workflow_state_redis_url)
        second_redis = await connect_shared_state_redis(workflow_state_redis_url)
        first, second = ProjectionCache(first_redis), ProjectionCache(second_redis)
        tenant = f"umapreplica{uuid.uuid4().hex[:6]}:main"
        documents = [{"id": str(i), "title": None, "text": None} for i in range(5)]
        builds = []

        def build():
            builds.append(1)
            return atlas_projection.build_map(documents, np.eye(5, 8))

        try:
            generation = await second.generation(tenant, "p")
            before = await second.get(tenant, "p", 5, generation, build)
            assert await second.get(tenant, "p", 5, generation, build) is before
            assert await first.invalidate(tenant, "p") == generation + 1
            current = await second.generation(tenant, "p")
            after = await second.get(tenant, "p", 5, current, build)
            assert (len(builds), after is before, after.generation) == (
                2,
                False,
                generation + 1,
            )
        finally:
            await first_redis.aclose()
            await second_redis.aclose()

    def test_worker_processes_reading_a_cold_map_at_once_build_it_once(
        self, workflow_state_redis_url
    ):
        """Three worker processes ask for the same map at the same moment:
        one builds it, and all three serve that one layout."""
        tenant = f"umapworkers{uuid.uuid4().hex[:6]}:main"
        builds_key = f"cogniverse:test:umap-builds:{tenant}"
        read = _in_workers(3, workflow_state_redis_url, tenant, 0, builds_key)
        redis = SyncRedis.from_url(workflow_state_redis_url)
        try:
            assert int(redis.get(builds_key)) == 1
        finally:
            redis.delete(builds_key)
            redis.close()
        assert [r == read[0] for r in read] == [True] * 3
        assert read[0][2] == 0

    async def test_another_worker_serves_the_stored_map_until_it_is_invalidated(
        self, workflow_state_redis_url
    ):
        """A map built by one worker process is served by the next one
        without a build; once invalidated, the next worker builds the new
        generation."""
        tenant = f"umapreuse{uuid.uuid4().hex[:6]}:main"
        builds_key = f"cogniverse:test:umap-builds:{tenant}"
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        cache = ProjectionCache(redis)
        try:
            generation = await cache.generation(tenant, "p")
            [first] = _in_workers(
                1, workflow_state_redis_url, tenant, generation, builds_key
            )
            [again] = _in_workers(
                1, workflow_state_redis_url, tenant, generation, builds_key
            )
            assert again == first
            assert int(await redis.get(builds_key)) == 1

            current = await cache.invalidate(tenant, "p")
            [rebuilt] = _in_workers(
                1, workflow_state_redis_url, tenant, current, builds_key
            )
            assert int(await redis.get(builds_key)) == 2
            assert (rebuilt[0], rebuilt[2]) == (first[0], generation + 1)
            assert rebuilt[1] > first[1]
        finally:
            await redis.delete(builds_key)
            await redis.aclose()


class _Planted:
    """Unpickling one touches ``marker``."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (Path(self.marker),))


class TestStoredLayout:
    def test_a_stored_map_is_a_numpy_archive_read_without_pickle(
        self, workflow_state_redis_url
    ):
        """The stored map is base64 of a numpy archive that loads with
        pickle refused: the layout and reducer state as arrays, the rest as
        JSON."""
        tenant = f"umapformat{uuid.uuid4().hex[:6]}:main"
        builds_key = f"cogniverse:test:umap-builds:{tenant}"
        [read] = _in_workers(1, workflow_state_redis_url, tenant, 0, builds_key)
        redis = SyncRedis.from_url(workflow_state_redis_url)
        layout_key = LAYOUT_KEY.format(
            tenant=tenant, profile="p", limit=5, generation=0
        )
        try:
            stored = redis.get(layout_key)
        finally:
            redis.delete(builds_key, layout_key)
            redis.close()
        with np.load(io.BytesIO(base64.b64decode(stored)), allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
        meta = json.loads(arrays.pop("meta").tobytes())
        assert sorted(arrays) == [
            "clusters",
            "coords",
            "reducer/_raw_data",
            "reducer/_rhos",
            "reducer/_sigmas",
            "reducer/embedding_",
            "reducer/graph_.data",
            "reducer/graph_.indices",
            "reducer/graph_.indptr",
            "vectors",
        ]
        assert arrays["coords"].tolist() == read[0]
        assert np.array_equal(arrays["reducer/_raw_data"], np.eye(5, 8))
        assert (
            meta["documents"],
            meta["cluster_names"],
            meta["computed_at"],
            meta["generation"],
            meta["source"],
            meta["reducer"]["values"]["metric"],
            meta["reducer"]["sparse"],
            meta["reducer"]["tuples"],
        ) == (
            [{"id": str(i), "title": None, "text": None} for i in range(5)],
            [],
            read[1],
            0,
            None,
            "euclidean",
            {"graph_": [5, 5]},
            ["precomputed_knn"],
        )

    async def test_a_pickled_entry_is_discarded_unread_and_the_map_built_again(
        self, workflow_state_redis_url, tmp_path, caplog
    ):
        """An entry that is not a stored archive (a pickle, as an older
        runtime wrote) is never unpickled: it is deleted, the map is built
        and the archive stored in its place."""
        marker = tmp_path / "unpickled"
        tenant = f"umappickle{uuid.uuid4().hex[:6]}:main"
        layout_key = LAYOUT_KEY.format(
            tenant=tenant, profile="p", limit=5, generation=0
        )
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        await redis.set(
            layout_key,
            base64.b64encode(
                zlib.compress(pickle.dumps(_Planted(str(marker))))
            ).decode(),
        )
        cache = ProjectionCache(redis)
        documents = [{"id": str(i), "title": None, "text": None} for i in range(5)]
        builds = []

        def build():
            builds.append(1)
            return atlas_projection.build_map(documents, np.eye(5, 8))

        try:
            with caplog.at_level(
                logging.WARNING, logger="cogniverse_runtime.atlas_projection"
            ):
                built = await cache.get(tenant, "p", 5, 0, build)
            stored = await redis.get(layout_key)
        finally:
            await redis.delete(layout_key)
            await redis.aclose()
        assert (marker.exists(), builds, built.generation) == (False, [1], 0)
        assert (
            atlas_projection._decoded(stored).coords.tolist() == built.coords.tolist()
        )
        assert [
            r.getMessage().split(": ", 1)[0]
            for r in caplog.records
            if r.name == "cogniverse_runtime.atlas_projection"
        ] == [f"Discarding the unreadable embedding atlas map {layout_key}"]


class TestFaultContract:
    def test_an_unreachable_cache_is_an_error_not_a_fresh_build(
        self, config_manager, schema_loader, tenant
    ):
        with TestClient(
            _app(config_manager, schema_loader, "redis://127.0.0.1:9/0", connect=False)
        ) as dead:
            response = _umap(dead, tenant)
        assert (response.status_code, response.json()["detail"]) == (
            503,
            {
                "error": "atlas_cache_unavailable",
                "message": "The embedding atlas cache could not be reached.",
                "failure": "ConnectionError",
                "tenant_id": tenant,
                "profile": "notes",
            },
        )

    async def test_a_cache_lost_after_the_generation_read_is_an_error_not_a_build(
        self,
    ):
        """Redis stops answering between the generation read and the stored
        layout: the read fails naming the outage and nothing is laid out."""
        cache = ProjectionCache(Redis.from_url("redis://127.0.0.1:9/0"))
        tenant = f"umaplost{uuid.uuid4().hex[:6]}:main"
        documents = [{"id": str(i), "title": None, "text": None} for i in range(5)]
        builds = []

        def build():
            builds.append(1)
            return atlas_projection.build_map(documents, np.eye(5, 8))

        with pytest.raises(AtlasCacheUnavailableError) as raised:
            await cache.get(tenant, "p", 5, 3, build)
        assert (builds, str(raised.value), type(raised.value.__cause__)) == (
            [],
            f"the embedding atlas cache did not answer for {tenant}/p (limit 5): "
            "ConnectionError: Error 111 connecting to 127.0.0.1:9. Connect call "
            "failed ('127.0.0.1', 9).",
            RedisConnectionError,
        )

    def test_a_cache_lost_mid_read_answers_503_and_lays_nothing_out(
        self, config_manager, schema_loader, tenant, workflow_state_redis_url
    ):
        """The generation reads, then the stored-layout read fails: the route
        answers 503 naming the outage, never a map built without the cache."""
        builds = []
        real_build = atlas_projection.build_map

        def counted(*args, **kwargs):
            builds.append(1)
            return real_build(*args, **kwargs)

        class _LayoutReadFails(ProjectionCache):
            async def _stored(self, layout_key):
                raise RedisConnectionError("layout read refused")

        @asynccontextmanager
        async def lifespan(_app):
            redis = await connect_shared_state_redis(workflow_state_redis_url)
            previous = atlas_projection.projection_cache()
            set_projection_cache(_LayoutReadFails(redis))
            try:
                yield
            finally:
                set_projection_cache(previous)
                await redis.aclose()

        app = FastAPI(lifespan=lifespan)
        app.include_router(embedding_atlas.router, prefix="/admin/tenant")
        admin.set_config_manager(config_manager)
        admin.set_schema_loader(schema_loader)
        with (
            TestClient(app) as faulty,
            patch.object(atlas_projection, "build_map", counted),
        ):
            response = _umap(faulty, tenant, limit=7)
        assert (builds, response.status_code, response.json()["detail"]) == (
            [],
            503,
            {
                "error": "atlas_cache_unavailable",
                "message": "The embedding atlas cache could not be reached.",
                "failure": "ConnectionError",
                "tenant_id": tenant,
                "profile": "notes",
            },
        )

    def test_a_refused_document_read_is_an_error_and_is_not_kept(
        self, client, config_manager, vespa_instance, tenant
    ):
        from cogniverse_core.registries.backend_registry import BackendRegistry

        client.delete(
            f"/admin/tenant/{tenant}/embeddings/atlas/umap", params={"profile": "notes"}
        )
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
                refused = _umap(client, tenant, limit=200)
            finally:
                system = config_manager.get_system_config()
                system.backend_port = http_port
                config_manager.set_system_config(system)
                registry.clear_instances()
        assert proxy.entered.is_set()
        assert (refused.status_code, refused.json()["detail"]) == (
            502,
            {
                "error": "embedding_export_failed",
                "message": "Reading the documents of profile 'notes' failed; the "
                "runtime log names the cause.",
                "failure": "RuntimeError",
                "tenant_id": tenant,
                "profile": "notes",
            },
        )
        recovered = _umap(client, tenant, limit=200)
        assert recovered.status_code == 200, recovered.text
        assert len(recovered.json()["points"]) == len(TEXTS)

    def test_a_down_query_encoder_is_an_error_naming_the_profile(
        self, client, config_manager, tenant
    ):
        _umap(client, tenant)
        system = config_manager.get_system_config()
        previous = dict(system.inference_service_urls)
        system.inference_service_urls["colbert_pylate"] = "http://127.0.0.1:9"
        config_manager.set_system_config(system)
        try:
            response = _umap(client, tenant, queries=["rivers"])
        finally:
            system = config_manager.get_system_config()
            system.inference_service_urls = previous
            config_manager.set_system_config(system)
        assert response.status_code == 502, response.text
        detail = response.json()["detail"]
        assert {k: detail[k] for k in ("error", "message", "tenant_id", "profile")} == {
            "error": "query_encoding_failed",
            "message": "The queries could not be encoded with profile 'notes'; the "
            "runtime log names the cause.",
            "tenant_id": tenant,
            "profile": "notes",
        }
