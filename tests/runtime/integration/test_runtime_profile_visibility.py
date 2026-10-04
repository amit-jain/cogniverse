"""Runtime-added profiles resolve per tenant, on every worker process.

Each worker is its own process running the runtime's real startup
(``cogniverse_runtime.main.lifespan``) with its own config manager, backend
registry and shared search backend. Profiles are written through the admin
routes of one worker and searched through the shared search backend of each
worker, against the test Vespa with a document fed through the tenant's
ingestion backend. A worker reads another worker's profile write within the
config manager's documented staleness bound.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import multiprocessing
import os
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.ingestion_client import document_namespace

pytestmark = pytest.mark.integration

_CONTEXT = multiprocessing.get_context("spawn")
STARTUP_TIMEOUT_S = 600
CALL_TIMEOUT_S = 300
# A write made on another worker is served within this bound.
VISIBILITY_BOUND_S = (
    inspect.signature(ConfigManager).parameters["scoped_config_max_staleness_s"].default
)
MEMORY_SCHEMA = "agent_memories"
EMBEDDING_DIMS = 768


def _search(tenant_id: str, profile: str, vector: list) -> tuple:
    """Search this worker's shared search backend as ``tenant_id``."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_runtime.routers import admin

    backend = BackendRegistry.get_instance().get_search_backend(
        name="vespa",
        config_manager=admin.get_config_manager_dependency(),
        schema_loader=admin.get_schema_loader_dependency(),
    )
    try:
        results = backend.search(
            {
                "query": "visibility probe",
                "type": "document",
                "profile": profile,
                "strategy": "semantic_search",
                "tenant_id": tenant_id,
                "top_k": 5,
                "query_embeddings": np.asarray(vector, dtype=np.float32),
            }
        )
    except Exception as exc:
        return ("refused", type(exc).__name__, str(exc))
    return ("hits", sorted(result.document.id for result in results))


def _ingest(tenant_id: str, profile: str, doc_id: str, text: str, vector: list):
    """Feed one embedded document into the schema ``profile`` names, the way
    the ingestion pipeline feeds what it embedded."""
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_runtime.routers import admin
    from cogniverse_sdk.document import Document

    config_manager = admin.get_config_manager_dependency()
    profile_config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    schema_name = profile_config.get("backend")["profiles"][profile]["schema_name"]
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        "vespa",
        tenant_id=tenant_id,
        config_manager=config_manager,
        schema_loader=admin.get_schema_loader_dependency(),
    )
    document = Document(
        id=doc_id,
        text_content=text,
        metadata={
            "user_id": "visibility",
            "agent_id": "visibility",
            "created_at": int(time.time()),
        },
    )
    document.add_embedding(
        "embedding",
        np.asarray(vector, dtype=np.float32),
        {"type": "float", "raw": True},
    )
    return backend.ingest_documents([document], schema_name=schema_name)


async def _serve(conn) -> None:
    import httpx

    from cogniverse_runtime import main as runtime_main

    async with runtime_main.lifespan(runtime_main.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=runtime_main.app),
            base_url="http://worker",
            timeout=CALL_TIMEOUT_S,
        ) as client:
            conn.send(("ready", os.getpid()))
            while True:
                command, args = await asyncio.to_thread(conn.recv)
                if command == "stop":
                    break
                try:
                    if command == "request":
                        method, path, body, params = args
                        response = await client.request(
                            method, path, json=body, params=params
                        )
                        conn.send(("ok", (response.status_code, response.json())))
                    elif command == "search":
                        conn.send(("ok", await asyncio.to_thread(_search, *args)))
                    elif command == "ingest":
                        conn.send(("ok", await asyncio.to_thread(_ingest, *args)))
                    else:
                        conn.send(("error", f"unknown command {command!r}"))
                except Exception as exc:
                    conn.send(("error", f"{type(exc).__name__}: {exc}"))


def _worker_main(conn, env: dict) -> None:
    os.environ.update(env)
    asyncio.run(_serve(conn))


class _Worker:
    """One runtime worker process, driven over a pipe."""

    def __init__(self, name: str, env: dict):
        self.name = name
        self._conn, child = _CONTEXT.Pipe()
        self.process = _CONTEXT.Process(
            target=_worker_main, args=(child, env), name=name, daemon=True
        )
        self.process.start()
        child.close()
        self.pid = None

    def wait_ready(self) -> None:
        status, self.pid = self._receive(STARTUP_TIMEOUT_S)
        assert (status, self.pid) == ("ready", self.process.pid)

    def _receive(self, timeout: float):
        if not self._conn.poll(timeout):
            raise AssertionError(
                f"worker {self.name} did not answer within {timeout}s "
                f"(exit code {self.process.exitcode})"
            )
        return self._conn.recv()

    def _call(self, command: str, *args):
        self._conn.send((command, args))
        status, value = self._receive(CALL_TIMEOUT_S)
        assert status == "ok", f"worker {self.name} {command} failed: {value}"
        return value

    def request(self, method: str, path: str, body=None, params=None):
        return self._call("request", method, path, body, params)

    def search(self, tenant_id: str, profile: str, vector: list) -> tuple:
        return self._call("search", tenant_id, profile, vector)

    def ingest(self, tenant_id, profile, doc_id, text, vector) -> dict:
        return self._call("ingest", tenant_id, profile, doc_id, text, vector)

    def stop(self) -> None:
        if self.process.is_alive():
            self._conn.send(("stop", ()))
            self.process.join(120)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(30)
        self._conn.close()


@pytest.fixture(scope="module")
def workers(vespa_instance, workflow_state_redis_url, semantic_embedder_env):
    env = {
        **semantic_embedder_env,
        "BACKEND_URL": "http://localhost",
        "BACKEND_PORT": str(vespa_instance["http_port"]),
        "REDIS_URL": workflow_state_redis_url,
        "COGNIVERSE_SANDBOX_POLICY": "disabled",
        "COGNIVERSE_MEMORY_LIFECYCLE_DISABLED": "1",
    }
    started = [_Worker("worker-1", env), _Worker("worker-2", env)]
    try:
        for worker in started:
            worker.wait_ready()
        yield SimpleNamespace(one=started[0], two=started[1])
    finally:
        for worker in started:
            worker.stop()


def _tenant(label: str) -> str:
    return canonical_tenant_id(f"pvis{label}{uuid.uuid4().hex[:8]}")


def _profile_body(tenant_id: str, profile_name: str, *, deploy_schema: bool) -> dict:
    return {
        "profile_name": profile_name,
        "tenant_id": tenant_id,
        "type": "document",
        "schema_name": MEMORY_SCHEMA,
        "embedding_model": "lightonai/DenseOn",
        "pipeline_config": {},
        "strategies": {},
        "embedding_type": "single_vector",
        "schema_config": {"embedding_dims": EMBEDDING_DIMS},
        "deploy_schema": deploy_schema,
    }


def _not_found(profile_name: str, outcome: tuple) -> list:
    """The profiles a "not found" refusal lists; fails on any other outcome."""
    prefix = f"Requested profile '{profile_name}' not found. Available profiles: "
    assert outcome[:2] == ("refused", "ValueError"), outcome
    assert outcome[2].startswith(prefix), outcome
    return ast.literal_eval(outcome[2][len(prefix) :])


def _poll(worker, tenant_id, profile, vector, accept, since: float) -> tuple:
    """Search until ``accept(outcome)``; the outcome and seconds since ``since``.

    Stops at twice the visibility bound, so a worker that never converges
    reports its last outcome and an elapsed time past the bound.
    """
    while True:
        outcome = worker.search(tenant_id, profile, vector)
        elapsed = time.monotonic() - since
        if accept(outcome) or elapsed > 2 * VISIBILITY_BOUND_S:
            return outcome, elapsed
        time.sleep(0.5)


def _tenant_schema(tenant_id: str) -> str:
    return f"{MEMORY_SCHEMA}_{tenant_id.replace(':', '_')}"


def _wait_until_queryable(http_port: int, schema: str, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        answer = requests.get(
            f"http://localhost:{http_port}/search/",
            params={"yql": f"select * from {schema} where true limit 0"},
            timeout=5,
        )
        if answer.status_code == 200 and "errors" not in answer.json().get("root", {}):
            return
        time.sleep(1)
    raise AssertionError(f"Vespa did not activate {schema} within {timeout_s}s")


@pytest.fixture(scope="module")
def corpus(workers, vespa_instance):
    """Tenant A, with a profile added at runtime on worker 1 and one document
    fed into its schema through the ingestion backend."""
    tenant = _tenant("a")
    profile = f"pvis_base_{uuid.uuid4().hex[:8]}"
    vector = np.random.default_rng(7).random(EMBEDDING_DIMS).astype(np.float32)
    doc_id = f"pvis_doc_{uuid.uuid4().hex[:8]}"
    text = f"runtime profile visibility {doc_id}"
    created = workers.one.request(
        "POST", "/admin/profiles", _profile_body(tenant, profile, deploy_schema=True)
    )
    _wait_until_queryable(vespa_instance["http_port"], _tenant_schema(tenant))
    fed = workers.one.ingest(tenant, profile, doc_id, text, vector.tolist())
    return SimpleNamespace(
        tenant=tenant,
        profile=profile,
        vector=vector,
        doc_id=doc_id,
        text=text,
        created=created,
        fed=fed,
    )


def test_a_document_fed_into_a_runtime_added_profile_is_stored_embedded_and_searchable(
    workers, corpus, vespa_instance
):
    status, body = corpus.created
    assert status == 201, body
    assert (body["profile_name"], body["tenant_id"], body["schema_deployed"]) == (
        corpus.profile,
        corpus.tenant,
        True,
    )
    assert body["tenant_schema_name"] == _tenant_schema(corpus.tenant)
    assert corpus.fed == {
        "success_count": 1,
        "failed_count": 0,
        "failed_documents": [],
        "total_documents": 1,
    }

    stored = requests.get(
        f"http://localhost:{vespa_instance['http_port']}/document/v1/"
        f"{document_namespace(_tenant_schema(corpus.tenant))}/"
        f"{_tenant_schema(corpus.tenant)}/docid/{corpus.doc_id}",
        timeout=10,
    )
    assert stored.status_code == 200, stored.text[:300]
    fields = stored.json()["fields"]
    assert (fields["text"], fields["user_id"], fields["agent_id"]) == (
        corpus.text,
        "visibility",
        "visibility",
    )
    assert np.array_equal(
        np.asarray(fields["embedding"]["values"], dtype=np.float32), corpus.vector
    )

    assert workers.one.search(
        corpus.tenant, corpus.profile, corpus.vector.tolist()
    ) == ("hits", [corpus.doc_id])
    outcome, _ = _poll(
        workers.two,
        corpus.tenant,
        corpus.profile,
        corpus.vector.tolist(),
        lambda found: found == ("hits", [corpus.doc_id]),
        since=time.monotonic(),
    )
    assert outcome == ("hits", [corpus.doc_id])


def test_another_tenant_cannot_resolve_or_see_a_runtime_added_profile(
    workers, corpus, vespa_instance
):
    other = _tenant("b")
    reader = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    try:
        visible_to_other = sorted(
            get_config(tenant_id=other, config_manager=reader).get("backend")[
                "profiles"
            ]
        )
    finally:
        reader.store.close()
    assert corpus.profile not in visible_to_other

    for worker in (workers.one, workers.two):
        listed = _not_found(
            corpus.profile,
            worker.search(other, corpus.profile, corpus.vector.tolist()),
        )
        assert sorted(listed) == visible_to_other
        absent = f"pvis_absent_{uuid.uuid4().hex[:8]}"
        listed = _not_found(
            absent, worker.search(other, absent, corpus.vector.tolist())
        )
        assert sorted(listed) == visible_to_other


def test_a_profile_added_on_one_worker_resolves_there_at_once_and_on_another_within_the_bound(
    workers, corpus
):
    profile = f"pvis_added_{uuid.uuid4().hex[:8]}"
    vector = corpus.vector.tolist()
    # Worker 2 holds the tenant's profiles from before the add.
    assert profile not in _not_found(
        profile, workers.two.search(corpus.tenant, profile, vector)
    )

    status, body = workers.one.request(
        "POST",
        "/admin/profiles",
        _profile_body(corpus.tenant, profile, deploy_schema=False),
    )
    written = time.monotonic()
    assert status == 201, body

    assert workers.one.search(corpus.tenant, profile, vector) == (
        "hits",
        [corpus.doc_id],
    )
    outcome, elapsed = _poll(
        workers.two,
        corpus.tenant,
        profile,
        vector,
        lambda found: found == ("hits", [corpus.doc_id]),
        since=written,
    )
    assert outcome == ("hits", [corpus.doc_id])
    assert elapsed < VISIBILITY_BOUND_S


def test_a_profile_deleted_on_one_worker_stops_resolving_on_another_within_the_bound(
    workers, corpus
):
    profile = f"pvis_deleted_{uuid.uuid4().hex[:8]}"
    vector = corpus.vector.tolist()
    status, body = workers.two.request(
        "POST",
        "/admin/profiles",
        _profile_body(corpus.tenant, profile, deploy_schema=False),
    )
    assert status == 201, body
    assert workers.two.search(corpus.tenant, profile, vector) == (
        "hits",
        [corpus.doc_id],
    )
    # Worker 1 serves the profile before it deletes it.
    outcome, _ = _poll(
        workers.one,
        corpus.tenant,
        profile,
        vector,
        lambda found: found == ("hits", [corpus.doc_id]),
        since=time.monotonic(),
    )
    assert outcome == ("hits", [corpus.doc_id])

    status, body = workers.one.request(
        "DELETE", f"/admin/profiles/{profile}", params={"tenant_id": corpus.tenant}
    )
    deleted = time.monotonic()
    assert status == 200, body
    assert profile not in _not_found(
        profile, workers.one.search(corpus.tenant, profile, vector)
    )

    outcome, elapsed = _poll(
        workers.two,
        corpus.tenant,
        profile,
        vector,
        lambda found: found[0] == "refused",
        since=deleted,
    )
    assert profile not in _not_found(profile, outcome)
    assert elapsed < VISIBILITY_BOUND_S
