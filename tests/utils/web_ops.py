"""The runtime the web client's operations views talk to, for browser tests.

The tenant admin, tenant self-service, approvals, orchestration annotation,
profile admin, agents, events and ingestion routers are mounted at the paths the
runtime mounts them on, over the caller's real config store and schema loader,
with a cluster-events channel of their own so tenant deletes and session
closes reach this worker the way they reach a runtime replica, a config
events channel of their own that config writes publish on, and a task
event store on the same Redis that uploads open their tasks in and tenant
deletes cancel them through.
The annotation queue is a real Redis queue under ``annotation_queue_prefix``,
and the embedding atlas cache keeps its generations in the same Redis.
Given an ingest processor, the ingestion worker's claim loop runs on the
server's loop against ``REDIS_URL`` with the same task event store, as a
worker pod runs it, so a cancelled job stops before it starts.

``serve_token_embedder`` answers the OpenAI ``/v1/embeddings`` contract the
memory store's DenseOn client speaks, for hosts that cannot fetch DenseOn.
``memory_on_vespa`` builds a Mem0 manager per tenant on real Vespa behind a
fault proxy, embedding with it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Awaitable, Callable, Dict, Iterable, Iterator, Optional, Tuple

from fastapi import FastAPI

from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_core.common.tenant_utils import parse_tenant_id
from cogniverse_core.memory.manager import Mem0MemoryManager, affirm_memory_profile
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.admin.models import Tenant
from cogniverse_runtime.atlas_projection import ProjectionCache, set_projection_cache
from cogniverse_runtime.cluster_events import (
    CONFIGS_CHANGED,
    ClusterEvents,
    release_held_configs,
)
from cogniverse_runtime.harness_keys import HarnessKeyStore
from cogniverse_runtime.ingestion_worker import status_api
from cogniverse_runtime.ingestion_worker import worker as ingest_worker
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis
from cogniverse_runtime.ingestion_worker.worker import WorkerConfig, _claim_loop
from cogniverse_runtime.routers import (
    admin,
    agents,
    approvals,
    config_entries,
    embedding_atlas,
    ingestion,
    openai_compat,
    orchestration_annotations,
    routing_decisions,
    telemetry_metrics,
    tenant,
)
from cogniverse_runtime.routers import events as events_router
from cogniverse_runtime.shared_state import connect_shared_state_redis
from cogniverse_runtime.task_events import TaskEventStore
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.vespa_test_helpers import deploy_tenant_schema
from tests.utils.web_client import serve_app


@contextmanager
def serve_ops_runtime(
    config_manager,
    schema_loader,
    redis_url: str,
    *,
    ingest_processor: Optional[Callable[..., Awaitable[dict]]] = None,
    annotation_queue_prefix: Optional[str] = None,
) -> Iterator[str]:
    """Serve the operations routes on a real uvicorn socket; yields its URL."""
    previous = (tm._config_manager, tm._schema_loader)
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    tenant.set_config_manager(config_manager)
    approvals.set_config_manager(config_manager)
    agents.set_agent_registry(
        AgentRegistry(tenant_id="default", config_manager=config_manager)
    )

    @asynccontextmanager
    async def cluster_events(_app):
        # Started on the server's loop, as the runtime starts its own.
        events = ClusterEvents(
            redis_url,
            f"web-ops-test-{uuid.uuid4().hex[:8]}",
            {
                "tenant_deleted": tm.release_deleted_tenant,
                "tenant_tier_set": tm.release_tenant_tier,
                "backend_profiles_changed": admin.release_backend_profiles,
                "session_closed": admin.sweep_closed_session,
            },
            channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        )
        await events.start()
        config_events = ClusterEvents(
            redis_url,
            f"web-ops-test-{uuid.uuid4().hex[:8]}",
            {CONFIGS_CHANGED: release_held_configs},
            channel=f"cogniverse:test-config-events:{uuid.uuid4().hex[:8]}",
        )
        await config_events.start()
        config_entries.set_config_events(config_events)
        shared_state = await connect_shared_state_redis(redis_url)
        set_projection_cache(ProjectionCache(shared_state))
        task_events = TaskEventStore(shared_state)
        task_events.start()
        agents.set_task_event_store(task_events)
        events_router.set_task_event_store(task_events)
        ingestion.set_task_event_store(task_events)
        tm.set_cluster_events(events)
        tm.set_task_event_store(task_events)
        admin.set_cluster_events(events)
        agents.set_annotation_queue(
            AnnotationQueue(
                shared_state,
                key_prefix=annotation_queue_prefix
                or f"web-ops-test:annotation-queue:{uuid.uuid4().hex[:8]}",
            )
        )
        stop = asyncio.Event()
        worker = None
        if ingest_processor is not None:
            ingest_worker._task_events = task_events
            config = WorkerConfig()
            config.claim_block_ms = 200
            worker = asyncio.create_task(
                _claim_loop(
                    await get_redis(config.redis_url),
                    config,
                    stop,
                    processor=ingest_processor,
                )
            )
        try:
            yield
        finally:
            stop.set()
            if worker is not None:
                await asyncio.wait_for(worker, timeout=20)
                ingest_worker._task_events = None
            await close_redis()
            agents.set_annotation_queue(None)
            tm.set_cluster_events(None)
            tm.set_task_event_store(None)
            admin.set_cluster_events(None)
            agents.set_task_event_store(None)
            events_router.set_task_event_store(None)
            ingestion.set_task_event_store(None)
            await task_events.close()
            set_projection_cache(None)
            await shared_state.aclose()
            await events.close()
            config_entries.set_config_events(None)
            await config_events.close()

    app = FastAPI(lifespan=cluster_events)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(config_entries.router, prefix="/admin")
    app.include_router(tm.router, prefix="/admin")
    app.include_router(tenant.router, prefix="/admin/tenant")
    app.include_router(approvals.router, prefix="/admin/tenant")
    app.include_router(orchestration_annotations.router, prefix="/admin/tenant")
    app.include_router(telemetry_metrics.router, prefix="/admin/tenant")
    app.include_router(routing_decisions.router, prefix="/admin/tenant")
    app.include_router(embedding_atlas.router, prefix="/admin/tenant")
    app.include_router(agents.router, prefix="/agents")
    app.include_router(events_router.router, prefix="/events")
    app.include_router(ingestion.router, prefix="/ingestion")
    app.include_router(status_api.router, prefix="/ingestion")
    app.dependency_overrides[ingestion.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[ingestion.get_schema_loader_dependency] = lambda: (
        schema_loader
    )
    try:
        with serve_app(app) as url:
            yield url
    finally:
        tm.set_config_manager(previous[0])
        tm.set_schema_loader(previous[1])
        admin.reset_dependencies()
        tenant.set_config_manager(None)
        approvals.set_config_manager(None)
        BackendRegistry.get_instance().clear_instances()


@contextmanager
def harness_key_admin(app: FastAPI, config_manager) -> Iterator[HarnessKeyStore]:
    """Mount on ``app`` the routes the web server acts for a tenant through,
    and resolve the keys it mints on ``/ag-ui``; yields the key store.

    The harness-key admin runs over ``config_manager``'s store. The tenant
    registry is mounted without a metadata backend, so it answers 503 and
    the server mints a tenant's key without confirming the tenant.
    """
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tm.router, prefix="/admin")
    previous = (tm._config_manager, tm._schema_loader)
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(None)
    admin.set_config_manager(config_manager)
    keys = HarnessKeyStore(config_manager.store)
    openai_compat.set_key_resolver(keys.resolve)
    try:
        yield keys
    finally:
        openai_compat.set_key_resolver(None)
        admin.reset_dependencies()
        tm.set_config_manager(previous[0])
        tm.set_schema_loader(previous[1])


def register_tenant(tenant_id: str) -> str:
    """Write ``tenant_id``'s row in the tenant registry the operations
    runtime serves, as ``POST /admin/tenants`` writes it but without deploying
    base schemas, so the web client's tenant check confirms it; returns the
    canonical id. A test seeds the tenants its views are chosen for."""
    org_id, tenant_name = parse_tenant_id(tenant_id)
    tenant = Tenant(
        tenant_full_id=f"{org_id}:{tenant_name}",
        org_id=org_id,
        tenant_name=tenant_name,
        created_at=int(time.time() * 1000),
        created_by="web-ops-test",
        status="active",
        schemas_deployed=[],
    )
    # Every field but the in-memory ``config``, which the registry does not
    # store.
    fields = {k: v for k, v in asdict(tenant).items() if k != "config"}
    with tm.metadata_backend() as backend:
        written = backend.create_metadata_document(
            schema="tenant_metadata", doc_id=tenant.tenant_full_id, fields=fields
        )
    assert written, f"the tenant registry did not take {tenant.tenant_full_id}"
    return tenant.tenant_full_id


EMBEDDING_DIMS = 768
_PROMPT = re.compile(r"^(document|query): ")


def token_embedding(text: str) -> list[float]:
    """A unit vector summing one signed hashed axis per word of ``text``,
    without the DenseOn prompt prefix: texts sharing words lie close."""
    vector = [0.0] * EMBEDDING_DIMS
    for word in re.findall(r"[a-z0-9]+", _PROMPT.sub("", text.lower())):
        digest = hashlib.sha256(word.encode()).digest()
        axis = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMS
        vector[axis] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


@contextmanager
def serve_token_embedder() -> Iterator[str]:
    """Serve ``token_embedding`` at ``/v1/embeddings``; yields the base URL."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 (http.server API)
            request = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            inputs = request["input"]
            inputs = [inputs] if isinstance(inputs, str) else inputs
            body = json.dumps(
                {
                    "object": "list",
                    "model": request["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": token_embedding(text),
                        }
                        for i, text in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }
            ).encode()
            self.send_response(200 if self.path == "/v1/embeddings" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def memory_on_vespa(
    vespa_instance: dict, config_manager, tenants: Iterable[str]
) -> Iterator[Tuple[Dict[str, Mem0MemoryManager], InterceptFaultProxy]]:
    """A Mem0 manager per tenant on real Vespa, reached through a fault proxy
    and embedding with ``serve_token_embedder``; yields the managers by tenant
    and the proxy."""
    tenants = list(tenants)
    affirm_memory_profile(config_manager)
    with (
        InterceptFaultProxy(vespa_instance["base_url"]) as proxy,
        serve_token_embedder() as embedder_url,
    ):
        endpoints = dict(vespa_instance, http_port=proxy.port)
        managers = {}
        for tenant_id in tenants:
            Mem0MemoryManager._instances.pop(tenant_id, None)
            deploy_tenant_schema(
                endpoints,
                tenant_id=tenant_id,
                base_schema_name="agent_memories",
                config_manager=config_manager,
            )
            manager = Mem0MemoryManager(tenant_id)
            manager.initialize(
                backend_host="http://127.0.0.1",
                backend_port=proxy.port,
                backend_config_port=vespa_instance["config_port"],
                base_schema_name="agent_memories",
                llm_model="memory-test-unused",
                embedding_model="lightonai/DenseOn",
                llm_base_url="http://127.0.0.1:9",
                embedder_base_url=embedder_url,
                auto_create_schema=False,
                config_manager=config_manager,
                schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            )
            managers[tenant_id] = manager
        try:
            yield managers, proxy
        finally:
            for tenant_id in tenants:
                Mem0MemoryManager._instances.pop(tenant_id, None)
