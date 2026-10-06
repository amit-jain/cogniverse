"""The runtime the web client's operations views talk to, for browser tests.

The tenant admin, tenant self-service, approvals, orchestration annotation,
profile admin, agents and ingestion routers are mounted at the paths the
runtime mounts them on, over the caller's real config store and schema loader,
with a cluster-events channel of their own so tenant deletes and session
closes reach this worker the way they reach a runtime replica.
The annotation queue is a real Redis queue under ``annotation_queue_prefix``.
Given an ingest processor, the ingestion worker's claim loop runs on the
server's loop against ``REDIS_URL``, as a worker pod runs it.

``serve_token_embedder`` answers the OpenAI ``/v1/embeddings`` contract the
memory store's DenseOn client speaks, for hosts that cannot fetch DenseOn.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Awaitable, Callable, Iterator, Optional

from fastapi import FastAPI

from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_runtime.ingestion_worker import status_api
from cogniverse_runtime.ingestion_worker.redis_client import close_redis, get_redis
from cogniverse_runtime.ingestion_worker.worker import WorkerConfig, _claim_loop
from cogniverse_runtime.routers import (
    admin,
    agents,
    approvals,
    ingestion,
    orchestration_annotations,
    routing_decisions,
    telemetry_metrics,
    tenant,
)
from cogniverse_runtime.shared_state import connect_shared_state_redis
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
                "session_closed": admin.sweep_closed_session,
            },
            channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        )
        await events.start()
        tm.set_cluster_events(events)
        admin.set_cluster_events(events)
        shared_state = await connect_shared_state_redis(redis_url)
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
            await close_redis()
            agents.set_annotation_queue(None)
            await shared_state.aclose()
            tm.set_cluster_events(None)
            admin.set_cluster_events(None)
            await events.close()

    app = FastAPI(lifespan=cluster_events)
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tm.router, prefix="/admin")
    app.include_router(tenant.router, prefix="/admin/tenant")
    app.include_router(approvals.router, prefix="/admin/tenant")
    app.include_router(orchestration_annotations.router, prefix="/admin/tenant")
    app.include_router(telemetry_metrics.router, prefix="/admin/tenant")
    app.include_router(routing_decisions.router, prefix="/admin/tenant")
    app.include_router(agents.router, prefix="/agents")
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
