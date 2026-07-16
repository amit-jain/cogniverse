"""Sync Vespa I/O on async paths runs off the event loop.

The wiki routes, the ingestion worker's graph upsert, and the lifecycle
scheduler's pin lookup call synchronous manager methods that do blocking Vespa
round-trips (the graph feed can even ``time.sleep`` during a convergence race).
Run inline, each stalls the whole loop. These tests prove a concurrent ticker
keeps firing while the blocking call is in flight — i.e. it was offloaded.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


async def _ticks_during(slow_coro_factory) -> int:
    """Run slow_coro_factory() while a 10ms ticker counts; return tick count.

    If the awaited work blocks the loop, the ticker is starved (~0-1 ticks);
    if it was offloaded via to_thread, the ticker keeps firing (many ticks).
    """
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    await slow_coro_factory()
    stop.set()
    await t
    return ticks


def _blocking(duration: float):
    def _fn(*a, **k):
        time.sleep(duration)
        return MagicMock()

    return _fn


@pytest.mark.asyncio
async def test_wiki_search_route_offloads_blocking_search(monkeypatch):
    from cogniverse_runtime.routers import wiki

    wm = MagicMock()
    wm.search = _blocking(0.3)
    monkeypatch.setattr(wiki, "get_wiki_manager_for_tenant", lambda t: wm)
    req = wiki.WikiSearchRequest(query="q", tenant_id="acme:acme", top_k=5)

    ticks = await _ticks_during(lambda: wiki.search_wiki(req))

    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s wiki search — the blocking Vespa "
        "search ran on the event loop"
    )


@pytest.mark.asyncio
async def test_lifecycle_pin_lookup_offloaded(monkeypatch):
    from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler

    warm = MagicMock()
    warm.tenant_id = "acme:acme"
    warm.cleanup_with_schema.return_value = {}

    sched = LifecycleScheduler(
        get_warm_managers=lambda: [warm],
        registry=MagicMock(),
        pin_lookup=_blocking(0.3),
    )

    ticks = await _ticks_during(sched.tick_once)

    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s pin lookup — it ran on the loop"
    )


@pytest.mark.asyncio
async def test_kg_extraction_and_face_pipeline_offloaded(monkeypatch):
    """Per-segment GLiNER/claim extraction (240s HTTP timeouts) and the face
    pipeline (per-keyframe HTTP + CPU clustering) dwarf the already-offloaded
    graph upsert — inline they froze the worker loop for minutes, deferring
    SIGTERM until k8s SIGKILLed the pod mid-extraction."""
    from types import SimpleNamespace

    from cogniverse_runtime.routers import ingestion

    records = [
        SimpleNamespace(
            text="seg text", segment_anchor=SimpleNamespace(segment_id=f"s{i}")
        )
        for i in range(2)
    ]
    monkeypatch.setattr(
        ingestion, "_iter_segments_for_graph", lambda pr, sdid: iter(records)
    )
    monkeypatch.setattr(
        ingestion, "_lookup_artifact_manager", lambda t, cm: MagicMock()
    )
    monkeypatch.setattr(
        ingestion, "_resolve_tenant_llm_config", lambda t, cm: MagicMock()
    )
    monkeypatch.setattr(
        ingestion, "_lookup_face_embed_endpoint", lambda cm: "http://face:1"
    )

    face_calls = []

    def blocking_face(**kwargs):
        face_calls.append(kwargs["source_doc_id"])
        time.sleep(0.25)
        return [], []

    monkeypatch.setattr(ingestion, "_run_face_pipeline", blocking_face)

    async def fake_backrefs(**kwargs):
        return None

    monkeypatch.setattr(ingestion, "_write_backrefs_to_content", fake_backrefs)

    extract_calls = []

    class StubDocExtractor:
        def __init__(self, **kwargs):
            pass

        def extract_from_text(self, **kwargs):
            extract_calls.append(kwargs["segment_anchor"].segment_id)
            time.sleep(0.25)
            return SimpleNamespace(nodes=[], edges=[])

    class StubResult:
        def __init__(self, source_doc_id="", nodes=(), edges=(), file_sha256=None):
            self.source_doc_id = source_doc_id
            self.nodes = list(nodes)
            self.edges = list(edges)
            self.file_sha256 = file_sha256

    class StubLinker:
        def link(self, combined):
            return combined

    mgr = MagicMock()
    mgr.upsert.return_value = {"nodes_upserted": 0, "edges_upserted": 0}
    mgr._backend = MagicMock()
    graph_router = SimpleNamespace(_graph_manager_factory=lambda t: mgr)

    ticks = await _ticks_during(
        lambda: ingestion._extract_graph_per_segment_inner(
            processing_results={},
            source_doc_id="doc1",
            tenant_id="acme:acme",
            config_manager=MagicMock(),
            DocExtractor=StubDocExtractor,
            ClaimExtractor=MagicMock,
            CrossModalLinker=StubLinker,
            ExtractionResult=StubResult,
            graph_router=graph_router,
        )
    )

    assert extract_calls == ["s0", "s1"]
    assert face_calls == ["doc1"]
    assert ticks >= 30, (
        f"event loop starved during KG extraction: only {ticks} ticks — "
        "the extractor/face calls ran inline on the loop"
    )


@pytest.mark.asyncio
async def test_admin_schema_deploy_offloaded(monkeypatch):
    """schema_registry.deploy_schema blocks through prepareandactivate +
    convergence sleeps; called inline it froze every request on the API loop
    (the tenant-manager already offloads the same call)."""
    from types import SimpleNamespace

    from cogniverse_runtime.admin.profile_models import SchemaDeploymentRequest
    from cogniverse_runtime.routers import admin

    cm = MagicMock()
    cm.get_backend_profile.return_value = SimpleNamespace(schema_name="wiki_pages")

    backend = MagicMock()
    backend.schema_exists.return_value = False
    backend.schema_registry.deploy_schema = _blocking(0.3)
    backend.get_tenant_schema_name.return_value = "wiki_pages_acme_acme"
    monkeypatch.setattr(
        admin.BackendRegistry,
        "get_instance",
        classmethod(
            lambda cls: SimpleNamespace(get_ingestion_backend=lambda *a, **k: backend)
        ),
    )

    req = SchemaDeploymentRequest(tenant_id="acme:acme")
    ticks = await _ticks_during(
        lambda: admin.deploy_profile_schema(
            "wiki_semantic", req, config_manager=cm, schema_loader=MagicMock()
        )
    )

    assert ticks >= 15, f"event loop starved during schema deploy: only {ticks} ticks"


@pytest.mark.asyncio
async def test_coding_sandbox_exec_offloaded():
    """CodingAgent._execute_in_sandbox runs the sync gRPC sandbox exec (and the
    connectivity probe) off the loop. Run inline, the write+run execs (up to
    30s + 300s) plus the .available TCP probe froze the whole API loop and
    tripped k8s liveness mid-task."""
    from cogniverse_agents.coding_agent import CodingAgent

    class _BlockingSandbox:
        @property
        def available(self):
            time.sleep(0.1)  # blocking connectivity probe (socket.create_connection)
            return True

        def exec_in_sandbox(self, agent_type, command, timeout_seconds):
            time.sleep(0.15)  # blocking gRPC session exec
            return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    agent = object.__new__(CodingAgent)
    agent._sandbox_manager = _BlockingSandbox()

    ticks = await _ticks_during(
        lambda: agent._execute_in_sandbox(
            file_path="/workspace/solution.py",
            code="print('hi')",
            test_command="python solution.py",
            language="python",
        )
    )

    assert ticks >= 30, (
        f"event loop starved during sandbox exec: only {ticks} ticks — the sync "
        "gRPC exec/probe ran on the loop"
    )
