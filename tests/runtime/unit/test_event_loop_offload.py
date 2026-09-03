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

from cogniverse_agents.graph.doc_extractor import ClaimExtractionResult
from cogniverse_agents.graph.graph_schema import ExtractionResult

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

    from cogniverse_agents.graph.graph_schema import DOCUMENT_MODALITY
    from cogniverse_runtime.routers import ingestion

    records = [
        SimpleNamespace(
            text="seg text",
            segment_anchor=SimpleNamespace(
                segment_id=f"s{i}", modality=DOCUMENT_MODALITY
            ),
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

    entity_calls = []
    claim_calls = []

    class StubDocExtractor:
        def __init__(self, **kwargs):
            pass

        def extract_entities_from_text(self, **kwargs):
            entity_calls.append(kwargs["segment_anchor"].segment_id)
            time.sleep(0.25)
            return SimpleNamespace(nodes=[], per_chunk_entity_names=[])

        def extract_claims_from_text(self, **kwargs):
            claim_calls.append(kwargs["segment_anchor"].segment_id)
            time.sleep(0.25)
            return ClaimExtractionResult()

    StubResult = ExtractionResult

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

    # Both passes ran for both segments; the two-pass fan-out completes them
    # concurrently, so assert the set (call order is not deterministic).
    assert set(entity_calls) == {"s0", "s1"}
    assert set(claim_calls) == {"s0", "s1"}
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


@pytest.mark.asyncio
async def test_graph_read_route_offloads_factory_resolution(monkeypatch):
    """A read route's manager-factory resolution runs off the loop. On a cold
    tenant the resolution does a blocking config read (and on the write path a
    seconds-long schema deploy); run inline it froze the whole API loop."""
    from types import SimpleNamespace

    from cogniverse_runtime.routers import graph as graph_router

    def _blocking_factory(tenant_id, deploy=True):
        time.sleep(0.3)  # blocking config read / schema deploy
        return SimpleNamespace(search_nodes=lambda q, top_k=10: [])

    saved = graph_router._graph_manager_factory
    graph_router.set_graph_manager_factory(_blocking_factory)

    async def _noop(tenant_id):
        return None

    monkeypatch.setattr(
        "cogniverse_core.common.tenant_utils.assert_tenant_exists", _noop
    )
    try:
        ticks = await _ticks_during(
            lambda: graph_router.search_nodes(tenant_id="acme:acme", q="x", top_k=5)
        )
    finally:
        graph_router._graph_manager_factory = saved

    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s graph-manager resolution — it ran on "
        "the event loop"
    )


@pytest.mark.asyncio
async def test_tenant_create_memory_offloaded(monkeypatch):
    """POST a user memory runs the blocking mem0 add (embedder HTTP + Vespa) off
    the loop — run inline it stalled every concurrent request."""
    from cogniverse_runtime.routers import tenant

    mgr = MagicMock()
    mgr.add_memory = _blocking(0.3)
    monkeypatch.setattr(tenant, "_get_memory_manager", lambda tid: mgr)

    req = tenant.MemoryCreateRequest(text="remember this")
    ticks = await _ticks_during(lambda: tenant.create_memory("acme:acme", req))
    assert ticks >= 10, f"only {ticks} ticks — add_memory ran on the loop"


@pytest.mark.asyncio
async def test_tenant_clear_memories_offloaded(monkeypatch):
    from cogniverse_runtime.routers import tenant

    mgr = MagicMock()
    mgr.clear_agent_memory = _blocking(0.3)
    monkeypatch.setattr(tenant, "_get_memory_manager", lambda tid: mgr)

    ticks = await _ticks_during(
        lambda: tenant.clear_memories("acme:acme", category=None)
    )
    assert ticks >= 10, f"only {ticks} ticks — clear_agent_memory ran on the loop"


@pytest.mark.asyncio
async def test_search_route_offloads_config_resolution(monkeypatch):
    """POST /search resolves the ConfigUtils ensure-chain (system + routing +
    telemetry + backend config reads) and builds the SearchService off the
    loop. A cold config read does synchronous Vespa I/O; run inline it stalled
    every concurrent request before the already-offloaded search even began."""
    from contextlib import contextmanager

    from cogniverse_runtime.routers import search as search_router

    async def _noop_tenant(tenant_id):
        return None

    monkeypatch.setattr(search_router, "assert_tenant_exists", _noop_tenant)

    @contextmanager
    def _noop_span(*a, **k):
        yield MagicMock()

    tm = MagicMock()
    tm.span = _noop_span
    tm.session_span = _noop_span
    monkeypatch.setattr(
        search_router, "get_telemetry_manager", lambda *args, **kwargs: tm
    )

    def _slow_get(*a, **k):
        time.sleep(0.3)  # cold ConfigUtils ensure-chain read
        return "video_profile"

    slow_config = MagicMock()
    slow_config.get = _slow_get
    monkeypatch.setattr(search_router, "get_config", lambda **k: slow_config)

    svc = MagicMock()
    svc.search.return_value = []
    monkeypatch.setattr(search_router, "SearchService", lambda **k: svc)

    req = search_router.SearchRequest(query="q", tenant_id="acme:acme")
    ticks = await _ticks_during(
        lambda: search_router.search(
            req, config_manager=MagicMock(), schema_loader=MagicMock()
        )
    )
    assert ticks >= 10, (
        f"only {ticks} ticks during a 0.3s config resolution — the ConfigUtils "
        "ensure-chain ran on the event loop"
    )


@pytest.mark.asyncio
async def test_tenant_get_memory_manager_init_offloaded(monkeypatch):
    """First-touch _get_memory_manager (Memory.from_config + possible schema
    deploy) runs off the loop; inline it stalled every concurrent request."""
    from cogniverse_runtime.routers import tenant

    monkeypatch.setattr(tenant, "_get_memory_manager", _blocking(0.3))
    req = tenant.MemoryCreateRequest(text="remember this")
    ticks = await _ticks_during(lambda: tenant.create_memory("acme:acme", req))
    assert ticks >= 10, f"only {ticks} ticks — the mem0 lazy-init ran on the loop"


@pytest.mark.asyncio
async def test_admin_pin_service_init_offloaded(monkeypatch):
    """_get_pin_service lazy-inits Mem0 (blocking Memory.from_config + schema
    deploy on a cold tenant); offload it off the loop like the mem op after it."""
    from cogniverse_runtime.routers import admin

    def _blocking_svc(*a, **k):
        time.sleep(0.3)
        svc = MagicMock()
        svc.list_pins.return_value = []
        return svc

    async def _no_overrides(tenant_id):
        return {}

    monkeypatch.setattr(admin, "_load_pin_quotas", _no_overrides)
    monkeypatch.setattr(admin, "_get_pin_service", _blocking_svc)
    ticks = await _ticks_during(lambda: admin.list_pins("acme:acme"))
    assert ticks >= 10, f"only {ticks} ticks — the pin-service init ran on the loop"


@pytest.mark.asyncio
async def test_knowledge_bind_graph_and_inject_offloaded(monkeypatch):
    """_inject_memory (Mem0 lazy-init) and _bind_graph (get_graph_manager — a
    blocking config read / schema deploy on a cold tenant) run off the loop."""
    from types import SimpleNamespace

    from cogniverse_runtime.routers import knowledge

    class _StubAgent:
        async def _process_impl(self, inp):
            return SimpleNamespace(model_dump=lambda: {"ok": True})

    monkeypatch.setattr(
        "cogniverse_agents.citation_tracing_agent.CitationTracingAgent",
        lambda **k: _StubAgent(),
    )
    monkeypatch.setattr(knowledge, "_inject_memory", lambda *a, **k: None)
    monkeypatch.setattr(knowledge, "_bind_graph", _blocking(0.3))

    body = knowledge.CitationTraceRequest(memory_id="m1")
    ticks = await _ticks_during(lambda: knowledge.citation_trace("acme:acme", body))
    assert ticks >= 10, f"only {ticks} ticks — _bind_graph ran on the event loop"


@pytest.mark.asyncio
async def test_coding_code_search_offloaded(monkeypatch):
    """The coding agent's code-context search (SearchService build + encoder
    inference + Vespa HTTP) runs off the loop — inline it froze every request
    while the coding agent fetched context."""
    from cogniverse_runtime.agent_dispatcher import AgentDispatcher

    d = object.__new__(AgentDispatcher)
    d._config_manager = MagicMock()
    d._schema_loader = MagicMock()

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config", lambda **k: MagicMock()
    )

    class _SlowService:
        def __init__(self, **k):
            pass

        def search(self, **k):
            time.sleep(0.3)  # encoder inference + Vespa HTTP
            return []

    monkeypatch.setattr("cogniverse_agents.search.service.SearchService", _SlowService)

    ticks = await _ticks_during(lambda: d._code_search("q", "acme:acme"))
    assert ticks >= 10, f"only {ticks} ticks — code search ran on the event loop"


def _stub_admin_memory_manager(monkeypatch, **methods):
    """Patch the admin routes' Mem0MemoryManager with a recording stub whose
    given methods are installed verbatim."""
    import cogniverse_core.memory.manager as manager_mod

    mgr = MagicMock()
    mgr.memory = object()
    for name, fn in methods.items():
        setattr(mgr, name, fn)
    cls = MagicMock(return_value=mgr)
    monkeypatch.setattr(manager_mod, "Mem0MemoryManager", cls)
    return mgr


@pytest.mark.asyncio
async def test_admin_delete_memory_offloaded(monkeypatch):
    from cogniverse_runtime.routers import admin

    _stub_admin_memory_manager(monkeypatch, delete_memory=_blocking(0.3))
    ticks = await _ticks_during(lambda: admin.admin_delete_memory("acme:acme", "mem-1"))
    assert ticks >= 10, f"only {ticks} ticks — delete_memory ran on the loop"


@pytest.mark.asyncio
async def test_admin_clear_memories_offloaded(monkeypatch):
    from cogniverse_runtime.routers import admin

    _stub_admin_memory_manager(monkeypatch, clear_agent_memory=_blocking(0.3))
    ticks = await _ticks_during(
        lambda: admin.admin_clear_memories("acme:acme", type=None)
    )
    assert ticks >= 10, f"only {ticks} ticks — clear_agent_memory ran on the loop"


@pytest.mark.asyncio
async def test_admin_drop_session_offloaded(monkeypatch):
    from cogniverse_runtime.routers import admin

    def _slow_drop(*a, **k):
        time.sleep(0.3)
        return {"chat": 2}

    _stub_admin_memory_manager(monkeypatch, drop_session=_slow_drop)
    ticks = await _ticks_during(lambda: admin.admin_drop_session("acme:acme", "sess-1"))
    assert ticks >= 10, f"only {ticks} ticks — drop_session ran on the loop"


@pytest.mark.asyncio
async def test_admin_close_session_sweep_offloaded(monkeypatch):
    from types import SimpleNamespace

    from cogniverse_runtime.routers import admin

    def _slow_drop(*a, **k):
        time.sleep(0.3)
        return {"chat": 1}

    warm = SimpleNamespace(
        tenant_id="acme:acme", memory=object(), drop_session=_slow_drop
    )
    import cogniverse_core.memory.manager as manager_mod

    monkeypatch.setattr(
        manager_mod,
        "Mem0MemoryManager",
        SimpleNamespace(_instances={"acme:acme": warm}),
    )
    ticks = await _ticks_during(lambda: admin.admin_close_session("sess-1"))
    assert ticks >= 10, f"only {ticks} ticks — the warm-tenant sweep ran on the loop"


def _stub_pin_service(monkeypatch, **methods):
    from cogniverse_runtime.routers import admin

    svc = MagicMock()
    for name, fn in methods.items():
        setattr(svc, name, fn)

    async def _svc_for(tenant_id):
        return svc

    monkeypatch.setattr(admin, "_pin_service_for", _svc_for)
    return svc


@pytest.mark.asyncio
async def test_admin_pin_offloaded(monkeypatch):
    from types import SimpleNamespace

    from cogniverse_runtime.routers import admin

    def _slow_pin(*a, **k):
        time.sleep(0.3)
        return SimpleNamespace(
            memory_id="p1",
            target_memory_id="m1",
            target_kind="chat",
            pinned_by=SimpleNamespace(value="user"),
            pinned_by_actor="alice",
        )

    _stub_pin_service(monkeypatch, pin=_slow_pin)
    body = admin.PinCreateRequest(
        target_kind="chat", pinned_by="user", actor_id="alice"
    )
    ticks = await _ticks_during(lambda: admin.pin_memory("acme:acme", "m1", body))
    assert ticks >= 10, f"only {ticks} ticks — svc.pin ran on the loop"


@pytest.mark.asyncio
async def test_admin_unpin_offloaded(monkeypatch):
    from cogniverse_runtime.routers import admin

    def _slow_unpin(*a, **k):
        time.sleep(0.3)
        return 1

    _stub_pin_service(monkeypatch, unpin=_slow_unpin)
    body = admin.PinUnpinRequest(requester_role="user", actor_id="alice")
    ticks = await _ticks_during(lambda: admin.unpin_memory("acme:acme", "m1", body))
    assert ticks >= 10, f"only {ticks} ticks — svc.unpin ran on the loop"


@pytest.mark.asyncio
async def test_admin_list_pins_call_offloaded(monkeypatch):
    from cogniverse_runtime.routers import admin

    def _slow_list(*a, **k):
        time.sleep(0.3)
        return []

    _stub_pin_service(monkeypatch, list_pins=_slow_list)
    ticks = await _ticks_during(lambda: admin.list_pins("acme:acme"))
    assert ticks >= 10, f"only {ticks} ticks — svc.list_pins ran on the loop"


@pytest.mark.asyncio
async def test_admin_promote_to_org_trunk_offloaded(monkeypatch):
    from types import SimpleNamespace

    import cogniverse_core.memory.federation as federation_mod
    from cogniverse_runtime.routers import admin

    def _slow_promote(*a, **k):
        time.sleep(0.3)
        return SimpleNamespace(
            source_memory_id="m1",
            promoted_memory_id="trunk-m1",
            org_trunk_tenant_id="acme:_org_trunk",
        )

    svc = MagicMock()
    svc.promote_to_org_trunk = _slow_promote
    monkeypatch.setattr(
        federation_mod, "FederationService", MagicMock(return_value=svc)
    )

    pin_svc = MagicMock()
    pin_svc._mm.memory.get = lambda mid: {"user_id": "acme:acme", "id": mid}

    async def _svc_for(tenant_id):
        return pin_svc

    monkeypatch.setattr(admin, "_pin_service_for", _svc_for)
    body = admin.PromoteToOrgTrunkRequest(actor_role="org_admin", actor_id="alice")
    ticks = await _ticks_during(
        lambda: admin.promote_to_org_trunk("acme:acme", "m1", body)
    )
    assert ticks >= 10, f"only {ticks} ticks — promote_to_org_trunk ran on the loop"


@pytest.mark.asyncio
async def test_admin_restore_memory_offloaded(monkeypatch):
    from cogniverse_runtime.routers import admin

    def _slow_restore(*a, **k):
        time.sleep(0.3)
        return True

    pin_svc = MagicMock()
    pin_svc._mm.restore_archived_memory = _slow_restore

    async def _svc_for(tenant_id):
        return pin_svc

    monkeypatch.setattr(admin, "_pin_service_for", _svc_for)
    ticks = await _ticks_during(lambda: admin.restore_memory("acme:acme", "m1"))
    assert ticks >= 10, f"only {ticks} ticks — restore_archived_memory ran on the loop"


@pytest.mark.asyncio
async def test_tenant_list_jobs_offloaded(monkeypatch):
    from cogniverse_runtime.routers import tenant

    cm = MagicMock()

    def _slow_list(*a, **k):
        time.sleep(0.3)
        return []

    cm.store.list_configs = _slow_list
    monkeypatch.setattr(tenant, "_require_config_manager", lambda: cm)

    ticks = await _ticks_during(lambda: tenant.list_jobs("acme:acme"))
    assert ticks >= 10, f"only {ticks} ticks — list_configs ran on the loop"
