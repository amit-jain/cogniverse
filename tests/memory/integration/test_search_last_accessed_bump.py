"""P2.3 — search_memory bumps last_accessed on every returned hit.

Without this wire, the lifecycle scheduler can prune an actively-queried
memory because its ``last_accessed`` (or fallback ``updated_at``) never
moves. The plan's recency-aware retention policies depend on a read
actually advancing the recency signal.

We assert the wire by spying on ``Mem0.update`` — that's the call
``_bump_last_accessed_for_hits`` makes per hit. (The Vespa-side
``last_accessed`` write is a Mem0/backend concern; what this commit
owns is the dispatcher → bump-helper → mem0.update wire.)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager

pytestmark = pytest.mark.integration


class _SpyMemory:
    """Mem0 stand-in that records every search + update call.

    Tracks ``metadata`` passed to update so the test can verify
    last_accessed actually rides through, not just that update was
    called. The previous version of the test only recorded
    ``(memory_id, data)`` and missed the bug where the metadata dict
    was built then discarded.
    """

    def __init__(self, hits: List[Dict[str, Any]]):
        self._hits = hits
        self.update_calls: List[Dict[str, Any]] = []

    def search(self, query, *, user_id, agent_id, limit, filters):
        return {"results": list(self._hits)}

    def update(self, memory_id, data=None, metadata=None):
        self.update_calls.append(
            {"memory_id": memory_id, "data": data, "metadata": metadata}
        )


def _fresh_mm(monkeypatch) -> Mem0MemoryManager:
    """Build a manager bypassing initialize() — only the search/bump path
    needs to be live, so the LLM/Vespa wiring is unnecessary.
    """
    Mem0MemoryManager._instances.clear()
    mm = Mem0MemoryManager(tenant_id="p23_tenant")
    # Reset the singleton's __init__ guard so re-running tests starts fresh.
    mm._initialized = True
    mm.tenant_id = "p23_tenant"
    mm.config = None
    mm._knowledge_registry = None
    return mm


class TestBumpHelper:
    def test_calls_update_once_per_hit(self, monkeypatch):
        mm = _fresh_mm(monkeypatch)
        spy = _SpyMemory(
            [
                {"id": "a", "memory": "x"},
                {"id": "b", "memory": "y"},
                {"id": "c", "memory": "z"},
            ]
        )
        mm.memory = spy

        out = mm.search_memory(
            query="anything",
            tenant_id="p23_tenant",
            agent_name="p23_agent",
            top_k=5,
        )
        assert len(out) == 3
        assert sorted(call["memory_id"] for call in spy.update_calls) == [
            "a",
            "b",
            "c",
        ]
        # F1.2 — every update call must carry a metadata dict with
        # last_accessed populated. The previous bug built the metadata
        # then dropped it on the floor; without this assertion the bump
        # was a no-op against the persistence layer.
        for call in spy.update_calls:
            md = call["metadata"]
            assert isinstance(md, dict), (
                f"metadata must be a dict; got {type(md)!r}. The bump "
                "helper must pass the augmented metadata so last_accessed "
                "actually persists."
            )
            assert "last_accessed" in md, (
                "metadata must contain a fresh last_accessed timestamp; "
                f"got keys {sorted(md.keys())}. Without this, the lifecycle "
                "scheduler cannot distinguish recently-accessed memories "
                "from stale ones and prunes active rows."
            )

    def test_skips_hits_without_id(self, monkeypatch):
        mm = _fresh_mm(monkeypatch)
        spy = _SpyMemory(
            [
                {"memory": "no id field"},  # skipped
                {"id": "a", "memory": "with id"},  # bumped
            ]
        )
        mm.memory = spy
        mm.search_memory(
            query="x",
            tenant_id="p23_tenant",
            agent_name="p23_agent",
            top_k=5,
        )
        assert len(spy.update_calls) == 1
        assert spy.update_calls[0]["memory_id"] == "a"

    def test_search_succeeds_when_update_raises(self, monkeypatch):
        # Best-effort contract: update failure must not break the search.
        class _BrokenSpy(_SpyMemory):
            def update(self, memory_id, data=None, metadata=None):
                raise RuntimeError("simulated mem0 update failure")

        mm = _fresh_mm(monkeypatch)
        spy = _BrokenSpy([{"id": "a", "memory": "x"}])
        mm.memory = spy

        out = mm.search_memory(
            query="x",
            tenant_id="p23_tenant",
            agent_name="p23_agent",
            top_k=5,
        )
        # Search still returns the hit.
        assert len(out) == 1
        assert out[0]["id"] == "a"

    def test_no_hits_no_updates(self, monkeypatch):
        mm = _fresh_mm(monkeypatch)
        spy = _SpyMemory([])
        mm.memory = spy
        out = mm.search_memory(
            query="x",
            tenant_id="p23_tenant",
            agent_name="p23_agent",
            top_k=5,
        )
        assert out == []
        assert spy.update_calls == []


class TestRealVespaSchemaIntegration:
    """Real Vespa: prove the bump helper runs against live retrieval.

    We seed via the spy seam (Mem0 strips arbitrary metadata in this
    env — see P2.1 boundary-spy rationale), then invoke search_memory
    against the real Vespa-backed manager and assert update was called.
    """

    def test_real_vespa_search_invokes_bump_for_each_hit(
        self, shared_memory_vespa, shared_denseon
    ):
        # Reuse the test_tenant pre-deployed schema.
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_vespa.config.config_store import VespaConfigStore
        from tests.utils.llm_config import get_llm_model

        Mem0MemoryManager._instances.clear()
        config_store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
        )
        cm = ConfigManager(store=config_store)
        cm.set_system_config(
            SystemConfig(
                backend_url="http://localhost",
                backend_port=shared_memory_vespa["http_port"],
                inference_service_urls={"denseon": shared_denseon},
            )
        )
        mm = Mem0MemoryManager(tenant_id="test_tenant")
        mm.initialize(
            backend_host="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            backend_config_port=shared_memory_vespa["config_port"],
            base_schema_name="agent_memories",
            llm_model=get_llm_model(),
            embedding_model="lightonai/DenseOn",
            llm_base_url="http://localhost:11434",
            embedder_base_url=shared_denseon,
            auto_create_schema=False,
            config_manager=cm,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        )

        # Write one memory, then count update() invocations on search.
        memory_id = mm.add_memory(
            content="P2.3 — actively-queried fact about X",
            tenant_id="test_tenant",
            agent_name="p23_real_agent",
            metadata={},
            infer=False,
        )
        assert memory_id is not None

        update_count = {"n": 0}
        real_update = mm.memory.update

        def _spy_update(*args, **kwargs):
            update_count["n"] += 1
            try:
                return real_update(*args, **kwargs)
            except Exception:
                # Some Mem0 builds reject minimal calls; still counts as
                # an attempted bump for wire-coverage.
                return None

        mm.memory.update = _spy_update  # type: ignore[method-assign]
        try:
            hits = mm.search_memory(
                query="P2.3",
                tenant_id="test_tenant",
                agent_name="p23_real_agent",
                top_k=5,
            )
            assert len(hits) >= 1, (
                "real Vespa search must return at least the seeded memory"
            )
            assert update_count["n"] == len(hits), (
                "search_memory must bump last_accessed once per returned hit; "
                f"got {update_count['n']} updates for {len(hits)} hits"
            )
        finally:
            mm.memory.update = real_update  # type: ignore[method-assign]
            try:
                mm.clear_agent_memory("test_tenant", "p23_real_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
