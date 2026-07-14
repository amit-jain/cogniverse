"""search_memory bumps last_accessed on every returned hit.

Without this wire, the lifecycle scheduler can prune an actively-queried
memory because its ``last_accessed`` (or fallback ``updated_at``) never
moves. The plan's recency-aware retention policies depend on a read
actually advancing the recency signal.

We assert the wire by spying on the vector store's partial update —
that's the call ``_bump_last_accessed_for_hits`` makes per hit, with
``vector=None`` so the stored embedding is preserved and nothing is
re-embedded. Mem0's ``Memory.update`` must NOT be used here: it
re-embeds the memory text on every call, which made each search pay
``top_k`` embedder round-trips just to stamp a timestamp.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager

pytestmark = pytest.mark.integration


class _SpyVectorStore:
    """Vector-store stand-in recording partial-update calls.

    ``update_calls`` records one entry per bumped memory regardless of the
    path taken; ``round_trips`` counts backend calls — the batched path must
    stamp N hits in ONE round-trip, not N.
    """

    def __init__(self):
        self.update_calls: List[Dict[str, Any]] = []
        self.round_trips = 0

    def update(self, vector_id, vector=None, payload=None):
        self.round_trips += 1
        self.update_calls.append(
            {"vector_id": vector_id, "vector": vector, "payload": payload}
        )

    def update_many(self, items):
        self.round_trips += 1
        for vector_id, vector, payload in items:
            self.update_calls.append(
                {"vector_id": vector_id, "vector": vector, "payload": payload}
            )


class _SpyMemory:
    """Mem0 stand-in that records searches, exposes a spy vector store for
    the bump path, and records any (forbidden) ``Memory.update`` calls —
    that path re-embeds the memory text per call.
    """

    def __init__(self, hits: List[Dict[str, Any]]):
        self._hits = hits
        self.vector_store = _SpyVectorStore()
        self.reembedding_update_calls: List[str] = []

    @property
    def update_calls(self) -> List[Dict[str, Any]]:
        return self.vector_store.update_calls

    def search(self, query, *, user_id, agent_id, limit, filters):
        return {"results": list(self._hits)}

    def update(self, memory_id, data=None, metadata=None):
        self.reembedding_update_calls.append(memory_id)


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
    def test_bumps_every_hit_in_one_round_trip(self, monkeypatch):
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
        assert spy.vector_store.round_trips == 1, (
            "the bump must batch all hits into one backend round-trip — "
            "per-hit updates cost top_k sequential HTTP calls per search"
        )
        assert spy.reembedding_update_calls == [], (
            "the bump must go through the vector store's partial update, "
            "not Memory.update — the latter re-embeds every hit"
        )
        assert sorted(call["vector_id"] for call in spy.update_calls) == [
            "a",
            "b",
            "c",
        ]
        for call in spy.update_calls:
            assert call["vector"] is None, (
                "bump must not write an embedding — vector=None keeps the "
                "stored tensor intact"
            )
        # every update call must carry a metadata dict with
        # last_accessed populated. The previous bug built the metadata
        # then dropped it on the floor; without this assertion the bump
        # was a no-op against the persistence layer.
        for call in spy.update_calls:
            md = (call["payload"] or {}).get("metadata")
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
        assert spy.update_calls[0]["vector_id"] == "a"

    def test_search_succeeds_when_update_raises(self, monkeypatch):
        # Best-effort contract: update failure must not break the search.
        class _BrokenStore(_SpyVectorStore):
            def update(self, vector_id, vector=None, payload=None):
                raise RuntimeError("simulated store update failure")

            def update_many(self, items):
                raise RuntimeError("simulated store batch update failure")

        class _BrokenSpy(_SpyMemory):
            def __init__(self, hits):
                super().__init__(hits)
                self.vector_store = _BrokenStore()

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

    def test_recently_bumped_hit_is_skipped(self, monkeypatch):
        """A hit stamped within the bump interval must not be re-written —
        recency needs day-scale fidelity, not a Vespa write per request."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        recent = (now - timedelta(seconds=60)).isoformat()
        stale = (now - timedelta(hours=2)).isoformat()

        mm = _fresh_mm(monkeypatch)
        spy = _SpyMemory(
            [
                {"id": "fresh", "memory": "x", "metadata": {"last_accessed": recent}},
                {
                    "id": "stale",
                    "memory": "y",
                    "metadata": {"last_accessed": stale, "topic": "k8s"},
                },
            ]
        )
        mm.memory = spy
        mm.search_memory(
            query="x",
            tenant_id="p23_tenant",
            agent_name="p23_agent",
            top_k=5,
        )
        assert [c["vector_id"] for c in spy.update_calls] == ["stale"]
        md = spy.update_calls[0]["payload"]["metadata"]
        assert md["topic"] == "k8s", "existing metadata keys must be preserved"
        assert md["last_accessed"] != stale

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
        assert spy.vector_store.round_trips == 0

    def test_store_without_update_many_falls_back_to_per_hit(self, monkeypatch):
        """Duck compatibility: a vector store exposing only update() still
        gets every hit stamped, one call per hit."""

        class _SingleOnlyStore:
            def __init__(self):
                self.update_calls: List[Dict[str, Any]] = []
                self.round_trips = 0

            def update(self, vector_id, vector=None, payload=None):
                self.round_trips += 1
                self.update_calls.append(
                    {"vector_id": vector_id, "vector": vector, "payload": payload}
                )

        class _SingleOnlySpy(_SpyMemory):
            def __init__(self, hits):
                super().__init__(hits)
                self.vector_store = _SingleOnlyStore()

        mm = _fresh_mm(monkeypatch)
        spy = _SingleOnlySpy([{"id": "a", "memory": "x"}, {"id": "b", "memory": "y"}])
        mm.memory = spy

        mm.search_memory(
            query="x",
            tenant_id="p23_tenant",
            agent_name="p23_agent",
            top_k=5,
        )
        assert sorted(c["vector_id"] for c in spy.update_calls) == ["a", "b"]
        assert spy.vector_store.round_trips == 2


class TestRealVespaSchemaIntegration:
    """Real Vespa: prove search bumps last_accessed and the value persists.

    Seed a memory, capture the baseline timestamp from Vespa, run a
    search, then re-read from Vespa and assert ``last_accessed`` advanced.
    Anything weaker (spying on update) only proves the wire fired, not
    that Mem0 + the BackendVectorStore actually persisted the new value.
    """

    def test_real_vespa_search_persists_last_accessed_bump(
        self, shared_memory_vespa, shared_denseon
    ):
        # Reuse the test_tenant pre-deployed schema.
        import time
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_vespa.config.config_store import VespaConfigStore
        from tests.utils.llm_config import get_llm_base_url, get_llm_model

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
            llm_base_url=get_llm_base_url(),
            embedder_base_url=shared_denseon,
            auto_create_schema=False,
            config_manager=cm,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        )

        try:
            memory_id = mm.add_memory(
                content="actively-queried fact about X",
                tenant_id="test_tenant",
                agent_name="p23_real_agent",
                metadata={"kind": "external_doc"},
                infer=False,
            )
            assert memory_id is not None

            # Baseline: read what the write actually persisted.
            pre = mm.get_all_memories(
                tenant_id="test_tenant", agent_name="p23_real_agent"
            )
            seeded = next((r for r in pre if str(r.get("id")) == memory_id), None)
            assert seeded is not None, (
                f"seeded memory {memory_id} not visible via get_all_memories — "
                "the write didn't reach the read path"
            )
            seeded_md = seeded.get("metadata") or {}
            assert isinstance(seeded_md, dict), (
                f"metadata round-trip broken: expected dict, got {type(seeded_md)!r}. "
                "BackendVectorStore must deserialize metadata_ on read."
            )
            assert seeded_md.get("kind") == "external_doc", (
                "kind written at add_memory time must round-trip through Vespa; "
                f"got metadata={seeded_md!r}"
            )
            baseline_last_accessed = seeded_md.get("last_accessed")

            # Sleep just long enough that any bump produces a strictly
            # greater timestamp than the baseline (which add_memory writes
            # via the same mechanism).
            time.sleep(1.1)

            hits = mm.search_memory(
                query="P2.3",
                tenant_id="test_tenant",
                agent_name="p23_real_agent",
                top_k=5,
            )
            assert len(hits) >= 1, (
                "real Vespa search must return at least the seeded memory"
            )

            # Round-trip assertion: last_accessed must be present AND
            # strictly newer than the baseline. This is what proves
            # search → bump-helper → mem0.update → BackendVectorStore.update
            # → Vespa actually wrote the new timestamp.
            post = mm.get_all_memories(
                tenant_id="test_tenant", agent_name="p23_real_agent"
            )
            updated = next((r for r in post if str(r.get("id")) == memory_id), None)
            assert updated is not None, (
                f"memory {memory_id} disappeared after search bump"
            )
            md = updated.get("metadata") or {}
            assert isinstance(md, dict), (
                f"metadata round-trip broken on re-read: got {type(md)!r}"
            )
            assert "last_accessed" in md, (
                "search must persist a fresh last_accessed in Vespa metadata; "
                f"got keys {sorted(md.keys())}. Without persistence, the "
                "lifecycle scheduler still sees a stale recency signal."
            )
            from datetime import datetime

            def _to_dt(value):
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return datetime.fromtimestamp(value)
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

            new_dt = _to_dt(md["last_accessed"])
            assert new_dt is not None, (
                f"could not parse last_accessed={md['last_accessed']!r}"
            )
            base_dt = _to_dt(baseline_last_accessed)
            if base_dt is not None:
                assert new_dt > base_dt, (
                    f"last_accessed did not advance: baseline={baseline_last_accessed!r}, "
                    f"after-search={md['last_accessed']!r}. The bump was "
                    "wired but the persisted value didn't change — likely "
                    "metadata round-trip is dropping the field."
                )
        finally:
            try:
                mm.clear_agent_memory("test_tenant", "p23_real_agent")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
