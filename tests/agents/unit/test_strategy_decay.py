"""Unit tests — bump-on-dedup, retrieval downweighting, retirement hook."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock

from cogniverse_agents.optimizer.strategy_learner import (
    DEDUP_SIMILARITY_THRESHOLD,
    STRATEGY_AGENT_NAME,
    Strategy,
    StrategyLearner,
)
from cogniverse_core.memory.schema import (
    Retention,
    _retire_unconfirmed_strategy,
    build_default_registry,
)


class FakeMemoryManager:
    """Mem0MemoryManager-shaped stub that tracks add/delete/search calls."""

    def __init__(self, search_results=None):
        self._next_id = 0
        self.store: Dict[str, Dict[str, Any]] = {}
        self.add_calls: list[Dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self._search_results = search_results or []
        # Mem0-shaped object so the learner's `is None` checks pass.
        self.memory = MagicMock()

    def add_memory(self, *, content, tenant_id, agent_name, metadata=None, infer=False):
        self._next_id += 1
        mid = f"m{self._next_id}"
        record = {
            "id": mid,
            "memory": content,
            "tenant_id": tenant_id,
            "agent_name": agent_name,
            "metadata": dict(metadata or {}),
        }
        self.store[mid] = record
        self.add_calls.append(record)
        return mid

    def delete_memory(self, *, memory_id, tenant_id, agent_name):
        self.delete_calls.append(memory_id)
        return self.store.pop(memory_id, None) is not None

    def search_memory(self, *, query, tenant_id, agent_name, top_k=5, filters=None):
        return list(self._search_results)


def _strategy(text: str = "use ColPali for video", confirmation_count: int = 1):
    return Strategy(
        text=text,
        applies_when="query is video search",
        agent="search_agent",
        level="user",
        confidence=0.7,
        source="llm_distillation",
        tenant_id="acme",
        trace_count=5,
        confirmation_count=confirmation_count,
    )


class TestStrategyMetadata:
    def test_kind_is_learned_strategy(self):
        meta = _strategy().to_metadata()
        assert meta["kind"] == "learned_strategy"

    def test_includes_confirmation_fields(self):
        meta = _strategy(confirmation_count=4).to_metadata()
        assert meta["confirmation_count"] == 4
        assert meta["last_confirmed_at"]


class TestBumpOnDedup:
    def test_dedup_hit_replaces_with_bumped_record(self):
        # Existing memory near-identical to the incoming strategy.
        existing_strategy = _strategy(confirmation_count=2)
        existing_memory = {
            "id": "m_existing",
            "memory": existing_strategy.to_memory_content(),
            "metadata": existing_strategy.to_metadata(),
        }
        mm = FakeMemoryManager(search_results=[existing_memory])
        learner = StrategyLearner(mm, tenant_id="acme")

        ok = learner._store_strategy(_strategy(confirmation_count=1))
        assert ok is True

        # The existing memory must have been deleted...
        assert "m_existing" in mm.delete_calls
        # ...and replaced with a new memory whose count is bumped to 3.
        bumped_records = [
            r for r in mm.add_calls if r["metadata"]["confirmation_count"] == 3
        ]
        assert len(bumped_records) == 1, f"got {[r['metadata'] for r in mm.add_calls]}"
        # trace_count accumulates across bumps so the agent can see total weight.
        assert bumped_records[0]["metadata"]["trace_count"] >= 5

    def test_no_dedup_hit_writes_fresh_strategy(self):
        # Search returns a memory that does NOT overlap.
        unrelated = {
            "id": "m_unrelated",
            "memory": "completely different unrelated strategy text about audio",
            "metadata": Strategy(
                text="use whisper for audio",
                applies_when="audio query",
                agent="search_agent",
                level="user",
                confidence=0.6,
                source="llm_distillation",
                tenant_id="acme",
                trace_count=3,
            ).to_metadata(),
        }
        mm = FakeMemoryManager(search_results=[unrelated])
        learner = StrategyLearner(mm, tenant_id="acme")

        ok = learner._store_strategy(_strategy())
        assert ok is True
        # No delete (existing was unrelated).
        assert mm.delete_calls == []
        # Fresh add with confirmation_count=1.
        added = mm.add_calls[-1]
        assert added["metadata"]["confirmation_count"] == 1

    def test_overlap_below_threshold_treated_as_distinct(self):
        # Identical-prefix but different tail; word-level Jaccard is below threshold.
        mostly_different = {
            "id": "m_x",
            "memory": "I prefer the following approach for search_agent: completely "
            "unrelated audio strategy about whisper. I use this when listening.",
            "metadata": Strategy(
                text="use whisper for audio",
                applies_when="audio query",
                agent="search_agent",
                level="user",
                confidence=0.6,
                source="llm_distillation",
                tenant_id="acme",
                trace_count=3,
            ).to_metadata(),
        }
        mm = FakeMemoryManager(search_results=[mostly_different])
        learner = StrategyLearner(mm, tenant_id="acme")

        learner._store_strategy(_strategy())
        # Fresh write, no bump.
        assert mm.delete_calls == []


class TestRetrievalDecay:
    def _build_strategies(self):
        """Mix of strategies: high+old, low+old (target for downweight), low+new."""
        now = datetime.utcnow()
        ts_old = (now - timedelta(days=30)).isoformat()
        ts_new = now.isoformat()

        return [
            {
                "memory": "high-confirm old strategy",
                "score": 0.6,
                "metadata": {
                    "kind": "learned_strategy",
                    "confidence": 0.6,
                    "confirmation_count": 5,
                    "created_at": ts_old,
                },
            },
            {
                "memory": "low-confirm old strategy",
                "score": 0.7,
                "metadata": {
                    "kind": "learned_strategy",
                    "confidence": 0.7,
                    "confirmation_count": 1,
                    "created_at": ts_old,
                },
            },
            {
                "memory": "low-confirm fresh strategy",
                "score": 0.5,
                "metadata": {
                    "kind": "learned_strategy",
                    "confidence": 0.5,
                    "confirmation_count": 1,
                    "created_at": ts_new,
                },
            },
        ]

    def test_low_confirmation_old_strategy_downweighted_below_high_confirmation(
        self,
    ):
        strategies = self._build_strategies()
        ranked = StrategyLearner.rank_strategies_with_decay(
            strategies,
            low_confirmation_threshold=3,
            downweight_age_days=14,
            downweight_factor=0.5,
        )

        ids = [s["memory"] for s in ranked]
        # high-confirm old (0.6) must outrank low-confirm old (0.7 * 0.5 = 0.35).
        assert ids.index("high-confirm old strategy") < ids.index(
            "low-confirm old strategy"
        )
        # low-confirm fresh keeps its original score 0.5; not downweighted.
        assert ranked[0]["memory"] == "high-confirm old strategy"

    def test_empty_input_returns_empty(self):
        assert StrategyLearner.rank_strategies_with_decay([]) == []

    def test_string_metadata_accepted(self):
        strategies = self._build_strategies()
        # Stringify one of them to mimic Vespa round-trip.
        strategies[0]["metadata"] = json.dumps(strategies[0]["metadata"])
        # Must not raise.
        ranked = StrategyLearner.rank_strategies_with_decay(strategies)
        assert len(ranked) == 3


class TestRetirementHook:
    def test_unconfirmed_old_strategy_retired(self):
        old_ts = (datetime.utcnow() - timedelta(days=45)).isoformat()
        memory = {
            "id": "m1",
            "memory": "fading strategy",
            "metadata": {
                "kind": "learned_strategy",
                "confirmation_count": 1,
                "created_at": old_ts,
            },
        }
        schema = build_default_registry().get("learned_strategy")
        assert _retire_unconfirmed_strategy(memory, schema) is True

    def test_recent_strategy_kept_even_if_unconfirmed(self):
        recent_ts = datetime.utcnow().isoformat()
        memory = {
            "id": "m1",
            "memory": "fresh strategy",
            "metadata": {
                "kind": "learned_strategy",
                "confirmation_count": 1,
                "created_at": recent_ts,
            },
        }
        schema = build_default_registry().get("learned_strategy")
        assert _retire_unconfirmed_strategy(memory, schema) is False

    def test_confirmed_strategy_kept_even_when_old(self):
        old_ts = (datetime.utcnow() - timedelta(days=90)).isoformat()
        memory = {
            "id": "m1",
            "memory": "battle-tested strategy",
            "metadata": {
                "kind": "learned_strategy",
                "confirmation_count": 7,
                "created_at": old_ts,
            },
        }
        schema = build_default_registry().get("learned_strategy")
        assert _retire_unconfirmed_strategy(memory, schema) is False

    def test_non_learned_strategy_kind_ignored(self):
        old_ts = (datetime.utcnow() - timedelta(days=90)).isoformat()
        memory = {
            "id": "m1",
            "memory": "an unrelated old fact",
            "metadata": {
                "kind": "entity_fact",
                "confirmation_count": 1,
                "created_at": old_ts,
            },
        }
        schema = build_default_registry().get("learned_strategy")
        assert _retire_unconfirmed_strategy(memory, schema) is False

    def test_string_metadata_accepted(self):
        old_ts = (datetime.utcnow() - timedelta(days=90)).isoformat()
        memory = {
            "id": "m1",
            "memory": "x",
            "metadata": json.dumps(
                {
                    "kind": "learned_strategy",
                    "confirmation_count": 1,
                    "created_at": old_ts,
                }
            ),
        }
        schema = build_default_registry().get("learned_strategy")
        assert _retire_unconfirmed_strategy(memory, schema) is True


class TestRegistryWiring:
    def test_default_registry_attaches_retirement_hook(self):
        s = build_default_registry().get("learned_strategy")
        assert s.retention is Retention.SCHEMA_DRIVEN
        assert callable(s.cleanup_hook)

    def test_dedup_threshold_constant_in_band(self):
        """Sanity: the dedup threshold is in (0,1] so the bump path triggers."""
        assert 0 < DEDUP_SIMILARITY_THRESHOLD <= 1
        # Retain identifier so import isn't dead code.
        assert STRATEGY_AGENT_NAME == "_strategy_store"
