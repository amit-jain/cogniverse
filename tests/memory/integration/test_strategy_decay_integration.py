"""strategy bump-on-dedup + retirement against real Mem0+Vespa."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cogniverse_agents.optimizer.strategy_learner import (
    Strategy,
    StrategyLearner,
)
from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "a8_strategy_tenant"
AGENT = "search_agent"


@pytest.fixture(scope="module")
def memory_env(shared_memory_vespa, shared_denseon):
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
    mm = Mem0MemoryManager(tenant_id=TENANT)
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

    yield mm

    try:
        mm.clear_agent_memory(TENANT, f"_strategy_store_{AGENT}")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _strategy(text: str, *, confirmation_count: int = 1) -> Strategy:
    return Strategy(
        text=text,
        applies_when="query is video search",
        agent=AGENT,
        level="user",
        confidence=0.7,
        source="llm_distillation",
        tenant_id=TENANT,
        trace_count=5,
        confirmation_count=confirmation_count,
    )


def test_dedup_bumps_confirmation_count_on_real_vespa(memory_env):
    """Repeated stores of the same strategy bump confirmation_count in Vespa."""
    mm = memory_env
    learner = StrategyLearner(mm, tenant_id=TENANT)

    strat = _strategy("use ColPali for video search and rank by patch similarity")
    learner._store_strategy(strat)
    # Allow Vespa indexing to settle so the second store sees the first.
    time.sleep(2)
    learner._store_strategy(strat)
    time.sleep(2)
    learner._store_strategy(strat)
    time.sleep(2)

    # Read back via the standard search path; the surviving record should
    # carry confirmation_count >= 3.
    matches = mm.search_memory(
        query="ColPali video patch",
        tenant_id=TENANT,
        agent_name=f"_strategy_store_{AGENT}",
        top_k=10,
    )
    strategy_records = [
        m for m in matches if (m.get("metadata") or {}).get("type") == "strategy"
    ]
    assert strategy_records, f"expected strategy records; got {matches}"

    counts = []
    for m in strategy_records:
        meta = m.get("metadata") or {}
        if isinstance(meta, str):
            import json as _json

            meta = _json.loads(meta)
        counts.append(int(meta.get("confirmation_count", 1)))

    # At least one record carries the bumped confirmation count from
    # repeated dedup hits (>= 3 after the three store calls).
    assert max(counts) >= 3, (
        f"expected at least one record with confirmation_count >= 3; got {counts}"
    )


@pytest.mark.asyncio
async def test_unconfirmed_aged_strategies_retired_via_lifecycle(memory_env):
    """Schema cleanup hook removes a single-confirmation strategy older than 30d."""
    mm = memory_env

    # Seed an old single-confirmation strategy by manipulating time.time() on add.
    real_time = time.time
    target_epoch = float(int(real_time()) - 45 * 24 * 3600)  # 45 days old
    try:
        time.time = lambda: target_epoch  # type: ignore[assignment]
        old_id = mm.add_memory(
            content="ancient unconfirmed strategy that should retire",
            tenant_id=TENANT,
            agent_name=f"_strategy_store_{AGENT}",
            metadata={
                "kind": "learned_strategy",
                "type": "strategy",
                "confirmation_count": 1,
                "created_at": (datetime.utcnow() - timedelta(days=45)).isoformat(),
            },
            infer=False,
        )
    finally:
        time.time = real_time  # type: ignore[assignment]

    # Seed a fresh single-confirmation strategy (must NOT retire).
    fresh_id = mm.add_memory(
        content="fresh unconfirmed strategy that should survive",
        tenant_id=TENANT,
        agent_name=f"_strategy_store_{AGENT}",
        metadata={
            "kind": "learned_strategy",
            "type": "strategy",
            "confirmation_count": 1,
            "created_at": datetime.utcnow().isoformat(),
        },
        infer=False,
    )

    # Run a single tick of the schema-driven scheduler.
    scheduler = LifecycleScheduler(
        get_warm_managers=Mem0MemoryManager._instances.values,
        registry=build_default_registry(),
    )
    await scheduler.tick_once()

    # Old one is gone, fresh one remains.
    surviving = {
        m["id"]
        for m in mm.get_all_memories(
            tenant_id=TENANT, agent_name=f"_strategy_store_{AGENT}"
        )
    }
    assert old_id not in surviving, (
        "ancient unconfirmed strategy should have been retired by lifecycle hook"
    )
    assert fresh_id in surviving, "fresh unconfirmed strategy must NOT be retired"
