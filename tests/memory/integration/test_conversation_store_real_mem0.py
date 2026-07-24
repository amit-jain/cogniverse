"""Real Mem0 (Vespa-backed) round-trip for ConversationStore.

The dispatcher relies on ConversationStore to persist and reload per-chat
turns. This exercises the real Mem0 store, not a mock: store turns, reload
them in order, confirm contexts are isolated and the window is bounded.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.conversation import ConversationStore
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model


def _build_manager(*, shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()
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
    mm = Mem0MemoryManager(tenant_id=SYSTEM_TENANT_ID)
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
    return mm


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_and_reload_turns_in_order_real_mem0(
    shared_memory_vespa, shared_denseon
):
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    store = ConversationStore(mm, SYSTEM_TENANT_ID)
    ctx = f"chat{uuid.uuid4().hex[:10]}"

    store.store_turn(ctx, "user", "what is colpali")
    store.store_turn(ctx, "assistant", "a late-interaction retrieval model")
    store.store_turn(ctx, "user", "how many dimensions")

    history = store.get_history(ctx)
    assert history == [
        {"role": "user", "content": "what is colpali"},
        {"role": "assistant", "content": "a late-interaction retrieval model"},
        {"role": "user", "content": "how many dimensions"},
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_contexts_isolated_real_mem0(shared_memory_vespa, shared_denseon):
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    store = ConversationStore(mm, SYSTEM_TENANT_ID)
    a = f"chat{uuid.uuid4().hex[:10]}"
    b = f"chat{uuid.uuid4().hex[:10]}"

    store.store_turn(a, "user", "belongs to A")
    store.store_turn(b, "user", "belongs to B")

    assert store.get_history(a) == [{"role": "user", "content": "belongs to A"}]
    assert store.get_history(b) == [{"role": "user", "content": "belongs to B"}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_history_window_is_bounded_real_mem0(shared_memory_vespa, shared_denseon):
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    store = ConversationStore(mm, SYSTEM_TENANT_ID)
    ctx = f"chat{uuid.uuid4().hex[:10]}"

    for i in range(14):
        store.store_turn(ctx, "user", f"turn {i}")

    history = store.get_history(ctx, max_turns=10)
    assert len(history) == 10
    assert history[0]["content"] == "turn 4"
    assert history[-1]["content"] == "turn 13"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_quiet_context_survives_busy_neighbor_real_mem0(
    shared_memory_vespa, shared_denseon
):
    """A quiet context reloads complete when a neighbor floods the partition.

    get_history must narrow to the context server-side and page through
    every matching turn. A plain enumerate-then-filter sees only the newest
    100 rows, so an older context's turns fall off the page and its history
    reloads empty once the partition grows past 100.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    store = ConversationStore(mm, SYSTEM_TENANT_ID)
    quiet = f"chat{uuid.uuid4().hex[:10]}"
    busy = f"chat{uuid.uuid4().hex[:10]}"

    store.store_turn(quiet, "user", "quiet q1")
    store.store_turn(quiet, "assistant", "quiet a1")
    store.store_turn(quiet, "user", "quiet q2")
    # Push the quiet context strictly before the flood so the second-grained
    # created_at cannot tie its turns into the newest-100 page.
    time.sleep(1.2)
    for i in range(101):
        store.store_turn(busy, "user", f"busy {i}")

    assert store.get_history(quiet) == [
        {"role": "user", "content": "quiet q1"},
        {"role": "assistant", "content": "quiet a1"},
        {"role": "user", "content": "quiet q2"},
    ]
    # The flooded context's most-recent window is exact and ordered, proving
    # the paginated within-context read returns every matching turn.
    busy_hist = store.get_history(busy, max_turns=5)
    assert [t["content"] for t in busy_hist] == [f"busy {i}" for i in range(96, 101)]
