"""EPHEMERAL_SESSION cleanup integration against real Mem0 + Vespa.

Verifies:
  * write rejection when an EPHEMERAL_SESSION kind is missing session_id
  * drop_session deletes only the matching session's EPHEMERAL_SESSION memories
  * memories tagged with the same session_id but a non-session kind survive
  * empty session_id raises ValueError
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
    Retention,
    SchemaViolationError,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "l20_session_tenant"
AGENT = "l20_session_agent"
SESSION_KIND = "ephemeral_session_test_kind"
PERMANENT_KIND = "session_drop_permanent_kind"


def _registry_for_test() -> KnowledgeRegistry:
    """Registry with an EPHEMERAL_SESSION kind plus a permanent kind sharing session_id."""
    reg = build_default_registry()
    reg.register(
        KnowledgeSchema(
            kind=SESSION_KIND,
            retention=Retention.EPHEMERAL_SESSION,
            provenance_required=False,
            # Session-scoped kinds forbid pinning at the schema level;
            # the gate in KnowledgeSchema.__post_init__ rejects anything
            # else. Use NOBODY to construct.
            pinnable_by=Pinnable.NOBODY,
        ),
        replace=True,
    )
    reg.register(
        KnowledgeSchema(
            kind=PERMANENT_KIND,
            retention=Retention.PERMANENT,
            provenance_required=False,
            pinnable_by=Pinnable.USER,
        ),
        replace=True,
    )
    return reg


@pytest.fixture(scope="module")
def session_env(shared_memory_vespa, shared_denseon):
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
    registry = _registry_for_test()
    mm = Mem0MemoryManager(tenant_id=TENANT)
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
        knowledge_registry=registry,
    )

    yield mm, registry

    try:
        mm.clear_agent_memory(TENANT, AGENT)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def test_ephemeral_session_write_without_session_id_is_rejected(session_env):
    mm, _registry = session_env
    with pytest.raises(SchemaViolationError, match="session_id"):
        mm.add_memory(
            content="this write has no session_id",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": SESSION_KIND},
            infer=False,
        )


def test_drop_session_deletes_only_matching_session_kind(session_env):
    mm, registry = session_env

    # Three EPHEMERAL_SESSION memories under session "s_alpha", two under "s_beta".
    alpha_ids = [
        mm.add_memory(
            content=f"alpha session memory {i}",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": SESSION_KIND, "session_id": "s_alpha"},
            infer=False,
        )
        for i in range(3)
    ]
    beta_ids = [
        mm.add_memory(
            content=f"beta session memory {i}",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": SESSION_KIND, "session_id": "s_beta"},
            infer=False,
        )
        for i in range(2)
    ]
    # A permanent memory tagged with the same session_id — must survive.
    permanent_id = mm.add_memory(
        content="permanent record carrying session_id metadata",
        tenant_id=TENANT,
        agent_name=AGENT,
        metadata={"kind": PERMANENT_KIND, "session_id": "s_alpha"},
        infer=False,
    )

    deleted = mm.drop_session("s_alpha", registry)

    assert deleted.get(SESSION_KIND, 0) == 3, deleted
    assert PERMANENT_KIND not in deleted, deleted

    surviving = {m["id"] for m in mm.get_all_memories(TENANT, AGENT)}
    for aid in alpha_ids:
        assert aid is None or aid not in surviving, (
            f"alpha session memory {aid} must be hard-deleted"
        )
    for bid in beta_ids:
        assert bid in surviving, f"beta session memory {bid} must survive"
    assert permanent_id in surviving, "permanent kind must survive drop_session"


def test_drop_session_empty_id_raises(session_env):
    mm, registry = session_env
    with pytest.raises(ValueError, match="non-empty session_id"):
        mm.drop_session("", registry)
    with pytest.raises(ValueError, match="non-empty session_id"):
        mm.drop_session("   ", registry)


def test_drop_session_unknown_session_returns_empty(session_env):
    mm, registry = session_env
    deleted = mm.drop_session("session_that_does_not_exist", registry)
    assert deleted == {}, deleted
