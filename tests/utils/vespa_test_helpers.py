"""Schema-deploy helpers for the consolidated shared_vespa.

Tests that need a data schema (video_colpali, code_lateon, agent_memories,
etc.) tenant-scoped to themselves call into one of these helpers from a
fixture. The actual deploy goes through ``SchemaRegistry.deploy_schema``,
which handles tenant-name normalization and merge-with-existing-schemas
correctly — these helpers just wire it up to the shared_vespa endpoints.

End-of-test wipe is deliberately not provided here: tenant_ids derived
from ``tenant_helpers.py`` are unique per module, so schemas from two
different test modules don't collide. The shared_vespa container is torn
down at session end, taking everything with it.

If a specific test needs to assert on Vespa state after explicit wipe
(schema lifecycle tests do), call ``SchemaRegistry.delete_schema`` directly
rather than adding a wipe helper here that other tests would mis-use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore

_SCHEMAS_DIR = Path("configs/schemas")


def make_config_manager(shared_vespa: Dict[str, Any]) -> ConfigManager:
    """Build a ConfigManager bound to the shared_vespa container.

    Sets ``SystemConfig.backend_url/backend_port`` so any code path that
    later resolves a Vespa endpoint via the manager points at the shared
    container, not at production defaults.
    """
    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_vespa["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_vespa["http_port"],
        )
    )
    return cm


def deploy_tenant_schema(
    shared_vespa: Dict[str, Any],
    *,
    tenant_id: str,
    base_schema_name: str,
    config_manager: ConfigManager | None = None,
    force: bool = False,
) -> str:
    """Deploy ``base_schema_name`` for ``tenant_id`` against shared_vespa.

    Uses the canonical SchemaRegistry pathway so it co-deploys with any
    schemas already present (per the design of ``deploy_schema`` —
    collects existing + adds new + redeploys atomically). Returns the
    full tenant-scoped schema name (e.g. ``agent_memories_<tenant>``).

    ``force=True`` re-runs the merge-and-redeploy even when the schema is
    already registered; tests that assert the merge itself preserves peer
    tenants need the deploy to actually execute.
    """
    if config_manager is None:
        config_manager = make_config_manager(shared_vespa)

    schema_loader = FilesystemSchemaLoader(_SCHEMAS_DIR)
    backend_config = {
        "url": "http://localhost",
        "port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
    }

    registry = BackendRegistry.get_instance()
    backend = registry.get_ingestion_backend(
        name="vespa",
        config={"backend": backend_config},
        config_manager=config_manager,
        schema_loader=schema_loader,
        tenant_id=tenant_id,
    )
    return backend.schema_registry.deploy_schema(
        tenant_id=tenant_id,
        base_schema_name=base_schema_name,
        force=force,
    )


def schema_full_name(base_schema_name: str, tenant_id: str) -> str:
    """The naming convention SchemaRegistry uses for tenant-scoped schemas.

    Mirrors ``schema_registry.py::deploy_schema`` exactly: it canonicalizes
    the tenant_id (``test`` → ``test:test``) before replacing colons with
    underscores, so a bare tenant id resolves to the same double-suffixed
    name deploy produces (``knowledge_graph_test`` → ``knowledge_graph_test_test``).
    Tests that need the deployed schema name without going through deploy
    (to construct a Vespa query/probe) use this so the rule lives in one place.
    """
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    return f"{base_schema_name}_{canonical_tenant_id(tenant_id).replace(':', '_')}"


def load_raw_schema_json(base_schema_name: str) -> Dict[str, Any]:
    """Read a base schema definition from configs/schemas/.

    Useful for tests that need to inspect the raw schema (field names,
    rank profiles) before or after deploy.
    """
    path = _SCHEMAS_DIR / f"{base_schema_name}_schema.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No schema definition for base name {base_schema_name!r} at {path}"
        )
    return json.loads(path.read_text())


def schema_tensor_dim(base_schema_name: str, field_name: str) -> int:
    """The ``v[N]`` width declared for a tensor field in configs/schemas/.

    Tests that hand-feed documents size their vectors from this rather than
    a literal, so a model swap that changes the embedding width surfaces as
    a schema/encoder mismatch instead of a 400 from every feed.
    """
    schema = load_raw_schema_json(base_schema_name)
    for field in schema["document"]["fields"]:
        if field["name"] != field_name:
            continue
        match = re.search(r"v\[(\d+)\]", field["type"])
        if match is None:
            raise ValueError(
                f"{base_schema_name}.{field_name} is {field['type']!r}, "
                f"which declares no v[N] dimension"
            )
        return int(match.group(1))
    raise KeyError(f"{base_schema_name} has no field named {field_name!r}")
