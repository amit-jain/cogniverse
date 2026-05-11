"""Memory integration test fixtures.

Re-exports the ``dspy_lm`` and ``_dspy_lm_instance`` fixtures from
``tests/agents/integration/conftest.py`` so memory-side integration
tests can exercise the real DSPy LM (no stubs). Pytest only walks UP
from a test file's directory; without this re-export, tests under
``tests/memory/integration/`` would not see the LM fixture defined
under ``tests/agents/integration/``.
"""

from __future__ import annotations

import logging

import pytest

# Re-export both fixtures by name. Pytest discovers them via the symbol
# table of the conftest module, so a plain import is enough.
from tests.agents.integration.conftest import (  # noqa: F401
    _dspy_lm_instance,
    dspy_lm,
)

logger = logging.getLogger(__name__)


# Map base-schema-name → schema-definition JSON file. Used by the
# reconciler below to load real definitions for orphan schemas; with a
# real definition the deploy safety check can reconstruct + retain
# them. An empty stub would still cause the redeploy parser to fail.
_BASE_SCHEMA_FILES = {
    "agent_memories": "agent_memories_schema.json",
    "wiki_pages": "wiki_pages_schema.json",
    "provenance": "provenance_schema.json",
    "knowledge_graph": "knowledge_graph_schema.json",
    "video_colpali_smol500_mv_frame": "video_colpali_smol500_mv_frame_schema.json",
    "video_videoprism_base_mv_chunk_30s": "video_videoprism_base_mv_chunk_30s_schema.json",
    "video_colqwen_omni_mv_chunk_30s": "video_colqwen_omni_mv_chunk_30s_schema.json",
}


def _load_schema_definition(base_schema_name: str, full_schema_name: str) -> str | None:
    """Load the schema JSON file for ``base_schema_name`` and patch the
    ``name``/``document.name`` to the tenant-scoped ``full_schema_name``.

    Returns the JSON string, or ``None`` if we don't have a known file
    for the base name (caller should skip registering that one — better
    to let the deploy refuse than register a stub the parser will choke
    on).
    """
    import json as _json
    from pathlib import Path as _Path

    file_name = _BASE_SCHEMA_FILES.get(base_schema_name)
    if file_name is None:
        return None
    schema_path = _Path("configs/schemas") / file_name
    if not schema_path.exists():
        return None
    try:
        schema_json = _json.loads(schema_path.read_text())
        schema_json["name"] = full_schema_name
        if "document" in schema_json and isinstance(schema_json["document"], dict):
            schema_json["document"]["name"] = full_schema_name
        return _json.dumps(schema_json)
    except Exception as exc:
        logger.warning(
            f"Failed to load schema definition for {base_schema_name}: {exc}"
        )
        return None


@pytest.fixture(autouse=True)
def _reconcile_registry_with_vespa(shared_memory_vespa):
    """Pre-test: register every Vespa-side tenant schema in the
    ``BackendRegistry._shared_schema_registry`` singleton so the deploy
    safety check can find a definition for each one.

    Without this, once any earlier fixture clears the singleton mid-
    session the registry's view (empty) diverges from Vespa's actual
    state (accumulated tenant schemas). The next ``deploy_schema``
    call then hits ``Refusing to deploy: Vespa has schemas X that are
    not in SchemaRegistry and cannot be reconstructed`` and dies.
    Reconciling here means later deploys see a registry whose
    ``_get_all_schemas()`` includes every Vespa-side schema, so the
    safety check reconstructs them via the JSON loader instead of
    refusing.

    This does NOT touch Vespa data — pin records, memories, and
    provenance writes from earlier tests survive into later tests
    that depend on them. The fixture only updates the registry's
    in-memory view.
    """
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    sm = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=shared_memory_vespa["config_port"],
    )
    try:
        deployed = sm.list_deployed_document_types()
    except Exception as exc:
        logger.warning(f"reconcile probe failed; skipping: {exc}")
        yield
        return

    registry = BackendRegistry._shared_schema_registry
    if registry is None:
        # No registry to reconcile against. The first test of the
        # session that calls Mem0MemoryManager.initialize creates one;
        # subsequent tests will hit this fixture again and reconcile.
        yield
        return

    known_full_names = {info.full_schema_name for info in registry._get_all_schemas()}

    for full_name in deployed:
        if full_name in sm._PROTECTED_SCHEMAS or full_name in known_full_names:
            continue
        for base in _BASE_SCHEMA_FILES:
            if full_name.startswith(base + "_"):
                tenant_id = full_name[len(base) + 1 :]
                schema_def = _load_schema_definition(base, full_name)
                if schema_def is None:
                    break
                try:
                    registry.register_schema(
                        tenant_id=tenant_id,
                        base_schema_name=base,
                        full_schema_name=full_name,
                        schema_definition=schema_def,
                    )
                except Exception as exc:
                    logger.warning(f"reconcile: failed to register {full_name}: {exc}")
                break
    yield
