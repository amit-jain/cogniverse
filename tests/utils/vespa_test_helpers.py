"""Schema-deploy and content-feed helpers for the consolidated shared_vespa.

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
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_vespa.config.config_store import VespaConfigStore

if TYPE_CHECKING:
    from cogniverse_vespa.ingestion_client import VespaPyClient

_PROFILES_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"
_SCHEMAS_DIR = _PROFILES_PATH.parent / "schemas"


def shipped_profile(
    *,
    profile_type: str,
    embedding_type: str,
    process_type: str | None = None,
    extract_keyframes: bool | None = None,
) -> BackendProfileConfig:
    """Select one shipped profile by capabilities and resolve its base schema."""
    profiles = {
        name: BackendProfileConfig.from_dict(name, data)
        for name, data in json.loads(_PROFILES_PATH.read_text())["backend"][
            "profiles"
        ].items()
    }
    matches = {
        name: profile
        for name, profile in profiles.items()
        if profile.type == profile_type
        and profile.embedding_type == embedding_type
        and (process_type is None or profile.process_type == process_type)
        and (
            extract_keyframes is None
            or profile.pipeline_config.get("extract_keyframes") == extract_keyframes
        )
    }
    if len(matches) != 1:
        criteria = {
            "profile_type": profile_type,
            "embedding_type": embedding_type,
            "process_type": process_type,
            "extract_keyframes": extract_keyframes,
        }
        raise ValueError(
            f"Expected exactly one shipped profile for {criteria}; "
            f"matched {sorted(matches)!r}"
        )
    profile = next(iter(matches.values()))
    schema = load_raw_schema_json(profile.schema_name)
    if schema["name"] != profile.schema_name:
        raise ValueError(
            f"Schema file for {profile.schema_name!r} declares {schema['name']!r}"
        )
    return profile


def make_config_manager(
    shared_vespa: Dict[str, Any],
    *,
    inference_service_urls: Dict[str, str] | None = None,
) -> ConfigManager:
    """Build a ConfigManager bound to the shared_vespa container.

    Sets ``SystemConfig.backend_url/backend_port`` so any code path that
    later resolves a Vespa endpoint via the manager points at the shared
    container, not at production defaults. ``inference_service_urls`` maps
    a profile's ``inference_services.embedding`` name onto the test-owned
    sidecar serving it, which is what search-side encoder resolution reads.
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
            inference_service_urls=dict(inference_service_urls or {}),
        )
    )
    return cm


def make_ingestion_client(
    *,
    schema_name: str,
    http_port: int,
    schema_loader: FilesystemSchemaLoader,
    base_schema_name: str | None = None,
) -> "VespaPyClient":
    """A connected ``VespaPyClient`` for ``schema_name`` on the test Vespa.

    This is the client production's ``VespaBackend.ingest_documents()`` drives;
    building it directly skips the tenant-management wrapper while keeping the
    same ``process()`` + ``_feed_prepared_batch()`` path. ``base_schema_name``
    names the ``configs/schemas`` file behind a tenant-scoped ``schema_name``.
    """
    from cogniverse_vespa.ingestion_client import VespaPyClient

    client = VespaPyClient(
        config={
            "schema_name": schema_name,
            "base_schema_name": base_schema_name or schema_name,
            "url": "http://localhost",
            "port": http_port,
            "schema_loader": schema_loader,
        }
    )
    client.connect()
    return client


class IngestionBackendAdapter:
    """Adapts ``VespaPyClient`` to the ``ingest_documents()`` interface
    ``EmbeddingGeneratorImpl`` calls.

    ``VespaBackend.ingest_documents()`` does the same two steps internally:
    ``client.process(doc)`` then ``client._feed_prepared_batch()``.
    """

    def __init__(self, vespa_client):
        self._client = vespa_client

    def ingest_documents(self, documents, schema_name):
        prepared = [self._client.process(doc) for doc in documents]
        success, failed = self._client._feed_prepared_batch(prepared)
        return {
            "success_count": success,
            "failed_count": len(failed),
            "failed_documents": failed,
            "total_documents": len(documents),
        }


def feed_text_documents(
    *,
    backend_client,
    schema_name: str,
    inference_url: str,
    model_name: str,
    documents: Iterable[Dict[str, str]],
    corpus_id: str,
) -> Any:
    """Feed ``documents`` into ``schema_name`` with real served embeddings.

    Runs the production ``EmbeddingGeneratorImpl`` against the served model at
    ``inference_url`` and hands the Documents to ``backend_client``, which is
    either a ``VespaBackend`` or an ``IngestionBackendAdapter``. Each entry
    supplies ``id``, ``title`` and ``text``; the document id is the caller's,
    so re-feeding the same corpus overwrites in place rather than accumulating.
    """
    from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
        EmbeddingGeneratorImpl,
    )

    generator = EmbeddingGeneratorImpl(
        config={
            "embedding_model": model_name,
            "embedding_type": "multi_vector",
            "model_loader": "colbert",
            "schema_name": schema_name,
            "inference_services": {"embedding": "colbert_pylate"},
            "remote_inference_url": inference_url,
        },
        backend_client=backend_client,
    )
    segments: List[Dict[str, Any]] = [
        {
            "document_id": entry["id"],
            "extracted_text": f"{entry['title']}. {entry['text']}",
            "filename": entry["title"],
            "document_type": "txt",
            "path": f"/{corpus_id}/{entry['id']}.txt",
            "page_count": 1,
        }
        for entry in documents
    ]
    return generator.generate_embeddings(
        {"video_id": corpus_id, "document_files": segments},
        output_dir=Path("/tmp"),
    )


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
