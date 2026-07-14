"""
Mem0 Memory Manager

Mem0-based memory system using Vespa backend.
Provides simple, persistent agent memory with multi-tenant support.
Each tenant gets dedicated Vespa schema for memory isolation.
"""

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_core.memory.schema import KnowledgeRegistry

# Disable Mem0's telemetry BEFORE importing mem0
os.environ["MEM0_TELEMETRY"] = "False"

from mem0 import Memory

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.memory._timestamps import to_epoch_seconds
from cogniverse_foundation.caching import TenantLRUCache

logger = logging.getLogger(__name__)


# Max number of distinct tenants whose Mem0 instances are kept warm in
# this process. Each instance carries a Vespa client and an LLM config,
# so an unbounded dict leaks ~50-200MB per tenant across long-running
# suites. Override with COGNIVERSE_TENANT_CACHE_CAPACITY to widen the
# working set in multi-tenant servers.
_DEFAULT_TENANT_CACHE_CAPACITY = 16


def _tenant_cache_capacity() -> int:
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "COGNIVERSE_TENANT_CACHE_CAPACITY",
                    _DEFAULT_TENANT_CACHE_CAPACITY,
                )
            ),
        )
    except (TypeError, ValueError):
        return _DEFAULT_TENANT_CACHE_CAPACITY


def _on_tenant_evicted(tenant_id: str, instance: "Mem0MemoryManager") -> None:
    """Drop references the evicted instance held to Mem0/backend state.

    Mem0.Memory keeps a reference to the vector store + LLM config. We
    null those refs so the GC can reclaim them; actual network clients
    release on the next cycle (pyvespa uses httpx.Client which closes
    on deref).
    """
    try:
        instance.memory = None
        instance.config = None
        instance._initialized = False
        logger.info("Evicted Mem0MemoryManager for tenant %s", tenant_id)
    except Exception as exc:
        logger.debug("Mem0 eviction cleanup for %s: %s", tenant_id, exc)


# Vespa standard ports — single source of truth for config port derivation.
# Duplicated from cogniverse_vespa.config_utils to avoid cross-layer import
# (core cannot depend on vespa).
_VESPA_DEFAULT_DATA_PORT = 8080
_VESPA_DEFAULT_CONFIG_PORT = 19071
_VESPA_CONFIG_PORT_OFFSET = (
    _VESPA_DEFAULT_CONFIG_PORT - _VESPA_DEFAULT_DATA_PORT
)  # 10991


def _calculate_config_port(data_port: int) -> int:
    """Derive Vespa config port from data port using standard offset."""
    if data_port == _VESPA_DEFAULT_DATA_PORT:
        return _VESPA_DEFAULT_CONFIG_PORT
    return data_port + _VESPA_CONFIG_PORT_OFFSET


# Register backend as a supported vector store provider in Mem0
def _register_backend_provider():
    """Register backend-agnostic vector store provider in Mem0"""
    import sys

    from mem0.configs.base import VectorStoreConfig
    from mem0.utils.factory import VectorStoreFactory

    # Make BackendConfig available for mem0 import
    from cogniverse_core.memory import backend_config

    sys.modules["mem0.configs.vector_stores.backend"] = backend_config
    logger.debug("Registered backend config module for mem0 import")

    # Register in VectorStoreConfig._provider_configs (access via private attrs)
    provider_configs_attr = VectorStoreConfig.__private_attributes__[
        "_provider_configs"
    ]
    if "backend" not in provider_configs_attr.default:
        provider_configs_attr.default["backend"] = "BackendConfig"
        logger.info("Registered backend in VectorStoreConfig._provider_configs")

    # Register BackendVectorStore in factory
    if "backend" not in VectorStoreFactory.provider_to_class:
        VectorStoreFactory.provider_to_class["backend"] = (
            "cogniverse_core.memory.backend_vector_store.BackendVectorStore"
        )
        logger.info("Registered BackendVectorStore in Mem0 factory")


# Register on module import
_register_backend_provider()


class Mem0MemoryManager:
    """
    Memory manager using Mem0 with Vespa vector store backend.

    Provides:
    - Multi-tenant memory isolation via schema-per-tenant
    - Per-agent memory namespacing within tenant
    - Persistent storage in Vespa
    - Semantic search via embeddings
    - Simple API without Letta's complexity

    Each tenant gets dedicated Vespa schema: agent_memories_{tenant_id}
    """

    # Per-tenant LRU cache. Bounded so a test suite that creates a fresh
    # tenant per test (or a multi-tenant server with churn) can't push the
    # working set past capacity without evicting the oldest tenant.
    _instances: TenantLRUCache["Mem0MemoryManager"] = TenantLRUCache(
        capacity=_tenant_cache_capacity(),
        on_evict=_on_tenant_evicted,
    )

    def __new__(cls, tenant_id: str):
        """Per-tenant singleton pattern (LRU-bounded)."""

        def _build() -> "Mem0MemoryManager":
            instance = super(Mem0MemoryManager, cls).__new__(cls)
            instance._initialized = False
            logger.info(
                "Created new Mem0MemoryManager instance for tenant: %s",
                tenant_id,
            )
            return instance

        return cls._instances.get_or_set(tenant_id, _build)

    def __init__(self, tenant_id: str):
        """Initialize memory manager for specific tenant (only once per tenant)"""
        if self._initialized:
            return

        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id
        self.memory: Optional[Memory] = None
        self.config: Optional[Dict[str, Any]] = None
        # knowledge schema registry. When set (via initialize),
        # add_memory enforces schema.provenance_required and auto-attaches
        # an initial trust score derived from the provenance + schema's
        # default_trust. Unset = no enforcement, no trust.
        self._knowledge_registry: Optional[object] = None

        self._initialized = True
        logger.info(f"Mem0MemoryManager initialized for tenant: {tenant_id}")

    def initialize(
        self,
        backend_host: str,
        backend_port: int,
        llm_model: str,
        embedding_model: str,
        llm_base_url: str,
        embedder_base_url: str,
        config_manager,
        schema_loader,
        llm_api_key: str = "not-required",
        backend_config_port: Optional[int] = None,
        base_schema_name: str = "agent_memories",
        auto_create_schema: bool = True,
        knowledge_registry: Optional[object] = None,
    ) -> None:
        """
        Initialize Mem0 with backend using tenant-specific schema.

        Args:
            backend_host: Backend endpoint URL (e.g. "http://localhost")
            backend_port: Backend data endpoint port
            llm_model: LLM model name for memory extraction
            embedding_model: Embedding model name for memory search
                (DenseOn served by the colbert_pylate sidecar in dense mode)
            llm_base_url: LLM endpoint URL. Any OpenAI-compatible server
                (OAI-compat local servers, OpenAI, hosted) works. ``/v1`` suffix is
                added automatically when missing.
            embedder_base_url: OpenAI-compatible /v1/embeddings endpoint —
                separate from the LLM endpoint because the embedder always
                runs DenseOn on the dedicated denseon sidecar pod.
            config_manager: ConfigManager instance
            schema_loader: SchemaLoader instance
            llm_api_key: API key sent to ``llm_base_url``. Defaults to
                ``"not-required"`` for local OAI-compat backends
                that don't authenticate; pass a real key for hosted
                providers.
            backend_config_port: Backend config endpoint port (default: 19071)
            base_schema_name: Base schema name (default: agent_memories)
            auto_create_schema: Auto-deploy tenant schema if not exists

        Raises:
            ValueError: If tenant_id not set
        """
        if not self.tenant_id:
            raise ValueError("tenant_id must be set before initialize()")

        # Idempotency: the dispatcher runs initialize_memory per dispatched
        # request (via MemoryAwareMixin), and this manager is a per-tenant
        # singleton — rebuilding Memory.from_config each time reconstructed
        # the embedder/LLM/vector-store stack for identical wiring.
        fingerprint = (
            backend_host,
            backend_port,
            llm_model,
            embedding_model,
            llm_base_url,
            embedder_base_url,
            llm_api_key,
            backend_config_port,
            base_schema_name,
        )
        if (
            self.memory is not None
            and getattr(self, "_init_fingerprint", None) == fingerprint
        ):
            if knowledge_registry is not None:
                self._knowledge_registry = knowledge_registry
            return

        # Get backend instance for memory operations
        from cogniverse_core.registries.backend_registry import get_backend_registry
        from cogniverse_foundation.config.utils import get_config

        config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
        backend_type = config.get("backend_type", "vespa")
        registry = get_backend_registry()

        # Get backend instance with full config including profiles
        # Add agent_memories profile since it may not be in config
        profiles_raw = config.get("profiles", {})

        # Ensure profiles is a dict (handle list format or other edge cases)
        if isinstance(profiles_raw, dict):
            profiles = profiles_raw
        elif isinstance(profiles_raw, list):
            profiles = {
                p.get("name", f"profile_{i}"): p
                for i, p in enumerate(profiles_raw)
                if isinstance(p, dict)
            }
        else:
            profiles = {}

        if base_schema_name not in profiles:
            # Minimal profile for agent_memories. Shape must match
            # `BackendProfileConfig.to_dict()` (see unified_config.py:429) so
            # it survives round-trip through ConfigStore unchanged.
            memory_profile = {
                "type": "memory",
                "model": "lightonai/DenseOn",
                "embedding_model": "lightonai/DenseOn",
                "embedding_dims": 768,
                "encoder": "denseon",
                "strategy": "semantic_search",
                "schema_name": base_schema_name,
                "embedding_type": "dense",
                "schema_config": {"embedding_dims": 768},
            }
            profiles[base_schema_name] = memory_profile

            # Persist the profile through ConfigManager so the shared search
            # backend picks it up via the profile_change_listener wired in
            # main.py. Without this step the profile was only known to the
            # tenant-specific ingestion backend — writes landed in Vespa but
            # reads returned "profile not found" from the shared search cache.
            try:
                from cogniverse_foundation.config.unified_config import (
                    BackendProfileConfig,
                )

                profile_config = BackendProfileConfig.from_dict(
                    base_schema_name, memory_profile
                )
                config_manager.add_backend_profile(
                    profile_config,
                    tenant_id=SYSTEM_TENANT_ID,
                    service="backend",
                )
            except Exception as exc:
                # Idempotent re-adds surface as store-level warnings but
                # shouldn't fail Mem0 init. The in-process dict we just
                # built is still used below for the ingestion backend.
                logger.debug(
                    "ConfigManager.add_backend_profile for '%s' raised "
                    "(may be a harmless re-register): %s",
                    base_schema_name,
                    exc,
                )

        config_backend = config.get("backend", {})
        # Strip url/port from config.json's backend section — the explicit
        # backend_host/backend_port parameters are authoritative. Letting
        # config.json values leak through causes port mismatches in tests
        # and multi-instance deployments.
        config_backend_clean = {
            k: v for k, v in config_backend.items() if k not in ("url", "port")
        }
        backend_section = {
            "profiles": profiles,
            **config_backend_clean,
        }

        backend_config_dict = {
            "url": backend_host,  # VespaBackend expects "url", not "backend_url"
            "port": backend_port,  # VespaBackend expects "port", not "backend_port"
            "config_port": backend_config_port or _calculate_config_port(backend_port),
            "schema_name": base_schema_name,  # Base schema name for operations (backend handles tenant transformation)
            "backend": backend_section,  # Backend section for search operations
            "profiles": profiles,  # Also keep at top level for config merging
            "default_profiles": config.get("default_profiles", {}),
        }

        # Create tenant-specific backend for memory operations
        # Each tenant gets their own memory schema (agent_memories_{tenant_id})
        backend = registry.get_ingestion_backend(
            backend_type,
            tenant_id=self.tenant_id,
            config=backend_config_dict,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        # Keep a handle so the provenance store + future side-stores can
        # talk to Vespa directly without going through Mem0's vector
        # store wrapper.
        self._backend = backend

        # Get tenant-specific schema name
        tenant_schema_name = backend.get_tenant_schema_name(
            self.tenant_id, base_schema_name
        )

        # Deploy tenant schema if needed
        if auto_create_schema:
            backend.schema_registry.deploy_schema(
                tenant_id=self.tenant_id, base_schema_name=base_schema_name
            )
            logger.info(f"Ensured tenant schema exists: {tenant_schema_name}")
            # Deploy the per-tenant provenance schema alongside the
            # memory schema. The walker reads from this schema only;
            # a deploy failure here breaks every audit / citation
            # path so it must surface, not be swallowed.
            backend.schema_registry.deploy_schema(
                tenant_id=self.tenant_id, base_schema_name="provenance"
            )
            logger.info(
                "Ensured tenant provenance schema exists: provenance_%s",
                self.tenant_id,
            )

        # Strip any litellm-style "provider/" prefix from model names — the
        # OpenAI-compatible endpoint expects the bare
        # model name. Leaving the litellm-prefixed model id would make the OAI-compat endpoint reply with
        # "model not found". Use the foundation helper so
        # HuggingFace ``Org/Name`` ids (e.g. Qwen/Qwen2.5-7B-Instruct) keep
        # their org segment — only known DSPy provider prefixes are stripped.
        from cogniverse_foundation.dspy import bare_model_name

        llm_provider_config = {
            "model": bare_model_name(llm_model),
            "temperature": 0.1,
        }

        # The embedder targets DenseOn's /v1/embeddings endpoint, independent
        # of the LLM provider (different pods, different models). Route through
        # the cogniverse DenseOn adapter rather than Mem0's stock openai
        # embedder so the sentence-transformers prompt prefix + L2-normalization
        # the pylate sidecar applied are restored client-side; otherwise stored
        # memory vectors drift against existing agent_memories rows.
        from cogniverse_core.memory.mem0_embedder import (
            DENSEON_PROVIDER,
            register_denseon_provider,
        )

        register_denseon_provider()
        embedder_url = embedder_base_url.rstrip("/")
        if not embedder_url.endswith("/v1"):
            embedder_url = f"{embedder_url}/v1"
        embedder_provider_config = {
            "model": bare_model_name(embedding_model),
            "openai_base_url": embedder_url,
            "api_key": "denseon",
        }

        # All supported LLM backends (OAI-compat local servers, OpenAI, hosted) speak
        # OpenAI-compatible /v1 chat completions, so we always use Mem0's
        # "openai" provider. Normalise the URL with a /v1 suffix when the
        # caller hasn't included one — matches the chart's primaryLLM
        # endpoint convention which omits /v1.
        base_url = llm_base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        llm_provider_config["openai_base_url"] = base_url
        llm_provider_config["api_key"] = llm_api_key

        self.config = {
            "llm": {
                "provider": "openai",
                "config": llm_provider_config,
            },
            "embedder": {
                "provider": DENSEON_PROVIDER,
                "config": embedder_provider_config,
            },
            "vector_store": {
                "provider": "backend",  # Backend-agnostic (not vespa-specific)
                "config": {
                    "collection_name": tenant_schema_name,  # Tenant-specific schema
                    "backend_client": backend,  # Pre-configured backend instance
                    "embedding_model_dims": 768,  # DenseOn output dimension
                    "tenant_id": self.tenant_id,  # Pass tenant_id directly
                    "profile": base_schema_name,  # Pass base schema/profile name
                },
            },
        }

        # Initialize Memory
        self.memory = Memory.from_config(self.config)
        self._init_fingerprint = fingerprint

        # stash the optional knowledge registry. When set, add_memory
        # enforces schema.provenance_required and computes initial trust.
        self._knowledge_registry = knowledge_registry

        # Indexed provenance store (built lazily on first use). Per-tenant
        # provenance documents land in their own Vespa schema so chain
        # walks can batch each BFS level into one query.
        self._provenance_store: Optional[object] = None

        logger.info(
            f"Mem0MemoryManager initialized for tenant {self.tenant_id} "
            f"with schema {tenant_schema_name} at {backend_host}:{backend_port}"
        )

    @property
    def provenance_store(self):
        """Lazy per-tenant ProvenanceStore wrapping the Vespa backend."""
        if (
            self._provenance_store is None
            and getattr(self, "_backend", None) is not None
        ):
            from cogniverse_core.memory.provenance_store import ProvenanceStore

            self._provenance_store = ProvenanceStore(
                backend=self._backend, tenant_id=self.tenant_id
            )
        return self._provenance_store

    def _enforce_schema_on_write(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate provenance + auto-attach initial trust per schema.

        Returns the (possibly-augmented) metadata dict. Pure function over
        ``metadata``; the caller passes the result on to ``Memory.add``.

        Behaviour matrix:
          * No knowledge_registry wired → return metadata unchanged.
          * No ``metadata.kind`` set → cannot resolve schema, return
            unchanged. (The registry's safe default would fire here in
            ``registry.get(kind)``, but without an explicit kind we don't
            know what was meant — better to surface than to guess.)
          * Schema requires provenance and metadata lacks it → raise
            :class:`SchemaViolationError`. The write never reaches Mem0.
          * Schema has ``default_trust`` and metadata already has a
            ``trust`` block → leave it alone (caller is overriding).
          * Schema has ``default_trust`` and no trust block yet → compute
            initial trust from the provenance + schema and attach it.
        """
        if self._knowledge_registry is None:
            return metadata
        kind = metadata.get("kind")
        if not kind:
            return metadata

        from cogniverse_core.memory.provenance import extract_from_memory
        from cogniverse_core.memory.schema import SchemaViolationError
        from cogniverse_core.memory.trust import (
            attach_trust_to_metadata,
            compute_initial_trust,
        )

        try:
            schema = self._knowledge_registry.get(kind)
        except Exception as exc:
            logger.debug("Schema lookup failed for kind=%r: %s", kind, exc)
            return metadata

        # Provenance is read off a synthetic memory-shaped dict because
        # extract_from_memory expects {"metadata": …} shape.
        provenance = extract_from_memory({"metadata": metadata})
        if schema.provenance_required and provenance is None:
            raise SchemaViolationError(
                f"kind={kind!r} requires provenance but metadata is missing "
                "the provenance block — refusing the write"
            )

        # EPHEMERAL_SESSION writes need session_id so drop_session can
        # find them at session-end. Schema raises SchemaViolationError.
        schema.validate_session_membership(metadata)

        if "trust" not in metadata:
            trust = compute_initial_trust(schema, provenance)
            metadata = attach_trust_to_metadata(metadata, trust)
        return metadata

    def add_memory(
        self,
        content: str,
        tenant_id: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> Optional[str]:
        """
        Add content to agent's memory.

        Args:
            content: Memory content to store
            tenant_id: Tenant identifier
            agent_name: Agent name
            metadata: Optional metadata
            infer: If True, Mem0 runs an LLM extraction pass before storing.
                If False, the content is stored verbatim. Set False for
                user-provided memories where the text is already curated.

        Returns:
            Memory ID of the stored memory, or None when Mem0 deliberately
            stored nothing (LLM found no extractable facts, or content
            was deduplicated against an existing memory). Callers that
            require storage to succeed should check for ``None`` and
            either retry with ``infer=False`` or treat as a no-op.

        Raises:
            RuntimeError: If the backend is not initialised.
        """
        if not self.memory:
            raise RuntimeError("Mem0MemoryManager not initialized")

        # schema enforcement. When a registry is wired, every write
        # is checked against the schema for the metadata.kind:
        #   * provenance_required=True → reject if no provenance attached
        #   * default_trust set → compute initial trust from provenance and
        #     attach to metadata so retrieval-time ranking has it.
        # When no registry is wired, the write proceeds without
        # enforcement.
        metadata = self._enforce_schema_on_write(metadata or {})

        result = self.memory.add(
            content,
            user_id=tenant_id,
            agent_id=agent_name,
            metadata=metadata,
            infer=infer,
        )
        logger.info(f"Mem0.add() returned: {result}")

        memory_id: Optional[str] = None
        if isinstance(result, dict):
            if result.get("id"):
                memory_id = str(result["id"])
            else:
                entries = result.get("results") or []
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("id"):
                        memory_id = str(entry["id"])
                        break
        elif isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict) and entry.get("id"):
                    memory_id = str(entry["id"])
                    break

        if not memory_id:
            # Empty results from Mem0 are legitimate — either the LLM
            # extraction found no facts (common with small local models)
            # or the content was deduplicated against an existing memory.
            # Surface as a warning, not an exception; callers that care
            # about actual storage should use ``infer=False``.
            logger.warning(
                "Mem0 stored no memory for %s/%s (infer=%s); raw response: %r",
                tenant_id,
                agent_name,
                infer,
                result,
            )
            return None

        logger.info(f"Added memory for {tenant_id}/{agent_name}: {memory_id}")

        # Persist provenance to the indexed Vespa store. The walker
        # reads from this store exclusively — a failure here means
        # this memory is unreachable from any chain walk, so raise
        # rather than silently dropping a citation edge.
        self._attach_indexed_provenance(memory_id, metadata)

        # detect contradictions on the write. The detector runs on every
        # knowledge write, persisting a ``conflict_set`` memory when the
        # new write disagrees with an existing same-subject memory. This
        # is what ContradictionReconciliationAgent consumes in production.
        # Best-effort: detection failure must not block the write that
        # already succeeded.
        try:
            self._detect_and_persist_contradictions(
                memory_id=memory_id,
                tenant_id=tenant_id,
                agent_name=agent_name,
                metadata=metadata,
                content=content,
            )
        except Exception as exc:
            logger.warning(
                "Contradiction detection failed for %s/%s/%s: %s",
                tenant_id,
                agent_name,
                memory_id,
                exc,
            )

        return memory_id

    def _attach_indexed_provenance(
        self, memory_id: str, metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Persist the in-band provenance to the indexed Vespa store."""
        if not isinstance(metadata, dict):
            return
        prov_payload = metadata.get("provenance")
        if not isinstance(prov_payload, dict):
            return
        store = self.provenance_store
        if store is None:
            return
        from cogniverse_core.memory.provenance import Provenance

        try:
            provenance = Provenance.from_metadata_payload(prov_payload)
        except (KeyError, ValueError) as exc:
            logger.debug(
                "Skipping provenance attach for %s — malformed payload: %s",
                memory_id,
                exc,
            )
            return
        store.attach(memory_id, provenance)

    def _detect_and_persist_contradictions(
        self,
        *,
        memory_id: str,
        tenant_id: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]],
        content: str,
    ) -> None:
        """Run ContradictionDetector against the new write + same-subject peers.

        Persists a ``conflict_set`` sentinel memory under
        ``_conflict_store`` for every distinct conflict surfaced. Skips
        the recursion case (the new write itself IS a conflict_set
        record) and the no-context cases (no registry, no subject_key).
        """
        if self._knowledge_registry is None:
            return  # detector is opt-in; skipped until a registry is wired
        if not isinstance(metadata, dict):
            return
        kind = metadata.get("kind")
        if kind == "conflict_set":
            return  # writing a conflict_set itself — don't recurse
        subject_key = metadata.get("subject_key")
        if not subject_key:
            return

        from cogniverse_core.memory.contradiction import (
            CONFLICT_AGENT_NAME,
            CONFLICT_RECORD_KIND,
            ContradictionDetector,
        )

        # Fetch only same-subject peers. subject_key is promoted to a
        # top-level Vespa field, so the backend filters server-side rather than
        # pulling the tenant's memories (capped at Mem0's limit, which could
        # also miss same-subject peers past the cap) and filtering in Python.
        try:
            raw = self.memory.get_all(
                user_id=tenant_id, filters={"subject_key": subject_key}
            )
        except Exception as exc:
            logger.warning("get_all for contradiction scan failed: %s", exc)
            return
        peers = raw.get("results", []) if isinstance(raw, dict) else (raw or [])

        candidates: List[Dict[str, Any]] = []
        for row in peers:
            if not isinstance(row, dict):
                continue
            row_meta = row.get("metadata") or {}
            if not isinstance(row_meta, dict):
                continue
            if row_meta.get("subject_key") != subject_key:
                continue
            if row_meta.get("kind") == CONFLICT_RECORD_KIND:
                continue  # never include conflict_set records as peers
            candidates.append(row)

        # Add the just-written memory to the candidate list — Mem0's
        # get_all may not be read-after-write consistent across all
        # backends, so include it explicitly.
        if not any(str(r.get("id")) == memory_id for r in candidates):
            candidates.append(
                {
                    "id": memory_id,
                    "memory": content,
                    "metadata": dict(metadata),
                }
            )

        detector = ContradictionDetector()
        new_conflicts = detector.detect(candidates)
        if not new_conflicts:
            return

        # Walk existing conflict_set memories for this subject so we don't
        # write a duplicate every time the same conflict re-surfaces.
        try:
            # subject_key is a promoted Vespa field — filter server-side
            # instead of pulling every conflict record per memory write.
            existing_blob = self.memory.get_all(
                user_id=tenant_id,
                agent_id=CONFLICT_AGENT_NAME,
                filters={"subject_key": subject_key},
            )
        except Exception as exc:
            # Can't know the subject's current conflict state — persisting
            # blind would duplicate the sentinel every time the same
            # conflict re-surfaces. Skip persistence for this write.
            logger.warning(
                "conflict-state read failed for subject_key=%s: %r — "
                "skipping contradiction persistence for this write",
                subject_key,
                exc,
            )
            return
        existing_rows = (
            existing_blob.get("results", [])
            if isinstance(existing_blob, dict)
            else (existing_blob or [])
        )
        # One conflict_set per subject_key: the existing record already
        # says "this subject is disputed". Subsequent writes that grow the
        # member list shouldn't spam the audit log — operators fetch
        # current members on demand from the live memories.
        subject_already_flagged = any(
            isinstance(row, dict)
            and isinstance(row.get("metadata"), dict)
            and row["metadata"].get("subject_key") == subject_key
            for row in existing_rows
        )
        if subject_already_flagged:
            return

        for conflict in new_conflicts:
            try:
                self.memory.add(
                    conflict.to_memory_content(),
                    user_id=tenant_id,
                    agent_id=CONFLICT_AGENT_NAME,
                    metadata=conflict.to_metadata_payload(),
                    infer=False,
                )
                logger.info(
                    "Persisted conflict_set for tenant=%s subject=%s members=%s",
                    tenant_id,
                    subject_key,
                    list(conflict.conflicting_memory_ids),
                )
            except Exception as exc:
                logger.warning("Failed to persist conflict_set: %s", exc)

    def search_memory(
        self,
        query: str,
        tenant_id: str,
        agent_name: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search agent's memory for relevant content.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            agent_name: Agent name
            top_k: Number of results to return
            filters: Optional Mem0 metadata filters (e.g. {"agent": "search_agent"}).
                Passed directly to memory.search() — supports Mem0's full filter
                syntax including exact-match, in-list, and logical operators.
            include_archived: when False (default), soft-deleted memories
                (``metadata.archived=true``) are filtered out post-fetch. Set
                True for admin / restore tooling that needs to surface them.

        Returns:
            List of matching memories with scores
        """
        if not self.memory:
            logger.warning("Mem0MemoryManager not initialized, returning empty results")
            return []

        try:
            results = self.memory.search(
                query,
                user_id=tenant_id,
                agent_id=agent_name,
                limit=top_k,
                filters=filters,
            )

            # Mem0 search might return dict with "results" key
            if isinstance(results, dict):
                actual_results = results.get("results", [])
                logger.info(
                    f"Found {len(actual_results)} memories for {tenant_id}/{agent_name} (from dict)"
                )
            else:
                logger.info(
                    f"Found {len(results)} memories for {tenant_id}/{agent_name}"
                )
                actual_results = results

            if not include_archived:
                actual_results = [
                    r
                    for r in actual_results
                    if not self._read_metadata(r).get("archived")
                ]

            # bump last_accessed on each hit so the lifecycle
            # scheduler doesn't prune actively-used memories. Best-effort:
            # logged-not-raised on backend failure (the search itself
            # succeeded, returning a stale recency signal is better than
            # erroring the whole call).
            self._bump_last_accessed_for_hits(actual_results)
            return actual_results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    # Re-stamps within this window are skipped — the lifecycle scheduler
    # reads recency at day scale, so per-request writes buy nothing.
    _LAST_ACCESSED_BUMP_INTERVAL_S = 900

    def _bump_last_accessed_for_hits(self, hits: List[Dict[str, Any]]) -> None:
        """Update each hit's ``last_accessed`` ISO timestamp.

        Best-effort: any failure (store doesn't expose update, backend
        rejects the partial-update, network blip) is logged at DEBUG and
        the search result is still returned. The lifecycle scheduler reads
        ``last_accessed`` from the metadata; missing or stale values cause
        it to fall back to ``created_at`` on the next tick.

        Routes through the vector store's partial update (``vector=None``)
        rather than Mem0's ``Memory.update`` — the latter re-embeds the
        memory text over HTTP on every call, which made each search pay
        ``top_k`` embedder round-trips just to stamp a timestamp. The
        vector store still owns schema-name routing and ``metadata_``
        serialization, and ``vector=None`` leaves the stored embedding
        untouched. All eligible hits go out in ONE ``update_many`` feed
        (falling back to per-hit ``update`` for stores without it). Hits
        stamped within the last ``_LAST_ACCESSED_BUMP_INTERVAL_S`` seconds
        are skipped entirely.
        """
        if not hits:
            return
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        memory = self.memory
        store = getattr(memory, "vector_store", None) if memory is not None else None
        if store is None:
            return
        has_batch = hasattr(store, "update_many")
        if not has_batch and not hasattr(store, "update"):
            return

        pending: List[tuple] = []
        for hit in hits:
            mid = hit.get("id")
            if not mid:
                continue
            try:
                metadata = hit.get("metadata") or {}
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                previous = metadata.get("last_accessed")
                if isinstance(previous, str):
                    try:
                        previous_dt = datetime.fromisoformat(
                            previous.replace("Z", "+00:00")
                        )
                        if previous_dt.tzinfo is None:
                            previous_dt = previous_dt.replace(tzinfo=timezone.utc)
                        age = (now - previous_dt).total_seconds()
                        if age < self._LAST_ACCESSED_BUMP_INTERVAL_S:
                            continue
                    except ValueError:
                        pass
                metadata["last_accessed"] = now_iso
                payload = {"data": hit.get("memory", ""), "metadata": metadata}
                if has_batch:
                    pending.append((mid, None, payload))
                else:
                    store.update(vector_id=mid, vector=None, payload=payload)
            except Exception as exc:
                logger.debug(
                    "last_accessed bump failed for memory_id=%s: %s",
                    mid,
                    exc,
                )
        if pending:
            try:
                # One feed for all hits — per-hit updates cost top_k
                # sequential round-trips per search.
                store.update_many(pending)
            except Exception as exc:
                logger.debug(
                    "last_accessed batch bump failed for %d memories: %s",
                    len(pending),
                    exc,
                )

    def get_all_memories(
        self,
        tenant_id: str,
        agent_name: str,
        include_archived: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for an agent.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name
            include_archived: when False (default), soft-deleted memories
                are excluded. Set True for admin restore tooling that
                needs to surface them.
            filters: Optional server-side filters (e.g. ``{"subject_key":
                ...}``) merged into the store query — prefer these over
                fetching everything and filtering in Python.

        Returns:
            List of all memories
        """
        if not self.memory:
            return []

        try:
            result = self.memory.get_all(
                user_id=tenant_id,
                agent_id=agent_name,
                filters=filters,
            )

            # Mem0 get_all returns {"results": [...]}
            memories = result.get("results", []) if isinstance(result, dict) else result

            if not include_archived:
                memories = [
                    m for m in memories if not self._read_metadata(m).get("archived")
                ]

            logger.info(
                f"Retrieved {len(memories)} total memories for {tenant_id}/{agent_name}"
            )
            return memories

        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    def delete_memory(
        self,
        memory_id: str,
        tenant_id: str,
        agent_name: str,
    ) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: Memory ID to delete
            tenant_id: Tenant identifier (not used, for API compatibility)
            agent_name: Agent name (not used, for API compatibility)

        Returns:
            Success status
        """
        if not self.memory:
            return False

        try:
            self.memory.delete(memory_id)

            logger.info(f"Deleted memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    def clear_agent_memory(
        self,
        tenant_id: str,
        agent_name: str,
    ) -> bool:
        """
        Clear all memory for an agent.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name

        Returns:
            Success status
        """
        if not self.memory:
            return False

        try:
            # Get all memories and delete them
            memories = self.get_all_memories(tenant_id, agent_name)

            failed_count = 0
            for memory in memories:
                # Memory can be a dict or a string ID
                if isinstance(memory, dict):
                    memory_id = memory.get("id")
                else:
                    memory_id = str(memory)

                if memory_id:
                    success = self.delete_memory(memory_id, tenant_id, agent_name)
                    if not success:
                        failed_count += 1

            if failed_count > 0:
                logger.error(
                    f"Failed to delete {failed_count}/{len(memories)} memories "
                    f"for {tenant_id}/{agent_name}"
                )
                return False

            logger.info(f"Cleared all memory for {tenant_id}/{agent_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear agent memory: {e}")
            return False

    def cleanup_with_schema(
        self,
        registry: "KnowledgeRegistry",
        pinned_memory_ids: Optional[set] = None,
    ) -> Dict[str, int]:
        """schema-driven cleanup that respects per-kind retention.

        Iterates every memory for this tenant; for each one, looks up its
        ``kind`` in the registry and applies the retention policy:

          * ``PERMANENT`` → never deleted.
          * ``EPHEMERAL_SESSION`` → not handled here (session lifecycle is
            owned by the caller's session manager).
          * ``EPHEMERAL_DAYS(N)`` → delete when ``created_at`` is older
            than ``N`` days.
          * ``SCHEMA_DRIVEN`` → delegate to ``schema.cleanup_hook`` when
            present; skip when no hook is registered.

        Pinned memory ids are skipped entirely (lifecycle never overrides
        an explicit pin). Memories with no ``kind`` metadata fall back to
        the registry's safe default, which is permanent — they are skipped.

        Args:
            registry: ``KnowledgeRegistry`` used to look up per-kind policy.
            pinned_memory_ids: ids that must NOT be deleted regardless of
                policy. Typically built from
                ``PinService.list_pins(tenant_id)`` upstream.

        Returns:
            Mapping of kind → deleted_count for the run. Useful for the
            scheduler's audit logs.
        """
        from cogniverse_core.memory.schema import (
            KnowledgeRegistry,  # noqa: F401 — runtime ref for type-checker fallback
            Retention,
        )

        if not self.memory:
            raise RuntimeError("Mem0MemoryManager not initialized")

        pinned_ids = pinned_memory_ids or set()
        now_epoch = int(time.time())
        deleted_by_kind: Dict[str, int] = {}

        result = self.memory.get_all(user_id=self.tenant_id)
        memories = result.get("results", []) if isinstance(result, dict) else result

        for memory in memories:
            if not isinstance(memory, dict):
                continue
            memory_id = memory.get("id")
            if not memory_id or memory_id in pinned_ids:
                continue

            kind = self._extract_kind(memory)
            schema = registry.get(kind)

            should_delete = False
            if schema.retention is Retention.PERMANENT:
                continue
            elif schema.retention is Retention.EPHEMERAL_SESSION:
                # Session lifecycle is event-driven (drop_session at session
                # end) not time-driven; the periodic scheduler does not
                # touch EPHEMERAL_SESSION memories.
                continue
            elif schema.retention is Retention.EPHEMERAL_DAYS:
                age_seconds = self._compute_age_seconds(memory, now_epoch)
                if age_seconds is None:
                    continue
                # soft-delete window: at TTL flip archived=true,
                # at 2× TTL hard-delete. Operators can restore in
                # between via the admin endpoint. retention_days is
                # validated > 0 in KnowledgeSchema.__post_init__.
                cutoff = (schema.retention_days or 0) * 86400
                hard_cutoff = cutoff * 2
                meta = self._read_metadata(memory)
                if age_seconds > hard_cutoff:
                    should_delete = True
                elif age_seconds > cutoff and not meta.get("archived"):
                    # Soft-delete: flip archived flag, do not remove.
                    self._archive_memory(
                        memory_id,
                        meta,
                        existing_data=memory.get("memory") or memory.get("text") or "",
                    )
                    deleted_by_kind[f"{kind}:archived"] = (
                        deleted_by_kind.get(f"{kind}:archived", 0) + 1
                    )
                    continue
            elif schema.retention is Retention.SCHEMA_DRIVEN:
                hook = schema.cleanup_hook
                if hook is None:
                    continue
                try:
                    should_delete = bool(hook(memory, schema))
                except Exception as exc:
                    logger.warning(
                        "cleanup_hook for kind=%s raised %s; skipping memory %s",
                        kind,
                        type(exc).__name__,
                        memory_id,
                    )
                    continue

            if should_delete:
                self.memory.delete(memory_id)
                deleted_by_kind[kind] = deleted_by_kind.get(kind, 0) + 1
                logger.debug(
                    "Schema-driven delete: kind=%s memory_id=%s policy=%s",
                    kind,
                    memory_id,
                    schema.retention.value,
                )

        if deleted_by_kind:
            logger.info(
                "Schema-driven cleanup for tenant %s: %s",
                self.tenant_id,
                deleted_by_kind,
            )
        return deleted_by_kind

    def drop_session(
        self,
        session_id: str,
        registry: "KnowledgeRegistry",
    ) -> Dict[str, int]:
        """Hard-delete every EPHEMERAL_SESSION memory tagged with this session.

        Walks the tenant's memories, matches on
        ``metadata.session_id == session_id`` AND
        ``schema.retention is EPHEMERAL_SESSION``, hard-deletes each match.

        Pinned memories ARE deleted by this call (a session-end cleanup is
        the user's explicit signal to drop their session state). Operators
        who want pinning to outlive a session should not pin
        EPHEMERAL_SESSION kinds in the first place — the schema's
        ``pinnable_by`` field is the right gate for that.

        Args:
            session_id: identifier passed in at write time. Required;
                empty string raises ValueError.
            registry: KnowledgeRegistry resolving each memory's kind to
                its schema. Required — no fallback "delete anything with
                this session_id" behaviour.

        Returns:
            ``{kind: deleted_count}`` for the run.
        """
        from cogniverse_core.memory.schema import Retention

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("drop_session requires a non-empty session_id")
        if not self.memory:
            raise RuntimeError("Mem0MemoryManager not initialized")

        # session_id is a promoted Vespa field written at insert time, so
        # the store returns only this session's rows — walking the tenant's
        # whole corpus cost O(memory count) per session end. When the
        # filtered query returns nothing, verify with the full scan: a
        # tenant schema deployed before the field existed makes the filter
        # yield an empty result (the store layer flattens the unknown-field
        # error), and converting those tenants' session cleanup into a
        # silent no-op is not acceptable. On current schemas the fallback
        # only runs for genuinely empty sessions, which cost the full scan
        # before this optimization too.
        deleted_by_kind: Dict[str, int] = {}
        result = self.memory.get_all(
            user_id=self.tenant_id, filters={"session_id": session_id}
        )
        memories = result.get("results", []) if isinstance(result, dict) else result
        if not memories:
            result = self.memory.get_all(user_id=self.tenant_id)
            memories = result.get("results", []) if isinstance(result, dict) else result
            if any(
                self._read_metadata(m).get("session_id") == session_id
                for m in memories
                if isinstance(m, dict)
            ):
                logger.warning(
                    "drop_session(%s): server-side session_id filter returned "
                    "nothing but a scan found matching rows — the tenant "
                    "schema for %s likely predates the session_id field; "
                    "redeploy it to restore filtered cleanup.",
                    session_id,
                    self.tenant_id,
                )

        for memory in memories:
            if not isinstance(memory, dict):
                continue
            memory_id = memory.get("id")
            if not memory_id:
                continue

            meta = self._read_metadata(memory)
            if meta.get("session_id") != session_id:
                continue

            kind = self._extract_kind(memory)
            schema = registry.get(kind)
            if schema.retention is not Retention.EPHEMERAL_SESSION:
                continue

            self.memory.delete(memory_id)
            deleted_by_kind[kind] = deleted_by_kind.get(kind, 0) + 1

        if deleted_by_kind:
            logger.info(
                "drop_session(%s) for tenant %s: %s",
                session_id,
                self.tenant_id,
                deleted_by_kind,
            )
        return deleted_by_kind

    @staticmethod
    def _extract_kind(memory: Dict[str, Any]) -> str:
        """Read ``metadata.kind`` from a Mem0 memory dict, tolerating shapes."""
        meta = memory.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                return "_unknown"
        if isinstance(meta, dict):
            return str(meta.get("kind") or "_unknown")
        return "_unknown"

    @staticmethod
    def _read_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive metadata reader — handles dict / JSON-string / None shapes."""
        meta = memory.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                return {}
        return meta if isinstance(meta, dict) else {}

    def _archive_memory(
        self,
        memory_id: str,
        meta: Dict[str, Any],
        existing_data: str = "",
    ) -> None:
        """soft-delete: flip metadata.archived=true with a timestamp.

        The lifecycle scheduler calls this when a memory hits its TTL but
        not yet 2*TTL. The memory remains queryable via opt-in
        ``include_archived=True`` paths and via the admin restore
        endpoint; default reads filter it out.
        """
        from datetime import datetime, timezone

        new_meta = dict(meta)
        new_meta["archived"] = True
        new_meta["archived_at"] = datetime.now(timezone.utc).isoformat()
        try:
            # Mem0.update requires `data` — pass the current content so
            # the memory text is preserved.
            self.memory.update(
                memory_id=memory_id,
                data=existing_data,
                metadata=new_meta,
            )
            logger.info(
                "Soft-deleted (archived) memory %s for tenant %s",
                memory_id,
                self.tenant_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to archive memory %s: %s — falling through to hard-delete on next tick",
                memory_id,
                exc,
            )

    def restore_archived_memory(self, memory_id: str) -> bool:
        """admin restore: clear the archived flag on a soft-deleted memory.

        Returns True when the flag was cleared; False when the memory
        wasn't found or wasn't archived. Callers (the admin endpoint)
        translate False into a 404/409 as appropriate.
        """
        if not self.memory:
            return False
        try:
            blob = self.memory.get_all(user_id=self.tenant_id)
        except Exception as exc:
            logger.warning("restore: get_all failed: %s", exc)
            return False
        rows = blob.get("results", []) if isinstance(blob, dict) else (blob or [])
        target = next((r for r in rows if str(r.get("id")) == memory_id), None)
        if target is None:
            return False
        meta = self._read_metadata(target)
        if not meta.get("archived"):
            return False
        meta = dict(meta)
        meta.pop("archived", None)
        meta.pop("archived_at", None)
        existing_data = target.get("memory") or target.get("text") or ""
        try:
            self.memory.update(memory_id=memory_id, data=existing_data, metadata=meta)
        except Exception as exc:
            logger.warning("restore: update failed for %s: %s", memory_id, exc)
            return False
        logger.info(
            "Restored archived memory %s for tenant %s", memory_id, self.tenant_id
        )
        return True

    @staticmethod
    def _compute_age_seconds(memory: Dict[str, Any], now_epoch: int) -> Optional[int]:
        created_epoch = to_epoch_seconds(memory.get("created_at"))
        if created_epoch is None:
            return None
        return max(0, now_epoch - created_epoch)

    def update_memory(
        self,
        memory_id: str,
        content: str,
        tenant_id: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID to update
            content: New content
            tenant_id: Tenant identifier (not used by Mem0 update API)
            agent_name: Agent name (not used by Mem0 update API)
            metadata: Optional metadata

        Returns:
            Success status
        """
        if not self.memory:
            return False

        try:
            # Mem0's update() only accepts memory_id and data (content)
            # It does NOT accept user_id or agent_id
            self.memory.update(
                memory_id,
                data=content,
            )

            logger.info(f"Updated memory {memory_id} for {tenant_id}/{agent_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False

    def health_check(self) -> bool:
        """
        Check if memory manager is healthy.

        Returns:
            Health status
        """
        if not self.memory:
            return False

        try:
            # Memory is initialized - considered healthy
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_memory_stats(
        self,
        tenant_id: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        Get memory statistics for an agent.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name

        Returns:
            Memory statistics
        """
        if not self.memory:
            return {"total_memories": 0, "enabled": False}

        try:
            memories = self.get_all_memories(tenant_id, agent_name)

            return {
                "total_memories": len(memories),
                "enabled": True,
                "tenant_id": tenant_id,
                "agent_name": agent_name,
            }

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"total_memories": 0, "enabled": True, "error": str(e)}
