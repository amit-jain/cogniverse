"""
Mem0 Memory Manager

Mem0-based memory system using Vespa backend.
Provides simple, persistent agent memory with multi-tenant support.
Each tenant gets dedicated Vespa schema for memory isolation.
"""

import logging
import os
from typing import Any, Dict, List, Optional

# Disable Mem0's telemetry BEFORE importing mem0
os.environ["MEM0_TELEMETRY"] = "False"

from mem0 import Memory

logger = logging.getLogger(__name__)


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
    provider_configs_attr = VectorStoreConfig.__private_attributes__["_provider_configs"]
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

    # Per-tenant instances cache
    _instances: Dict[str, "Mem0MemoryManager"] = {}
    _instances_lock = None  # Will be initialized as threading.Lock()

    def __new__(cls, tenant_id: str):
        """Per-tenant singleton pattern"""
        import threading

        # Initialize lock on first call
        if cls._instances_lock is None:
            cls._instances_lock = threading.Lock()

        with cls._instances_lock:
            if tenant_id not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[tenant_id] = instance
                logger.info(
                    f"Created new Mem0MemoryManager instance for tenant: {tenant_id}"
                )
            return cls._instances[tenant_id]

    def __init__(self, tenant_id: str):
        """Initialize memory manager for specific tenant (only once per tenant)"""
        if self._initialized:
            return

        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id
        self.memory: Optional[Memory] = None
        self.config: Optional[Dict[str, Any]] = None

        self._initialized = True
        logger.info(f"Mem0MemoryManager initialized for tenant: {tenant_id}")

    def initialize(
        self,
        backend_host: str = "localhost",
        backend_port: int = 8080,
        backend_config_port: Optional[int] = None,
        base_schema_name: str = "agent_memories",
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434/v1",
        auto_create_schema: bool = True,
        config_manager = None,
        schema_loader = None,
    ) -> None:
        """
        Initialize Mem0 with backend using tenant-specific schema.

        Args:
            backend_host: Backend endpoint host
            backend_port: Backend data endpoint port (default: 8080)
            backend_config_port: Backend config endpoint port (default: 19071)
            base_schema_name: Base schema name (default: agent_memories)
            llm_model: Ollama model name (default: llama3.2)
            embedding_model: Ollama embedding model (default: nomic-embed-text)
            ollama_base_url: Ollama OpenAI-compatible API endpoint
            auto_create_schema: Auto-deploy tenant schema if not exists
            config_manager: ConfigManager instance (REQUIRED)
            schema_loader: SchemaLoader instance (REQUIRED)

        Raises:
            ValueError: If tenant_id not set
        """
        if not self.tenant_id:
            raise ValueError("tenant_id must be set before initialize()")

        # Get backend instance for memory operations
        from cogniverse_core.config.manager import ConfigManager
        from cogniverse_core.config.utils import get_config
        from cogniverse_core.registries.backend_registry import get_backend_registry

        config_manager = ConfigManager()
        config = get_config(tenant_id="default", config_manager=config_manager)
        backend_type = config.get("backend_type", "vespa")
        registry = get_backend_registry()

        # Get backend instance with full config including profiles
        # Add agent_memories profile since it may not be in config
        profiles = config.get("profiles", {})
        if base_schema_name not in profiles:
            # Add minimal profile for agent_memories schema
            profiles[base_schema_name] = {
                "type": "memory",
                "model": "nomic-embed-text",
                "embedding_dims": 768,
                "encoder": "ollama",
                "strategy": "semantic_search",  # Default to semantic search for memories
            }

        # Build backend section with ollama_base_url and profiles for encoder initialization
        backend_section = {
            "ollama_base_url": ollama_base_url,  # Required for Ollama encoder
            "profiles": profiles,  # Profiles for VespaSearchBackend._initialize_encoders()
            **config.get("backend", {}),  # Merge any existing backend config
        }

        backend_config_dict = {
            "url": backend_host,  # VespaBackend expects "url", not "backend_url"
            "port": backend_port,  # VespaBackend expects "port", not "backend_port"
            "config_port": backend_config_port or 19071,
            "schema_name": base_schema_name,  # Base schema name for operations (backend handles tenant transformation)
            "backend": backend_section,  # Backend section for search operations
            "profiles": profiles,  # Also keep at top level for config merging
            "default_profiles": config.get("default_profiles", {}),
        }

        # Create tenant-specific backend for memory operations
        # Each tenant gets their own memory schema (agent_memories_{tenant_id})
        backend = registry.get_ingestion_backend(
            backend_type, tenant_id=self.tenant_id, config=backend_config_dict,
            config_manager=config_manager, schema_loader=schema_loader
        )

        # Get tenant-specific schema name
        tenant_schema_name = backend.get_tenant_schema_name(
            self.tenant_id, base_schema_name
        )

        # Deploy tenant schema if needed
        if auto_create_schema:
            backend.schema_registry.deploy_schema(
                tenant_id=self.tenant_id,
                base_schema_name=base_schema_name
            )
            logger.info(f"Ensured tenant schema exists: {tenant_schema_name}")

        # Configure Mem0 with Ollama provider and backend-agnostic storage
        self.config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": llm_model,
                    "temperature": 0.1,
                    "ollama_base_url": "http://localhost:11434",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": embedding_model,
                    "ollama_base_url": "http://localhost:11434",
                },
            },
            "vector_store": {
                "provider": "backend",  # Backend-agnostic (not vespa-specific)
                "config": {
                    "collection_name": tenant_schema_name,  # Tenant-specific schema
                    "backend_client": backend,  # Pre-configured backend instance
                    "embedding_model_dims": 768,  # nomic-embed-text dimensions
                    "tenant_id": self.tenant_id,  # Pass tenant_id directly
                    "profile": base_schema_name,  # Pass base schema/profile name
                },
            },
        }

        # Initialize Memory
        self.memory = Memory.from_config(self.config)

        logger.info(
            f"Mem0MemoryManager initialized for tenant {self.tenant_id} "
            f"with schema {tenant_schema_name} at {backend_host}:{backend_port}"
        )

    def add_memory(
        self,
        content: str,
        tenant_id: str,
        agent_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add content to agent's memory.

        Args:
            content: Memory content to store
            tenant_id: Tenant identifier
            agent_name: Agent name
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        if not self.memory:
            raise RuntimeError("Mem0MemoryManager not initialized")

        try:
            # Mem0 uses user_id and agent_id for namespacing
            result = self.memory.add(
                content,
                user_id=tenant_id,
                agent_id=agent_name,
                metadata=metadata or {},
            )

            # Log what Mem0 returned for debugging
            logger.info(f"Mem0.add() returned: {result}")

            # Handle different return types from Mem0
            if isinstance(result, dict):
                # Could be {"id": "...", ...} or {"results": [...]}
                if "id" in result:
                    memory_id = result["id"]
                elif "results" in result and result["results"]:
                    memory_id = result["results"][0].get(
                        "id", str(result["results"][0])
                    )
                else:
                    logger.warning(
                        f"Mem0 returned dict without id or results: {result}"
                    )
                    memory_id = str(result)
            elif isinstance(result, list) and result:
                memory_id = (
                    result[0].get("id")
                    if isinstance(result[0], dict)
                    else str(result[0])
                )
            else:
                memory_id = str(result) if result else None

            if memory_id:
                logger.info(f"Added memory for {tenant_id}/{agent_name}: {memory_id}")
            else:
                logger.warning(f"Memory added but no ID returned: {result}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    def search_memory(
        self,
        query: str,
        tenant_id: str,
        agent_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search agent's memory for relevant content.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            agent_name: Agent name
            top_k: Number of results to return

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
            )

            # Mem0 search might return dict with "results" key
            if isinstance(results, dict):
                actual_results = results.get("results", [])
                logger.info(
                    f"Found {len(actual_results)} memories for {tenant_id}/{agent_name} (from dict)"
                )
                return actual_results
            else:
                logger.info(
                    f"Found {len(results)} memories for {tenant_id}/{agent_name}"
                )
                return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    def get_all_memories(
        self,
        tenant_id: str,
        agent_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for an agent.

        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name

        Returns:
            List of all memories
        """
        if not self.memory:
            return []

        try:
            result = self.memory.get_all(
                user_id=tenant_id,
                agent_id=agent_name,
            )

            # Mem0 get_all returns {"results": [...]}
            memories = result.get("results", []) if isinstance(result, dict) else result

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

            for memory in memories:
                # Memory can be a dict or a string ID
                if isinstance(memory, dict):
                    memory_id = memory.get("id")
                else:
                    memory_id = str(memory)

                if memory_id:
                    self.delete_memory(memory_id, tenant_id, agent_name)

            logger.info(f"Cleared all memory for {tenant_id}/{agent_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear agent memory: {e}")
            return False

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
