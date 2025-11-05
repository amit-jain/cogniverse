"""
Memory-Aware Mixin for Agents

Provides standard memory interface for all agents using Mem0.
Handles context retrieval, memory updates, and lifecycle management.
"""

import logging
from typing import Any, Dict, Optional

from cogniverse_core.memory.manager import Mem0MemoryManager

logger = logging.getLogger(__name__)


class MemoryAwareMixin:
    """
    Mixin class that adds memory capabilities to agents.

    Usage:
        class MyAgent(MemoryAwareMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.initialize_memory("my_agent")

            def process_query(self, query):
                # Get relevant context from memory
                context = self.get_relevant_context(query)

                # Process with context
                result = self._process_with_context(query, context)

                # Update memory
                self.update_memory(f"Processed query: {query}")

                return result
    """

    def __init__(self):
        """Initialize memory-aware mixin"""
        self.memory_manager: Optional[Mem0MemoryManager] = None
        self.agent_name: Optional[str] = None
        self.tenant_id: Optional[str] = None
        self._memory_initialized: bool = False

    def initialize_memory(
        self,
        agent_name: str,
        tenant_id: str,
        backend_host: str = "localhost",
        backend_port: int = 8080,
        backend_config_port: Optional[int] = None,
        auto_create_schema: bool = True,
        config_manager = None,
        schema_loader = None,
    ) -> bool:
        """
        Initialize memory for this agent

        Args:
            agent_name: Name of the agent
            tenant_id: Tenant identifier (REQUIRED - no default)
            backend_host: Backend endpoint host
            backend_port: Backend data endpoint port
            backend_config_port: Backend config endpoint port (default: 19071)
            auto_create_schema: Auto-deploy tenant schema if not exists (default: True)
            config_manager: ConfigManager instance (REQUIRED)
            schema_loader: SchemaLoader instance (REQUIRED)

        Returns:
            Success status

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        try:
            self.agent_name = agent_name
            self.tenant_id = tenant_id

            # Get tenant-specific memory manager instance
            self.memory_manager = Mem0MemoryManager(tenant_id=tenant_id)

            # Initialize if not already done
            if self.memory_manager.memory is None:
                self.memory_manager.initialize(
                    backend_host=backend_host,
                    backend_port=backend_port,
                    backend_config_port=backend_config_port,
                    base_schema_name="agent_memories",
                    auto_create_schema=auto_create_schema,
                    config_manager=config_manager,
                    schema_loader=schema_loader,
                )

            self._memory_initialized = True
            logger.info(f"Memory initialized for {agent_name} (tenant: {tenant_id})")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory for {agent_name}: {e}")
            self._memory_initialized = False
            return False

    def is_memory_enabled(self) -> bool:
        """Check if memory is enabled and initialized"""
        return self._memory_initialized and self.memory_manager is not None

    def get_relevant_context(
        self, query: str, top_k: Optional[int] = 5
    ) -> Optional[str]:
        """
        Get relevant context from memory for a query

        Args:
            query: Query to search memory for
            top_k: Number of results to retrieve

        Returns:
            Formatted context string or None
        """
        if not self.is_memory_enabled():
            return None

        if not self.agent_name or not self.memory_manager:
            return None

        try:
            # Search memory with Mem0
            results = self.memory_manager.search_memory(
                query=query,
                tenant_id=self.tenant_id,
                agent_name=self.agent_name,
                top_k=top_k,
            )

            if not results:
                return None

            # Format context - Mem0 returns list of dicts with 'memory' key
            context_parts = []
            for i, result in enumerate(results, 1):
                # Mem0 returns memory content in 'memory' field
                memory_text = result.get("memory", result.get("text", str(result)))
                context_parts.append(f"{i}. {memory_text}")

            context = "\n\n".join(context_parts)

            logger.info(f"Retrieved {len(results)} memories for {self.agent_name}")

            return context

        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return None

    def update_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add content to agent's memory

        Args:
            content: Content to store
            metadata: Optional metadata

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        if not self.agent_name or not self.memory_manager:
            return False

        try:
            memory_id = self.memory_manager.add_memory(
                content=content,
                tenant_id=self.tenant_id,
                agent_name=self.agent_name,
                metadata=metadata,
            )

            if memory_id:
                logger.debug(f"Updated memory for {self.agent_name}: {memory_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False

    def get_memory_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current memory state

        Returns:
            Memory state dictionary with stats or None
        """
        if not self.is_memory_enabled():
            return None

        if not self.agent_name or not self.memory_manager:
            return None

        try:
            return self.memory_manager.get_memory_stats(
                tenant_id=self.tenant_id, agent_name=self.agent_name
            )

        except Exception as e:
            logger.error(f"Failed to get memory state: {e}")
            return None

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Alias for get_memory_state() for API compatibility"""
        return self.get_memory_state()

    def clear_memory(self) -> bool:
        """
        Clear all memory for this agent

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        if not self.agent_name or not self.memory_manager:
            return False

        try:
            success = self.memory_manager.clear_agent_memory(
                tenant_id=self.tenant_id, agent_name=self.agent_name
            )

            if success:
                logger.info(f"Cleared memory for {self.agent_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False

    def inject_context_into_prompt(self, prompt: str, query: str) -> str:
        """
        Inject relevant memory context into a prompt

        Args:
            prompt: Base prompt
            query: Query to search memory for

        Returns:
            Prompt with injected context
        """
        if not self.is_memory_enabled():
            return prompt

        context = self.get_relevant_context(query)

        if not context:
            return prompt

        # Inject context into prompt
        enhanced_prompt = f"""{prompt}

## Relevant Context from Memory:
{context}

## Current Query:
{query}
"""

        return enhanced_prompt

    def remember_success(
        self, query: str, result: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Remember a successful interaction

        Args:
            query: The query that was successful
            result: The result that was produced
            metadata: Optional metadata about the success

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        # Format success memory - direct factual statement for Mem0's LLM
        success_content = (
            f"SUCCESS - Successfully answered: {query}. The answer was: {result}"
        )

        return self.update_memory(success_content, metadata)

    def remember_failure(
        self, query: str, error: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Remember a failed interaction to avoid repeating mistakes

        Args:
            query: The query that failed
            error: Error message or description
            metadata: Optional metadata about the failure

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        # Format failure memory - direct factual statement for Mem0's LLM
        failure_content = (
            f"FAILURE - Failed attempt: {query}. Error encountered: {error}"
        )

        return self.update_memory(failure_content, metadata)

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory status

        Returns:
            Summary dictionary
        """
        summary = {
            "enabled": self.is_memory_enabled(),
            "agent_name": self.agent_name,
            "tenant_id": self.tenant_id,
            "initialized": self._memory_initialized,
        }

        if self.is_memory_enabled():
            memory_state = self.get_memory_state()
            if memory_state:
                summary["total_memories"] = memory_state.get("total_memories", 0)

        return summary
