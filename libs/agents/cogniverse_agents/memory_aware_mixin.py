"""
Memory-Aware Mixin for Agents

Provides standard memory interface for all agents using Mem0.
Handles context retrieval, memory updates, and lifecycle management.
"""

import logging
from typing import Any, Dict, List, Optional

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

    def __init__(self, **kwargs):
        """Initialize memory-aware mixin.

        Accepts **kwargs to work with cooperative multiple inheritance (MRO).
        """
        super().__init__(**kwargs)

        self.memory_manager: Optional[Mem0MemoryManager] = None
        self._memory_agent_name: Optional[str] = None
        # Note: tenant_id is now a property in AgentBase from deps
        # We store it separately for memory operations only if needed
        self._memory_tenant_id: Optional[str] = None
        self._memory_initialized: bool = False

    def set_tenant_for_context(self, tenant_id: str) -> None:
        """Set tenant_id for instruction injection without full memory init."""
        self._memory_tenant_id = tenant_id

    def initialize_memory(
        self,
        agent_name: str,
        tenant_id: str,
        embedder_base_url: str,
        *,
        llm_model: str,
        backend_host: str = "localhost",
        backend_port: int = 8080,
        embedding_model: str = "lightonai/DenseOn",
        llm_base_url: str = "http://localhost:11434",
        llm_api_key: str = "not-required",
        config_manager=None,
        schema_loader=None,
        backend_config_port: Optional[int] = None,
        auto_create_schema: bool = True,
    ) -> bool:
        """
        Initialize memory for this agent.

        Args:
            agent_name: Name of the agent
            tenant_id: Tenant identifier (REQUIRED - no default)
            embedder_base_url: OpenAI-compatible /v1/embeddings endpoint of
                the denseon sidecar (separate from the LLM endpoint).
            llm_model: LLM model name for memory extraction (REQUIRED - must
                come from llm_config.primary, no fallback default).
            backend_host: Backend endpoint host
            backend_port: Backend endpoint port
            embedding_model: Embedding model name for memory search
            llm_base_url: OpenAI-compatible LLM endpoint. ``/v1`` suffix
                added automatically when missing.
            llm_api_key: API key sent to ``llm_base_url``; defaults to
                ``"not-required"`` for local Ollama / vLLM.
            config_manager: ConfigManager instance (REQUIRED for dependency injection)
            schema_loader: SchemaLoader instance (REQUIRED for dependency injection)
            backend_config_port: Backend config endpoint port (default: 19071)
            auto_create_schema: Auto-deploy tenant schema if not exists

        Returns:
            Success status

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        try:
            self._memory_agent_name = agent_name
            self._memory_tenant_id = tenant_id

            self.memory_manager = Mem0MemoryManager(tenant_id=tenant_id)

            if self.memory_manager.memory is None:
                self.memory_manager.initialize(
                    backend_host=backend_host,
                    backend_port=backend_port,
                    llm_model=llm_model,
                    embedding_model=embedding_model,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    embedder_base_url=embedder_base_url,
                    base_schema_name="agent_memories",
                    auto_create_schema=auto_create_schema,
                    config_manager=config_manager,
                    schema_loader=schema_loader,
                    backend_config_port=backend_config_port,
                )

            self._memory_initialized = True
            logger.info(
                f"Memory initialized for {self._memory_agent_name} (tenant: {self._memory_tenant_id})"
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize memory for {self._memory_agent_name}: {e}"
            )
            self._memory_initialized = False
            return False

    def is_memory_enabled(self) -> bool:
        """Check if memory is enabled and initialized.

        Uses getattr with defaults so subclasses with broken cooperative
        __init__ chains (where MemoryAwareMixin.__init__ never ran) still
        report False instead of raising AttributeError.
        """
        return bool(
            getattr(self, "_memory_initialized", False)
            and getattr(self, "memory_manager", None) is not None
        )

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

        if not self._memory_agent_name or not self.memory_manager:
            return None

        try:
            results = self.memory_manager.search_memory(
                query=query,
                tenant_id=self._memory_tenant_id,
                agent_name=self._memory_agent_name,
                top_k=top_k,
            )

            if not results:
                return None

            # P2.2 — apply trust ranking and per-schema reconciliation when
            # a knowledge registry is wired into the manager. Legacy code
            # paths that don't set ``_knowledge_registry`` see no behaviour
            # change (the helpers no-op on missing trust/contradiction).
            results = self._apply_trust_and_reconcile(results)

            context_parts = []
            for i, result in enumerate(results, 1):
                memory_text = result.get("memory", result.get("text", str(result)))
                context_parts.append(f"{i}. {memory_text}")

            context = "\n\n".join(context_parts)

            logger.info(
                f"Retrieved {len(results)} memories for {self._memory_agent_name}"
            )

            return context

        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return None

    def _apply_trust_and_reconcile(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank by trust × confidence and reconcile per-schema (P2.2 wire).

        Composes:
          * :func:`rank_with_trust` (A.4) — moves high-trust hits to the
            top so the agent's prompt sees them first;
          * :class:`ContradictionDetector` + :func:`reconcile` (A.3) — when
            two hits on the same ``subject_key`` disagree, the schema's
            ``contradiction_policy`` decides which survive.

        Behaviour matrix:
          * No knowledge_registry on the underlying manager → return
            ``results`` unchanged (legacy behaviour).
          * Hits without trust → still ranked, but their trust defaults
            to 0.5 inside ``rank_with_trust`` so they sort fairly.
          * Hits with the same subject_key but distinct content → grouped
            into a ConflictSet and reconciled per the schema's policy.
        """
        registry = getattr(self.memory_manager, "_knowledge_registry", None)
        if registry is None:
            return results

        from cogniverse_core.memory.contradiction import (
            ContradictionDetector,
            reconcile,
        )
        from cogniverse_core.memory.schema import ContradictionPolicy
        from cogniverse_core.memory.trust import rank_with_trust

        ranked = rank_with_trust(results)

        # Group hits by metadata.kind so each kind's reconciliation policy
        # applies independently. Hits without a kind fall through.
        by_kind: Dict[str, List[Dict[str, Any]]] = {}
        unkinded: List[Dict[str, Any]] = []
        for r in ranked:
            meta = r.get("metadata") or {}
            kind = meta.get("kind") if isinstance(meta, dict) else None
            if kind:
                by_kind.setdefault(kind, []).append(r)
            else:
                unkinded.append(r)

        out: List[Dict[str, Any]] = list(unkinded)
        detector = ContradictionDetector()
        for kind, group in by_kind.items():
            try:
                schema = registry.get(kind)
                policy = schema.contradiction_policy
            except Exception:
                policy = ContradictionPolicy.LATEST_WINS
            conflicts = detector.detect(group)
            if not conflicts:
                out.extend(group)
                continue
            out.extend(reconcile(group, policy))
        return out

    def update_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> bool:
        """
        Add content to agent's memory

        Args:
            content: Content to store
            metadata: Optional metadata
            infer: If True (default), Mem0 runs an LLM extraction pass
                before storing. Pass ``infer=False`` to store verbatim
                when content is already a curated factual statement —
                this avoids empty results from small local LLMs that
                can't reliably extract facts from short sentences.

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        if not self._memory_agent_name or not self.memory_manager:
            return False

        try:
            memory_id = self.memory_manager.add_memory(
                content=content,
                tenant_id=self._memory_tenant_id,
                agent_name=self._memory_agent_name,
                metadata=metadata,
                infer=infer,
            )

            if memory_id:
                logger.debug(
                    f"Updated memory for {self._memory_agent_name}: {memory_id}"
                )
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

        if not self._memory_agent_name or not self.memory_manager:
            return None

        try:
            return self.memory_manager.get_memory_stats(
                tenant_id=self._memory_tenant_id, agent_name=self._memory_agent_name
            )

        except Exception as e:
            logger.error(f"Failed to get memory state: {e}")
            return None

    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Return memory stats — delegates to get_memory_state()."""
        return self.get_memory_state()

    def clear_memory(self) -> bool:
        """
        Clear all memory for this agent

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        if not self._memory_agent_name or not self.memory_manager:
            return False

        try:
            success = self.memory_manager.clear_agent_memory(
                tenant_id=self._memory_tenant_id, agent_name=self._memory_agent_name
            )

            if success:
                logger.info(f"Cleared memory for {self._memory_agent_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False

    def get_strategies(self, query: str, top_k: int = 5) -> Optional[str]:
        """Retrieve learned strategies relevant to a query.

        Performs two-level retrieval (org + user) via StrategyLearner
        and returns formatted strategies for agent context injection.

        Args:
            query: Query to find relevant strategies for
            top_k: Number of strategies to retrieve

        Returns:
            Formatted strategy text or None
        """
        if not self.is_memory_enabled():
            return None

        from cogniverse_agents.optimizer.strategy_learner import StrategyLearner

        learner = StrategyLearner(
            memory_manager=self.memory_manager,
            tenant_id=self._memory_tenant_id,
        )
        strategies = learner.get_strategies_for_agent(
            query=query,
            agent_name=self._memory_agent_name,
            top_k=top_k,
        )
        if strategies:
            return StrategyLearner.format_strategies_for_context(strategies)
        return None

    def _get_tenant_instructions(self) -> Optional[str]:
        """Load tenant instructions from ConfigStore."""
        tenant_id = getattr(self, "_memory_tenant_id", None)
        if not tenant_id:
            return None
        try:
            import json

            from cogniverse_foundation.config.utils import create_default_config_manager
            from cogniverse_sdk.interfaces.config_store import ConfigScope

            cm = create_default_config_manager()
            entry = cm.store.get_config(
                tenant_id=tenant_id,
                scope=ConfigScope.SYSTEM,
                service="tenant_instructions",
                config_key="system_prompt",
            )
            if entry and entry.config_value:
                value = entry.config_value
                if isinstance(value, dict):
                    return value.get("text", "") or None
                if isinstance(value, str):
                    try:
                        return json.loads(value).get("text", "") or None
                    except (json.JSONDecodeError, AttributeError):
                        return value or None
        except Exception:
            pass
        return None

    def inject_context_into_prompt(self, prompt: str, query: str) -> str:
        """
        Inject tenant instructions, learned strategies, and memory context
        into a prompt.

        Instructions are always loaded (from ConfigStore, no memory needed).
        Strategies and memory context require memory to be initialized.
        """
        instructions = self._get_tenant_instructions()

        context = None
        strategies = None
        if self.is_memory_enabled():
            context = self.get_relevant_context(query)
            strategies = self.get_strategies(query)

        if not context and not strategies and not instructions:
            return prompt

        parts = [prompt]

        if instructions:
            parts.append(f"## Tenant Instructions\n{instructions}")

        if strategies:
            parts.append(strategies)

        if context:
            parts.append(f"## Relevant Context from Memory:\n{context}")

        parts.append(f"## Current Query:\n{query}")

        return "\n\n".join(parts)

    def remember_success(
        self,
        query: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> bool:
        """
        Remember a successful interaction

        Args:
            query: The query that was successful
            result: The result that was produced
            metadata: Optional metadata about the success
            infer: Passed through to ``update_memory``. Default True
                preserves LLM-based fact extraction.

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        # Format success memory - direct factual statement for Mem0's LLM
        success_content = (
            f"SUCCESS - Successfully answered: {query}. The answer was: {result}"
        )

        return self.update_memory(success_content, metadata, infer=infer)

    def remember_failure(
        self,
        query: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> bool:
        """
        Remember a failed interaction to avoid repeating mistakes

        Args:
            query: The query that failed
            error: Error message or description
            metadata: Optional metadata about the failure
            infer: Passed through to ``update_memory``. Default True
                preserves LLM-based fact extraction.

        Returns:
            Success status
        """
        if not self.is_memory_enabled():
            return False

        # Format failure memory - direct factual statement for Mem0's LLM
        failure_content = (
            f"FAILURE - Failed attempt: {query}. Error encountered: {error}"
        )

        return self.update_memory(failure_content, metadata, infer=infer)

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory status

        Returns:
            Summary dictionary
        """
        summary = {
            "enabled": self.is_memory_enabled(),
            "agent_name": self._memory_agent_name,
            "tenant_id": self._memory_tenant_id,
            "initialized": self._memory_initialized,
        }

        if self.is_memory_enabled():
            memory_state = self.get_memory_state()
            if memory_state:
                summary["total_memories"] = memory_state.get("total_memories", 0)

        return summary
