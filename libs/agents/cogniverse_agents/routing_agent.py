"""
Routing Agent — Thin DSPy Decision-Maker

Routes enriched queries to the appropriate execution agent.
Entity extraction, query enhancement, and profile selection are handled
by dedicated upstream A2A agents. This agent only makes the routing
DECISION: given pre-enriched data, which agent should handle it?
"""

import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import ConfigDict, Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.routing.dspy_relationship_router import (
    DSPyAdvancedRoutingModule,
)
from cogniverse_agents.routing.dspy_routing_signatures import (
    BasicQueryAnalysisSignature,
)
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

# =============================================================================
# Type-Safe Input/Output/Dependencies Models
# =============================================================================


class RoutingInput(AgentInput):
    """Type-safe input for routing agent.

    Accepts pre-enriched data from upstream A2A agents
    (EntityExtractionAgent, QueryEnhancementAgent).
    """

    query: str = Field(..., description="Original user query")
    enhanced_query: Optional[str] = Field(
        None, description="Enhanced query from QueryEnhancementAgent"
    )
    entities: Optional[List[Dict[str, Any]]] = Field(
        None, description="Entities from EntityExtractionAgent"
    )
    relationships: Optional[List[Dict[str, Any]]] = Field(
        None, description="Relationships from EntityExtractionAgent"
    )
    tenant_id: str = Field(..., description="Tenant identifier (required)")
    context: Optional[str] = Field(None, description="Optional context information")


class RoutingOutput(AgentOutput):
    """Type-safe output from routing agent.

    Slim output focused on the routing decision only.
    Entities, relationships, and query enhancement are handled by upstream agents.

    Deprecated fields (enhanced_query, entities, relationships, query_variants)
    are kept as empty defaults for backward compatibility with downstream consumers
    (search_agent, summarizer_agent, etc.) that still read them. These will be
    removed once downstream agents are updated to receive enrichment from
    upstream A2A agents directly.
    """

    query: str = Field(..., description="Original query")
    recommended_agent: str = Field(..., description="Selected execution agent")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Routing confidence")
    reasoning: str = Field("", description="Reasoning for the decision")
    fallback_agents: List[str] = Field(
        default_factory=list, description="Fallback agents if primary fails"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Deprecated — kept for backward compat with downstream consumers
    enhanced_query: str = Field("", description="Deprecated: use upstream QueryEnhancementAgent")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Deprecated: use upstream EntityExtractionAgent"
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list, description="Deprecated: use upstream EntityExtractionAgent"
    )
    query_variants: List[Dict[str, str]] = Field(
        default_factory=list, description="Deprecated: use upstream QueryEnhancementAgent"
    )

    model_config = ConfigDict(extra="allow")

    @property
    def extracted_entities(self) -> List[Dict[str, Any]]:
        """Deprecated: backward compat for downstream consumers."""
        return self.entities

    @property
    def extracted_relationships(self) -> List[Dict[str, Any]]:
        """Deprecated: backward compat for downstream consumers."""
        return self.relationships

    @property
    def routing_metadata(self) -> Dict[str, Any]:
        """Deprecated: backward compat for downstream consumers."""
        return self.metadata


class RoutingDeps(AgentDeps):
    """Type-safe dependencies for routing agent.

    Slim deps focused on routing decision configuration only.
    Preprocessing deps (GLiNER, SpaCy, GRPO, etc.) removed — those live
    in dedicated upstream agents.
    """

    telemetry_config: Any = Field(None, description="Telemetry configuration")
    llm_config: Optional[LLMEndpointConfig] = Field(
        default=None,
        description="LLM endpoint configuration. If None, resolved from config.json at init.",
    )
    confidence_threshold: float = Field(0.5, description="Min confidence threshold")
    enable_memory: bool = Field(False, description="Enable memory (requires Mem0)")
    memory_backend_host: Optional[str] = Field(
        None, description="Backend host for memory storage"
    )
    memory_backend_port: Optional[int] = Field(
        None, description="Backend port for memory storage"
    )
    memory_llm_model: Optional[str] = Field(
        None, description="LLM model for memory extraction"
    )
    memory_embedding_model: Optional[str] = Field(
        None, description="Embedding model for memory search"
    )
    memory_llm_base_url: Optional[str] = Field(
        None, description="LLM API base URL for memory"
    )
    memory_config_manager: Optional[Any] = Field(
        None, description="ConfigManager for memory"
    )
    memory_schema_loader: Optional[Any] = Field(
        None, description="SchemaLoader for memory"
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class RoutingAgent(
    A2AAgent[RoutingInput, RoutingOutput, RoutingDeps],
    MemoryAwareMixin,
    TenantAwareAgentMixin,
):
    """Thin routing agent — DSPy-powered decision-maker.

    Receives pre-enriched queries (with entities, relationships, enhanced_query)
    from upstream A2A agents and decides which execution agent should handle
    the query. All preprocessing (entity extraction, query enhancement,
    profile selection, GRPO optimization) lives in dedicated agents.
    """

    def __init__(
        self,
        deps: RoutingDeps,
        registry=None,
        port: int = 8001,
    ):
        self.telemetry_config = deps.telemetry_config
        self.registry = registry
        self.logger = logging.getLogger(__name__)

        self._initialize_telemetry_manager()
        self._configure_dspy(deps)
        self._initialize_routing_module()

        # Memory system (lazy per-tenant init)
        if deps.enable_memory:
            for field in (
                "memory_backend_host",
                "memory_backend_port",
                "memory_llm_model",
                "memory_embedding_model",
                "memory_llm_base_url",
                "memory_config_manager",
                "memory_schema_loader",
            ):
                if getattr(deps, field) is None:
                    raise ValueError(
                        f"enable_memory=True but {field} is None — "
                        "all memory_* fields are required when memory is enabled"
                    )
            self._memory_config = {
                "backend_host": deps.memory_backend_host,
                "backend_port": deps.memory_backend_port,
                "llm_model": deps.memory_llm_model,
                "embedding_model": deps.memory_embedding_model,
                "llm_base_url": deps.memory_llm_base_url,
                "config_manager": deps.memory_config_manager,
                "schema_loader": deps.memory_schema_loader,
            }
            self._memory_initialized_tenants: set = set()

        a2a_config = A2AAgentConfig(
            agent_name="routing_agent",
            agent_description="DSPy-powered routing decision-maker",
            capabilities=["intelligent_routing", "query_analysis", "agent_orchestration"],
            port=port,
        )

        super().__init__(
            deps=deps,
            config=a2a_config,
            dspy_module=self.routing_module,
        )

    def _load_artifact(self) -> None:
        """Load optimized DSPy routing module from artifact store.

        Called by the dispatcher after telemetry_manager and _artifact_tenant_id
        are injected — not from __init__ (telemetry_manager is not yet available).
        """
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            import asyncio
            import json
            from concurrent.futures import ThreadPoolExecutor

            from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

            tenant_id = getattr(self, "_artifact_tenant_id", "default")
            provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
            am = ArtifactManager(provider, tenant_id)

            async def _load():
                return await am.load_blob("model", "routing_decision")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, _load())
                    blob = future.result()
            else:
                blob = asyncio.run(_load())

            if blob:
                state = json.loads(blob)
                self.routing_module.load_state(state)
                self.logger.info("RoutingAgent loaded optimized DSPy module from artifact")
        except Exception as e:
            self.logger.debug("No routing artifact to load (using defaults): %s", e)

    def _initialize_telemetry_manager(self) -> None:
        """Initialize telemetry manager for span emission."""
        if self.telemetry_config and getattr(self.telemetry_config, "enabled", False):
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            self.telemetry_manager = TelemetryManager(config=self.telemetry_config)
            self.logger.info("Telemetry manager initialized")
        else:
            self.telemetry_manager = None

    def _configure_dspy(self, deps: RoutingDeps) -> None:
        """Configure DSPy LM instance (scoped via dspy.context, not global).

        If deps.llm_config is None, resolves from config.json via ConfigManager.
        Disables qwen3 thinking mode which breaks DSPy field parsing.
        """
        llm_config = deps.llm_config

        if llm_config is None:
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
                get_config,
            )

            cm = create_default_config_manager()
            config = get_config(tenant_id="default", config_manager=cm)
            llm_config = config.get_llm_config().resolve("routing_agent")
            self.logger.info(
                f"Resolved LLM config from config.json: {llm_config.model}"
            )

        if "qwen3" in llm_config.model or "qwen-3" in llm_config.model:
            if not llm_config.extra_body or "think" not in llm_config.extra_body:
                import dataclasses

                llm_config = dataclasses.replace(
                    llm_config, extra_body={"think": False}
                )
                self.logger.info("Disabled qwen3 thinking mode for DSPy compatibility")

        self._dspy_lm = create_dspy_lm(llm_config)
        self.logger.info(
            f"Created DSPy LM: {llm_config.model} at {llm_config.api_base}"
        )

    def _initialize_routing_module(self) -> None:
        """Initialize DSPy routing module."""
        try:
            self.routing_module = DSPyAdvancedRoutingModule(analysis_module=None)
            self.logger.info("DSPy advanced routing module initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize routing module: {e}")
            self._create_fallback_routing_module()

    def _create_fallback_routing_module(self) -> None:
        """Create fallback routing module for graceful degradation."""

        class FallbackRoutingModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.analyze = dspy.ChainOfThought(BasicQueryAnalysisSignature)

            def forward(self, query: str, context: str | None = None, **kwargs):
                try:
                    return self.analyze(query=query)
                except Exception:
                    return dspy.Prediction(
                        primary_intent="search",
                        needs_video_search=True,
                        recommended_agent="search_agent",
                        confidence=0.5,
                    )

        self.routing_module = FallbackRoutingModule()
        self.logger.warning("Using fallback routing module")

    def _get_telemetry_provider(self, tenant_id: str):
        """Get telemetry provider for a tenant."""
        if self.telemetry_manager is not None:
            return self.telemetry_manager.get_provider(tenant_id=tenant_id)
        raise RuntimeError(
            "Telemetry manager not initialized — cannot create artifact storage"
        )

    def _ensure_memory_for_tenant(self, tenant_id: str) -> None:
        """Initialize memory for a tenant if not already done."""
        if not self.deps.enable_memory:
            return
        if tenant_id in self._memory_initialized_tenants:
            return
        try:
            self.initialize_memory(
                agent_name="routing_agent",
                tenant_id=tenant_id,
                backend_host=self._memory_config["backend_host"],
                backend_port=self._memory_config["backend_port"],
                llm_model=self._memory_config["llm_model"],
                embedding_model=self._memory_config["embedding_model"],
                llm_base_url=self._memory_config["llm_base_url"],
                config_manager=self._memory_config["config_manager"],
                schema_loader=self._memory_config["schema_loader"],
            )
            self._memory_initialized_tenants.add(tenant_id)
            self.logger.info(f"Memory initialized for tenant: {tenant_id}")
        except Exception as e:
            self.logger.error(
                f"Failed to initialize memory for tenant {tenant_id}: {e}"
            )

    async def route_query(
        self,
        query: str,
        enhanced_query: str | None = None,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        context: str | None = None,
        tenant_id: str | None = None,
    ) -> RoutingOutput:
        """Route a query to the appropriate execution agent.

        Args:
            query: Original user query
            enhanced_query: Pre-enriched query from QueryEnhancementAgent
            entities: Entities from EntityExtractionAgent
            relationships: Relationships from EntityExtractionAgent
            context: Optional additional context
            tenant_id: Tenant identifier (required)

        Returns:
            RoutingOutput with recommended agent and confidence
        """
        if not tenant_id:
            raise ValueError(
                "tenant_id is required for routing operations. "
                "Tenant isolation is mandatory — cannot default to 'unknown'."
            )

        self.set_tenant_for_context(tenant_id)

        with_span = self.telemetry_manager is not None

        try:
            routing_info = await self._make_routing_decision(
                query=query,
                enhanced_query=enhanced_query,
                entities=entities,
                relationships=relationships,
                context=context,
                tenant_id=tenant_id,
            )

            decision = RoutingOutput(
                query=query,
                recommended_agent=routing_info["recommended_agent"],
                confidence=routing_info["confidence"],
                reasoning=routing_info["reasoning"],
                fallback_agents=routing_info.get("fallback_agents", []),
                metadata={"tenant_id": tenant_id},
            )

            if with_span:
                try:
                    with self.telemetry_manager.span(
                        "cogniverse.routing",
                        tenant_id=tenant_id,
                        attributes={
                            "routing.query": query[:200],
                            "routing.recommended_agent": decision.recommended_agent,
                            "routing.primary_intent": routing_info.get("primary_intent", ""),
                            "routing.confidence": decision.confidence,
                            "routing.reasoning": decision.reasoning[:200],
                        },
                    ):
                        pass
                except Exception as e:
                    self.logger.debug("Failed to emit routing span: %s", e)

            self.logger.info(
                f"Query routed to {decision.recommended_agent} "
                f"(confidence: {decision.confidence:.3f})"
            )

            return decision

        except Exception as e:
            self.logger.error(f"Routing failed for query '{query}': {e}")
            return RoutingOutput(
                query=query,
                recommended_agent="search_agent",
                confidence=0.2,
                reasoning=f"Fallback routing due to error: {e}",
                fallback_agents=["summarizer_agent", "detailed_report_agent"],
                metadata={"error": str(e), "fallback": True},
            )

    async def _make_routing_decision(
        self,
        query: str,
        enhanced_query: str | None,
        entities: list[dict] | None,
        relationships: list[dict] | None,
        context: str | None,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Make DSPy-powered routing decision using pre-enriched data."""
        try:
            routing_context = ""
            if context:
                routing_context += f"Context: {context}. "
            if entities:
                entity_str = ", ".join(e.get("text", "") for e in entities[:5])
                routing_context += f"Entities: {entity_str}. "
            if relationships:
                rel_str = "; ".join(
                    f"{r.get('subject', '')} {r.get('relation', '')} {r.get('object', '')}"
                    for r in relationships[:3]
                )
                routing_context += f"Relationships: {rel_str}. "

            # Inject full context stack (instructions + strategies + memory)
            if self.is_memory_enabled():
                self._ensure_memory_for_tenant(tenant_id)
            routing_context = self.inject_context_into_prompt(
                prompt=routing_context, query=query
            )

            effective_query = enhanced_query or query
            available_agents = self.registry.list_agents() if self.registry else None

            with dspy.context(lm=self._dspy_lm):
                dspy_result = await self.call_dspy(
                    self.routing_module,
                    output_field="recommended_agent",
                    query=effective_query,
                    context=routing_context,
                    available_agents=available_agents,
                )

            # Extract routing info from DSPy result
            routing_decision = getattr(dspy_result, "routing_decision", {})
            recommended_agent = (
                routing_decision.get("primary_agent")
                if isinstance(routing_decision, dict)
                else getattr(dspy_result, "recommended_agent", "search_agent")
            )

            raw_confidence = getattr(
                dspy_result,
                "overall_confidence",
                getattr(dspy_result, "confidence", 0.5),
            )

            return {
                "recommended_agent": recommended_agent,
                "confidence": self._parse_confidence(raw_confidence),
                "reasoning": self._extract_reasoning(dspy_result),
                "primary_intent": getattr(dspy_result, "primary_intent", "search"),
            }

        except Exception as e:
            self.logger.error(f"DSPy routing decision failed: {e}")
            return {
                "recommended_agent": "search_agent",
                "confidence": 0.3,
                "reasoning": f"Fallback routing due to error: {e}",
                "primary_intent": "search",
            }

    @staticmethod
    def _parse_confidence(value: Any) -> float:
        """Parse confidence value from DSPy result to float in [0, 1]."""
        try:
            if isinstance(value, (int, float)):
                return min(max(float(value), 0.0), 1.0)
            if isinstance(value, str):
                cleaned = value.strip().rstrip("%")
                parsed = float(cleaned)
                if parsed > 1.0:
                    parsed /= 100.0
                return min(max(parsed, 0.0), 1.0)
        except (ValueError, TypeError):
            pass
        return 0.5

    @staticmethod
    def _extract_reasoning(dspy_result) -> str:
        """Extract reasoning string from DSPy prediction.

        DSPyAdvancedRoutingModule returns reasoning_chain (list of strings).
        DSPyBasicRoutingModule returns reasoning (str).
        """
        reasoning = getattr(dspy_result, "reasoning_chain", None)
        if reasoning is None:
            reasoning = getattr(dspy_result, "reasoning", "")
        if isinstance(reasoning, list):
            return " ".join(str(r) for r in reasoning)
        return str(reasoning)

    # =========================================================================
    # A2A protocol: _process_impl and _get_agent_skills
    # =========================================================================

    async def _process_impl(self, input_data: RoutingInput) -> RoutingOutput:
        """Core processing logic for routing.

        The A2A base class handles conversion from A2A protocol format
        to RoutingInput and from RoutingOutput back to A2A format.
        """
        return await self.route_query(
            query=input_data.query,
            enhanced_query=input_data.enhanced_query,
            entities=input_data.entities,
            relationships=input_data.relationships,
            context=input_data.context,
            tenant_id=input_data.tenant_id,
        )

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return A2A skill descriptors."""
        return [
            {
                "id": "route_query",
                "name": "Route Query",
                "description": (
                    "Route an enriched query to the appropriate execution agent. "
                    "Accepts pre-extracted entities, relationships, and enhanced query."
                ),
                "tags": ["routing", "decision", "dspy"],
                "examples": [
                    "Route 'find videos of robots' to the best agent",
                    "Determine which agent should handle a summarization request",
                ],
            }
        ]


# =============================================================================
# Factory functions
# =============================================================================


def create_routing_agent(
    telemetry_config: Any,
    port: int = 8001,
    **kwargs: Any,
) -> RoutingAgent:
    """Factory function to create RoutingAgent with typed dependencies."""
    deps = RoutingDeps(
        telemetry_config=telemetry_config,
        **kwargs,
    )
    return RoutingAgent(deps=deps, port=port)


# =============================================================================
# Standalone entry point
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from dataclasses import dataclass

    @dataclass
    class MockTelemetryConfig:
        enabled: bool = False

    async def main():
        deps = RoutingDeps(
            telemetry_config=MockTelemetryConfig(enabled=False),
            llm_config=LLMEndpointConfig(
                model="ollama/qwen3:4b",
                api_base="http://localhost:11434",
                temperature=0.1,
                max_tokens=1000,
            ),
            confidence_threshold=0.5,
        )

        agent = RoutingAgent(deps=deps, port=8001)

        test_queries = [
            "Show me videos of robots playing soccer in tournaments",
            "Summarize the key findings from the latest AI research papers",
            "Generate a detailed report on renewable energy trends",
        ]

        for query in test_queries:
            print(f"\nProcessing: {query}")
            decision = await agent.route_query(
                query=query, tenant_id="demo-tenant"
            )
            print(
                f"Route to: {decision.recommended_agent} "
                f"(confidence: {decision.confidence:.3f})"
            )
            print(f"Reasoning: {decision.reasoning}")

    asyncio.run(main())
