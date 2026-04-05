"""Agent dispatch logic — routes tasks to in-process agent implementations.

Extracted from routers/agents.py so both the REST endpoint and the A2A
executor can share the same dispatch logic without duplication.

Multi-turn support: The context dict may carry 'context_id' and
'conversation_history' (list of {role, content} dicts) which are
forwarded to routing/orchestration and search paths.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cogniverse_core.registries.agent_registry import AgentRegistry

if TYPE_CHECKING:
    from cogniverse_runtime.sandbox_manager import SandboxManager
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Routes agent tasks to the correct in-process agent implementation.

    Each dispatch call instantiates a fresh agent (stateless, per-request)
    and executes the task against it.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config_manager: ConfigManager,
        schema_loader: SchemaLoader,
        sandbox_manager: "SandboxManager | None" = None,
    ) -> None:
        self._registry = agent_registry
        self._config_manager = config_manager
        self._schema_loader = schema_loader
        self._sandbox_manager = sandbox_manager
        self._query_rewriter = None

    async def dispatch(
        self,
        agent_name: str,
        query: str,
        context: Dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Dispatch a task to the named agent and return the result.

        Raises:
            ValueError: If agent is not found or has no supported execution path.
        """
        if context is None:
            context = {}

        agent = self._registry.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        tenant_id = context.get("tenant_id", "default")
        capabilities = set(agent.capabilities)

        conversation_history = context.get("conversation_history", [])

        if "routing" in capabilities:
            result = await self._execute_routing_task(query, context, tenant_id)
        elif capabilities & {"search", "video_search", "retrieval"}:
            result = await self._execute_search_task(
                query, tenant_id, top_k, conversation_history=conversation_history
            )
        elif capabilities & {"image_search", "visual_analysis"}:
            result = await self._execute_image_search_task(query, tenant_id, top_k)
        elif capabilities & {"audio_analysis", "transcription"}:
            result = await self._execute_audio_search_task(query, tenant_id, top_k)
        elif capabilities & {"document_analysis", "pdf_processing"}:
            result = await self._execute_document_search_task(query, tenant_id, top_k)
        elif capabilities & {"detailed_report"}:
            result = await self._execute_detailed_report_task(query, tenant_id)
        elif capabilities & {"summarization", "text_generation"}:
            result = await self._execute_summarization_task(query, tenant_id)
        elif capabilities & {"text_analysis", "sentiment", "classification"}:
            result = await self._execute_text_analysis_task(query, context, tenant_id)
        elif "deep_research" in capabilities:
            result = await self._execute_deep_research_task(query, tenant_id)
        elif "coding" in capabilities:
            result = await self._execute_coding_task(query, tenant_id, context)
        else:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path. "
                f"Capabilities: {agent.capabilities}"
            )

        entities = result.get("entities", [])
        turn_count = len(conversation_history or []) // 2 + 1
        asyncio.create_task(
            self._maybe_auto_file_wiki(query, result, entities, agent_name, tenant_id, turn_count)
        )
        return result

    async def _maybe_auto_file_wiki(
        self,
        query: str,
        response: Dict[str, Any],
        entities: List[str],
        agent_name: str,
        tenant_id: str,
        turn_count: int,
    ) -> None:
        """Fire-and-forget wiki auto-filing. Non-fatal: logs and returns on any error."""
        try:
            from cogniverse_runtime.routers.wiki import _wiki_manager

            if _wiki_manager is None:
                return

            if not _wiki_manager._should_auto_file(entities, agent_name, turn_count):
                return

            response_text = str(response.get("answer", response))
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _wiki_manager.save_session(
                    query=query,
                    response=response_text,
                    entities=entities,
                    agent_name=agent_name,
                ),
            )
        except Exception:
            logger.warning("Wiki auto-filing failed (non-fatal)", exc_info=True)

    def create_streaming_agent(
        self, agent_name: str, query: str, tenant_id: str
    ) -> tuple:
        """Create a streaming-capable agent and its typed input.

        Returns (agent, typed_input) for use with agent.process(typed_input, stream=True).
        All agents support streaming via emit_progress() and call_dspy().
        """
        agent_entry = self._registry.get_agent(agent_name)
        if not agent_entry:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        capabilities = set(agent_entry.capabilities)

        if capabilities & {"summarization", "text_generation"}:
            from cogniverse_agents.summarizer_agent import (
                SummarizerAgent,
                SummarizerDeps,
                SummarizerInput,
            )

            deps = SummarizerDeps(tenant_id=tenant_id)
            agent = SummarizerAgent(deps=deps, config_manager=self._config_manager)
            typed_input = SummarizerInput(
                query=query, search_results=[], summary_type="general"
            )
            return agent, typed_input

        if capabilities & {"routing"}:
            from cogniverse_agents.routing_agent import (
                RoutingAgent,
                RoutingDeps,
                RoutingInput,
            )
            from cogniverse_foundation.config.utils import get_config
            from cogniverse_foundation.telemetry.config import TelemetryConfig

            config = get_config(
                tenant_id=tenant_id, config_manager=self._config_manager
            )
            llm_config = config.get_llm_config().resolve("routing_agent")
            deps = RoutingDeps(
                telemetry_config=TelemetryConfig(enabled=False),
                llm_config=llm_config,
            )
            agent = RoutingAgent(deps=deps, registry=self._registry)
            typed_input = RoutingInput(query=query, tenant_id=tenant_id)
            return agent, typed_input

        if capabilities & {"detailed_report"}:
            from cogniverse_agents.detailed_report_agent import (
                DetailedReportAgent,
                DetailedReportDeps,
                DetailedReportInput,
            )

            deps = DetailedReportDeps(tenant_id=tenant_id)
            agent = DetailedReportAgent(deps=deps, config_manager=self._config_manager)
            typed_input = DetailedReportInput(query=query, search_results=[])
            return agent, typed_input

        if capabilities & {"search", "video_search", "retrieval"}:
            from cogniverse_agents.search_agent import (
                SearchAgent,
                SearchAgentDeps,
                SearchInput,
            )

            system_config = self._config_manager.get_system_config()
            deps = SearchAgentDeps(
                tenant_id=tenant_id,
                backend_url=system_config.backend_url,
                backend_port=system_config.backend_port,
            )
            agent = SearchAgent(
                deps=deps,
                schema_loader=self._schema_loader,
                config_manager=self._config_manager,
            )
            typed_input = SearchInput(query=query, tenant_id=tenant_id)
            return agent, typed_input

        raise ValueError(
            f"Agent '{agent_name}' streaming not configured. "
            f"Capabilities: {agent_entry.capabilities}"
        )

    async def _execute_search_task(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        from cogniverse_agents.search.service import SearchService
        from cogniverse_foundation.config.utils import get_config

        # Rewrite query using conversation history to resolve anaphoric references.
        # If the DSPy rewriter fails (e.g., no LM configured), propagate with
        # a clear error rather than silently searching with the unresolved query.
        resolved_query = query
        if conversation_history:
            resolved_query = await self._rewrite_query_with_history(
                query, conversation_history
            )
            if resolved_query != query:
                logger.info(f"Query rewritten: '{query}' -> '{resolved_query}'")

        config = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
        search_service = SearchService(
            config=config,
            config_manager=self._config_manager,
            schema_loader=self._schema_loader,
        )

        profile = config.get("default_profile", "video_colpali_smol500_mv_frame")

        results = search_service.search(
            query=resolved_query,
            profile=profile,
            tenant_id=tenant_id,
            top_k=top_k,
            ranking_strategy="float_float",
        )

        result_list = [r.to_dict() for r in results]
        result_count = len(result_list)

        if result_count > 0:
            message = f"Found {result_count} results for '{resolved_query}'"
        else:
            message = f"No results found for '{resolved_query}'"

        response: Dict[str, Any] = {
            "status": "success",
            "agent": "search_agent",
            "message": message,
            "results_count": result_count,
            "results": result_list,
            "profile": profile,
        }

        if resolved_query != query:
            response["original_query"] = query
            response["rewritten_query"] = resolved_query

        return response

    async def _rewrite_query_with_history(
        self, query: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Rewrite a query that may contain anaphoric references using conversation history.

        Uses a DSPy ChainOfThought module to resolve references like 'that',
        'those', 'more' etc. Raises on failure — callers must handle explicitly.
        """
        from cogniverse_agents.search_agent import (
            ConversationalQueryRewriteModule,
        )

        if self._query_rewriter is None:
            self._query_rewriter = ConversationalQueryRewriteModule()

        history_lines = []
        for turn in conversation_history[-5:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")[:200]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        result = await self._query_rewriter.acall(
            query=query, conversation_history=history_text
        )
        rewritten = result.rewritten_query.strip()
        return rewritten if rewritten else query

    async def _execute_routing_task(
        self, query: str, context: Dict[str, Any], tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.config.utils import get_config

        config = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
        llm_config = config.get_llm_config()
        llm_endpoint = llm_config.resolve("routing_agent")

        routing_config = config.get("routing_agent", {})
        memory_enabled = routing_config.get("enable_memory", False)

        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        global_tm = get_telemetry_manager()
        telemetry_config = global_tm.config

        deps_kwargs: Dict[str, Any] = {
            "telemetry_config": telemetry_config,
            "llm_config": llm_endpoint,
            "enable_memory": memory_enabled,
            "enable_advanced_optimization": routing_config.get(
                "enable_advanced_optimization", False
            ),
        }

        if memory_enabled:
            _sys_cfg = self._config_manager.get_system_config()
            backend_url = _sys_cfg.backend_url
            backend_port = _sys_cfg.backend_port
            deps_kwargs.update(
                {
                    "memory_backend_host": backend_url,
                    "memory_backend_port": backend_port,
                    "memory_llm_model": llm_endpoint.model,
                    "memory_embedding_model": routing_config.get(
                        "memory_embedding_model", "nomic-embed-text"
                    ),
                    "memory_llm_base_url": llm_endpoint.api_base,
                    "memory_config_manager": self._config_manager,
                    "memory_schema_loader": self._schema_loader,
                }
            )

        deps = RoutingDeps(**deps_kwargs)
        # RoutingAgent.__init__ loads GLiNER + spaCy models (sync, CPU-bound).
        # Run in thread to avoid blocking the event loop and health probes.
        import asyncio
        agent = await asyncio.to_thread(
            RoutingAgent, deps=deps, registry=self._registry
        )

        action = context.get("action")
        if action == "optimize_routing":
            examples = context.get("examples", [])
            if examples:
                return await self._handle_routing_optimization(
                    agent, context, tenant_id
                )
            return await self._run_optimization_cycle(agent, tenant_id)
        elif action == "get_optimization_status":
            return self._handle_optimization_status(agent, tenant_id)

        result = await agent.route_query(
            query=query,
            context=context.get("context"),
            tenant_id=tenant_id,
        )

        needs_orchestration = result.metadata.get("needs_orchestration", False)

        if needs_orchestration:
            from cogniverse_agents.multi_agent_orchestrator import (
                MultiAgentOrchestrator,
            )
            from cogniverse_foundation.telemetry.manager import get_telemetry_manager

            runtime_base_url = "http://localhost:8000"
            available_agents = {}
            for name in self._registry.list_agents():
                agent_ep = self._registry.get_agent(name)
                if agent_ep and name != "routing_agent":
                    available_agents[name] = {
                        "capabilities": agent_ep.capabilities,
                        "endpoint": runtime_base_url,
                        "timeout_seconds": agent_ep.timeout,
                    }

            orchestrator = MultiAgentOrchestrator(
                tenant_id=tenant_id,
                telemetry_manager=get_telemetry_manager(),
                routing_agent=agent,
                available_agents=available_agents,
            )

            orch_result = await orchestrator.process_complex_query(
                query=result.enhanced_query or query,
                context=context.get("context"),
                conversation_history=context.get("conversation_history"),
            )

            return {
                "status": "success",
                "agent": "routing_agent",
                "message": f"Orchestrated '{query}' via multi-agent workflow",
                "recommended_agent": result.recommended_agent,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "enhanced_query": result.enhanced_query,
                "needs_orchestration": True,
                "orchestration_result": orch_result,
                "metadata": result.metadata,
            }

        recommended = result.recommended_agent
        effective_query = result.enhanced_query or query
        conversation_history = context.get("conversation_history", [])

        downstream_result = await self._execute_downstream_agent(
            agent_name=recommended,
            query=effective_query,
            tenant_id=tenant_id,
            top_k=context.get("top_k", 10),
            conversation_history=conversation_history,
        )

        return {
            "status": "success",
            "agent": "routing_agent",
            "message": f"Routed '{query}' to {recommended}",
            "recommended_agent": recommended,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "enhanced_query": result.enhanced_query,
            "entities": result.entities,
            "relationships": result.relationships,
            "query_variants": result.query_variants,
            "metadata": result.metadata,
            "downstream_result": downstream_result,
        }

    async def _handle_routing_optimization(
        self, agent, context: Dict[str, Any], tenant_id: str
    ) -> Dict[str, Any]:
        """Trigger routing optimization with provided examples."""
        examples = context.get("examples", [])
        if not examples:
            return {
                "status": "insufficient_data",
                "message": "No routing examples provided",
                "training_examples": 0,
            }

        optimizer = agent._get_optimizer(tenant_id)
        if not optimizer:
            return {
                "status": "error",
                "message": "Advanced optimization not enabled for this tenant",
            }

        try:
            for example in examples:
                await optimizer.record_routing_experience(
                    query=example.get("query", ""),
                    entities=example.get("entities", []),
                    relationships=example.get("relationships", []),
                    enhanced_query=example.get("enhanced_query", ""),
                    chosen_agent=example.get("chosen_agent", "search_agent"),
                    routing_confidence=example.get("confidence", 0.5),
                    search_quality=example.get("search_quality", 0.5),
                    agent_success=example.get("agent_success", True),
                    processing_time=example.get("processing_time", 0.0),
                )

            # Explicitly trigger DSPy optimization compile after recording all examples
            await optimizer._run_optimization_step()
            await optimizer._persist_data()

            return {
                "status": "optimization_triggered",
                "message": "Routing experiences recorded for optimization",
                "training_examples": len(examples),
                "optimizer": "AdvancedRoutingOptimizer",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _handle_optimization_status(self, agent, tenant_id: str) -> Dict[str, Any]:
        """Get optimization status from routing agent."""
        optimizer = agent._get_optimizer(tenant_id)
        if not optimizer:
            return {
                "status": "inactive",
                "message": "Advanced optimization not enabled",
                "optimizer_ready": False,
                "metrics": {},
            }

        stats = agent.get_routing_statistics()
        return {
            "status": "active",
            "optimizer_ready": True,
            "metrics": stats,
        }

    async def _run_optimization_cycle(self, agent, tenant_id: str) -> Dict[str, Any]:
        """Run full optimization cycle from accumulated traces.

        Creates an OptimizationOrchestrator that reads routing spans from
        Phoenix, annotates low-quality decisions, feeds back to optimizer,
        and triggers DSPy compile when enough data exists.
        """
        from cogniverse_agents.routing.optimization_orchestrator import (
            OptimizationOrchestrator,
        )

        optimizer = agent._get_optimizer(tenant_id)
        if not optimizer:
            return {
                "status": "error",
                "message": "Advanced optimization not enabled for this tenant",
            }

        telemetry_provider = agent._get_telemetry_provider(tenant_id)

        try:
            from cogniverse_agents.routing.config import AutomationRulesConfig
            from cogniverse_foundation.config.utils import get_config

            rules_config = get_config(
                tenant_id=tenant_id, config_manager=self._config_manager
            )
            rules_data = rules_config.get("automation_rules", {})
            automation_rules = (
                AutomationRulesConfig.from_dict(rules_data) if rules_data else None
            )

            from cogniverse_runtime.routers.agents import get_annotation_queue

            orchestrator = OptimizationOrchestrator(
                llm_config=agent.deps.llm_config,
                telemetry_provider=telemetry_provider,
                tenant_id=tenant_id,
                automation_rules=automation_rules,
                annotation_queue=get_annotation_queue(),
            )

            results = await orchestrator.run_once()

            return {
                "status": "optimization_triggered",
                "message": "Full optimization cycle completed from traces",
                "cycle_results": results,
                "spans_evaluated": results.get("span_evaluation", {}).get(
                    "spans_processed", 0
                ),
                "annotations_generated": results.get("annotations_generated", 0),
                "optimizer": "OptimizationOrchestrator",
            }
        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _execute_downstream_agent(
        self,
        agent_name: str,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute the downstream agent that the router recommended.

        Re-uses the existing _execute_*_task methods based on the agent's
        capabilities, passing conversation_history through for query rewrite.
        """
        agent = self._registry.get_agent(agent_name)
        if not agent:
            raise ValueError(
                f"Routing recommended '{agent_name}' but it is not in the registry. "
                "Register the agent under its exact name before routing to it."
            )

        capabilities = set(agent.capabilities)

        if capabilities & {"search", "video_search", "retrieval"}:
            return await self._execute_search_task(
                query,
                tenant_id,
                top_k,
                conversation_history=conversation_history,
            )
        elif capabilities & {"image_search", "visual_analysis"}:
            return await self._execute_image_search_task(query, tenant_id, top_k)
        elif capabilities & {"audio_analysis", "transcription"}:
            return await self._execute_audio_search_task(query, tenant_id, top_k)
        elif capabilities & {"document_analysis", "pdf_processing"}:
            return await self._execute_document_search_task(query, tenant_id, top_k)
        elif capabilities & {"detailed_report"}:
            return await self._execute_detailed_report_task(query, tenant_id)
        elif capabilities & {"summarization", "text_generation"}:
            return await self._execute_summarization_task(query, tenant_id)
        elif capabilities & {"text_analysis", "sentiment", "classification"}:
            return await self._execute_text_analysis_task(
                query, {"tenant_id": tenant_id}, tenant_id
            )
        elif "coding" in capabilities:
            return await self._execute_coding_task(
                query, tenant_id, {"tenant_id": tenant_id}
            )
        else:
            raise ValueError(
                f"Routed agent '{agent_name}' has no supported execution path. "
                f"Capabilities: {agent.capabilities}"
            )

    def _get_vespa_endpoint(self, tenant_id: str) -> str:
        system_config = self._config_manager.get_system_config()
        return f"{system_config.backend_url}:{system_config.backend_port}"

    async def _execute_summarization_task(
        self, query: str, tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.summarizer_agent import (
            SummarizerAgent,
            SummarizerDeps,
            SummaryRequest,
        )

        deps = SummarizerDeps(tenant_id=tenant_id)
        agent = SummarizerAgent(deps=deps, config_manager=self._config_manager)

        request = SummaryRequest(
            query=query,
            search_results=[],
            summary_type="general",
        )
        result = await agent.summarize(request)

        return {
            "status": "success",
            "agent": "summarizer_agent",
            "message": f"Generated summary for '{query}'",
            "result": dataclasses.asdict(result),
        }

    async def _execute_text_analysis_task(
        self, query: str, context: Dict[str, Any], tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

        agent = TextAnalysisAgent(
            tenant_id=tenant_id, config_manager=self._config_manager
        )

        analysis_type = context.get("analysis_type", "summary")
        result = agent.analyze_text(text=query, analysis_type=analysis_type)

        return {
            "status": "success",
            "agent": "text_analysis_agent",
            "message": f"Completed {analysis_type} analysis for '{query}'",
            "result": result,
        }

    async def _execute_detailed_report_task(
        self, query: str, tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.detailed_report_agent import (
            DetailedReportAgent,
            DetailedReportDeps,
            ReportRequest,
        )

        deps = DetailedReportDeps(tenant_id=tenant_id)
        agent = DetailedReportAgent(deps=deps, config_manager=self._config_manager)

        request = ReportRequest(
            query=query,
            search_results=[],
            report_type="comprehensive",
        )
        result = await agent.generate_report(request)

        return {
            "status": "success",
            "agent": "detailed_report_agent",
            "message": f"Generated detailed report for '{query}'",
            "result": dataclasses.asdict(result),
        }

    async def _execute_image_search_task(
        self, query: str, tenant_id: str, top_k: int
    ) -> Dict[str, Any]:
        from cogniverse_agents.image_search_agent import (
            ImageSearchAgent,
            ImageSearchDeps,
        )

        vespa_endpoint = self._get_vespa_endpoint(tenant_id)
        deps = ImageSearchDeps(
            vespa_endpoint=vespa_endpoint,
            tenant_id=tenant_id,
        )
        agent = ImageSearchAgent(deps=deps)

        results = await agent.search_images(query=query, limit=top_k)

        result_list = [dataclasses.asdict(r) for r in results]
        return {
            "status": "success",
            "agent": "image_search_agent",
            "message": f"Found {len(result_list)} images for '{query}'",
            "results_count": len(result_list),
            "results": result_list,
        }

    async def _execute_audio_search_task(
        self, query: str, tenant_id: str, top_k: int
    ) -> Dict[str, Any]:
        from cogniverse_agents.audio_analysis_agent import (
            AudioAnalysisAgent,
            AudioAnalysisDeps,
        )

        vespa_endpoint = self._get_vespa_endpoint(tenant_id)
        deps = AudioAnalysisDeps(
            vespa_endpoint=vespa_endpoint,
            tenant_id=tenant_id,
        )
        agent = AudioAnalysisAgent(deps=deps)

        results = await agent.search_audio(query=query, limit=top_k)

        result_list = [dataclasses.asdict(r) for r in results]
        return {
            "status": "success",
            "agent": "audio_analysis_agent",
            "message": f"Found {len(result_list)} audio results for '{query}'",
            "results_count": len(result_list),
            "results": result_list,
        }

    async def _execute_document_search_task(
        self, query: str, tenant_id: str, top_k: int
    ) -> Dict[str, Any]:
        from cogniverse_agents.document_agent import (
            DocumentAgent,
            DocumentAgentDeps,
        )

        vespa_endpoint = self._get_vespa_endpoint(tenant_id)
        deps = DocumentAgentDeps(
            vespa_endpoint=vespa_endpoint,
            tenant_id=tenant_id,
        )
        agent = DocumentAgent(deps=deps)

        results = await agent.search_documents(query=query, limit=top_k)

        result_list = [dataclasses.asdict(r) for r in results]
        return {
            "status": "success",
            "agent": "document_agent",
            "message": f"Found {len(result_list)} documents for '{query}'",
            "results_count": len(result_list),
            "results": result_list,
        }

    async def _execute_deep_research_task(
        self, query: str, tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.deep_research_agent import (
            DeepResearchAgent,
            DeepResearchDeps,
            DeepResearchInput,
        )

        deps = DeepResearchDeps(tenant_id=tenant_id)

        async def search_fn(query: str, tenant_id: str):
            result = await self._execute_search_task(query, tenant_id, top_k=10)
            return result.get("results", [])

        agent = DeepResearchAgent(deps=deps, search_fn=search_fn)
        input_data = DeepResearchInput(query=query, tenant_id=tenant_id)
        result = await agent.process(input_data)

        return {
            "status": "success",
            "agent": "deep_research_agent",
            "message": f"Research complete for '{query}'",
            "result": result.model_dump(),
        }

    async def _execute_coding_task(
        self,
        query: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import dspy

        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )
        from cogniverse_foundation.config.llm_factory import create_dspy_lm
        from cogniverse_foundation.config.utils import get_config

        config = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
        coding_lm = create_dspy_lm(config.get_llm_config().resolve("coding_agent"))
        dspy.configure(lm=coding_lm)

        deps = CodingDeps(
            tenant_id=tenant_id,
            sandbox_manager=self._sandbox_manager,
        )

        async def search_fn(query: str, tenant_id: str):
            """Search code using the code_lateon_mv profile."""
            from cogniverse_agents.search.service import SearchService

            search_config = get_config(
                tenant_id=tenant_id, config_manager=self._config_manager
            )
            search_service = SearchService(
                config=search_config,
                config_manager=self._config_manager,
                schema_loader=self._schema_loader,
            )
            results = search_service.search(
                query=query,
                profile="code_lateon_mv",
                tenant_id=tenant_id,
                top_k=10,
            )
            return [r.to_dict() for r in results]

        agent = CodingAgent(
            deps=deps,
            search_fn=search_fn,
            sandbox_manager=self._sandbox_manager,
        )

        ctx = context or {}
        input_data = CodingInput(
            task=query,
            codebase_path=ctx.get("codebase_path", ""),
            tenant_id=tenant_id,
            max_iterations=ctx.get("max_iterations", 5),
            language=ctx.get("language", "python"),
        )
        result = await agent.process(input_data)

        return {
            "status": "success",
            "agent": "coding_agent",
            "message": f"Coding task complete for '{query}'",
            "result": result.model_dump(),
        }
