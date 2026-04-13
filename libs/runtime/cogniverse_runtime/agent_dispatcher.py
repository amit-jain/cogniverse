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

    def _init_agent_memory(
        self, agent: Any, agent_name: str, tenant_id: str
    ) -> None:
        """Auto-initialize MemoryAwareMixin for any agent that supports it.

        Audit fix #14 — the dispatcher previously constructed agents without
        ever calling ``initialize_memory()``, so even agents that inherited
        MemoryAwareMixin had ``is_memory_enabled() == False`` and silently
        skipped strategy/memory injection.

        This helper checks at runtime whether the constructed agent inherits
        the mixin and, if so, runs ``initialize_memory`` with the dispatcher's
        own config_manager and schema_loader. Silently no-ops for agents that
        don't inherit the mixin (e.g. ImageSearchAgent). Errors during init
        are logged but never raised — memory is best-effort enrichment, not a
        hard dependency.
        """
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        if not isinstance(agent, MemoryAwareMixin):
            return

        # Always set tenant for instruction lookup, even if full memory init fails.
        try:
            agent.set_tenant_for_context(tenant_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "set_tenant_for_context failed for %s: %s", agent_name, exc
            )

        sys_cfg = self._config_manager.get_system_config()
        try:
            agent.initialize_memory(
                agent_name=agent_name,
                tenant_id=tenant_id,
                backend_host=sys_cfg.backend_url,
                backend_port=sys_cfg.backend_port,
                llm_model=getattr(sys_cfg, "llm_model", "qwen3:4b"),
                embedding_model=getattr(
                    sys_cfg, "embedding_model", "nomic-embed-text"
                ),
                llm_base_url=getattr(
                    sys_cfg, "base_url", "http://localhost:11434"
                ),
                config_manager=self._config_manager,
                schema_loader=self._schema_loader,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize memory for %s (tenant=%s): %s",
                agent_name,
                tenant_id,
                exc,
            )

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

        # Action-based dispatch: optimization actions bypass normal routing
        action = context.get("action")
        if action in ("optimize_routing", "get_optimization_status", "optimization_cycle_from_traces"):
            return await self._execute_optimization_action(
                action, query, context, tenant_id
            )

        if capabilities & {"gateway", "routing"}:
            result = await self._execute_gateway_task(query, context, tenant_id)
        elif "orchestration" in capabilities:
            result = await self._execute_orchestration_task(
                query, context, tenant_id
            )
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
            # Generic A2A dispatch — any registered agent can be called.
            # Import the agent class from its registered path, instantiate,
            # and call _process_impl. This is how preprocessing agents
            # (entity_extraction, query_enhancement, profile_selection)
            # are executed when the OrchestratorAgent calls them via HTTP.
            result = await self._execute_generic_agent(
                agent_name, query, context, tenant_id
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
        """Fire-and-forget wiki auto-filing. Non-fatal: logs and returns on any error.

        Audit fix #12 — resolves the per-tenant WikiManager via the factory
        rather than the deleted ``_wiki_manager`` singleton, so each tenant's
        auto-filed pages land in their own wiki.
        """
        try:
            from cogniverse_runtime.routers import wiki as wiki_router

            if wiki_router._wiki_manager_factory is None:
                return

            wm = wiki_router._wiki_manager_factory(tenant_id)
            if wm is None:
                return

            if not wm._should_auto_file(entities, agent_name, turn_count):
                return

            response_text = str(response.get("answer", response))
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: wm.save_session(
                    query=query,
                    response=response_text,
                    entities=entities,
                    agent_name=agent_name,
                ),
            )
        except Exception:
            logger.warning("Wiki auto-filing failed (non-fatal)", exc_info=True)

    async def _execute_generic_agent(
        self,
        agent_name: str,
        query: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Generic A2A agent execution via dynamic class import.

        Handles any registered agent by importing its class from
        ConfigLoader.AGENT_CLASSES, instantiating with default deps,
        and calling process(). This is the path taken by preprocessing
        agents (entity_extraction, query_enhancement, profile_selection)
        when the OrchestratorAgent calls them via A2A HTTP.
        """
        import importlib

        from cogniverse_runtime.config_loader import ConfigLoader

        class_path = ConfigLoader.AGENT_CLASSES.get(agent_name)
        if not class_path:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path "
                f"(not in AGENT_CLASSES)"
            )

        module_path, class_name = class_path.split(":")
        module = importlib.import_module(module_path)
        agent_cls = getattr(module, class_name)

        # Find the Deps class — convention: same module, class name ends with "Deps"
        deps_cls = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and attr_name.endswith("Deps")
                and attr_name != "AgentDeps"
            ):
                deps_cls = attr
                break

        if deps_cls is None:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path "
                f"(no Deps class in {module_path} — agent uses legacy mixin pattern)"
            )

        # Instantiate with default deps
        deps = deps_cls()
        agent = agent_cls(deps=deps)
        self._init_agent_memory(agent, agent_name, tenant_id)

        # Inject global TelemetryManager for span emission
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        agent.telemetry_manager = get_telemetry_manager()
        agent._artifact_tenant_id = tenant_id
        if hasattr(agent, "_load_artifact"):
            agent._load_artifact()

        # Find the Input class — convention: same module, class name ends with "Input"
        input_cls = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and attr_name.endswith("Input")
                and attr_name != "AgentInput"
            ):
                input_cls = attr
                break

        if input_cls is None:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path "
                f"(no Input class in {module_path} — agent uses legacy mixin pattern)"
            )

        # Build input — pass query + tenant_id + any extra context fields
        input_kwargs = {"query": query}
        if "tenant_id" in input_cls.model_fields:
            input_kwargs["tenant_id"] = tenant_id

        # Forward upstream agent results from context (e.g., entities from entity extraction)
        for key in ("entities", "relationships", "enhanced_query"):
            if key in context and key in input_cls.model_fields:
                input_kwargs[key] = context[key]

        typed_input = input_cls(**input_kwargs)
        result = await agent.process(typed_input)

        # Convert pydantic model to dict
        if hasattr(result, "model_dump"):
            return {"status": "success", "agent": agent_name, **result.model_dump()}
        return {"status": "success", "agent": agent_name, "result": str(result)}

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

        if capabilities & {"gateway"}:
            from cogniverse_agents.gateway_agent import (
                GatewayAgent,
                GatewayDeps,
                GatewayInput,
            )

            if not hasattr(self, "_gateway_agent") or self._gateway_agent is None:
                deps = GatewayDeps()
                self._gateway_agent = GatewayAgent(deps=deps)
            typed_input = GatewayInput(query=query, tenant_id=tenant_id)
            return self._gateway_agent, typed_input

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

        if capabilities & {"coding"}:
            from cogniverse_agents.coding_agent import (
                CodingAgent,
                CodingDeps,
                CodingInput,
            )
            from cogniverse_foundation.config.llm_factory import create_dspy_lm
            from cogniverse_foundation.config.utils import get_config

            config = get_config(
                tenant_id=tenant_id, config_manager=self._config_manager
            )
            coding_lm = create_dspy_lm(config.get_llm_config().resolve("coding_agent"))

            deps = CodingDeps(
                tenant_id=tenant_id,
                sandbox_manager=self._sandbox_manager,
            )

            async def search_fn(q: str, tid: str):
                from cogniverse_agents.search.service import SearchService

                try:
                    cfg = get_config(tenant_id=tid, config_manager=self._config_manager)
                    svc = SearchService(
                        config=cfg,
                        config_manager=self._config_manager,
                        schema_loader=self._schema_loader,
                    )
                    return [r.to_dict() for r in svc.search(
                        query=q, profile="code_lateon_mv", tenant_id=tid, top_k=10,
                    )]
                except Exception as exc:
                    logger.info("Code search unavailable, proceeding without context: %s", exc)
                    return []

            agent = CodingAgent(
                deps=deps,
                search_fn=search_fn,
                sandbox_manager=self._sandbox_manager,
            )
            agent._dspy_lm = coding_lm  # type: ignore[attr-defined]
            typed_input = CodingInput(task=query, tenant_id=tenant_id)
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

    async def _execute_gateway_task(
        self, query: str, context: Dict[str, Any], tenant_id: str,
    ) -> Dict[str, Any]:
        """Route query through GatewayAgent for triage.

        Simple queries are dispatched directly to the target execution agent.
        Complex queries are forwarded to OrchestratorAgent for multi-agent
        coordination.
        """
        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )

        if not hasattr(self, "_gateway_agent") or self._gateway_agent is None:
            deps = GatewayDeps()
            self._gateway_agent = GatewayAgent(deps=deps)
            # Inject global TelemetryManager for span emission
            from cogniverse_foundation.telemetry.manager import get_telemetry_manager

            self._gateway_agent.telemetry_manager = get_telemetry_manager()
            self._gateway_agent._artifact_tenant_id = tenant_id
            self._gateway_agent._load_artifact()

        input_data = GatewayInput(query=query, tenant_id=tenant_id)
        result = await self._gateway_agent._process_impl(input_data)

        if result.complexity == "complex":
            return await self._execute_orchestration_task(
                query, context, tenant_id,
                gateway_context={
                    "modality": result.modality,
                    "generation_type": result.generation_type,
                    "confidence": result.confidence,
                },
            )

        # Simple: route directly to the execution agent
        conversation_history = context.get("conversation_history", [])
        downstream = await self._execute_downstream_agent(
            agent_name=result.routed_to,
            query=query,
            tenant_id=tenant_id,
            top_k=context.get("top_k", 10),
            conversation_history=conversation_history,
        )
        return {
            "status": "success",
            "agent": "gateway_agent",
            "message": f"Routed '{query[:50]}' to {result.routed_to} (simple)",
            "gateway": {
                "complexity": result.complexity,
                "modality": result.modality,
                "generation_type": result.generation_type,
                "routed_to": result.routed_to,
                "confidence": result.confidence,
            },
            "downstream_result": downstream,
        }

    async def _execute_orchestration_task(
        self,
        query: str,
        context: Dict[str, Any],
        tenant_id: str,
        gateway_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute full orchestration pipeline via OrchestratorAgent."""
        from cogniverse_agents.orchestrator_agent import (
            OrchestratorAgent,
            OrchestratorDeps,
            OrchestratorInput,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        deps = OrchestratorDeps()
        tm = get_telemetry_manager()

        # Create WorkflowIntelligence so OrchestratorAgent can load
        # workflow templates from the artifact store at startup.
        workflow_intelligence = None
        if tm is not None:
            try:
                from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

                provider = tm.get_provider(tenant_id=tenant_id)
                workflow_intelligence = WorkflowIntelligence(provider, tenant_id)
            except Exception as e:
                logger.debug("WorkflowIntelligence init failed (non-fatal): %s", e)

        agent = OrchestratorAgent(
            deps=deps,
            registry=self._registry,
            config_manager=self._config_manager,
            workflow_intelligence=workflow_intelligence,
        )
        self._init_agent_memory(agent, "orchestrator_agent", tenant_id)
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        input_data = OrchestratorInput(
            query=query,
            tenant_id=tenant_id,
            session_id=context.get("session_id"),
            conversation_history=context.get("conversation_history"),
        )

        result = await agent._process_impl(input_data)

        return {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": f"Orchestrated '{query[:50]}' via A2A pipeline",
            "orchestration_result": (
                result.model_dump() if hasattr(result, "model_dump") else vars(result)
            ),
            "gateway_context": gateway_context,
        }

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
        self._init_agent_memory(agent, "summarizer_agent", tenant_id)

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
        self._init_agent_memory(agent, "detailed_report_agent", tenant_id)

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
        self._init_agent_memory(agent, "document_agent", tenant_id)

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
        self._init_agent_memory(agent, "deep_research_agent", tenant_id)

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

        deps = CodingDeps(
            tenant_id=tenant_id,
            sandbox_manager=self._sandbox_manager,
        )

        async def search_fn(query: str, tenant_id: str):
            """Search code using the code_lateon_mv profile.

            Returns empty list if code search is unavailable (e.g. encoder
            not yet registered for the code model), so coding tasks can
            proceed without indexed context.
            """
            from cogniverse_agents.search.service import SearchService

            try:
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
            except Exception as exc:
                logger.info(
                    "Code search unavailable, proceeding without context: %s", exc
                )
                return []

        agent = CodingAgent(
            deps=deps,
            search_fn=search_fn,
            sandbox_manager=self._sandbox_manager,
        )
        # Audit fix #14 — auto-init memory so the coding agent receives
        # learned strategies and tenant memories in inject_context_into_prompt.
        self._init_agent_memory(agent, "coding_agent", tenant_id)

        ctx = context or {}
        input_data = CodingInput(
            task=query,
            codebase_path=ctx.get("codebase_path", ""),
            tenant_id=tenant_id,
            max_iterations=ctx.get("max_iterations", 5),
            language=ctx.get("language", "python"),
        )
        with dspy.context(lm=coding_lm):
            result = await agent.process(input_data)

        return {
            "status": "success",
            "agent": "coding_agent",
            "message": f"Coding task complete for '{query}'",
            "result": result.model_dump(),
        }

    async def _execute_optimization_action(
        self,
        action: str,
        query: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Handle optimization actions (optimize_routing, get_optimization_status).

        Creates an AdvancedRoutingOptimizer backed by real telemetry, records
        provided examples, and triggers optimization or returns status.
        """
        from cogniverse_agents.routing.advanced_optimizer import (
            AdvancedRoutingOptimizer,
        )
        from cogniverse_foundation.config.utils import get_config
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        config = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
        llm_cfg = config.get_llm_config().resolve("routing_agent")
        telemetry_manager = get_telemetry_manager()
        telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

        optimizer = AdvancedRoutingOptimizer(
            tenant_id=tenant_id,
            llm_config=llm_cfg,
            telemetry_provider=telemetry_provider,
        )

        if action == "get_optimization_status":
            status = optimizer.get_optimization_status()
            return {
                "status": "active",
                "optimizer_ready": True,
                "metrics": status,
            }

        # optimize_routing / optimization_cycle_from_traces
        examples = context.get("examples", [])

        if examples:
            for ex in examples:
                await optimizer.record_routing_experience(
                    query=ex.get("query", ""),
                    entities=[],
                    relationships=[],
                    enhanced_query=ex.get("query", ""),
                    chosen_agent=ex.get("chosen_agent", "search_agent"),
                    routing_confidence=ex.get("confidence", 0.5),
                    search_quality=ex.get("search_quality", 0.5),
                    agent_success=ex.get("agent_success", True),
                    processing_time=ex.get("processing_time", 1.0),
                )

            return {
                "status": "optimization_triggered",
                "training_examples": len(examples),
                "optimizer": "AdvancedRoutingOptimizer",
            }

        # No examples: run automated optimization cycle from traces
        from cogniverse_agents.routing.routing_span_evaluator import (
            RoutingSpanEvaluator,
        )

        evaluator = RoutingSpanEvaluator(
            optimizer=optimizer, tenant_id=tenant_id
        )
        cycle_results = await evaluator.run_evaluation_cycle()
        spans_evaluated = cycle_results.get("spans_evaluated", 0)

        return {
            "status": "optimization_triggered",
            "optimizer": "OptimizationOrchestrator",
            "cycle_results": cycle_results,
            "spans_evaluated": spans_evaluated,
        }
