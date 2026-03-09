"""Agent endpoints - unified interface for all agent operations."""

import dataclasses
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level dependencies (injected from main.py)
_agent_registry: Optional[AgentRegistry] = None
_config_manager: Optional[ConfigManager] = None
_schema_loader: Optional[SchemaLoader] = None


def set_agent_registry(registry: AgentRegistry) -> None:
    """Inject AgentRegistry dependency for router endpoints"""
    global _agent_registry
    _agent_registry = registry
    logger.info("AgentRegistry injected into agents router")


def set_agent_dependencies(
    config_manager: ConfigManager, schema_loader: SchemaLoader
) -> None:
    """Inject config_manager and schema_loader for in-process agent execution."""
    global _config_manager, _schema_loader
    _config_manager = config_manager
    _schema_loader = schema_loader
    logger.info("Agent dependencies (config_manager, schema_loader) injected")


def get_registry() -> AgentRegistry:
    """Get the injected registry or raise error"""
    if _agent_registry is None:
        raise RuntimeError(
            "AgentRegistry not initialized. Call set_agent_registry() first."
        )
    return _agent_registry


class AgentTask(BaseModel):
    """Task request for agent processing."""

    agent_name: str
    query: str
    context: Dict[str, Any] = {}
    top_k: int = 10


class AgentRegistrationData(BaseModel):
    """Agent self-registration data for Curated Registry pattern"""

    name: str
    url: str
    capabilities: List[str] = []
    health_endpoint: str = "/health"
    process_endpoint: str = "/tasks/send"
    timeout: int = 30


@router.post("/register", status_code=201)
async def register_agent(data: AgentRegistrationData) -> Dict[str, Any]:
    """
    Register an agent in the curated registry (A2A pattern).

    Agents call this endpoint during startup to self-register.
    """
    registry = get_registry()

    success = registry.register_agent_from_data(data.dict())

    if not success:
        raise HTTPException(
            status_code=400, detail=f"Failed to register agent '{data.name}'"
        )

    return {
        "status": "registered",
        "agent": data.name,
        "url": data.url,
        "capabilities": data.capabilities,
    }


@router.get("/")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents."""
    registry = get_registry()
    agents = registry.list_agents()

    return {
        "count": len(agents),
        "agents": agents,
    }


@router.get("/stats")
async def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics including health status"""
    registry = get_registry()
    return registry.get_registry_stats()


@router.get("/by-capability/{capability}")
async def find_agents_by_capability(capability: str) -> Dict[str, Any]:
    """
    Find agents by capability (A2A Curated Registry pattern).

    Enables capability-based agent discovery.
    """
    registry = get_registry()
    agents = registry.find_agents_by_capability(capability)

    return {
        "capability": capability,
        "count": len(agents),
        "agents": [
            {
                "name": agent.name,
                "url": agent.url,
                "capabilities": agent.capabilities,
                "health_status": agent.health_status,
            }
            for agent in agents
        ],
    }


@router.get("/{agent_name}")
async def get_agent_info(agent_name: str) -> Dict[str, Any]:
    """Get information about a specific agent."""
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "name": agent.name,
        "url": agent.url,
        "capabilities": agent.capabilities,
        "health_status": agent.health_status,
        "health_endpoint": agent.health_endpoint,
        "process_endpoint": agent.process_endpoint,
    }


@router.delete("/{agent_name}", status_code=200)
async def unregister_agent(agent_name: str) -> Dict[str, Any]:
    """Unregister an agent from the registry"""
    registry = get_registry()

    success = registry.unregister_agent(agent_name)

    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "status": "unregistered",
        "agent": agent_name,
    }


@router.get("/{agent_name}/card")
async def get_agent_card(agent_name: str) -> Dict[str, Any]:
    """Get agent card (A2A protocol) for a specific agent."""
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return {
        "name": agent.name,
        "url": agent.url,
        "version": "1.0",
        "capabilities": agent.capabilities,
        "endpoints": {
            "health": agent.health_endpoint,
            "process": agent.process_endpoint,
            "info": f"/agents/{agent_name}",
        },
    }


@router.post("/{agent_name}/process")
async def process_agent_task(agent_name: str, task: AgentTask) -> Dict[str, Any]:
    """
    Process a task with a specific agent.

    In unified runtime mode, this endpoint executes agent logic in-process
    by routing to the appropriate service (search, LLM, etc.).
    """
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    if _config_manager is None or _schema_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Agent dependencies not configured. Runtime not fully initialized.",
        )

    tenant_id = task.context.get("tenant_id", "default")
    capabilities = set(agent.capabilities)

    # Order matters: check specific capabilities before general ones.
    if "routing" in capabilities:
        return await _execute_routing_task(task, tenant_id)
    elif capabilities & {"search", "video_search", "retrieval"}:
        return await _execute_search_task(task, tenant_id)
    elif capabilities & {"image_search", "visual_analysis"}:
        return await _execute_image_search_task(task, tenant_id)
    elif capabilities & {"audio_analysis", "transcription"}:
        return await _execute_audio_search_task(task, tenant_id)
    elif capabilities & {"document_analysis", "pdf_processing"}:
        return await _execute_document_search_task(task, tenant_id)
    elif capabilities & {"detailed_report"}:
        return await _execute_detailed_report_task(task, tenant_id)
    elif capabilities & {"summarization", "text_generation"}:
        return await _execute_summarization_task(task, tenant_id)
    elif capabilities & {"text_analysis", "sentiment", "classification"}:
        return await _execute_text_analysis_task(task, tenant_id)
    else:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Agent '{agent_name}' has no supported execution path in "
                f"unified runtime mode. Capabilities: {agent.capabilities}"
            ),
        )


async def _execute_search_task(task: AgentTask, tenant_id: str) -> Dict[str, Any]:
    """Execute a search task using the SearchService."""
    from cogniverse_agents.search.service import SearchService
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=_config_manager)
    search_service = SearchService(
        config=config,
        config_manager=_config_manager,
        schema_loader=_schema_loader,
    )

    profile = config.get("default_profile", "video_colpali_smol500_mv_frame")

    results = search_service.search(
        query=task.query,
        profile=profile,
        tenant_id=tenant_id,
        top_k=task.top_k,
        ranking_strategy="float_float",
    )

    result_list = [r.to_dict() for r in results]
    result_count = len(result_list)

    if result_count > 0:
        message = f"Found {result_count} results for '{task.query}'"
    else:
        message = f"No results found for '{task.query}'"

    return {
        "status": "success",
        "agent": "search_agent",
        "message": message,
        "results_count": result_count,
        "results": result_list,
        "profile": profile,
    }


async def _execute_routing_task(
    task: AgentTask, tenant_id: str
) -> Dict[str, Any]:
    """Execute routing via the actual RoutingAgent.

    Instantiates RoutingAgent with RoutingDeps (same pattern as
    SummarizerAgent/DetailedReportAgent). Memory, query enhancement,
    entity extraction, and agent selection are all handled by the
    RoutingAgent's own pipeline (MemoryAwareMixin + route_query()).
    """
    from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig
    from cogniverse_foundation.config.utils import get_config
    from cogniverse_foundation.telemetry.config import TelemetryConfig

    config = get_config(tenant_id=tenant_id, config_manager=_config_manager)

    llm_config_dict = config.get("llm_config", {})
    primary = llm_config_dict.get("primary", {})

    llm_endpoint = LLMEndpointConfig(
        model=primary.get("model", "ollama/smollm3:3b"),
        api_base=primary.get("api_base", "http://localhost:11434"),
        temperature=primary.get("temperature", 0.1),
        max_tokens=primary.get("max_tokens", 1000),
    )

    routing_config = config.get("routing_agent", {})
    memory_enabled = routing_config.get("enable_memory", False)

    deps_kwargs: Dict[str, Any] = {
        "telemetry_config": TelemetryConfig(enabled=False),
        "llm_config": llm_endpoint,
        "enable_memory": memory_enabled,
    }

    if memory_enabled:
        backend_url = config.get("backend_url", "http://localhost")
        backend_port = config.get("backend_port", 8080)
        deps_kwargs.update(
            {
                "memory_backend_host": backend_url,
                "memory_backend_port": backend_port,
                "memory_llm_model": primary.get("model", "qwen3:4b"),
                "memory_embedding_model": routing_config.get(
                    "memory_embedding_model", "nomic-embed-text"
                ),
                "memory_llm_base_url": primary.get(
                    "api_base", "http://localhost:11434"
                ),
                "memory_config_manager": _config_manager,
                "memory_schema_loader": _schema_loader,
            }
        )

    deps = RoutingDeps(**deps_kwargs)
    agent = RoutingAgent(deps=deps)

    result = await agent.route_query(
        query=task.query,
        context=task.context.get("context"),
        tenant_id=tenant_id,
    )

    needs_orchestration = result.metadata.get("needs_orchestration", False)

    if needs_orchestration:
        from cogniverse_agents.multi_agent_orchestrator import MultiAgentOrchestrator
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        # All agents are co-located in this runtime, use localhost:8000
        registry = get_registry()
        runtime_base_url = "http://localhost:8000"
        available_agents = {}
        for agent_name in registry.list_agents():
            agent_ep = registry.get_agent(agent_name)
            if agent_ep and agent_name != "routing_agent":
                available_agents[agent_name] = {
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
            query=result.enhanced_query or task.query,
            context=task.context.get("context"),
        )

        return {
            "status": "success",
            "agent": "routing_agent",
            "message": f"Orchestrated '{task.query}' via multi-agent workflow",
            "recommended_agent": result.recommended_agent,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "enhanced_query": result.enhanced_query,
            "needs_orchestration": True,
            "orchestration_result": orch_result,
            "metadata": result.metadata,
        }

    return {
        "status": "success",
        "agent": "routing_agent",
        "message": f"Routed '{task.query}' to {result.recommended_agent}",
        "recommended_agent": result.recommended_agent,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "enhanced_query": result.enhanced_query,
        "entities": result.entities,
        "relationships": result.relationships,
        "query_variants": result.query_variants,
        "metadata": result.metadata,
    }


def _get_vespa_endpoint(tenant_id: str) -> str:
    """Extract backend_url:backend_port from config for Vespa-backed agents."""
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id=tenant_id, config_manager=_config_manager)
    url = config.get("backend_url", "http://localhost")
    port = config.get("backend_port", 8080)
    return f"{url}:{port}"


async def _execute_summarization_task(
    task: AgentTask, tenant_id: str
) -> Dict[str, Any]:
    """Execute a summarization task using SummarizerAgent."""
    from cogniverse_agents.summarizer_agent import (
        SummarizerAgent,
        SummarizerDeps,
        SummaryRequest,
    )

    deps = SummarizerDeps(tenant_id=tenant_id)
    agent = SummarizerAgent(deps=deps, config_manager=_config_manager)

    request = SummaryRequest(
        query=task.query,
        search_results=[],
        summary_type="general",
    )
    result = await agent.summarize(request)

    return {
        "status": "success",
        "agent": "summarizer_agent",
        "message": f"Generated summary for '{task.query}'",
        "result": dataclasses.asdict(result),
    }


async def _execute_text_analysis_task(
    task: AgentTask, tenant_id: str
) -> Dict[str, Any]:
    """Execute a text analysis task using TextAnalysisAgent."""
    from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

    agent = TextAnalysisAgent(tenant_id=tenant_id, config_manager=_config_manager)

    analysis_type = task.context.get("analysis_type", "summary")
    result = agent.analyze_text(text=task.query, analysis_type=analysis_type)

    return {
        "status": "success",
        "agent": "text_analysis_agent",
        "message": f"Completed {analysis_type} analysis for '{task.query}'",
        "result": result,
    }


async def _execute_detailed_report_task(
    task: AgentTask, tenant_id: str
) -> Dict[str, Any]:
    """Execute a detailed report task using DetailedReportAgent."""
    from cogniverse_agents.detailed_report_agent import (
        DetailedReportAgent,
        DetailedReportDeps,
        ReportRequest,
    )

    deps = DetailedReportDeps(tenant_id=tenant_id)
    agent = DetailedReportAgent(deps=deps, config_manager=_config_manager)

    request = ReportRequest(
        query=task.query,
        search_results=[],
        report_type="comprehensive",
    )
    result = await agent.generate_report(request)

    return {
        "status": "success",
        "agent": "detailed_report_agent",
        "message": f"Generated detailed report for '{task.query}'",
        "result": dataclasses.asdict(result),
    }


async def _execute_image_search_task(task: AgentTask, tenant_id: str) -> Dict[str, Any]:
    """Execute an image search task using ImageSearchAgent."""
    from cogniverse_agents.image_search_agent import (
        ImageSearchAgent,
        ImageSearchDeps,
    )

    vespa_endpoint = _get_vespa_endpoint(tenant_id)
    deps = ImageSearchDeps(
        vespa_endpoint=vespa_endpoint,
        tenant_id=tenant_id,
    )
    agent = ImageSearchAgent(deps=deps)

    results = await agent.search_images(query=task.query, limit=task.top_k)

    result_list = [dataclasses.asdict(r) for r in results]
    return {
        "status": "success",
        "agent": "image_search_agent",
        "message": f"Found {len(result_list)} images for '{task.query}'",
        "results_count": len(result_list),
        "results": result_list,
    }


async def _execute_audio_search_task(task: AgentTask, tenant_id: str) -> Dict[str, Any]:
    """Execute an audio search task using AudioAnalysisAgent."""
    from cogniverse_agents.audio_analysis_agent import (
        AudioAnalysisAgent,
        AudioAnalysisDeps,
    )

    vespa_endpoint = _get_vespa_endpoint(tenant_id)
    deps = AudioAnalysisDeps(
        vespa_endpoint=vespa_endpoint,
        tenant_id=tenant_id,
    )
    agent = AudioAnalysisAgent(deps=deps)

    results = await agent.search_audio(query=task.query, limit=task.top_k)

    result_list = [dataclasses.asdict(r) for r in results]
    return {
        "status": "success",
        "agent": "audio_analysis_agent",
        "message": f"Found {len(result_list)} audio results for '{task.query}'",
        "results_count": len(result_list),
        "results": result_list,
    }


async def _execute_document_search_task(
    task: AgentTask, tenant_id: str
) -> Dict[str, Any]:
    """Execute a document search task using DocumentAgent."""
    from cogniverse_agents.document_agent import (
        DocumentAgent,
        DocumentAgentDeps,
    )

    vespa_endpoint = _get_vespa_endpoint(tenant_id)
    deps = DocumentAgentDeps(
        vespa_endpoint=vespa_endpoint,
        tenant_id=tenant_id,
    )
    agent = DocumentAgent(deps=deps)

    results = await agent.search_documents(query=task.query, limit=task.top_k)

    result_list = [dataclasses.asdict(r) for r in results]
    return {
        "status": "success",
        "agent": "document_agent",
        "message": f"Found {len(result_list)} documents for '{task.query}'",
        "results_count": len(result_list),
        "results": result_list,
    }


@router.post("/{agent_name}/upload")
async def upload_file_to_agent(
    agent_name: str,
    file: UploadFile = File(...),
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Upload a file for agent processing (e.g., video/image search).

    Note: This endpoint is for routing to agents. For direct agent execution,
    agents should expose their own upload endpoints.
    """
    registry = get_registry()

    agent = registry.get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    raise HTTPException(
        status_code=501,
        detail=f"Direct file upload not supported. Call agent at {agent.url}/upload",
    )
