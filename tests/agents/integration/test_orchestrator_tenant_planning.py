"""The orchestrator plans only with agents that can serve the tenant.

A video-only tenant's orchestration planned image search and audio analysis
steps, which could only answer that the tenant has no such content. The real
OrchestratorAgent and AgentRegistry run here over a ConfigManager on the test
Vespa config store; tenants are seeded the way registration and profile
creation leave them.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from unittest.mock import Mock

import dspy
import pytest

from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.memory_store import (
    register_deployed_schema,
    unregister_deployed_schema,
)

pytestmark = pytest.mark.integration

_SHIPPED = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)
_SERVICE_URLS = {
    service: "http://inference.invalid"
    for data in _SHIPPED["backend"]["profiles"].values()
    for service in [(data.get("inference_services") or {}).get("embedding")]
    if service
}
VIDEO_ONLY_DROPPED = {"image_search_agent", "audio_analysis_agent", "document_agent"}
DOCUMENT_ONLY_DROPPED = {"search_agent", "image_search_agent", "audio_analysis_agent"}


def _tenant(label: str) -> str:
    return f"planning_{label}_{uuid.uuid4().hex[:8]}"


def _config_manager(http_port: int) -> ConfigManager:
    config_manager = ConfigManager(
        store=VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
    )
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=http_port,
            inference_service_urls=dict(_SERVICE_URLS),
        )
    )
    return config_manager


# The placeholder registry rows this module wrote to the shared Vespa's config
# store; removed when the module ends, so no later deploy reconstructs them.
_REGISTERED: list[tuple[str, str]] = []


def _register(config_manager, tenant_id: str, base_schema_name: str) -> None:
    register_deployed_schema(config_manager, tenant_id, base_schema_name)
    _REGISTERED.append((tenant_id, base_schema_name))


@pytest.fixture(scope="module")
def config_manager(shared_vespa):
    config_manager = _config_manager(shared_vespa["http_port"])
    yield config_manager
    while _REGISTERED:
        unregister_deployed_schema(config_manager, *_REGISTERED.pop())


def _video_tenant(config_manager) -> str:
    """Registered: the built-in video schema deployed, no stored profile."""
    tenant_id = _tenant("video")
    _register(config_manager, tenant_id, "video_colpali_smol500_mv_frame")
    return tenant_id


def _document_tenant(config_manager) -> str:
    """Stores a document profile and deployed only its schema."""
    tenant_id = _tenant("document")
    data = _SHIPPED["backend"]["profiles"]["document_text_semantic"]
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict("document_text_semantic", data),
        tenant_id=tenant_id,
    )
    _register(config_manager, tenant_id, data["schema_name"])
    return tenant_id


def _orchestrator(config_manager, tenant_id: str) -> OrchestratorAgent:
    """The orchestrator over a registry holding every enabled shipped agent,
    registered as the runtime's config loader registers them."""
    registry = AgentRegistry(tenant_id=tenant_id, config_manager=config_manager)
    for name, agent in _SHIPPED["agents"].items():
        if agent.get("enabled", True):
            assert registry.register_agent_from_data(
                {
                    "name": name,
                    "url": agent["url"],
                    "capabilities": agent.get("capabilities", []),
                    "process_endpoint": f"/agents/{name}/process",
                }
            )
    return OrchestratorAgent(
        deps=OrchestratorDeps(),
        registry=registry,
        config_manager=config_manager,
    )


@pytest.mark.asyncio
class TestPlanningFollowsTheTenantsModalities:
    async def test_a_video_tenant_is_planned_without_other_retrieval_agents(
        self, config_manager
    ):
        tenant_id = _video_tenant(config_manager)
        orchestrator = _orchestrator(config_manager, tenant_id)
        registered = orchestrator.registry.list_agents()

        planning = await orchestrator._planning_agents(tenant_id)

        assert VIDEO_ONLY_DROPPED <= set(registered)
        assert planning == [a for a in registered if a not in VIDEO_ONLY_DROPPED]

    async def test_a_document_tenant_is_planned_without_video_retrieval(
        self, config_manager
    ):
        tenant_id = _document_tenant(config_manager)
        orchestrator = _orchestrator(config_manager, tenant_id)
        registered = orchestrator.registry.list_agents()

        planning = await orchestrator._planning_agents(tenant_id)

        assert DOCUMENT_ONLY_DROPPED <= set(registered)
        assert planning == [a for a in registered if a not in DOCUMENT_ONLY_DROPPED]

    async def test_the_planner_is_offered_and_held_to_the_tenants_agents(
        self, config_manager
    ):
        """The LM sees only the agents the tenant can use, and an image step
        it names anyway is reported unavailable instead of run."""
        tenant_id = _video_tenant(config_manager)
        orchestrator = _orchestrator(config_manager, tenant_id)
        planning = await orchestrator._planning_agents(tenant_id)
        orchestrator.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="image_search_agent,search_agent,summarizer_agent",
                parallel_steps="",
                reasoning="find the clips, then summarize",
            )
        )

        plan = await orchestrator._create_plan(
            "Videos of people on a grassy field and what they do",
            available_agents=planning,
        )

        assert orchestrator.dspy_module.forward.call_args.kwargs[
            "available_agents"
        ] == ", ".join(planning)
        assert [step.agent_name for step in plan.steps] == [
            "search_agent",
            "summarizer_agent",
        ]
        assert plan.unavailable_agents == ["image_search_agent"]

    async def test_concurrent_plans_for_two_tenants_never_bleed(self, config_manager):
        video = _video_tenant(config_manager)
        document = _document_tenant(config_manager)
        orchestrator = _orchestrator(config_manager, video)
        registered = orchestrator.registry.list_agents()
        tenants = [video, document] * 5
        barrier = threading.Barrier(len(tenants))

        def plan_for(tenant_id: str) -> list[str]:
            barrier.wait()
            return asyncio.run(orchestrator._planning_agents(tenant_id))

        planned = await asyncio.gather(
            *(asyncio.to_thread(plan_for, tenant_id) for tenant_id in tenants)
        )

        assert (VIDEO_ONLY_DROPPED | DOCUMENT_ONLY_DROPPED) <= set(registered)
        assert (
            planned
            == [
                [a for a in registered if a not in VIDEO_ONLY_DROPPED],
                [a for a in registered if a not in DOCUMENT_ONLY_DROPPED],
            ]
            * 5
        )

    async def test_an_unreachable_config_store_fails_planning_naming_the_tenant(
        self, config_manager
    ):
        tenant_id = _video_tenant(config_manager)
        orchestrator = _orchestrator(config_manager, tenant_id)
        orchestrator._config_manager = _config_manager_on_dead_port()

        with pytest.raises(RuntimeError) as failure:
            await orchestrator._planning_agents(tenant_id)

        assert str(failure.value).startswith(
            f"Cannot plan for tenant {tenant_id!r}: its servable profiles could "
            "not be read (ConfigStoreUnavailableError: "
        )
        assert isinstance(failure.value.__cause__, ConfigStoreUnavailableError)


def _config_manager_on_dead_port() -> ConfigManager:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return ConfigManager(
        store=VespaConfigStore(backend_url="http://127.0.0.1", backend_port=port)
    )
