"""The orchestrator's query analysis uses the configured GLiNER model.

The model is the request tenant's ``RoutingConfigUnified.gliner_model`` (the
setting the dispatcher seeds the gateway with) and the endpoint is
``SystemConfig.inference_service_urls["gliner"]``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
    _request_tenant_id,
)
from cogniverse_agents.routing import dspy_relationship_router
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = pytest.mark.unit

GLINER_URL = "http://gliner.test:8080"
TENANT_A = "acme:production"
TENANT_B = "globex:production"


def _config_manager() -> ConfigManager:
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    manager.set_system_config(
        SystemConfig(inference_service_urls={"gliner": GLINER_URL})
    )
    manager.set_routing_config(
        RoutingConfigUnified(tenant_id=TENANT_A, gliner_model="acme/gliner-custom"),
        tenant_id=TENANT_A,
    )
    manager.set_routing_config(
        RoutingConfigUnified(tenant_id=TENANT_B, gliner_model="globex/gliner-small"),
        tenant_id=TENANT_B,
    )
    return manager


@pytest.fixture
def orchestrator():
    registry = Mock()
    registry.agents = {}
    with patch("dspy.ChainOfThought"):
        return OrchestratorAgent(
            deps=OrchestratorDeps(),
            registry=registry,
            config_manager=_config_manager(),
            port=8013,
        )


def _module_for(orchestrator, tenant_id):
    token = _request_tenant_id.set(tenant_id)
    try:
        return orchestrator._get_query_analysis_module()
    finally:
        _request_tenant_id.reset(token)


def test_the_tenants_configured_model_and_the_service_url_reach_the_extractor(
    orchestrator,
):
    module = _module_for(orchestrator, TENANT_A)

    assert (
        module.gliner_extractor.model_name,
        module.gliner_extractor.inference_url,
    ) == ("acme/gliner-custom", GLINER_URL)


def test_each_tenant_gets_its_own_configured_model(orchestrator):
    acme = _module_for(orchestrator, TENANT_A)
    globex = _module_for(orchestrator, TENANT_B)

    assert [acme.gliner_extractor.model_name, globex.gliner_extractor.model_name] == [
        "acme/gliner-custom",
        "globex/gliner-small",
    ]
    assert _module_for(orchestrator, TENANT_A) is acme


def test_concurrent_requests_build_one_module_per_configured_model(
    orchestrator, monkeypatch
):
    builds: list[tuple[str, str]] = []
    lock = threading.Lock()
    real = dspy_relationship_router.create_composable_query_analysis_module

    def counting(*, gliner_model, gliner_inference_url):
        with lock:
            builds.append((gliner_model, gliner_inference_url))
        return real(
            gliner_model=gliner_model, gliner_inference_url=gliner_inference_url
        )

    monkeypatch.setattr(
        dspy_relationship_router, "create_composable_query_analysis_module", counting
    )
    tenants = [TENANT_A, TENANT_B] * 8
    barrier = threading.Barrier(len(tenants))

    def resolve(tenant_id):
        barrier.wait(timeout=10)
        return _module_for(orchestrator, tenant_id)

    with ThreadPoolExecutor(max_workers=len(tenants)) as pool:
        modules = list(pool.map(resolve, tenants))

    assert sorted(builds) == [
        ("acme/gliner-custom", GLINER_URL),
        ("globex/gliner-small", GLINER_URL),
    ]
    assert len({id(module) for module in modules[0::2]}) == 1
    assert len({id(module) for module in modules[1::2]}) == 1


def test_an_unreadable_routing_config_raises_instead_of_using_a_default(
    orchestrator,
):
    def unreadable(tenant_id=None, service="gateway_agent"):
        raise RuntimeError("config store unreachable")

    orchestrator._config_manager.get_routing_config = unreadable

    with pytest.raises(RuntimeError, match="config store unreachable"):
        _module_for(orchestrator, TENANT_A)
