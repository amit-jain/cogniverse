"""The optimizer CLI's agent builders bind their ConfigManager at build time.

Each ``_build_cli_*`` helper constructs a production agent and hands it the
manager the CLI resolved for the tenant. Writing that manager straight onto
``agent._config_manager`` accepts ``None`` without a word: the builder returns
a callable, the run proceeds, and the missing manager surfaces much later as a
request-time error from a different module. The bind path refuses ``None`` at
the construction site and names the agent.

The second half pins that a bound manager is the manager the built agent reads
from: tenant instructions come out of the injected store and the process
singleton is never consulted.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pytest

from cogniverse_agents.memory_aware_mixin import TENANT_INSTRUCTIONS_LOADED
from cogniverse_core.agents.base import AgentConfigurationError
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.optimization_cli import (
    _build_cli_entity_extractor,
    _build_cli_profile_labeler,
    _build_cli_query_enhancer,
    _build_cli_routing_decider,
)
from cogniverse_runtime.routers.tenant import (
    _INSTRUCTIONS_KEY,
    _INSTRUCTIONS_SERVICE,
)
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:production"
GLINER_URL = "http://127.0.0.1:29071"

# builder -> the agent class it constructs, and whether that agent enriches
# prompts from tenant instructions.
BUILDERS: dict[str, tuple[Callable[..., Any], str, bool]] = {
    "entity_extractor": (_build_cli_entity_extractor, "EntityExtractionAgent", True),
    "routing_decider": (_build_cli_routing_decider, "GatewayAgent", False),
    "query_enhancer": (_build_cli_query_enhancer, "QueryEnhancementAgent", True),
    "profile_labeler": (_build_cli_profile_labeler, "ProfileSelectionAgent", True),
}

MEMORY_AWARE_BUILDERS = sorted(name for name, spec in BUILDERS.items() if spec[2])


def _manager(instructions: str) -> ConfigManager:
    """A real ConfigManager over an in-memory store, seeded the way the CLI
    resolves one: GLiNER endpoint plus the tenant's instruction text."""
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    manager.set_system_config(
        SystemConfig(inference_service_urls={"gliner": GLINER_URL})
    )
    manager.set_config_value(
        tenant_id=TENANT,
        scope=ConfigScope.SYSTEM,
        service=_INSTRUCTIONS_SERVICE,
        config_key=_INSTRUCTIONS_KEY,
        config_value={"text": instructions, "updated_at": "2026-01-01T00:00:00Z"},
    )
    return manager


class _SingletonProbe:
    """Stands in for ``get_config_manager_singleton`` and counts every call."""

    def __init__(self) -> None:
        self.calls = 0
        self.manager = _manager("SINGLETON-MUST-NOT-BE-READ")

    def __call__(self) -> ConfigManager:
        self.calls += 1
        return self.manager


@pytest.fixture
def singleton_probe(monkeypatch) -> _SingletonProbe:
    probe = _SingletonProbe()
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config_manager_singleton", probe
    )
    return probe


def _built_agent(callback: Callable[..., Any]) -> Any:
    """The agent the builder constructed, read off the callback it returned."""
    return inspect.getclosurevars(callback).nonlocals["agent"]


class TestBuilderRefusesAMissingManager:
    """``None`` is a construction bug, surfaced at the construction site."""

    @pytest.mark.parametrize("builder_name", sorted(BUILDERS), ids=sorted(BUILDERS))
    async def test_none_manager_raises_at_build_time(self, builder_name):
        builder, agent_name, _ = BUILDERS[builder_name]

        with pytest.raises(AgentConfigurationError) as excinfo:
            await builder(
                config_manager=None,
                telemetry_manager=None,
                tenant_id=TENANT,
            )

        assert str(excinfo.value) == (
            f"{agent_name} requires a config_manager; got None. Pass "
            "config_manager=<ConfigManager> to the constructor; the runtime "
            "injects its own manager."
        )


class TestBuiltAgentReadsTheBoundManager:
    """A bound manager is the manager the built agent serves from."""

    @pytest.mark.parametrize("builder_name", sorted(BUILDERS), ids=sorted(BUILDERS))
    async def test_agent_accessor_returns_the_injected_manager(
        self, builder_name, singleton_probe
    ):
        builder, _, _ = BUILDERS[builder_name]
        cm = _manager(f"INJECTED-INSTRUCTIONS-FOR-{builder_name}")

        callback = await builder(
            config_manager=cm, telemetry_manager=None, tenant_id=TENANT
        )
        agent = _built_agent(callback)

        assert agent.config_manager is cm
        assert agent._artifact_tenant_id == TENANT
        assert agent.artifact_load_status == "no_telemetry"
        assert singleton_probe.calls == 0

    @pytest.mark.parametrize(
        "builder_name", MEMORY_AWARE_BUILDERS, ids=MEMORY_AWARE_BUILDERS
    )
    async def test_tenant_instructions_come_from_the_bound_manager(
        self, builder_name, singleton_probe
    ):
        builder, _, _ = BUILDERS[builder_name]
        expected = f"INJECTED-INSTRUCTIONS-FOR-{builder_name}"
        cm = _manager(expected)

        callback = await builder(
            config_manager=cm, telemetry_manager=None, tenant_id=TENANT
        )
        agent = _built_agent(callback)
        agent.set_tenant_for_context(TENANT)

        assert agent._get_tenant_instructions() == (
            expected,
            TENANT_INSTRUCTIONS_LOADED,
        )
        assert singleton_probe.calls == 0
