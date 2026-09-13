"""An agent's own DSPy LM answers for one tenant, and caches under it.

``DynamicDSPyMixin`` builds the LM an agent binds for its own DSPy calls. That
LM reaches the model backend directly unless the semantic router is on, so
without the tenant on the LM nothing in the request distinguishes one tenant's
call from another's and the two share a cached answer.
"""

from __future__ import annotations

import pytest

from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

MESSAGES = [{"role": "user", "content": "rewrite: people exercising"}]


def _agent_config() -> AgentConfig:
    return AgentConfig(
        agent_name="text_analysis_agent",
        agent_version="1.0.0",
        agent_description="analysis",
        agent_url="http://agent.example",
        capabilities=["analyze"],
        skills=[],
        module_config=ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="Analyze"
        ),
        llm_model="google/gemma-4-e4b-it",
        llm_base_url="https://llm.example/v1",
        llm_api_key="not-required",
    )


class Agent(DynamicDSPyMixin):
    def __init__(self, tenant_id) -> None:
        self.tenant_id = tenant_id
        self.system_config = None
        self._configure_dspy_lm(_agent_config())


class TestTheAgentLmCarriesItsTenant:
    def test_the_lm_is_bound_to_the_agents_canonical_tenant(self):
        agent = Agent("acme")

        assert agent._dspy_lm.cache_tenant_id == "acme:acme"
        assert agent._dspy_lm.cache is False

    def test_two_tenants_agents_key_the_same_request_apart(self):
        acme = Agent("acme:prod")._dspy_lm
        globex = Agent("globex:prod")._dspy_lm

        assert acme.cache_key(MESSAGES, {}) != globex.cache_key(MESSAGES, {})
        assert acme.cache_key(MESSAGES, {}).split("|", 1)[0] == "acme:prod"
        assert globex.cache_key(MESSAGES, {}).split("|", 1)[0] == "globex:prod"

    def test_an_agent_with_no_tenant_disables_caching(self):
        agent = Agent("")

        assert agent._dspy_lm.cache_tenant_id is None
        assert agent._dspy_lm.cache is False
