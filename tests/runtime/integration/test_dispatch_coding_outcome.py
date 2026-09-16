"""The coding dispatch envelope reports the outcome the agent recorded.

Built on the real ``AgentDispatcher``, ``AgentRegistry`` and ``ConfigManager``
over the in-memory config store, driven with real ``CodingOutput`` values.
``CodingOutput.success`` is the canonical outcome field; ``error`` is its
detail.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

TENANT = "acme:acme"


class _StubSandboxManager:
    """Returns a fixed policy per agent, as SandboxManager does."""

    def __init__(self, policies: Optional[Dict[str, Dict[str, Any]]] = None):
        self._policies = policies or {}

    def get_policy(self, agent_name: str) -> Optional[Dict[str, Any]]:
        return self._policies.get(agent_name)


def _dispatcher() -> AgentDispatcher:
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    dispatcher = AgentDispatcher(
        agent_registry=AgentRegistry(tenant_id=TENANT, config_manager=config_manager),
        config_manager=config_manager,
        schema_loader=None,
        sandbox_manager=_StubSandboxManager(),
    )
    dispatcher._registry.register_agent(
        AgentEndpoint(
            name="coding_agent",
            url="http://localhost:8003",
            capabilities=["coding", "code_generation"],
        )
    )
    return dispatcher


@pytest.fixture
def coding_dispatcher(monkeypatch):
    """A dispatcher whose coding agent returns a scripted ``CodingOutput``."""
    import cogniverse_agents.coding_agent as coding_module

    dispatcher = _dispatcher()
    dispatcher._init_agent_memory = lambda *args, **kwargs: None
    scripted: List[Any] = []

    class _ScriptedCodingAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def process(self, input_data):
            return scripted[0]

    monkeypatch.setattr(coding_module, "CodingAgent", _ScriptedCodingAgent)
    return dispatcher, scripted


async def _envelope(coding_dispatcher, output) -> Dict[str, Any]:
    dispatcher, scripted = coding_dispatcher
    scripted.append(output)
    return await dispatcher._execute_coding_task("run the program", TENANT, {})


async def test_a_failed_run_without_a_detail_is_still_an_error(coding_dispatcher):
    """``success=False`` is the outcome even when no detail was recorded."""
    from cogniverse_agents.coding_agent import CodingOutput

    output = CodingOutput(plan="run it", success=False, summary="no detail recorded")

    envelope = await _envelope(coding_dispatcher, output)

    assert envelope["status"] == "error"
    assert envelope["error"] is None
    assert envelope["agent"] == "coding_agent"
    assert envelope["result"] == output.model_dump()


async def test_a_failed_run_carries_its_detail(coding_dispatcher):
    from cogniverse_agents.coding_agent import CodingOutput

    output = CodingOutput(
        plan="run it",
        success=False,
        error="Coding task failed after 2 iteration(s): Exit code: 7",
        summary="Coding task failed after 2 iteration(s): Exit code: 7",
    )

    envelope = await _envelope(coding_dispatcher, output)

    assert envelope["status"] == "error"
    assert envelope["error"] == (
        "Coding task failed after 2 iteration(s): Exit code: 7"
    )


async def test_a_successful_run_is_a_success(coding_dispatcher):
    from cogniverse_agents.coding_agent import CodingOutput

    output = CodingOutput(plan="run it", success=True, summary="Completed.")

    envelope = await _envelope(coding_dispatcher, output)

    assert envelope["status"] == "success"
    assert envelope["message"] == "Coding task complete for 'run the program'"
    assert envelope["result"] == output.model_dump()


async def test_pending_tool_calls_suspend_before_the_outcome_is_read(
    coding_dispatcher,
):
    """A workspace suspension asks for input; it is not an outcome yet."""
    from cogniverse_agents.coding_agent import CodingOutput

    output = CodingOutput(
        plan="edit it",
        success=False,
        pending_tool_calls=[{"id": "call_1", "name": "read_file", "arguments": "{}"}],
        continuation_state={"mode": "workspace", "plan": "edit it", "step": 1},
    )

    envelope = await _envelope(coding_dispatcher, output)

    assert envelope["status"] == "input_required"
    assert envelope["pending_tool_calls"] == output.pending_tool_calls
    assert envelope["continuation_state"] == output.continuation_state
