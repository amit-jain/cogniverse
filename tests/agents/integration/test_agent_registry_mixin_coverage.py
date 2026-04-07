"""Verify all agent classes inherit the EXTENDED MemoryAwareMixin.

Audit fix #16 — there were two MemoryAwareMixin classes: a BASE one in
``cogniverse_core.agents.memory_aware_mixin`` (no ``get_strategies``) and
an EXTENDED one in ``cogniverse_agents.memory_aware_mixin`` (with
``get_strategies`` and the strategies-aware ``inject_context_into_prompt``).

Nine of ten agents imported the BASE version and silently missed the
extended methods. The fix deleted the BASE version and redirected all
imports to the EXTENDED one. This test scans every importable agent
class in cogniverse_agents and asserts they have ``get_strategies`` —
catching any future regression where someone adds an agent that imports
from the wrong place (or forgets the mixin entirely).
"""

import importlib
import inspect
import pkgutil

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin


def _discover_agent_classes():
    """Walk cogniverse_agents and yield every class whose name ends in 'Agent'."""
    import cogniverse_agents

    discovered = []
    for module_info in pkgutil.iter_modules(
        cogniverse_agents.__path__, prefix="cogniverse_agents."
    ):
        if module_info.ispkg:
            continue
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not name.endswith("Agent"):
                continue
            if obj.__module__ != module_info.name:
                continue  # imported from elsewhere, skip
            discovered.append((name, obj))
    return discovered


_AGENT_CLASSES = _discover_agent_classes()

# Explicit allowlist of agents that genuinely don't need memory.
# Pure-retrieval agents that take a query and return Vespa results, with
# no learned strategies, no tenant-customizable behavior, and no
# multi-step reasoning. Adding an agent here is a deliberate decision —
# the test fails for any other agent that lacks the mixin.
_AGENTS_WITHOUT_MEMORY = {
    "AudioAnalysisAgent",  # pure speech-to-text + retrieval
    "ImageSearchAgent",  # pure ColPali retrieval
    "VideoSearchAgent",  # standalone class in video_agent_refactored.py
}


@pytest.mark.integration
class TestAgentRegistryMixinCoverage:
    def test_at_least_some_agents_discovered(self):
        """Sanity check the discovery itself: there must be at least
        a handful of agent classes in cogniverse_agents."""
        assert len(_AGENT_CLASSES) >= 5, (
            f"Expected to discover several agent classes in cogniverse_agents, "
            f"got {len(_AGENT_CLASSES)}: {[n for n, _ in _AGENT_CLASSES]}"
        )

    @pytest.mark.parametrize(
        "agent_name,agent_cls",
        _AGENT_CLASSES,
        ids=[n for n, _ in _AGENT_CLASSES],
    )
    def test_agent_inherits_extended_memory_mixin(self, agent_name, agent_cls):
        """Every agent must EITHER inherit the extended MemoryAwareMixin
        OR be on the explicit allowlist of pure-retrieval agents that
        don't need memory. Silent skips are not allowed — every new
        agent must make an explicit decision."""
        if agent_name in _AGENTS_WITHOUT_MEMORY:
            assert not issubclass(agent_cls, MemoryAwareMixin), (
                f"{agent_name} is on the no-memory allowlist but actually "
                f"inherits MemoryAwareMixin. Either remove it from the "
                f"allowlist or remove the mixin."
            )
            return

        assert issubclass(agent_cls, MemoryAwareMixin), (
            f"{agent_name} does not inherit MemoryAwareMixin and is not on "
            f"the no-memory allowlist. Add the mixin (and wire "
            f"inject_context_into_prompt in _process_impl) or, if this "
            f"agent genuinely doesn't need memory, add it to "
            f"_AGENTS_WITHOUT_MEMORY in this test file with a one-line "
            f"justification."
        )

        assert hasattr(agent_cls, "get_strategies"), (
            f"{agent_name} inherits MemoryAwareMixin but is missing "
            f"`get_strategies`. It's inheriting from the deleted BASE mixin "
            f"in cogniverse_core.agents.memory_aware_mixin. Fix the import "
            f"to `from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin`."
        )

        assert hasattr(agent_cls, "inject_context_into_prompt"), (
            f"{agent_name} is missing inject_context_into_prompt"
        )

    def test_main_agents_have_extended_mixin(self):
        """Pin the specific agents that should definitely inherit the
        extended mixin. If any of these stops inheriting from it, the
        audit fix has regressed."""
        from cogniverse_agents.coding_agent import CodingAgent
        from cogniverse_agents.detailed_report_agent import DetailedReportAgent
        from cogniverse_agents.routing_agent import RoutingAgent
        from cogniverse_agents.search_agent import SearchAgent
        from cogniverse_agents.summarizer_agent import SummarizerAgent

        for agent_cls in (
            RoutingAgent,
            SearchAgent,
            CodingAgent,
            SummarizerAgent,
            DetailedReportAgent,
        ):
            assert issubclass(agent_cls, MemoryAwareMixin), (
                f"{agent_cls.__name__} must inherit from the extended "
                f"MemoryAwareMixin"
            )
            assert callable(getattr(agent_cls, "get_strategies", None)), (
                f"{agent_cls.__name__} is missing get_strategies — it's "
                f"importing from the deleted BASE mixin"
            )
