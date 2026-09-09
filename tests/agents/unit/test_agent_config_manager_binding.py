"""The injected ConfigManager is the manager every agent actually reads.

``MemoryAwareMixin._get_tenant_instructions`` runs on the per-dispatch
enrichment path of every memory-aware agent. It reads ``self.config_manager``
— the single slot ``bind_config_manager`` writes. An agent that parked the
injected manager under any other name served tenant instructions from the
process singleton instead: a different config store, and on a runtime whose
singleton points at an unreachable backend, seconds of dead-port retries per
dispatch.

Three layers here:

* every discovered agent class resolves ``config_manager`` to the one
  accessor and never assigns over it (the drift guard);
* the read on a constructed agent returns the value seeded in the INJECTED
  manager's store while the singleton is never consulted;
* the manager's own failure contract (dead-port store) reaches the caller as
  ``unavailable`` rather than as a silent fall-back to the singleton.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import time
from typing import Any, Dict, List, Tuple

import pytest

from cogniverse_agents.memory_aware_mixin import (
    TENANT_INSTRUCTIONS_LOADED,
    TENANT_INSTRUCTIONS_UNAVAILABLE,
    MemoryAwareMixin,
)
from cogniverse_core.agents.base import AgentConfigurationError, ConfigManagerAware
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.routers.tenant import (
    _INSTRUCTIONS_KEY,
    _INSTRUCTIONS_SERVICE,
)
from cogniverse_sdk.interfaces.config_store import (
    ConfigScope,
    ConfigStoreUnavailableError,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = pytest.mark.unit

TENANT = "acme:production"
DEAD_PORT = 29071


def _manager(
    instructions: str | None, *, seed_inference: bool = False
) -> ConfigManager:
    """A real ConfigManager over an in-memory store, seeded with instructions."""
    store = InMemoryConfigStore()
    store.initialize()
    manager = ConfigManager(store=store)
    if seed_inference:
        manager.set_system_config(
            SystemConfig(
                inference_service_urls={
                    "vllm_colpali": "http://localhost:8000",
                    "vllm_asr": "http://localhost:8001",
                }
            )
        )
    if instructions:
        manager.set_config_value(
            tenant_id=TENANT,
            scope=ConfigScope.SYSTEM,
            service=_INSTRUCTIONS_SERVICE,
            config_key=_INSTRUCTIONS_KEY,
            config_value={"text": instructions, "updated_at": "2026-01-01T00:00:00Z"},
        )
    return manager


def _dead_port_manager() -> ConfigManager:
    """A real ConfigManager whose store is a Vespa that is not listening."""
    from cogniverse_vespa.config.config_store import VespaConfigStore

    return ConfigManager(
        store=VespaConfigStore(backend_url="http://127.0.0.1", backend_port=DEAD_PORT)
    )


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


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def _discover_agent_classes() -> List[Tuple[str, type]]:
    import cogniverse_agents

    discovered = []
    for module_info in pkgutil.iter_modules(
        cogniverse_agents.__path__, prefix="cogniverse_agents."
    ):
        if module_info.ispkg:
            continue
        module = importlib.import_module(module_info.name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not name.endswith("Agent") or obj.__module__ != module_info.name:
                continue
            discovered.append((name, obj))
    return sorted(discovered)


_AGENT_CLASSES = _discover_agent_classes()
_MEMORY_AGENTS = [(n, c) for n, c in _AGENT_CLASSES if issubclass(c, MemoryAwareMixin)]

# The exact set of memory-aware agents whose per-dispatch enrichment reads the
# injected manager. A new memory-aware agent belongs here: adding it is what
# makes the accessor guard below cover it.
EXPECTED_MEMORY_AGENTS = {
    "AuditExplanationAgent",
    "CitationTracingAgent",
    "CodingAgent",
    "ContradictionReconciliationAgent",
    "CrossTenantComparisonAgent",
    "DeepResearchAgent",
    "DetailedReportAgent",
    "DocumentAgent",
    "EntityExtractionAgent",
    "FederatedQueryAgent",
    "KnowledgeGraphTraversalAgent",
    "KnowledgeSummarizationAgent",
    "MultiDocumentSynthesisAgent",
    "OrchestratorAgent",
    "ProfileSelectionAgent",
    "QueryEnhancementAgent",
    "SearchAgent",
    "SummarizerAgent",
    "TemporalReasoningAgent",
    "TextAnalysisAgent",
}


class TestEveryAgentReadsTheOneAccessor:
    def test_memory_aware_agent_set_is_exactly_the_expected_set(self):
        assert {name for name, _ in _MEMORY_AGENTS} == EXPECTED_MEMORY_AGENTS

    @pytest.mark.parametrize(
        "agent_name,agent_cls", _MEMORY_AGENTS, ids=[n for n, _ in _MEMORY_AGENTS]
    )
    def test_config_manager_resolves_to_the_canonical_accessor(
        self, agent_name, agent_cls
    ):
        """No agent shadows ``config_manager`` with its own attribute."""
        assert (
            inspect.getattr_static(agent_cls, "config_manager")
            is ConfigManagerAware.__dict__["config_manager"]
        )

    @pytest.mark.parametrize(
        "agent_name,agent_cls", _MEMORY_AGENTS, ids=[n for n, _ in _MEMORY_AGENTS]
    )
    def test_agent_never_assigns_over_the_accessor(self, agent_name, agent_cls):
        """``self.config_manager = ...`` is the shape of the original defect:
        it parks the injected manager beside the slot the mixin reads."""
        tree = ast.parse(inspect.getsource(agent_cls))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and target.attr == "config_manager"
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ]
        assert offenders == [], (
            f"{agent_name} assigns self.config_manager at line(s) {offenders} "
            "instead of calling bind_config_manager()."
        )

    @pytest.mark.parametrize(
        "agent_name,agent_cls", _MEMORY_AGENTS, ids=[n for n, _ in _MEMORY_AGENTS]
    )
    def test_tenant_instruction_read_is_the_mixin_implementation(
        self, agent_name, agent_cls
    ):
        assert (
            agent_cls._get_tenant_instructions
            is MemoryAwareMixin._get_tenant_instructions
        )


# --------------------------------------------------------------------------
# executed reads on constructed agents
# --------------------------------------------------------------------------


def _build_search_agent(cm: ConfigManager):
    from unittest.mock import Mock

    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps

    return SearchAgent(
        deps=SearchAgentDeps(backend_url="http://localhost", backend_port=8080),
        schema_loader=Mock(),
        config_manager=cm,
    )


def _build_text_analysis_agent(cm: ConfigManager):
    from unittest.mock import patch

    from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

    with (
        patch(
            "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
        ),
        patch(
            "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature"
        ),
    ):
        return TextAnalysisAgent(tenant_id=TENANT, config_manager=cm)


def _build_audio_agent(cm: ConfigManager):
    from cogniverse_agents.audio_analysis_agent import (
        AudioAnalysisAgent,
        AudioAnalysisDeps,
    )

    return AudioAnalysisAgent(
        deps=AudioAnalysisDeps(tenant_id=TENANT, config_manager=cm)
    )


def _build_federated_query_agent(cm: ConfigManager):
    from cogniverse_agents.federated_query_agent import (
        FederatedQueryAgent,
        FederatedQueryDeps,
    )

    return FederatedQueryAgent(
        deps=FederatedQueryDeps(tenant_id=TENANT), config_manager=cm
    )


def _build_citation_tracing_agent(cm: ConfigManager):
    """Built from deps alone, then bound the way the runtime binds it."""
    from cogniverse_agents.citation_tracing_agent import (
        CitationTracingAgent,
        CitationTracingDeps,
    )

    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id=TENANT))
    agent.bind_config_manager(cm)
    return agent


_BUILDERS: Dict[str, Any] = {
    "SearchAgent": _build_search_agent,
    "TextAnalysisAgent": _build_text_analysis_agent,
    "FederatedQueryAgent": _build_federated_query_agent,
    "CitationTracingAgent": _build_citation_tracing_agent,
}


class TestInjectedManagerServesTenantInstructions:
    """The read lands on the injected manager's store, never the singleton."""

    @pytest.mark.parametrize("agent_name", sorted(_BUILDERS), ids=sorted(_BUILDERS))
    def test_instructions_come_from_the_injected_manager(
        self, agent_name, singleton_probe
    ):
        expected = f"INJECTED-INSTRUCTIONS-FOR-{agent_name}"
        cm = _manager(expected, seed_inference=True)

        agent = _BUILDERS[agent_name](cm)
        agent.set_tenant_for_context(TENANT)

        assert agent.config_manager is cm
        assert agent._get_tenant_instructions() == (
            expected,
            TENANT_INSTRUCTIONS_LOADED,
        )
        assert singleton_probe.calls == 0

    @pytest.mark.parametrize("agent_name", sorted(_BUILDERS), ids=sorted(_BUILDERS))
    def test_prompt_enrichment_carries_the_injected_instructions(
        self, agent_name, singleton_probe
    ):
        expected = f"INJECTED-INSTRUCTIONS-FOR-{agent_name}"
        cm = _manager(expected, seed_inference=True)

        agent = _BUILDERS[agent_name](cm)
        agent.set_tenant_for_context(TENANT)

        assert agent.inject_context_into_prompt("BASE PROMPT", "q") == (
            f"BASE PROMPT\n\n## Tenant Instructions\n{expected}\n\n## Current Query:\nq"
        )
        assert agent.last_tenant_instructions_status == TENANT_INSTRUCTIONS_LOADED
        assert singleton_probe.calls == 0


class TestNonMemoryAgentBinding:
    """AudioAnalysisAgent has no memory mixin; it reads the same accessor when
    it builds its shared search backend."""

    def test_audio_agent_binds_the_deps_manager(self, singleton_probe):
        cm = _manager(None)
        agent = _build_audio_agent(cm)
        assert agent.config_manager is cm
        assert singleton_probe.calls == 0


class TestInjectedManagerFaultContract:
    """A dead store on the INJECTED manager degrades that manager's read; it
    never silently reroutes the read to the process singleton."""

    def test_dead_port_store_reports_unavailable(self, singleton_probe):
        cm = _dead_port_manager()
        agent = _build_citation_tracing_agent(cm)
        agent.set_tenant_for_context(TENANT)

        assert agent._get_tenant_instructions() == (
            None,
            TENANT_INSTRUCTIONS_UNAVAILABLE,
        )
        assert singleton_probe.calls == 0

    def test_dead_port_manager_read_raises_rather_than_returning_no_data(self):
        cm = _dead_port_manager()
        with pytest.raises(ConfigStoreUnavailableError) as excinfo:
            cm.get_tenant_instructions_config(TENANT)
        assert f"port={DEAD_PORT}" in str(excinfo.value)

    def test_read_does_not_pay_the_unreachable_singleton_retry_budget(
        self, monkeypatch
    ):
        """A singleton pointed at an unreachable config store costs 3.76s of
        retries per read (5 attempts, measured on this host). The enrichment
        read runs on every dispatch, so reaching for the singleton instead of
        the injected manager put that budget on the request path. With the
        injected manager the same read is served from its store."""
        dead = _dead_port_manager()
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config_manager_singleton",
            lambda: dead,
        )
        cm = _manager("INJECTED-INSTRUCTIONS")
        agent = _build_citation_tracing_agent(cm)
        agent.set_tenant_for_context(TENANT)

        started = time.monotonic()
        result = agent._get_tenant_instructions()
        elapsed = time.monotonic() - started

        assert result == ("INJECTED-INSTRUCTIONS", TENANT_INSTRUCTIONS_LOADED)
        assert elapsed < 1.0, (
            f"tenant-instruction read took {elapsed:.2f}s — the dead-port "
            "singleton retry budget (3.76s measured) is back on the "
            "enrichment path"
        )


class TestConstructionWithoutAManagerRaises:
    """A missing manager is a construction bug, surfaced at construction."""

    def test_search_agent_without_manager(self):
        from unittest.mock import Mock

        from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps

        with pytest.raises(AgentConfigurationError) as excinfo:
            SearchAgent(
                deps=SearchAgentDeps(backend_url="http://localhost", backend_port=8080),
                schema_loader=Mock(),
                config_manager=None,
            )
        assert str(excinfo.value) == (
            "SearchAgent requires a config_manager; got None. Pass "
            "config_manager=<ConfigManager> to the constructor; the runtime "
            "injects its own manager."
        )

    def test_audio_agent_without_manager_in_deps(self):
        from cogniverse_agents.audio_analysis_agent import (
            AudioAnalysisAgent,
            AudioAnalysisDeps,
        )

        with pytest.raises(AgentConfigurationError) as excinfo:
            AudioAnalysisAgent(deps=AudioAnalysisDeps(tenant_id=TENANT))
        assert str(excinfo.value) == (
            "AudioAnalysisAgent requires a config_manager; got None. Pass "
            "config_manager=<ConfigManager> to the constructor; the runtime "
            "injects its own manager."
        )

    def test_text_analysis_agent_without_manager(self):
        from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

        with pytest.raises(AgentConfigurationError) as excinfo:
            TextAnalysisAgent(tenant_id=TENANT, config_manager=None)
        assert str(excinfo.value) == (
            "TextAnalysisAgent requires a config_manager; got None. Pass "
            "config_manager=<ConfigManager> to the constructor; the runtime "
            "injects its own manager."
        )

    def test_unbound_agent_read_names_the_agent_and_the_dependency(
        self, singleton_probe
    ):
        from cogniverse_agents.citation_tracing_agent import (
            CitationTracingAgent,
            CitationTracingDeps,
        )

        agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id=TENANT))
        agent.set_tenant_for_context(TENANT)
        with pytest.raises(AgentConfigurationError) as excinfo:
            agent._get_tenant_instructions()
        assert str(excinfo.value) == (
            "CitationTracingAgent has no config_manager bound: it was "
            "constructed without one. Pass config_manager=<ConfigManager> to "
            "the constructor, or call bind_config_manager() at build time."
        )
        assert singleton_probe.calls == 0
