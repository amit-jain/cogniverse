"""
Integration tests validating agents boot with NO env vars.

Validates:
- Zero os.getenv/os.environ in A2A agent __init__ or constructor paths
- Agents construct with explicit deps only
- tenant_id is NOT required at construction (arrives per-request)
- profile is NOT required at construction (arrives per-request)
- Only BootstrapConfig at startup boundary reads env vars
"""

from unittest.mock import Mock, patch

import pytest

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportDeps,
)
from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
)
from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
)
from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
)
from cogniverse_agents.query_enhancement_agent import (
    QueryEnhancementAgent,
    QueryEnhancementDeps,
)
from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
from cogniverse_agents.summarizer_agent import SummarizerAgent, SummarizerDeps


class TestAgentConstructionNoEnvVars:
    """Every A2A agent constructs with explicit deps — zero env var reads."""

    def test_entity_extraction_no_env_vars(self):
        """EntityExtractionAgent constructs without env vars."""
        with patch("dspy.ChainOfThought"):
            deps = EntityExtractionDeps()
            agent = EntityExtractionAgent(deps=deps, port=8010)

        assert agent.agent_name == "entity_extraction_agent"
        assert "entity_extraction" in agent.capabilities

    def test_profile_selection_no_env_vars(self):
        """ProfileSelectionAgent constructs without env vars."""
        with patch("dspy.ChainOfThought"):
            deps = ProfileSelectionDeps()
            agent = ProfileSelectionAgent(deps=deps, port=8011)

        assert agent.agent_name == "profile_selection_agent"
        assert "profile_selection" in agent.capabilities

    def test_query_enhancement_no_env_vars(self):
        """QueryEnhancementAgent constructs without env vars."""
        with patch("dspy.ChainOfThought"):
            deps = QueryEnhancementDeps()
            agent = QueryEnhancementAgent(deps=deps, port=8012)

        assert agent.agent_name == "query_enhancement_agent"
        assert "query_enhancement" in agent.capabilities

    def test_orchestrator_no_env_vars(self):
        """OrchestratorAgent constructs without env vars."""
        registry = Mock()
        registry.list_agents = Mock(return_value=[])
        registry.agents = {}

        with patch("dspy.ChainOfThought"):
            deps = OrchestratorDeps()
            agent = OrchestratorAgent(deps=deps, registry=registry, port=8013)

        assert agent.agent_name == "orchestrator_agent"
        assert "orchestration" in agent.capabilities

    def test_search_agent_no_env_vars(self):
        """SearchAgent constructs without env vars (requires schema_loader)."""
        mock_schema_loader = Mock()

        deps = SearchAgentDeps(
            backend_url="http://localhost",
            backend_port=8080,
        )
        agent = SearchAgent(
            deps=deps,
            schema_loader=mock_schema_loader,
            port=8002,
        )

        assert agent.agent_name == "search_agent"
        assert "search" in agent.capabilities

    def test_summarizer_no_env_vars(self):
        """SummarizerAgent constructs without env vars."""
        mock_config = {
            "llm": {"model": "test"},
            "inference": {
                "provider": "ollama",
                "model": "test-model",
                "local_endpoint": "http://localhost:11434",
            },
        }

        with patch(
            "cogniverse_foundation.config.utils.get_config",
            return_value=mock_config,
        ):
            with patch("dspy.ChainOfThought"):
                deps = SummarizerDeps()
                agent = SummarizerAgent(deps=deps, port=8003)

        assert agent.agent_name == "summarizer_agent"
        assert "summarization" in agent.capabilities

    def test_detailed_report_no_env_vars(self):
        """DetailedReportAgent constructs without env vars."""
        mock_config = {
            "llm": {"model": "test"},
            "inference": {
                "provider": "ollama",
                "model": "test-model",
                "local_endpoint": "http://localhost:11434",
            },
        }

        with patch(
            "cogniverse_foundation.config.utils.get_config",
            return_value=mock_config,
        ):
            with patch("dspy.ChainOfThought"):
                deps = DetailedReportDeps()
                agent = DetailedReportAgent(deps=deps, port=8004)

        assert agent.agent_name == "detailed_report_agent"
        assert "detailed_report" in agent.capabilities


class TestDepsHaveNoTenantId:
    """Verify Deps classes don't require tenant_id at construction."""

    def test_entity_extraction_deps_no_tenant(self):
        """EntityExtractionDeps creates without tenant_id."""
        deps = EntityExtractionDeps()
        assert (
            not hasattr(deps, "tenant_id") or deps.model_fields.get("tenant_id") is None
        )

    def test_orchestrator_deps_no_tenant(self):
        """OrchestratorDeps creates without tenant_id."""
        deps = OrchestratorDeps()
        assert (
            not hasattr(deps, "tenant_id") or deps.model_fields.get("tenant_id") is None
        )

    def test_search_agent_deps_no_tenant_required(self):
        """SearchAgentDeps doesn't require tenant_id."""
        deps = SearchAgentDeps(
            backend_url="http://localhost",
            backend_port=8080,
        )
        assert deps.backend_url == "http://localhost"
        assert deps.backend_port == 8080


class TestBootstrapConfigIsOnlyEnvBoundary:
    """BootstrapConfig is the ONLY place env vars are read."""

    def test_bootstrap_from_environment(self):
        """BootstrapConfig.from_environment() reads env vars at startup boundary."""
        from cogniverse_foundation.config.bootstrap import BootstrapConfig

        with patch.dict(
            "os.environ",
            {
                "BACKEND_URL": "http://test-vespa",
                "BACKEND_PORT": "19071",
            },
            clear=False,
        ):
            bootstrap = BootstrapConfig.from_environment()

        assert bootstrap.backend_url == "http://test-vespa"
        assert bootstrap.backend_port == 19071

    def test_bootstrap_requires_backend_url(self):
        """BootstrapConfig raises ValueError when BACKEND_URL is missing."""
        from cogniverse_foundation.config.bootstrap import BootstrapConfig

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                ValueError, match="BACKEND_URL environment variable is required"
            ):
                BootstrapConfig.from_environment()


class TestNoOsGetenvInAgentModules:
    """Static analysis: agent modules must not use os.getenv/os.environ."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "cogniverse_agents.orchestrator_agent",
            "cogniverse_agents.entity_extraction_agent",
            "cogniverse_agents.profile_selection_agent",
            "cogniverse_agents.query_enhancement_agent",
            "cogniverse_agents.search_agent",
            "cogniverse_agents.summarizer_agent",
            "cogniverse_agents.detailed_report_agent",
        ],
    )
    def test_no_os_getenv_in_module(self, module_path):
        """Agent module source code has no os.getenv or os.environ references."""
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        source = inspect.getsource(module)

        # Allow os.environ only in if __name__ == "__main__" blocks and startup_event
        # Strip those sections before checking
        lines = source.split("\n")
        in_startup_or_main = False
        clean_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("async def startup_event") or stripped.startswith(
                'if __name__ == "__main__"'
            ):
                in_startup_or_main = True
                continue
            if in_startup_or_main and (
                stripped.startswith("def ")
                or stripped.startswith("async def ")
                or stripped.startswith("class ")
                or (stripped.startswith("@") and "app." not in stripped)
            ):
                in_startup_or_main = False
            if not in_startup_or_main:
                clean_lines.append(line)

        non_startup_source = "\n".join(clean_lines)

        assert (
            "os.getenv" not in non_startup_source
        ), f"{module_path} uses os.getenv outside startup boundary"
        assert (
            "os.environ" not in non_startup_source
        ), f"{module_path} uses os.environ outside startup boundary"
