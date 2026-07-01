"""
Unit tests for TextAnalysisAgent
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cogniverse_agents.text_analysis_agent import (
    TextAnalysisAgent,
    _agent_instances,
    app,
    get_agent,
    set_config_manager,
)


class TestTextAnalysisAgent:
    """Test TextAnalysisAgent initialization and core functionality"""

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch(
        "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
    )
    @patch("cogniverse_foundation.config.utils.get_config")
    def test_initialization(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """Test agent initializes correctly with tenant_id"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # Initialize agent with config_manager
        agent = TextAnalysisAgent(
            tenant_id="test_tenant", config_manager=config_manager_memory
        )

        assert agent.tenant_id == "test_tenant"
        assert agent.config is not None
        assert agent.config.agent_name == "text_analysis_agent"
        assert "text_analysis" in agent.config.capabilities

        # Verify DSPy was initialized
        mock_initialize_dspy.assert_called_once()
        mock_register_signature.assert_called_once()

    def test_initialization_without_tenant_id_raises_error(self, config_manager_memory):
        """Test that initializing without tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            TextAnalysisAgent(tenant_id="", config_manager=config_manager_memory)

        with pytest.raises(ValueError, match="tenant_id is required"):
            TextAnalysisAgent(tenant_id=None, config_manager=config_manager_memory)

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch(
        "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
    )
    @patch("cogniverse_foundation.config.utils.get_config")
    def test_analyze_text(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """Test text analysis functionality"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        agent = TextAnalysisAgent(
            tenant_id="test_tenant", config_manager=config_manager_memory
        )
        # DynamicDSPyMixin.initialize_dynamic_dspy is patched out above, so
        # _dspy_lm would be unset and analyze_text would (correctly) raise.
        # Production always has it set via DynamicDSPyMixin; attach a sentinel
        # here so we exercise the analyze_text body. The per-tenant-lm contract
        # itself is asserted in test_analyze_text_uses_per_tenant_lm and in
        # tests/agents/unit/test_dspy_lm_context_binding.py.
        agent._dspy_lm = MagicMock(name="stub_lm")

        # Mock the get_or_create_module method to return a mock module
        mock_module = MagicMock()
        mock_result = MagicMock()
        mock_result.result = "Test analysis result"
        mock_result.confidence = "0.95"
        mock_module.return_value = mock_result

        with patch.object(agent, "get_or_create_module", return_value=mock_module):
            # Test analyze_text
            result = agent.analyze_text("Test text", "summary")

            assert result["result"] == "Test analysis result"
            assert result["confidence"] == 0.95
            assert result["analysis_type"] == "summary"
            mock_module.assert_called_once_with(
                text="Test text", analysis_type="summary"
            )

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch(
        "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
    )
    @patch("cogniverse_foundation.config.utils.get_config")
    def test_analyze_text_uses_per_tenant_lm(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """The per-tenant ``_dspy_lm`` must actually drive the module call —
        without ``dspy.context(lm=...)`` the call silently falls back to the
        global LM (or none on standalone), ignoring the tenant config."""
        import dspy

        mock_get_config.return_value = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        agent = TextAnalysisAgent(
            tenant_id="test_tenant", config_manager=config_manager_memory
        )
        sentinel = MagicMock(name="tenant_lm")
        agent._dspy_lm = sentinel

        captured = {}

        def fake_module(text, analysis_type):
            captured["lm"] = dspy.settings.lm
            result = MagicMock()
            result.result = "ok"
            result.confidence = "0.5"
            return result

        with patch.object(agent, "get_or_create_module", return_value=fake_module):
            agent.analyze_text("hi", "summary")

        assert captured["lm"] is sentinel

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch(
        "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
    )
    @patch("cogniverse_foundation.config.utils.get_config")
    def test_get_agent_factory(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
        config_manager_memory,
    ):
        """Test get_agent factory function creates and caches instances"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # Clear cache
        _agent_instances.clear()

        # Set config_manager for factory function
        set_config_manager(config_manager_memory)

        # First call creates new instance
        agent1 = get_agent("test_tenant_1")
        assert isinstance(agent1, TextAnalysisAgent)
        assert agent1.tenant_id == "test_tenant_1"

        # Second call returns cached instance
        agent2 = get_agent("test_tenant_1")
        assert agent2 is agent1

        # Different tenant creates new instance
        agent3 = get_agent("test_tenant_2")
        assert agent3 is not agent1
        assert agent3.tenant_id == "test_tenant_2"

    def test_get_agent_without_tenant_id_raises_error(self):
        """Test that get_agent without tenant_id raises ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            get_agent("")

    def test_lifespan_wires_config_manager(self, config_manager_memory, monkeypatch):
        """Standalone app startup must wire the ConfigManager so /analyze works.

        Before the fix the app had no lifespan and __main__ never called
        set_config_manager, so _config_manager stayed None: /health reported
        'initializing' and every /analyze raised RuntimeError -> HTTP 500. This
        drives the REAL lifespan via the TestClient context manager.
        """
        import cogniverse_agents.text_analysis_agent as ta_module

        monkeypatch.setattr(
            ta_module, "create_default_config_manager", lambda: config_manager_memory
        )
        monkeypatch.setattr(ta_module, "_config_manager", None)

        # Lifespan runs on context entry and wires the manager.
        with TestClient(app) as client:
            assert ta_module._config_manager is config_manager_memory
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "healthy"

        monkeypatch.setattr(ta_module, "_config_manager", None)

        with pytest.raises(ValueError, match="tenant_id is required"):
            get_agent(None)


class TestTextAnalysisEndpoints:
    """Test FastAPI endpoints"""

    def setup_method(self):
        """Clear agent cache before each test"""
        _agent_instances.clear()

    @pytest.fixture(autouse=True)
    def _wire_in_memory_config(self, config_manager_memory, monkeypatch):
        """Point the app lifespan at an in-memory ConfigManager.

        The real ``create_default_config_manager()`` builds a VespaConfigStore,
        so under ``TestClient`` a ``POST /analyze`` reaches
        ``get_agent -> TextAnalysisAgent.__init__ -> config_manager
        .get_agent_config(...)`` and blocks on a Vespa HTTP call that never
        answers in a unit-test sandbox. These tests must manage their own
        infrastructure, so the lifespan wires the in-memory manager instead.
        """
        import cogniverse_agents.text_analysis_agent as ta_module

        monkeypatch.setattr(
            ta_module, "create_default_config_manager", lambda: config_manager_memory
        )
        monkeypatch.setattr(ta_module, "_config_manager", None)

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch(
        "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
    )
    @patch("cogniverse_foundation.config.utils.get_config")
    def test_analyze_endpoint(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
    ):
        """Test /analyze endpoint accepts tenant_id and returns analysis"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # `with` runs the lifespan that sets the module config_manager.
        with TestClient(app) as client:
            # Mock the agent's analyze_text method
            with patch.object(TextAnalysisAgent, "analyze_text") as mock_analyze:
                mock_analyze.return_value = {
                    "result": "Test analysis result",
                    "confidence": 0.95,
                    "module_type": "predict",
                    "analysis_type": "summary",
                }

                # Params go in the JSON body, not the query string.
                response = client.post(
                    "/analyze",
                    json={
                        "text": "Test text to analyze",
                        "tenant_id": "test_tenant",
                        "analysis_type": "summary",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["analysis"]["result"] == "Test analysis result"
                assert data["analysis"]["confidence"] == 0.95
                mock_analyze.assert_called_once_with("Test text to analyze", "summary")

    @patch("cogniverse_foundation.config.utils.get_config")
    def test_analyze_endpoint_without_tenant_id_fails(self, mock_get_config):
        """Test that /analyze endpoint requires tenant_id parameter"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        client = TestClient(app)

        # Test without tenant_id - should fail with 422 (missing required field)
        response = client.post(
            "/analyze",
            json={
                "text": "Test text to analyze",
                "analysis_type": "summary",
            },
        )

        assert response.status_code == 422  # FastAPI validation error
        error_detail = response.json()["detail"]
        assert any(error["loc"] == ["body", "tenant_id"] for error in error_detail), (
            "Should have validation error for missing tenant_id"
        )

    @patch("cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.register_signature")
    @patch(
        "cogniverse_agents.text_analysis_agent.DynamicDSPyMixin.initialize_dynamic_dspy"
    )
    @patch("cogniverse_foundation.config.utils.get_config")
    def test_analyze_endpoint_error_handling(
        self,
        mock_get_config,
        mock_initialize_dspy,
        mock_register_signature,
    ):
        """Test that endpoint properly handles analysis errors"""
        mock_config = {
            "text_analysis_port": 8005,
            "llm_model": "gpt-4",
            "llm_base_url": "http://localhost:11434",
        }
        mock_get_config.return_value = mock_config

        # Configure TestClient to not re-raise server exceptions; `with` runs
        # the lifespan that sets the module config_manager.
        with TestClient(app, raise_server_exceptions=False) as client:
            # Mock analyze_text to raise exception
            with patch.object(TextAnalysisAgent, "analyze_text") as mock_analyze:
                mock_analyze.side_effect = RuntimeError("Analysis failed")

                response = client.post(
                    "/analyze",
                    json={
                        "text": "Test text",
                        "tenant_id": "test_tenant",
                        "analysis_type": "summary",
                    },
                )

                # Should return 500 error
                assert response.status_code == 500

    def test_health_endpoint(self):
        """GET /health reports agent status and capabilities (A2A probe)."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "text_analysis_agent"
        assert data["status"] in {"healthy", "initializing"}
        assert "summarization" in data["capabilities"]

    def test_agent_card_endpoint(self):
        """GET /agent.json returns the A2A discovery card."""
        client = TestClient(app)
        resp = client.get("/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TextAnalysisAgent"
        assert data["protocol"] == "a2a"
        assert data["url"] == "/analyze"
        assert set(data["capabilities"]) == {
            "text_analysis",
            "sentiment",
            "summarization",
            "entity_extraction",
        }
