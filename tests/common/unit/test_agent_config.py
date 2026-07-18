"""
Unit tests for AgentConfig and related schemas.
"""

import pytest

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)


class TestDSPyModuleType:
    """Test DSPyModuleType enum"""

    def test_invalid_module_type_raises_error(self):
        """Test invalid module type raises ValueError"""
        with pytest.raises(ValueError):
            DSPyModuleType("invalid_type")


class TestOptimizerType:
    """Test OptimizerType enum"""

    def test_invalid_optimizer_type_raises_error(self):
        """Test invalid optimizer type raises ValueError"""
        with pytest.raises(ValueError):
            OptimizerType("invalid_optimizer")


class TestAgentConfig:
    """Test AgentConfig dataclass"""

    def test_agent_config_to_dict(self):
        """Test AgentConfig serialization to dict"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="TestSignature"
        )

        config = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[{"name": "skill1"}],
            module_config=module_config,
            llm_api_key="secret_key",
        )

        config_dict = config.to_dict()

        assert config_dict["agent_name"] == "test_agent"
        assert config_dict["agent_version"] == "1.0.0"
        assert config_dict["module_config"]["module_type"] == "predict"
        assert config_dict["module_config"]["signature"] == "TestSignature"
        assert config_dict["llm_api_key"] == "***"  # Should be masked
        assert config_dict["optimizer_config"] is None

    def test_agent_config_from_dict(self):
        """Test AgentConfig deserialization from dict"""
        config_dict = {
            "agent_name": "test_agent",
            "agent_version": "1.0.0",
            "agent_description": "Test agent",
            "agent_url": "http://localhost:8000",
            "capabilities": ["test"],
            "skills": [],
            "module_config": {
                "module_type": "predict",
                "signature": "TestSignature",
                "max_retries": 3,
                "temperature": 0.7,
                "max_tokens": None,
                "custom_params": {},
            },
        }

        config = AgentConfig.from_dict(config_dict)

        assert config.agent_name == "test_agent"
        assert config.agent_version == "1.0.0"
        assert config.module_config.module_type == DSPyModuleType.PREDICT
        assert config.module_config.signature == "TestSignature"

    def test_agent_config_roundtrip(self):
        """Test AgentConfig serialization/deserialization roundtrip"""
        module_config = ModuleConfig(
            module_type=DSPyModuleType.REACT,
            signature="TestSignature",
            max_retries=5,
        )
        optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.COPRO, max_bootstrapped_demos=8
        )

        original = AgentConfig(
            agent_name="test_agent",
            agent_version="1.0.0",
            agent_description="Test agent",
            agent_url="http://localhost:8000",
            capabilities=["test"],
            skills=[],
            module_config=module_config,
            optimizer_config=optimizer_config,
            llm_model="gpt-4-turbo",
            llm_temperature=0.5,
        )

        # Serialize and deserialize
        config_dict = original.to_dict()
        restored = AgentConfig.from_dict(config_dict)

        # Verify all fields match
        assert restored.agent_name == original.agent_name
        assert restored.agent_version == original.agent_version
        assert restored.module_config.module_type == original.module_config.module_type
        assert restored.module_config.max_retries == original.module_config.max_retries
        assert (
            restored.optimizer_config.optimizer_type
            == original.optimizer_config.optimizer_type
        )
        assert (
            restored.optimizer_config.max_bootstrapped_demos
            == original.optimizer_config.max_bootstrapped_demos
        )
        assert restored.llm_model == original.llm_model
        assert restored.llm_temperature == original.llm_temperature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAgentConfigManagerCaching:
    """get_agent_config serves repeat reads from the manager's scoped TTL
    cache — it sits on the per-dispatch answer path (behavior toggles for
    every summarizer/report dispatch), so an uncached read cost one
    synchronous Vespa query per dispatch while sibling scopes (routing,
    telemetry) were cached. set_agent_config invalidates the cache so a
    fresh write is visible immediately."""

    def _manager_with_counting_store(self):
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        calls = {"get": 0}
        real_get = store.get_config

        def counting_get(*args, **kwargs):
            calls["get"] += 1
            return real_get(*args, **kwargs)

        store.get_config = counting_get
        return ConfigManager(store=store), calls

    def _config(self, description="v1"):
        return AgentConfig(
            agent_name="summarizer_agent",
            agent_version="1.0.0",
            agent_description=description,
            agent_url="http://x:8003",
            capabilities=["summarization"],
            skills=[],
            module_config=ModuleConfig(
                module_type=DSPyModuleType.PREDICT, signature="S"
            ),
        )

    def test_repeat_reads_hit_the_scoped_cache(self):
        cm, calls = self._manager_with_counting_store()
        cm.set_agent_config("acme:acme", "summarizer_agent", self._config("v1"))

        first = cm.get_agent_config("acme:acme", "summarizer_agent")
        second = cm.get_agent_config("acme:acme", "summarizer_agent")

        assert first.agent_description == "v1"
        assert second.agent_description == "v1"
        assert calls["get"] == 1, "second read must be served by the TTL cache"

    def test_absent_config_is_negative_cached(self):
        cm, calls = self._manager_with_counting_store()

        assert cm.get_agent_config("acme:acme", "summarizer_agent") is None
        assert cm.get_agent_config("acme:acme", "summarizer_agent") is None

        assert calls["get"] == 1, "the no-config case must be cached too"

    def test_set_invalidates_the_cache(self):
        cm, _ = self._manager_with_counting_store()
        cm.set_agent_config("acme:acme", "summarizer_agent", self._config("v1"))
        assert (
            cm.get_agent_config("acme:acme", "summarizer_agent").agent_description
            == "v1"
        )

        cm.set_agent_config("acme:acme", "summarizer_agent", self._config("v2"))

        assert (
            cm.get_agent_config("acme:acme", "summarizer_agent").agent_description
            == "v2"
        ), "a write must invalidate the cached read"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestTenantInstructionsCaching:
    """get_tenant_instructions_config serves repeat reads from the manager's
    scoped TTL cache. It sits on the per-dispatch enrichment path (every
    memory-aware agent loads tenant instructions before its LLM call), so an
    uncached read cost one synchronous store query per dispatch while the
    sibling scopes were cached. set_config_value invalidates the cache so a
    same-process instructions write is visible immediately."""

    def _manager_with_counting_store(self):
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        calls = {"get": 0}
        real_get = store.get_config

        def counting_get(*args, **kwargs):
            calls["get"] += 1
            return real_get(*args, **kwargs)

        store.get_config = counting_get
        return ConfigManager(store=store), calls

    @staticmethod
    def _set_instructions(cm, text):
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        cm.set_config_value(
            tenant_id="acme:acme",
            scope=ConfigScope.SYSTEM,
            service="tenant_instructions",
            config_key="system_prompt",
            config_value={"text": text, "updated_at": "2026-07-17T00:00:00+00:00"},
        )

    def test_repeat_reads_hit_the_scoped_cache(self):
        cm, calls = self._manager_with_counting_store()
        self._set_instructions(cm, "Always be helpful.")

        first = cm.get_tenant_instructions_config("acme:acme")
        second = cm.get_tenant_instructions_config("acme:acme")

        assert first == {
            "text": "Always be helpful.",
            "updated_at": "2026-07-17T00:00:00+00:00",
        }
        assert second == first
        assert calls["get"] == 1, "second read must be served by the TTL cache"

    def test_absent_instructions_are_negative_cached(self):
        cm, calls = self._manager_with_counting_store()

        assert cm.get_tenant_instructions_config("acme:acme") is None
        assert cm.get_tenant_instructions_config("acme:acme") is None

        assert calls["get"] == 1, "the no-instructions case must be cached too"

    def test_set_config_value_invalidates_the_cache(self):
        cm, _ = self._manager_with_counting_store()
        self._set_instructions(cm, "v1")
        assert cm.get_tenant_instructions_config("acme:acme")["text"] == "v1"

        self._set_instructions(cm, "v2")

        assert cm.get_tenant_instructions_config("acme:acme")["text"] == "v2", (
            "a write must invalidate the cached read"
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestConfigManagerReservedReads:
    """get_agent_config_history and get_config_value — reserved read surfaces
    with no production caller yet; pinned so they stay correct."""

    def _manager(self):
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        return ConfigManager(store=store)

    def test_agent_config_history_returns_versions(self):
        cm = self._manager()
        for i in range(3):
            cm.set_agent_config(
                "acme:acme",
                "summarizer_agent",
                AgentConfig(
                    agent_name="summarizer_agent",
                    agent_version=f"1.0.{i}",
                    agent_description=f"rev {i}",
                    agent_url="http://x",
                    capabilities=["summarization"],
                    skills=[],
                    module_config=ModuleConfig(
                        module_type=DSPyModuleType.PREDICT, signature="S"
                    ),
                ),
            )

        history = cm.get_agent_config_history("acme:acme", "summarizer_agent")

        assert [c.agent_version for c in history] == ["1.0.2", "1.0.1", "1.0.0"]

    def test_get_config_value_returns_value_or_default(self):
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        cm = self._manager()
        cm.set_config_value(
            tenant_id="acme:acme",
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="flag",
            config_value={"on": True},
        )

        assert cm.get_config_value(
            tenant_id="acme:acme",
            scope=ConfigScope.SYSTEM,
            service="runtime",
            config_key="flag",
        ) == {"on": True}
        assert (
            cm.get_config_value(
                tenant_id="acme:acme",
                scope=ConfigScope.SYSTEM,
                service="runtime",
                config_key="absent",
                default="fallback",
            )
            == "fallback"
        )
