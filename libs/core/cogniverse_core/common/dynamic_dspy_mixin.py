"""
Mixin for dynamic DSPy module and optimizer configuration at runtime.
"""

import logging
from typing import Any, Dict, Optional, Type

import dspy

from cogniverse_core.common.dspy_module_registry import (
    DSPyModuleRegistry,
    DSPyOptimizerRegistry,
)
from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    ModuleConfig,
    OptimizerConfig,
)
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.semantic_router import (
    apply_semantic_routing,
    resolve_semantic_router_config,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.dspy.model_format import ensure_provider_prefix

logger = logging.getLogger(__name__)


class DynamicDSPyMixin:
    """
    Mixin providing runtime DSPy module and optimizer configuration.

    Usage:
        class MyAgent(DynamicDSPyMixin):
            def __init__(self):
                config = AgentConfig(...)
                self.initialize_dynamic_dspy(config)
    """

    def initialize_dynamic_dspy(self, config: AgentConfig):
        """
        Initialize DSPy with dynamic configuration.

        Args:
            config: AgentConfig containing DSPy module and optimizer settings
        """
        self.agent_config = config

        # Configure DSPy LM
        self._configure_dspy_lm(config)

        # Store signature classes (must be set by subclass)
        self._signatures: Dict[str, Type[dspy.Signature]] = {}

        # Dynamic modules will be created on demand
        self._dynamic_modules: Dict[str, dspy.Module] = {}

        logger.info(
            f"Initialized dynamic DSPy for agent: {config.agent_name} "
            f"with module type: {config.module_config.module_type.value}"
        )

    def _configure_dspy_lm(self, config: AgentConfig):
        """
        Configure DSPy language model via centralized factory.

        Args:
            config: Agent configuration
        """
        # The LM endpoint/model/key are deployment-runtime, not agent state.
        # An AgentConfig persisted on an earlier deploy carries a stale/None
        # llm_base_url, so re-resolve from config.json's live llm_config.primary
        # (the complete model + in-cluster api_base + no-auth key the global
        # runtime LM uses), falling back to the AgentConfig. Without this the
        # persisted empty endpoint makes litellm silently target the public
        # OpenAI host and 401.
        model = config.llm_model
        base_url = config.llm_base_url
        api_key = config.llm_api_key
        system_config = getattr(self, "system_config", None)
        if system_config is not None and hasattr(system_config, "get_llm_config"):
            try:
                primary = system_config.get_llm_config().primary
                model = primary.model or model
                base_url = primary.api_base or base_url
                api_key = primary.api_key or api_key
            except Exception:  # noqa: BLE001 — degrade to the AgentConfig values
                logger.debug("No llm_config.primary; using AgentConfig LM fields")

        # The model id may be a BARE in-cluster name (e.g. "gemma3:4b"); litellm
        # rejects a bare id with "LLM Provider NOT provided", so attach the
        # openai-compatible provider prefix the in-cluster endpoint speaks.
        endpoint_config = LLMEndpointConfig(
            model=ensure_provider_prefix(model),
            api_base=base_url,
            # litellm's openai client refuses to dispatch with a null key even
            # against an in-cluster vLLM/Ollama endpoint that ignores auth —
            # hand it the no-auth sentinel (matches the chart + worker).
            api_key=api_key or "placeholder-no-auth-needed",
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens or 1000,
        )

        endpoint_config = self._route_through_semantic_router(endpoint_config)

        self._dspy_lm = create_dspy_lm(endpoint_config)
        logger.info(
            f"Created DSPy LM: {endpoint_config.model} @ {endpoint_config.api_base}"
        )

    def _route_through_semantic_router(
        self, endpoint: LLMEndpointConfig
    ) -> LLMEndpointConfig:
        """Route the LM endpoint through the semantic router when it is enabled.

        Reads ``SystemConfig.semantic_router`` via ``self.system_config``'s
        ``get_semantic_router`` accessor. Disabled (the default) returns the
        endpoint unchanged — the direct-to-backend path. When enabled,
        ``api_base`` is rewritten to the semantic router and the tenant-tier
        header is attached.

        ``resolve_semantic_router_config`` guards against a stray/mocked accessor
        (whose auto attributes look truthy) spuriously rewriting the endpoint.
        """
        system_config = getattr(self, "system_config", None)
        router = resolve_semantic_router_config(system_config)
        if not router.enabled:
            return endpoint

        tenant_id = (
            getattr(self, "tenant_id", None)
            or getattr(system_config, "tenant_id", "")
            or ""
        )
        return apply_semantic_routing(
            endpoint=endpoint,
            config=router,
            tenant_id=tenant_id,
        )

    def register_signature(self, name: str, signature: Type[dspy.Signature]):
        """
        Register a DSPy signature for dynamic module creation.

        Args:
            name: Signature name for lookup
            signature: DSPy signature class
        """
        self._signatures[name] = signature
        logger.debug(f"Registered signature: {name}")

    def create_module(
        self, signature_name: str, module_config: Optional[ModuleConfig] = None
    ) -> dspy.Module:
        """
        Create DSPy module dynamically based on configuration.

        Args:
            signature_name: Name of registered signature
            module_config: Optional override for module configuration

        Returns:
            DSPy module instance

        Raises:
            ValueError: If signature not registered
        """
        if signature_name not in self._signatures:
            raise ValueError(
                f"Signature {signature_name} not registered. "
                f"Available: {list(self._signatures.keys())}"
            )

        signature = self._signatures[signature_name]
        config = module_config or self.agent_config.module_config

        # Extract module creation parameters
        module_params = {
            "max_retries": config.max_retries,
            **config.custom_params,
        }

        # Create module using registry
        module = DSPyModuleRegistry.create_module(
            module_type=config.module_type, signature=signature, **module_params
        )

        # Cache for reuse
        self._dynamic_modules[signature_name] = module

        return module

    def get_or_create_module(self, signature_name: str) -> dspy.Module:
        """
        Get cached module or create new one.

        Args:
            signature_name: Name of registered signature

        Returns:
            DSPy module instance
        """
        if signature_name in self._dynamic_modules:
            return self._dynamic_modules[signature_name]

        return self.create_module(signature_name)

    def create_optimizer(
        self, optimizer_config: Optional[OptimizerConfig] = None
    ) -> Any:
        """
        Create DSPy optimizer dynamically based on configuration.

        Args:
            optimizer_config: Optional override for optimizer configuration

        Returns:
            DSPy optimizer instance

        Raises:
            ValueError: If no optimizer config available
        """
        config = optimizer_config or self.agent_config.optimizer_config

        if not config:
            raise ValueError("No optimizer configuration available")

        # Extract optimizer parameters
        optimizer_params = {
            "max_bootstrapped_demos": config.max_bootstrapped_demos,
            "max_labeled_demos": config.max_labeled_demos,
            **config.teacher_settings,
            **config.custom_params,
        }

        # Create optimizer using registry
        optimizer = DSPyOptimizerRegistry.create_optimizer(
            optimizer_type=config.optimizer_type, **optimizer_params
        )

        return optimizer

    def update_module_config(self, module_config: ModuleConfig):
        """
        Update module configuration at runtime.

        Args:
            module_config: New module configuration
        """
        self.agent_config.module_config = module_config

        # Clear cached modules to force recreation with new config
        self._dynamic_modules.clear()

        logger.info(f"Updated module config to: {module_config.module_type.value}")

    def update_optimizer_config(self, optimizer_config: OptimizerConfig):
        """
        Update optimizer configuration at runtime.

        Args:
            optimizer_config: New optimizer configuration
        """
        self.agent_config.optimizer_config = optimizer_config

        logger.info(
            f"Updated optimizer config to: {optimizer_config.optimizer_type.value}"
        )

    def get_module_info(self) -> Dict[str, Any]:
        """
        Get information about current module configuration.

        Returns:
            Dictionary with module configuration details
        """
        return {
            "module_type": self.agent_config.module_config.module_type.value,
            "registered_signatures": list(self._signatures.keys()),
            "cached_modules": list(self._dynamic_modules.keys()),
            "llm_model": self.agent_config.llm_model,
            "temperature": self.agent_config.llm_temperature,
        }

    def get_optimizer_info(self) -> Dict[str, Any]:
        """
        Get information about current optimizer configuration.

        Returns:
            Dictionary with optimizer configuration details
        """
        if not self.agent_config.optimizer_config:
            return {"optimizer_configured": False}

        return {
            "optimizer_configured": True,
            "optimizer_type": self.agent_config.optimizer_config.optimizer_type.value,
            "max_bootstrapped_demos": self.agent_config.optimizer_config.max_bootstrapped_demos,
            "max_labeled_demos": self.agent_config.optimizer_config.max_labeled_demos,
            "num_trials": self.agent_config.optimizer_config.num_trials,
        }
