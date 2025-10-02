"""
Mixin for dynamic DSPy module and optimizer configuration at runtime.
"""

import logging
from typing import Any, Dict, Optional, Type

import dspy

from src.common.agent_config import AgentConfig, ModuleConfig, OptimizerConfig
from src.common.dspy_module_registry import DSPyModuleRegistry, DSPyOptimizerRegistry

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

        # Optimizer instance
        self._optimizer: Optional[Any] = None

        logger.info(
            f"Initialized dynamic DSPy for agent: {config.agent_name} "
            f"with module type: {config.module_config.module_type.value}"
        )

    def _configure_dspy_lm(self, config: AgentConfig):
        """
        Configure DSPy language model.

        Args:
            config: Agent configuration
        """
        lm_params = {"model": config.llm_model}

        if config.llm_base_url:
            lm_params["api_base"] = config.llm_base_url

        if config.llm_api_key:
            lm_params["api_key"] = config.llm_api_key

        if config.llm_max_tokens:
            lm_params["max_tokens"] = config.llm_max_tokens

        lm = dspy.LM(**lm_params)
        dspy.settings.configure(lm=lm)

        logger.info(f"Configured DSPy LM: {config.llm_model}")

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

        self._optimizer = optimizer

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

        # Clear cached optimizer
        self._optimizer = None

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
