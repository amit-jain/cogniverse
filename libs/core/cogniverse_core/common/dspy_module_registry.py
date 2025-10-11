"""
DSPy Module Registry for dynamic module and optimizer instantiation.
"""

import logging
from typing import Any, Callable, Dict, Type

import dspy

from cogniverse_core.config.agent_config import DSPyModuleType, OptimizerType

logger = logging.getLogger(__name__)


class DSPyModuleRegistry:
    """Registry mapping module type strings to DSPy classes"""

    _module_registry: Dict[DSPyModuleType, Type[dspy.Module]] = {
        DSPyModuleType.PREDICT: dspy.Predict,
        DSPyModuleType.CHAIN_OF_THOUGHT: dspy.ChainOfThought,
        DSPyModuleType.REACT: dspy.ReAct,
        DSPyModuleType.MULTI_CHAIN_COMPARISON: dspy.MultiChainComparison,
        DSPyModuleType.PROGRAM_OF_THOUGHT: dspy.ProgramOfThought,
    }

    @classmethod
    def get_module_class(cls, module_type: DSPyModuleType) -> Type[dspy.Module]:
        """
        Get DSPy module class for the given type.

        Args:
            module_type: Module type enum

        Returns:
            DSPy module class

        Raises:
            ValueError: If module type is not registered
        """
        if module_type not in cls._module_registry:
            raise ValueError(
                f"Module type {module_type} not registered. "
                f"Available types: {list(cls._module_registry.keys())}"
            )

        return cls._module_registry[module_type]

    @classmethod
    def create_module(
        cls, module_type: DSPyModuleType, signature: Type[dspy.Signature], **kwargs
    ) -> dspy.Module:
        """
        Create DSPy module instance.

        Args:
            module_type: Type of module to create
            signature: DSPy signature class
            **kwargs: Additional module parameters

        Returns:
            Instantiated DSPy module

        Raises:
            ValueError: If module type is not registered
        """
        module_class = cls.get_module_class(module_type)

        try:
            module = module_class(signature, **kwargs)
            logger.info(
                f"Created DSPy module: {module_type.value} with signature {signature.__name__}"
            )
            return module
        except Exception as e:
            raise RuntimeError(
                f"Failed to create module {module_type.value}: {e}"
            ) from e

    @classmethod
    def register_module(
        cls, module_type: DSPyModuleType, module_class: Type[dspy.Module]
    ):
        """
        Register a custom DSPy module type.

        Args:
            module_type: Module type enum
            module_class: DSPy module class
        """
        cls._module_registry[module_type] = module_class
        logger.info(f"Registered custom module type: {module_type.value}")

    @classmethod
    def list_modules(cls) -> Dict[str, str]:
        """
        List all registered module types.

        Returns:
            Dictionary mapping module type names to class names
        """
        return {
            module_type.value: module_class.__name__
            for module_type, module_class in cls._module_registry.items()
        }


class DSPyOptimizerRegistry:
    """Registry mapping optimizer type strings to DSPy optimizer classes"""

    _optimizer_registry: Dict[OptimizerType, Callable] = {
        OptimizerType.BOOTSTRAP_FEW_SHOT: dspy.BootstrapFewShot,
        OptimizerType.LABELED_FEW_SHOT: dspy.LabeledFewShot,
        OptimizerType.BOOTSTRAP_FEW_SHOT_WITH_RANDOM_SEARCH: dspy.BootstrapFewShotWithRandomSearch,
        OptimizerType.COPRO: dspy.COPRO,
        OptimizerType.MIPRO_V2: dspy.MIPROv2,
    }

    @classmethod
    def get_optimizer_class(cls, optimizer_type: OptimizerType) -> Callable:
        """
        Get DSPy optimizer class for the given type.

        Args:
            optimizer_type: Optimizer type enum

        Returns:
            DSPy optimizer class

        Raises:
            ValueError: If optimizer type is not registered
        """
        if optimizer_type not in cls._optimizer_registry:
            raise ValueError(
                f"Optimizer type {optimizer_type} not registered. "
                f"Available types: {list(cls._optimizer_registry.keys())}"
            )

        return cls._optimizer_registry[optimizer_type]

    @classmethod
    def create_optimizer(cls, optimizer_type: OptimizerType, **kwargs) -> Any:
        """
        Create DSPy optimizer instance.

        Args:
            optimizer_type: Type of optimizer to create
            **kwargs: Optimizer parameters

        Returns:
            Instantiated DSPy optimizer

        Raises:
            ValueError: If optimizer type is not registered
        """
        optimizer_class = cls.get_optimizer_class(optimizer_type)

        try:
            optimizer = optimizer_class(**kwargs)
            logger.info(f"Created DSPy optimizer: {optimizer_type.value}")
            return optimizer
        except Exception as e:
            raise RuntimeError(
                f"Failed to create optimizer {optimizer_type.value}: {e}"
            ) from e

    @classmethod
    def register_optimizer(
        cls, optimizer_type: OptimizerType, optimizer_class: Callable
    ):
        """
        Register a custom DSPy optimizer type.

        Args:
            optimizer_type: Optimizer type enum
            optimizer_class: DSPy optimizer class
        """
        cls._optimizer_registry[optimizer_type] = optimizer_class
        logger.info(f"Registered custom optimizer type: {optimizer_type.value}")

    @classmethod
    def list_optimizers(cls) -> Dict[str, str]:
        """
        List all registered optimizer types.

        Returns:
            Dictionary mapping optimizer type names to class names
        """
        return {
            optimizer_type.value: optimizer_class.__name__
            for optimizer_type, optimizer_class in cls._optimizer_registry.items()
        }
